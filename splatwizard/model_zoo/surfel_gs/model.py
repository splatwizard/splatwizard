import logging
import math
import typing
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from splatwizard.modules.densify_mixin import DensificationAndPruneMixin
from splatwizard.config import PipelineParams
from splatwizard.scheduler import Scheduler, task
from splatwizard.rasterizer.surfel_gs import SurfelGaussianRasterizer, SurfelGaussianRasterizationSettings

from .config import SurfelGSModelParams, SurfelGSOptimizationParams
from splatwizard.modules.gaussian_model import GaussianModel

from splatwizard.utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func, build_scaling_rotation, build_rotation,
)
from ..._cmod.simple_knn import distCUDA2
from ...metrics.loss_utils import l1_func, ssim_func
from ...modules.dataclass import RenderResult, LossPack
from ...utils.graphics_utils import BasicPointCloud
from ...utils.misc import wrap_str
from ...utils.point_utils import depth_to_normal
from ...utils.sh_utils import eval_sh, RGB2SH


@dataclass
class SurfelGSRenderResult(RenderResult):
    step: int = 0
    rend_alpha: typing.Union[torch.Tensor, None] = None
    rend_normal: typing.Union[torch.Tensor, None] = None
    rend_dist: typing.Union[torch.Tensor, None] = None
    surf_depth: typing.Union[torch.Tensor, None] = None
    surf_normal: typing.Union[torch.Tensor, None] = None


class SurfelGS(DensificationAndPruneMixin, GaussianModel):

    def __init__(self, model_param: SurfelGSModelParams):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = model_param.sh_degree
        self.percent_dense = 0
        self.depth_ratio = model_param.depth_ratio
        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1),
                                        rotation).permute(0, 2, 1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:, :3, :3] = RS
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def register_pre_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: SurfelGSOptimizationParams):
        scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
        scheduler.register_task(range(0, opt.iterations, 1000), task=self.oneupSHdegree)

    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: SurfelGSOptimizationParams):
        scheduler.register_task(
            range(opt.densify_until_iter), task=self.add_densification_stats
        )
        scheduler.register_task(
            range(opt.densify_from_iter, opt.densify_until_iter, opt.densification_interval),
            task=self.densify_and_prune_task, logging=True
        )
        if ppl.white_background:
            scheduler.register_task(opt.densify_from_iter, task=self.reset_opacity, logging=True)
        scheduler.register_task(range(0, opt.densify_until_iter, opt.opacity_reset_interval),
                                task=self.reset_opacity, logging=True)

    @task
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    @task
    def update_learning_rate(self, iteration: int):
        """
        Learning rate scheduling per step
        Args:
            iteration:

        Returns:

        """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def render(self, viewpoint_camera, background, pipe, opt=None, step=0, scaling_modifier=1.0, override_color=None):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self.xyz, dtype=self.xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = SurfelGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            # pipe.debug
        )

        rasterizer = SurfelGaussianRasterizer(raster_settings=raster_settings)

        means3D = self.xyz
        means2D = screenspace_points
        opacity = self.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            # currently don't support normal consistency loss if use precomputed covariance
            splat2world = self.get_covariance(scaling_modifier)
            W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
            near, far = viewpoint_camera.znear, viewpoint_camera.zfar
            ndc2pix = torch.tensor([
                [W / 2, 0, 0, (W - 1) / 2],
                [0, H / 2, 0, (H - 1) / 2],
                [0, 0, far - near, near],
                [0, 0, 0, 1]]).float().cuda().T
            world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
            cov3D_precomp = (splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]]).permute(0, 2, 1).reshape(-1,
                                                                                                           9)  # column major
        else:
            scales = self.scaling
            rotations = self.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        pipe.convert_SHs_python = False
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = self.features.transpose(1, 2).view(-1, 3, (self.max_sh_degree + 1) ** 2)
                dir_pp = (self.xyz - viewpoint_camera.camera_center.repeat(self.features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.features
        else:
            colors_precomp = override_color

        rendered_image, radii, allmap = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        # rets = {"render": rendered_image,
        #         "viewspace_points": means2D,
        #         "visibility_filter": radii > 0,
        #         "radii": radii,
        #         }

        render_result = SurfelGSRenderResult(
            rendered_image=rendered_image,
            viewspace_points=means2D,
            visibility_filter=radii > 0,
            radii=radii,
            step=step
        )

        # additional regularizations
        render_alpha = allmap[1:2]

        # get normal map
        # transform normal from view space to world space
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(2,
                                                                                                                     0,
                                                                                                                     1)

        # get median depth map
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # get expected depth map
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

        # get depth distortion map
        render_dist = allmap[6:7]

        # psedo surface attributes
        # surf depth is either median or expected by setting depth_ratio to 1 or 0
        # for bounded scene, use median depth, i.e., depth_ratio = 1;
        # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
        surf_depth = render_depth_expected * (1 - self.depth_ratio) + self.depth_ratio * render_depth_median

        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
        surf_normal = surf_normal.permute(2, 0, 1)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        surf_normal = surf_normal * render_alpha.detach()

        # rets.update({
        #     'rend_alpha': render_alpha,
        #     'rend_normal': render_normal,
        #     'rend_dist': render_dist,
        #     'surf_depth': surf_depth,
        #     'surf_normal': surf_normal,
        # })

        render_result.rend_alpha = render_alpha
        render_result.rend_normal =render_normal
        render_result.rend_dist = render_dist
        render_result.surf_depth =surf_depth
        render_result.surf_normal = surf_normal

        return render_result


    def loss_func(self, viewpoint_cam, render_result: SurfelGSRenderResult, opt: SurfelGSOptimizationParams):
        image = render_result.rendered_image

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_func(image, gt_image)


        ssim_loss = (1.0 - ssim_func(image, gt_image))

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss # (1.0 - ssim_func(image, gt_image))
        iteration = render_result.step
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_result.rend_dist
        rend_normal = render_result.rend_normal
        surf_normal = render_result.surf_normal
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss
        loss_pack = LossPack(
            l1_loss=Ll1,
            ssim_loss=ssim_loss,
            loss=total_loss
        )
        return total_loss, loss_pack

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale: float, cam_infos=None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        logging.info(wrap_str("Number of points at initialisation : ", fused_point_cloud.shape[0]))

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device="cuda")

    @task
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.opacity, torch.ones_like(self.opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    @task
    def densify_and_prune_task(self, opt: SurfelGSOptimizationParams, step: int):
        size_threshold = 20 if step > opt.opacity_reset_interval else None
        self.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, self.spatial_lr_scale, size_threshold)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)



