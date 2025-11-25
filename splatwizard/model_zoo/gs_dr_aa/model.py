import typing
import math

from dataclasses import dataclass

import torch

from splatwizard.modules.densify_mixin import DensificationAndPruneMixin
from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard.modules.loss_mixin import LossMixin
from splatwizard.modules.render_mixin import RenderMixin
from splatwizard.scheduler import Scheduler, task


from .config import GSDRAAModelParams, GSDRAAOptimizationParams
from splatwizard.modules.gaussian_model import GaussianModel
from splatwizard.rasterizer.gs_dr_aa import GSDRAARasterizationSettings, GSDRAAGaussianRasterizer
from splatwizard.utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func,
)
from ...modules.dataclass import RenderResult
from ...utils.sh_utils import eval_sh


@dataclass
class GSDRAARenderResult(RenderResult):
    surf_depth: typing.Union[torch.Tensor, None] = None


class GSDRAA( LossMixin, DensificationAndPruneMixin, GaussianModel):

    def __init__(self, model_param: GSDRAAModelParams):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = model_param.sh_degree
        self.percent_dense = 0
        self.setup_functions()

    def register_pre_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: OptimizationParams):
        scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
        scheduler.register_task(range(0, opt.iterations, 1000), task=self.oneupSHdegree)

    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: GSDRAAOptimizationParams):
        # Densification
        # if iteration < opt.densify_until_iter:
        #     # Keep track of max radii in image-space for pruning
        #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
        #                                                          radii[visibility_filter])
        #     self.add_densification_stats(viewspace_point_tensor, visibility_filter)

        scheduler.register_task(range(opt.densify_until_iter), task=self.add_densification_stats)

        # if iteration < opt.densify_until_iter:
        #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
        #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
        #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
        scheduler.register_task(
            range(opt.densify_from_iter, opt.densify_until_iter, opt.densification_interval),
            task=self.densify_and_prune_task, logging=True
        )
        # if iteration < opt.densify_until_iter:
        #     if iteration % opt.opacity_reset_interval == 0 or (
        #             dataset.white_background and iteration == opt.densify_from_iter):
        #         gaussians.reset_opacity()
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

    # def render(self, viewpoint_camera, bg_color: torch.Tensor,
    #            pipe: PipelineParams, opt: OptimizationParams=None, step=0, scaling_modifier=1.0, override_color=None):
    #     """
    #     Render the scene.
    #
    #     Background tensor (bg_color) must be on GPU!
    #     """
    #
    #     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    #     screenspace_points = torch.zeros_like(
    #         self.xyz, dtype=self.xyz.dtype,
    #         requires_grad=True, device="cuda") + 0
    #     if self._training:
    #         try:
    #             screenspace_points.retain_grad()
    #         except:
    #             pass
    #
    #     # Set up rasterization configuration
    #     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    #     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    #
    #     raster_settings = Trim3DGSRasterizationSettings(
    #         image_height=int(viewpoint_camera.image_height),
    #         image_width=int(viewpoint_camera.image_width),
    #         tanfovx=tanfovx,
    #         tanfovy=tanfovy,
    #         bg=bg_color,
    #         scale_modifier=scaling_modifier,
    #         viewmatrix=viewpoint_camera.world_view_transform,
    #         projmatrix=viewpoint_camera.full_proj_transform,
    #         sh_degree=self.active_sh_degree,
    #         campos=viewpoint_camera.camera_center,
    #         prefiltered=False,
    #         record_transmittance=False,
    #         debug=pipe.debug
    #     )
    #
    #     rasterizer = Trim3DGSGaussianRasterizer(raster_settings=raster_settings)
    #
    #     means3D = self.xyz
    #     means2D = screenspace_points
    #     opacity = self.opacity
    #
    #     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    #     # scaling / rotation by the rasterizer.
    #     scales = None
    #     rotations = None
    #     cov3D_precomp = None
    #     if pipe.compute_cov3D_python:
    #         cov3D_precomp = self.get_covariance(scaling_modifier)
    #     else:
    #         scales = self.scaling
    #         rotations = self.rotation
    #
    #     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    #     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    #     shs = None
    #     colors_precomp = None
    #     if override_color is None:
    #         if pipe.convert_SHs_python:
    #             shs_view = self.features.transpose(1, 2).view(-1, 3, (self.max_sh_degree + 1) ** 2)
    #             dir_pp = (self.xyz - viewpoint_camera.camera_center.repeat(self.features.shape[0], 1))
    #             dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    #             sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
    #             colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #         else:
    #             shs = self.features
    #     else:
    #         colors_precomp = override_color
    #
    #     # Rasterize visible Gaussians to image, obtain their radii (on screen).
    #     color, out_extra_feats, median_depth, radii = rasterizer(
    #         means3D=means3D,
    #         means2D=means2D,
    #         shs=shs,
    #         colors_precomp=colors_precomp,
    #         opacities=opacity,
    #         scales=scales,
    #         rotations=rotations,
    #         cov3D_precomp=cov3D_precomp)
    #
    #     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    #     # They will be excluded from value updates used in the splitting criteria.
    #     # return {"render": rendered_image,
    #     #         "viewspace_points": screenspace_points,
    #     #         "visibility_filter": radii > 0,
    #     #         "radii": radii}
    #     # if opt is not None and opt.use_trained_exposure:
    #     #     exposure = self.get_exposure_from_name(viewpoint_camera.image_name)
    #     #     rendered_image = torch.matmul(
    #     #         rendered_image.permute(1, 2, 0),
    #     #         exposure[:3, :3]
    #     #     ).permute(2, 0, 1) + exposure[:3, 3, None, None]
    #
    #     # invdepths = -invdepths
    #     # invdepths = invdepths - invdepths.min()
    #     return GSDRAARenderResult(
    #         rendered_image=color,
    #         viewspace_points=screenspace_points,
    #         visibility_filter=radii > 0,
    #         radii=radii,
    #         surf_depth=median_depth
    #     )

    def render(self, viewpoint_camera, bg_color: torch.Tensor,
               pipe: PipelineParams, opt: OptimizationParams=None, step=0, scaling_modifier=1.0, override_color=None):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(
            self.xyz, dtype=self.xyz.dtype,
            requires_grad=True, device="cuda") + 0
        if self._training:
            try:
                screenspace_points.retain_grad()
            except:
                pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GSDRAARasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            antialiasing=False,
            debug=pipe.debug
        )

        rasterizer = GSDRAAGaussianRasterizer(raster_settings=raster_settings)

        means3D = self.xyz
        means2D = screenspace_points
        opacity = self.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = self.get_covariance(scaling_modifier)
        else:
            scales = self.scaling
            rotations = self.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
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

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, invdepths = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        # return {"render": rendered_image,
        #         "viewspace_points": screenspace_points,
        #         "visibility_filter": radii > 0,
        #         "radii": radii}
        # if opt is not None and opt.use_trained_exposure:
        #     exposure = self.get_exposure_from_name(viewpoint_camera.image_name)
        #     rendered_image = torch.matmul(
        #         rendered_image.permute(1, 2, 0),
        #         exposure[:3, :3]
        #     ).permute(2, 0, 1) + exposure[:3, 3, None, None]

        # invdepths = -invdepths
        # invdepths = invdepths - invdepths.min()
        return GSDRAARenderResult(
            rendered_image= rendered_image,
            viewspace_points=screenspace_points,
            visibility_filter=radii > 0,
            radii=radii,
            surf_depth=1 / invdepths
        )

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

    @task
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.opacity, torch.ones_like(self.opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # @task
    # def train_statis_task(self, render_result: RenderResult):
    #     self.add_densification_stats(render_result)

    @task
    def densify_and_prune_task(self, opt: GSDRAAOptimizationParams, step: int):
        size_threshold = 20 if step > opt.opacity_reset_interval else None
        self.densify_and_prune(opt.densify_grad_threshold, 0.005, self.spatial_lr_scale, size_threshold)



