import io
import pathlib
import pickle
import tempfile
import typing

import numpy as np
import torch
from loguru import logger
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F


# from fast_pytorch_kmeans import KMeans


from splatwizard.modules.densify_mixin import DensificationAndPruneMixin
from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard.modules.loss_mixin import LossMixin
from splatwizard.scheduler import Scheduler, task

from .config import ChimeraGSModelParams, ChimeraGSOptimizationParams, ChimeraGSPruneOptimizationParams, Stage, \
    ChimeraGSDistillOptimizationParams, ChimeraGSEncodeOptimizationParams
from splatwizard.modules.gaussian_model import GaussianModel

from splatwizard.utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func,
)
from dataclasses import dataclass

from ..lightgaussian.model import LightGaussian
from ...compression.gpcc import encode_anchor, decode_coordinate
from ...compression.morton import get_morton_order
from ...compression.quantizer import Quantize_anchor
from ...compression.vectree.vectree import VecTreeCodec, VecTreeConfig
from ...metrics.loss_utils import l1_func, union_ssim_func
from ...modules.dataclass import RenderResult, LossPack, ModelContext
from ...modules.render_mixin import FlashGSRenderMixin
from ...rasterizer.speedy_tcgs import SpeedyTCGaussianRasterizer, SpeedyTCGaussianRasterizationSettings
from ...rasterizer.speedy import SpeedyGaussianRasterizer, SpeedyGaussianRasterizationSettings
from ...modules.render_mixin.compress_renderer import CompressRenderMixin, CompressRenderResult
from ...scene import CameraIterator
from ...utils.misc import check_tensor
from ...utils.pose_utils import gaussian_poses

import math
from splatwizard.rasterizer.compress import CompressGaussianRasterizationSettings, CompressGaussianRasterizer
from ...utils.sh_utils import eval_sh


@dataclass
class LightGaussianDistillRenderResult(RenderResult):
    teacher_rendered_image: torch.Tensor = None

@dataclass
class LightGSContext(ModelContext):
    v_list: typing.Any = None
    mask: torch.Tensor = None


class ChimeraGS(FlashGSRenderMixin, CompressRenderMixin, LossMixin, DensificationAndPruneMixin, GaussianModel):

    def __init__(self, model_param: ChimeraGSModelParams= None):
        GaussianModel.__init__(self)
        self.active_sh_degree = 0
        self.max_sh_degree = model_param.sh_degree if model_param is not None else 3
        self.percent_dense = 0
        self.teacher_model = None
        vq_config_sh2 = VecTreeConfig(vq_ratio=model_param.sh2_vq_ratio)
        vq_config_sh3 = VecTreeConfig(sh_degree=3, vq_ratio=model_param.sh3_vq_ratio)
        self.vq_codec2: VecTreeCodec = VecTreeCodec(vq_config_sh2)
        self.vq_codec3: VecTreeCodec = VecTreeCodec(vq_config_sh3)
        self._context = LightGSContext()
        self._mask = None
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.setup_functions()
        self.decoded_version = False
        self.current_v_list = None

        FlashGSRenderMixin.__init__(self)

    # def register_pre_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: OptimizationParams):
    #     # assert opt.current_stage == Stage.PRUNE
    #     # scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
    #     # scheduler.register_task(range(0, opt.iterations, 1000), task=self.oneup_sh_degree)
    #     if opt.current_stage == Stage.DISTILL:
    #         # scheduler.register_task(1, task=self.onedown_sh_degree)
    #         scheduler.register_task(1, task=self.update_anchor_bound)

    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: ChimeraGSOptimizationParams):

        if opt.current_stage == Stage.PRUNE:
            opt = typing.cast(ChimeraGSPruneOptimizationParams, opt)
            scheduler.register_task(opt.prune_iterations, task=self.prune_task)
        elif opt.current_stage == Stage.DISTILL:
            scheduler.register_task(opt.iterations, task=self.calc_importance_score)

    def after_setup_hook(self, ppl: PipelineParams, opt: ChimeraGSOptimizationParams):
        if opt.current_stage == Stage.DISTILL:
            assert isinstance(opt, ChimeraGSDistillOptimizationParams)
            self.teacher_model = LightGaussian()
            self.teacher_model.load(opt.teacher_checkpoint)

            self.max_sh_degree = opt.new_max_sh_degree
        elif opt.current_stage == Stage.ENCODE:
            # config =
            opt = typing.cast(ChimeraGSEncodeOptimizationParams, opt)
            codec_config = VecTreeConfig(sh_degree=opt.new_max_sh_degree)
            self.vq_codec = VecTreeCodec(codec_config)


    @task
    def oneup_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    @task
    def onedown_sh_degree(self):
        logger.info('down sh degree')
        if self.active_sh_degree > self.max_sh_degree:
            self.active_sh_degree -= 1
            num_coeffs_to_keep = (self.active_sh_degree + 1) ** 2 - 1
        # ic(num_coeffs_to_keep)
        self._features_rest = self._features_rest.clone().detach()
        self._features_rest = self._features_rest[:,:num_coeffs_to_keep,:]
        self._features_rest.requires_grad = True


    def training_setup(self, opt: ChimeraGSOptimizationParams):
        self.percent_dense = opt.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")



        l = [
            {'params': [self._xyz], 'lr': opt.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': opt.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': opt.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': opt.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': opt.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': opt.rotation_lr, "name": "rotation"}
        ]

        if opt.current_stage == Stage.DISTILL:
            self._mask = nn.Parameter(torch.ones((self._xyz.shape[0], 1), device='cuda').requires_grad_(True) * (-4.5951))
            l.append({'params': [self._mask], 'lr': 1e-3, "name": "mask"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=opt.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=opt.position_lr_delay_mult,
                                                    max_steps=opt.position_lr_max_steps)

    # @property
    # def xyz(self):
    #     if self.decoded_version:
    #         return self._xyz
    #     anchor, quantized_v = Quantize_anchor.apply(self._xyz, self.x_bound_min, self.x_bound_max)
    #     return anchor

    def masked_render(self,
               viewpoint_camera,
               bg_color: torch.Tensor,
               pipe: PipelineParams,
               opt: OptimizationParams = None,
               step=0,
               scaling_modifier=1.0,
               override_color=None):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means

        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
                torch.zeros_like(
                    self.xyz, dtype=self.xyz.dtype, requires_grad=True, device="cuda"
                )
                + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = CompressGaussianRasterizationSettings(
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
            debug=pipe.debug,
            f_count=False,
        )

        rasterizer = CompressGaussianRasterizer(raster_settings=raster_settings)

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
                shs_view = self.features.transpose(1, 2).view(
                    -1, 3, (self.max_sh_degree + 1) ** 2
                )
                dir_pp = self.xyz - viewpoint_camera.camera_center.repeat(
                    self.features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.features
        else:
            colors_precomp = override_color

        if self._mask is not None:
            mask = ((torch.sigmoid(self._mask) > 0.01).float() - torch.sigmoid(self._mask)).detach() + torch.sigmoid(
                self._mask)

            sh_degree2 = 3 + 24
            sh_degree3 = 3 + 45
            # t = shs.std()
            # shs = shs + (torch.rand_like(shs) - 0.5) * shs[:, :9].detach().std() ** 2 * 0.811
            shs = torch.cat([shs[:, :9], shs[:, 9:] * mask.unsqueeze(1)], dim=1)

            t = shs[:10]

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        # return {
        #     "render": rendered_image,
        #     "viewspace_points": screenspace_points,
        #     "visibility_filter": radii > 0,
        #     "radii": radii,
        # }

        return CompressRenderResult(
            rendered_image=rendered_image,
            viewspace_points=screenspace_points,
            visibility_filter=radii > 0,
            radii=radii
        )

    def distill_render(self,
                       viewpoint_cam,
               bg_color: torch.Tensor,
               pipe: PipelineParams,
               opt: ChimeraGSDistillOptimizationParams=None,
               step=0,
               scaling_modifier=1.0,
               override_color=None):
        if opt.augmented_view and step % 3:
            viewpoint_cam = gaussian_poses(viewpoint_cam, mean=0, std_dev_translation=0.05, std_dev_rotation=0)
            student_render_pkg = self.masked_render( viewpoint_cam, bg_color, pipe)
            # student_image = student_render_pkg.rendered_image
            teacher_render_pkg = CompressRenderMixin.render(self.teacher_model,  viewpoint_cam, bg_color, pipe)
            teacher_image = teacher_render_pkg.rendered_image.detach()
        else:
            render_pkg = self.masked_render( viewpoint_cam, bg_color, pipe)
            student_render_pkg = render_pkg
            teacher_image = CompressRenderMixin.render(self.teacher_model,  viewpoint_cam, bg_color, pipe).rendered_image.detach()

        return LightGaussianDistillRenderResult(
            rendered_image=student_render_pkg.rendered_image,
            viewspace_points=student_render_pkg.viewspace_points,
            visibility_filter=student_render_pkg.visibility_filter,
            radii=student_render_pkg.radii,
            teacher_rendered_image=teacher_image,
        )

    def fast_render(self, viewpoint_camera,
               bg_color: torch.Tensor,
               pipe: PipelineParams,
               opt: OptimizationParams=None,
               step=0,
               scaling_modifier=1.0,
               override_color=None, scores=None):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self.xyz, dtype=self.xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = SpeedyTCGaussianRasterizationSettings(
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
            debug=pipe.debug
        )

        rasterizer = SpeedyTCGaussianRasterizer(raster_settings=raster_settings)

        means3D = self.xyz
        means2D = screenspace_points
        opacity = self.opacity

        # set scores to the correct size if not passed in
        if scores is None:
            scores = torch.zeros_like(opacity)

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
        rendered_image, radii, kernel_times = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scores=scores,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return RenderResult(
            rendered_image=rendered_image,
            viewspace_points=screenspace_points,
            visibility_filter=radii > 0,
            radii=radii
        )

    def render(self,
               viewpoint_camera,
               bg_color: torch.Tensor,
               pipe: PipelineParams,
               opt: OptimizationParams=None,
               step=0,
               scaling_modifier=1.0,
               override_color=None):
        if self.decoded_version:
            # return self.fast_render(viewpoint_camera, bg_color, pipe, opt, step, scaling_modifier, override_color)
            return FlashGSRenderMixin.render(self, viewpoint_camera, bg_color, pipe, opt, step, scaling_modifier, override_color)
        if opt is not None and opt.current_stage == Stage.DISTILL:
            return self.distill_render(viewpoint_camera, bg_color, pipe, opt, step, scaling_modifier, override_color)

        return self.masked_render(viewpoint_camera, bg_color, pipe, opt, step, scaling_modifier, override_color)
        # return CompressRenderMixin.render(self, viewpoint_camera, bg_color, pipe, opt, step, scaling_modifier, override_color)


    def loss_func(self, viewpoint_cam, render_result: RenderResult, opt: OptimizationParams) -> (torch.Tensor, LossPack):
        if opt.current_stage == Stage.PRUNE:
            return LossMixin.loss_func(self, viewpoint_cam, render_result, opt)
        assert opt.current_stage == Stage.DISTILL

        render_result = typing.cast(LightGaussianDistillRenderResult, render_result)
        opt = typing.cast(ChimeraGSDistillOptimizationParams, opt)
        Ll1 = l1_func(render_result.rendered_image, render_result.teacher_rendered_image)

        ssim_value = union_ssim_func(render_result.rendered_image, render_result.teacher_rendered_image, using_fused=opt.use_fused_ssim)

        ssim_loss = (1.0 - ssim_value)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

        # check_tensor(self._mask)

        # print( (torch.sigmoid(self._mask) <= 0.01).detach().sum().item() / self._mask.numel())
        loss = loss + opt.lmbda_mask * torch.mean((torch.sigmoid(self._mask)))
        loss_pack = LossPack(
            l1_loss=Ll1,
            ssim_loss=ssim_loss,
            loss=loss
        )

        return loss, loss_pack

    @torch.no_grad()
    def update_anchor_bound(self):
        x_bound_min = (torch.min(self._xyz, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self._xyz, dim=0, keepdim=True)[0]).detach()
        for c in range(x_bound_min.shape[-1]):
            x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 0.8
        for c in range(x_bound_max.shape[-1]):
            x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 0.8
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        logger.info('anchor_bound_updated')

    def capture(self):
        self._context.mask = self._mask
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict() if self.optimizer is not None else None,
            self.spatial_lr_scale,
            self.x_bound_min,
            self.x_bound_max,
            self._context
        )

    def restore(self, model_args, training_args: ChimeraGSOptimizationParams=None):
        if training_args is not None and training_args.current_stage == Stage.ENCODE:
           training_args = None

        try:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.x_bound_min,
                self.x_bound_max,
                context
            ) = model_args
        except:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                # self.x_bound_min,
                # self.x_bound_max,
                context
            ) = model_args
        if training_args is not None:
            self.training_setup(training_args)

        # Since training_setup will reset these parameters, we assign values to them manually
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if self.optimizer is not None:
            self.optimizer.load_state_dict(opt_dict)
        self._context = context  # type(self._context)(**context)
        self._mask = self._context.mask


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
    def densify_and_prune_task(self, opt: ChimeraGSPruneOptimizationParams, step: int):
        size_threshold = 20 if step > opt.opacity_reset_interval else None
        self.densify_and_prune(opt.densify_grad_threshold, 0.005, self.spatial_lr_scale, size_threshold)


    @task
    def prune_task(self, ppl: PipelineParams, opt: ChimeraGSPruneOptimizationParams, iteration: int, cam_iterator: CameraIterator):

        logger.info("Before prune iteration, number of gaussians: " + str(len(self.xyz)))

        # ic("Before prune iteration, number of gaussians: " + str(len(gaussians.get_xyz)))
        bg_color = [1, 1, 1] if ppl.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=ppl.device)

        i = opt.prune_iterations.index(iteration)
        gaussian_list, imp_list = self.prune_list(cam_iterator, ppl, background)

        if opt.prune_type == "important_score":
            self.prune_gaussians(
                (opt.prune_decay ** i) * opt.prune_percent, imp_list
            )
        elif opt.prune_type == "v_important_score":
            # normalize scale
            v_list = self.calculate_v_imp_score(imp_list, opt.v_pow)
            self.prune_gaussians(
                (opt.prune_decay ** i) * opt.prune_percent, v_list
            )
        elif opt.prune_type == "max_v_important_score":
            v_list = imp_list * torch.max(self.scaling, dim=1)[0]
            self.prune_gaussians(
                (opt.prune_decay ** i) * opt.prune_percent, v_list
            )
        elif opt.prune_type == "count":
            self.prune_gaussians(
                (opt.prune_decay ** i) * opt.prune_percent, gaussian_list
            )
        elif opt.prune_type == "opacity":
            self.prune_gaussians(
                (opt.prune_decay ** i) * opt.prune_percent,
                self.opacity.detach(),
            )

        else:
            raise Exception("Unsupportive pruning method")

        # ic("After prune iteration, number of gaussians: " + str(len(gaussians.get_xyz)))
        #
        # ic("after")
        # ic(gaussians.get_xyz.shape)
        # ic(len(gaussians.optimizer.param_groups[0]['params'][0]))
        logger.info("After prune iteration, number of gaussians: " + str(len(self.xyz)))

    @task
    def calc_importance_score(self, ppl: PipelineParams, opt: ChimeraGSDistillOptimizationParams, cam_iterator: CameraIterator):
        logger.info("Calculating importance score")
        # ic("Before prune iteration, number of gaussians: " + str(len(gaussians.get_xyz)))
        bg_color = [1, 1, 1] if ppl.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=ppl.device)

        # i = opt.prune_iterations.index(iteration)
        gaussian_list, imp_list = self.prune_list(cam_iterator, ppl, background)

        # gaussian_list, imp_list = prune_list(
        #     student_gaussians, student_scene, pipe, background
        # )
        v_list = self.calculate_v_imp_score(imp_list, opt.v_pow)
        self.context.v_list = v_list

    def calculate_v_imp_score(self, imp_list, v_pow):
        """
        :param gaussians: A data structure containing Gaussian components with a get_scaling method.
        :param imp_list: The importance scores for each Gaussian component.
        :param v_pow: The power to which the volume ratios are raised.
        :return: A list of adjusted values (v_list) used for pruning.
        """
        # Calculate the volume of each Gaussian component
        volume = torch.prod(self.scaling, dim=1)
        # Determine the kth_percent_largest value
        index = int(len(volume) * 0.9)
        sorted_volume, _ = torch.sort(volume, descending=True)
        kth_percent_largest = sorted_volume[index]
        # Calculate v_list
        v_list = torch.pow(volume / kth_percent_largest, v_pow)
        v_list = v_list * imp_list
        return v_list

    def prune_list(self, cam_iter, pipe, background):
        # viewpoint_stack = scene.getTrainCameras().copy()
        gaussian_list, imp_list = None, None
        # viewpoint_cam = viewpoint_stack.pop()
        viewpoint_cam = next(cam_iter)
        render_pkg: CompressRenderResult = self.count_render(viewpoint_cam, pipe, background)
        gaussian_list, imp_list = (
            render_pkg.gaussians_count,
            render_pkg.important_score,
        )

        # ic(dataset.model_path)
        for iteration, viewpoint_cam in enumerate(cam_iter):
            # Pick a random Camera
            # prunning
            # viewpoint_cam = viewpoint_stack.pop()
            render_pkg = self.count_render(viewpoint_cam, pipe, background)
            # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gaussians_count, important_score = (
                render_pkg.gaussians_count.detach(),
                render_pkg.important_score.detach(),
            )
            gaussian_list += gaussians_count
            imp_list += important_score
            # gc.collect()
        return gaussian_list, imp_list

    def densify(self, max_grad, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        torch.cuda.empty_cache()

    def prune_opacity(self, percent):
        sorted_tensor, _ = torch.sort(self.opacity, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (self.opacity <= value_nth_percentile).squeeze()

        # big_points_vs = self.max_radii2D > max_screen_size
        # big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        # prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune_gaussians(self, percent, import_score: list):
        # ic(import_score.shape)
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def build_attributes(self, sh_degree=3):

        if sh_degree == 3:
            sh_dim = 3 + 45

        elif sh_degree == 2:
            sh_dim = 3 + 24
        else:
            assert False

        sh_offset = sh_dim // 3

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest[:, :sh_offset - 1, :].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        return attributes

    @torch.no_grad()
    def _sort_morton(self):
        xyz_q = (
                (2 ** 21 - 1)
                * (self._xyz - self._xyz.min(0).values)
                / (self._xyz.max(0).values - self._xyz.min(0).values)
        ).long()
        order = get_morton_order(xyz_q).sort().indices
        self.reorder_gaussians(order)


    def reorder_gaussians(self, order):
        self._xyz = nn.Parameter(self._xyz[order], requires_grad=True)
        self._opacity = nn.Parameter(self._opacity[order], requires_grad=True)

        self._features_rest = nn.Parameter(
            self._features_rest[order], requires_grad=True
        )
        self._features_dc = nn.Parameter(
            self._features_dc[order], requires_grad=True
        )

        self._scaling = nn.Parameter(self._scaling[order], requires_grad=True)
        self._rotation = nn.Parameter(self._rotation[order], requires_grad=True)

        if self.current_v_list is not None:
            self.current_v_list = self.current_v_list[order]

    def mask_prune(self, mask):


        self._xyz = self._xyz[mask].clone()
        self._opacity = self._opacity[mask].clone()
        self._scaling = self._scaling[mask].clone()
        self._rotation = self._rotation[mask].clone()
        # self._mask = self._mask[mask]

        self._features_rest = self._features_rest[mask].clone()
        self._features_dc = self._features_dc[mask].clone()
        self.current_v_list = self.context.v_list[mask].clone()


    def encode(self, path: io.BufferedWriter):
        self.decoded_version = True
        backup = self.capture()

        mask = (torch.sigmoid(self._mask) <= 0.01).squeeze()

        features_part1 = self.features[mask]
        # print(mask.shape)
        self.mask_prune(mask)
        self._sort_morton()



        # self.mask_prune(selection)
        self.vq_codec2.setup(self.build_attributes(sh_degree=2))
        bs1 = self.vq_codec2.quantize(self.current_v_list.cpu())

        # self.max_sh_degree = 2
        # ply_data = self.vq_codec2.dequantize(bs1)
        # self.load_ply(None, ply_data)

        # features_part_new = self.features
        # plt.hist(features_part1[:, :9, :].flatten().cpu().detach().numpy(), bins=100)
        # plt.show()
        #
        # plt.hist(features_part_new.flatten().cpu().detach().numpy(), bins=100)
        # plt.show()
        #
        # plt.hist(features_part_new.flatten().cpu().detach().numpy() - features_part1[:, :9,:].flatten().cpu().detach().numpy(), bins=100)
        # plt.show()
        # exit()

        self.restore(backup)
        self.current_v_list = None

        self.mask_prune(~mask)
        self._sort_morton()

        self.vq_codec3.setup(self.build_attributes(sh_degree=3))
        bs2 = self.vq_codec3.quantize(self.current_v_list.cpu())
        #
        #
        pickle.dump([bs1, bs2], path)
        # path.write(bs)


    def concat_params(self, part1, part2):
        # return (
        #     self.active_sh_degree,
        #     self._xyz,
        #     self._features_dc,
        #     self._features_rest,
        #     self._scaling,
        #     self._rotation,
        #     self._opacity,
        #     self.max_radii2D,
        #     self.xyz_gradient_accum,
        #     self.denom,
        #     self.optimizer.state_dict() if self.optimizer is not None else None,
        #     self.spatial_lr_scale,
        #     self._context
        # )

        self._xyz = torch.concat([part1[1], part2[1]])
        self._features_dc = torch.concat([part1[2], part2[2]])

        sh2_feat_rest = part1[3]
        sh3_feat_rest = part2[3]

        sh2_feat_rest = torch.concat([
            sh2_feat_rest,
            torch.zeros(sh2_feat_rest.shape[0], sh3_feat_rest.shape[1] - sh2_feat_rest.shape[1], 3, device=sh2_feat_rest.device)
        ], dim=1)

        # feat_rest = torch.cat([sh2_feat_rest, sh3_feat_rest], dim=0)
        self._features_rest = torch.cat([sh2_feat_rest, sh3_feat_rest], dim=0)



        self._scaling = torch.concat([part1[4], part2[4]])
        self._rotation = torch.concat([part1[5], part2[5]])
        self._opacity = torch.concat([part1[6], part2[6]])




    def decode(self, path: io.BufferedReader):
        self.decoded_version = True
        bs1, bs2 =  pickle.load(path)

        self.max_sh_degree = 2
        ply_data = self.vq_codec2.dequantize(bs1)
        self.load_ply(None, ply_data)

        part1 = self.capture()


        self.max_sh_degree = 3
        ply_data = self.vq_codec3.dequantize(bs2)
        self.load_ply(None, ply_data)

        part2 = self.capture()

        self.concat_params(part1, part2)

        self._mask = None
        # self._sort_morton()
        # order = torch.randperm(self.xyz.shape[0], device=self.xyz.device)
        # self.reorder_gaussians(order)



