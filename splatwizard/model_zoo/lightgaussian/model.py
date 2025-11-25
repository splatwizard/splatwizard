import io
import typing

import numpy as np
import torch
from loguru import logger

from splatwizard.modules.densify_mixin import DensificationAndPruneMixin
from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard.modules.loss_mixin import LossMixin
from splatwizard.scheduler import Scheduler, task

from .config import LightGaussianParams, LightGaussianOptimizationParams, LightGaussianPruneOptimizationParams, Stage, \
    LightGaussianDistillOptimizationParams, LightGaussianEncodeOptimizationParams
from splatwizard.modules.gaussian_model import GaussianModel

from splatwizard.utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func,
)
from dataclasses import dataclass

from ...compression.vectree.vectree import VecTreeCodec, VecTreeConfig
from ...metrics.loss_utils import l1_func, union_ssim_func
from ...modules.dataclass import RenderResult, LossPack, ModelContext
from ...modules.render_mixin.compress_renderer import CompressRenderMixin, CompressRenderResult
from ...scene import CameraIterator
from ...utils.pose_utils import gaussian_poses


@dataclass
class LightGaussianDistillRenderResult(RenderResult):
    teacher_rendered_image: torch.Tensor = None

@dataclass
class LightGSContext(ModelContext):
    v_list: typing.Any = None


class LightGaussian(CompressRenderMixin, LossMixin, DensificationAndPruneMixin, GaussianModel):

    def __init__(self, model_param: LightGaussianParams= None):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = model_param.sh_degree if model_param is not None else 3
        self.percent_dense = 0
        self.teacher_model = None
        self.vq_codec: VecTreeCodec = VecTreeCodec()
        self._context = LightGSContext()
        self.setup_functions()

    def register_pre_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: OptimizationParams):
        # assert opt.current_stage == Stage.PRUNE
        # scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
        # scheduler.register_task(range(0, opt.iterations, 1000), task=self.oneup_sh_degree)
        if opt.current_stage == Stage.DISTILL:
            scheduler.register_task(1, task=self.onedown_sh_degree)

    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: LightGaussianOptimizationParams):

        if opt.current_stage == Stage.PRUNE:
            opt = typing.cast(LightGaussianPruneOptimizationParams, opt)
            scheduler.register_task(opt.prune_iterations, task=self.prune_task)
        elif opt.current_stage == Stage.DISTILL:
            scheduler.register_task(opt.iterations, task=self.calc_importance_score)

    def after_setup_hook(self, ppl: PipelineParams, opt: LightGaussianOptimizationParams):
        if opt.current_stage == Stage.DISTILL:
            assert isinstance(opt, LightGaussianDistillOptimizationParams)
            self.teacher_model = LightGaussian()
            self.teacher_model.load(opt.teacher_checkpoint)

            self.max_sh_degree = opt.new_max_sh_degree
        elif opt.current_stage == Stage.ENCODE:
            # config =
            opt = typing.cast(LightGaussianEncodeOptimizationParams, opt)
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


    def training_setup(self, opt: LightGaussianOptimizationParams):
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

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=opt.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=opt.position_lr_delay_mult,
                                                    max_steps=opt.position_lr_max_steps)

    def distill_render(self,
                       viewpoint_cam,
               bg_color: torch.Tensor,
               pipe: PipelineParams,
               opt: LightGaussianDistillOptimizationParams=None,
               step=0,
               scaling_modifier=1.0,
               override_color=None):
        if opt.augmented_view and step % 3:
            viewpoint_cam = gaussian_poses(viewpoint_cam, mean=0, std_dev_translation=0.05, std_dev_rotation=0)
            student_render_pkg = CompressRenderMixin.render(self, viewpoint_cam, bg_color, pipe)
            # student_image = student_render_pkg.rendered_image
            teacher_render_pkg = CompressRenderMixin.render(self.teacher_model,  viewpoint_cam, bg_color, pipe)
            teacher_image = teacher_render_pkg.rendered_image.detach()
        else:
            render_pkg = CompressRenderMixin.render(self, viewpoint_cam, bg_color, pipe)
            student_render_pkg = render_pkg
            teacher_image = CompressRenderMixin.render(self.teacher_model,  viewpoint_cam, bg_color, pipe).rendered_image.detach()

        return LightGaussianDistillRenderResult(
            rendered_image=student_render_pkg.rendered_image,
            viewspace_points=student_render_pkg.viewspace_points,
            visibility_filter=student_render_pkg.visibility_filter,
            radii=student_render_pkg.radii,
            teacher_rendered_image=teacher_image,
        )

    def render(self,
               viewpoint_camera,
               bg_color: torch.Tensor,
               pipe: PipelineParams,
               opt: OptimizationParams=None,
               step=0,
               scaling_modifier=1.0,
               override_color=None):
        if opt is not None and opt.current_stage == Stage.DISTILL:
            return self.distill_render(viewpoint_camera, bg_color, pipe, opt, step, scaling_modifier, override_color)

        return CompressRenderMixin.render(self, viewpoint_camera, bg_color, pipe, opt, step, scaling_modifier, override_color)


    def loss_func(self, viewpoint_cam, render_result: RenderResult, opt: OptimizationParams) -> (torch.Tensor, LossPack):
        if opt.current_stage == Stage.PRUNE:
            return LossMixin.loss_func(self, viewpoint_cam, render_result, opt)
        assert opt.current_stage == Stage.DISTILL

        render_result = typing.cast(LightGaussianDistillRenderResult, render_result)
        Ll1 = l1_func(render_result.rendered_image, render_result.teacher_rendered_image)

        ssim_value = union_ssim_func(render_result.rendered_image, render_result.teacher_rendered_image, using_fused=opt.use_fused_ssim)

        ssim_loss = (1.0 - ssim_value)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss_pack = LossPack(
            l1_loss=Ll1,
            ssim_loss=ssim_loss,
            loss=loss
        )

        return loss, loss_pack

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
    def densify_and_prune_task(self, opt: LightGaussianPruneOptimizationParams, step: int):
        size_threshold = 20 if step > opt.opacity_reset_interval else None
        self.densify_and_prune(opt.densify_grad_threshold, 0.005, self.spatial_lr_scale, size_threshold)


    @task
    def prune_task(self, ppl: PipelineParams, opt: LightGaussianPruneOptimizationParams, iteration: int, cam_iterator: CameraIterator):

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
    def calc_importance_score(self, ppl: PipelineParams, opt: LightGaussianDistillOptimizationParams, cam_iterator: CameraIterator):
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
        sorted_tensor, _ = torch.sort(self.get_opacity, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (self.get_opacity <= value_nth_percentile).squeeze()

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

    def build_attributes(self):

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        return attributes

    def encode(self, path: io.BufferedWriter):
        self.vq_codec.setup(self.build_attributes())
        bs = self.vq_codec.quantize(self.context.v_list.cpu())
        path.write(bs)


    def decode(self, path: io.BufferedReader):
        bs = path.read()
        ply_data = self.vq_codec.dequantize(bs)
        self.load_ply(None, ply_data)



