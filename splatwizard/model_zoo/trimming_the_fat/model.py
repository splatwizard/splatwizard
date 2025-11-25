from pathlib import Path

import torch
from loguru import logger

from splatwizard.modules.densify_mixin import DensificationAndPruneMixin
from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard.modules.dataclass import RenderResult
from splatwizard.modules.loss_mixin import LossMixin
from splatwizard.modules.render_mixin import RenderMixin
from splatwizard.scheduler import Scheduler, task
from splatwizard.model_zoo.trimming_the_fat.config import TTFModelParams, TTFOptimizationParams, Stage
from splatwizard.modules.gaussian_model import BaseGaussianModel, GaussianModel
from splatwizard.utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func
)



class TTF(RenderMixin, LossMixin, DensificationAndPruneMixin, GaussianModel):
    def __init__(self, model_param: TTFModelParams):
        super().__init__()
        self.max_sh_degree = model_param.sh_degree
        self.active_sh_degree = model_param.sh_degree


        self.total_entropy = torch.empty(0, requires_grad=True)

        self.optimizer = None
        self.percent_dense = 0

        self.scale_qbit = model_param.opacity_qbit
        self.rotation_qbit = model_param.rotation_qbit
        self.opacity_qbit = model_param.scale_qbit
        self.xyz_qbit = model_param.xyz_qbit
        self.setup_functions()

    def register_pre_task(self, scheduler, ppl: PipelineParams, opt: OptimizationParams):
        scheduler.register_task(
            range(opt.iterations),
            task=self.update_learning_rate,
        )
        # scheduler.register_task(
        #     range(0, opt.iterations, 1000),
        #     task=self.oneupSHdegree, logging=True
        # )
        if Path(ppl.init_checkpoint).suffix == ".pth":
            scheduler.register_task(1, task=self.gradient_aware_prune)

    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: TTFOptimizationParams):
        assert opt.current_stage == Stage.PRUNE, "Only support prune a pre-trained GS model"
        # Densification
        # if iteration < opt.densify_until_iter:
        #     # Keep track of max radii in image-space for pruning
        #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
        #                                                          radii[visibility_filter])
        #     self.add_densification_stats(viewspace_point_tensor, visibility_filter)

        # scheduler.register_task(range(opt.densify_until_iter), task=self.train_statis_task, )
        if Path(ppl.init_checkpoint).suffix != ".pth":
            scheduler.register_task(range(opt.densification_interval), task=self.add_densification_stats)

        # if iteration < opt.densify_until_iter:
        #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
        #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
        #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
        # scheduler.register_task(
        #     range(opt.densify_from_iter, opt.densify_until_iter, opt.densification_interval),
        #     task=self.densify_and_prune_task
        # )


        # if iteration < opt.densify_until_iter:
        #     if iteration % opt.opacity_reset_interval == 0 or (
        #             dataset.white_background and iteration == opt.densify_from_iter):
        #         gaussians.reset_opacity()
        # if ppl.white_background:
        #     scheduler.register_task(opt.densify_from_iter, task=self.reset_opacity)
        # scheduler.register_task(range(0, opt.densify_until_iter, opt.opacity_reset_interval),
        #                         task=self.reset_opacity)
        # # pruning interval
        # if iteration % 500 == 0 and iteration > 30_000 and iteration < 35_000:
        #     # gaussians.prune(pruning_level)
        #     gaussians.gradient_aware_prune(pruning_level)

        scheduler.register_task(
            range(
                opt.gradient_aware_prune_start,
                opt.gradient_aware_prune_end,
                opt.gradient_aware_prune_interval
            ),
            task=self.gradient_aware_prune,
        )

    # @task
    # def oneupSHdegree(self):
    #     if self.active_sh_degree < self.max_sh_degree:
    #         self.active_sh_degree += 1



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
        ''' Learning rate scheduling per step '''
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

    def create_pruning_mask(self, tensor, sparsity):
        """
        Creates a pruning mask with the specified sparsity level.

        Args:
            tensor: The tensor to be pruned.
            sparsity: The desired sparsity level (0.0 to 1.0).

        Returns:
            A PyTorch tensor with the same shape as the input tensor,
            where 1 indicates the element is kept and 0 indicates it is pruned.
        """
        num_elements = tensor.numel()
        num_to_prune = int(num_elements * sparsity)
        prune_indices = torch.randperm(num_elements)[:num_to_prune]
        mask = torch.ones_like(tensor)
        mask.view(-1)[prune_indices] = 0
        return mask.bool()

    def prune(self, prune_level):
        logger.info('Pruning.....')
        logger.info(f'Gaussians before Pruning: {self.opacity.shape}')
        opacity_level = torch.quantile(self.opacity, q=prune_level)
        prune_mask = (self.opacity < opacity_level).squeeze()
        # tensor = torch.randn(prune_mask.shape)
        # mask = self.create_pruning_mask(tensor, 0.6)
        self.prune_points(prune_mask)
        logger.info(f'Gaussians After Pruning: {self.opacity.shape}')
        torch.cuda.empty_cache()

    @task
    def gradient_aware_prune(self,  opt: TTFOptimizationParams):

        prune_level = opt.pruning_level
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads_level = torch.quantile(grads, q=prune_level)
        grads_mask = (grads < grads_level).squeeze()

        logger.info('Gradient Aware Pruning.....')
        logger.info(f'Gaussians before Pruning: {self.opacity.shape}')
        opacity_level = torch.quantile(self.opacity, q=prune_level)
        opacity_mask = (self.opacity < opacity_level).squeeze()
        prune_mask = torch.logical_and(grads_mask, opacity_mask)
        self.prune_points(prune_mask)
        logger.info(f'Gaussians After Pruning: {self.opacity.shape}')
        torch.cuda.empty_cache()

    def first_prune(self, min_opacity):
        logger.info('Pruning.....')
        logger.info(f'Gaussians before Pruning: {self.opacity.shape}')
        opacity_level = torch.quantile(self.opacity, q=0.9)
        prune_mask = (self.opacity < opacity_level).squeeze()
        # tensor = torch.randn(prune_mask.shape)
        # mask = self.create_pruning_mask(tensor, 0.6)
        self.prune_points(prune_mask)
        logger.info(f'Gaussians After Pruning: {self.opacity.shape}')
        torch.cuda.empty_cache()
