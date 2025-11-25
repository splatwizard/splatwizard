
import torch

from splatwizard.modules.densify_mixin import DensificationAndPruneMixin
from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard.modules.loss_mixin import LossMixin
from splatwizard.modules.render_mixin import RenderMixin
from splatwizard.scheduler import Scheduler, task


from .config import GSModelParams, GSOptimizationParams
from splatwizard.modules.gaussian_model import GaussianModel

from splatwizard.utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func,
)


class GSModel(RenderMixin, LossMixin, DensificationAndPruneMixin, GaussianModel):

    def __init__(self, model_param: GSModelParams):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = model_param.sh_degree
        self.percent_dense = 0
        self.setup_functions()

    def register_pre_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: OptimizationParams):
        scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
        scheduler.register_task(range(0, opt.iterations, 1000), task=self.oneupSHdegree)

    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: GSOptimizationParams):
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
    def densify_and_prune_task(self, opt: GSOptimizationParams, step: int):
        size_threshold = 20 if step > opt.opacity_reset_interval else None
        self.densify_and_prune(opt.densify_grad_threshold, 0.005, self.spatial_lr_scale, size_threshold)



