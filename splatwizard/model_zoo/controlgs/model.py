from dataclasses import dataclass

import torch
from loguru import logger
from torch import nn

from splatwizard.modules.densify_mixin import DensificationAndPruneMixin
from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard.modules.loss_mixin import LossMixin
from splatwizard.modules.render_mixin import RenderMixin
from splatwizard.scheduler import Scheduler, task


from .config import ControlGSModelParams, ControlGSOptimizationParams
from splatwizard.modules.gaussian_model import GaussianModel

from splatwizard.utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func, build_rotation,
)
from ..._cmod.fused_ssim import fused_ssim
from ...metrics.loss_utils import l1_func, ssim_func
from ...modules.dataclass import ModelContext, RenderResult, LossPack
from ...utils.graphics_utils import BasicPointCloud


@dataclass
class ControlGSContext(ModelContext):
    no_filtering_until: int = None
    before_pruned_gaussian_count: int = None
    prune_change_threshold: int = None
    lambda_opacity: float = None


class ControlGS(RenderMixin, DensificationAndPruneMixin, GaussianModel):

    def __init__(self, model_param: ControlGSModelParams):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = model_param.sh_degree
        self.percent_dense = 0
        self.setup_functions()
        self._split_count = torch.Tensor([])
        self._min_split_count = 0
        self._densification_level = 0
        self._context = ControlGSContext(
            no_filtering_until=0
        )

        self.exposure_mapping = None
        self.pretrained_exposures = None

        self._exposure = torch.empty(0)

    @property
    def split_count(self):
        return self._split_count

    @property
    def min_split_count(self):
        return self._min_split_count

    @property
    def densification_level(self):
        return self._densification_level

    @property
    def exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self.exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]

    def register_pre_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: OptimizationParams):
        scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
        scheduler.register_task(range(0, opt.iterations, 1000), task=self.oneup_sh_degree)

    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: ControlGSOptimizationParams):
        scheduler.register_task(range(0, opt.iterations, 100), task=self.prune_model )
        scheduler.register_task(range(opt.iterations), task=self.densify)

    @task
    def oneup_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, opt: ControlGSOptimizationParams):
        # self.percent_dense = opt.percent_dense
        # self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': opt.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': opt.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': opt.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': opt.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': opt.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': opt.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.exposure_optimizer = torch.optim.Adam([self.exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=opt.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=opt.position_lr_delay_mult,
                                                    max_steps=opt.position_lr_max_steps)

        self.exposure_scheduler_args = get_expon_lr_func(opt.exposure_lr_init,
                                                         opt.exposure_lr_final,
                                                         lr_delay_steps=opt.exposure_lr_delay_steps,
                                                         lr_delay_mult=opt.exposure_lr_delay_mult,
                                                         max_steps=opt.iterations)


        # Number of Gaussian points remaining after the last pruning
        self.context.before_pruned_gaussian_count = self.xyz.shape[0]
        self.context.num_removed = float('inf')
        # Initialize with the delay interval for consistent handling
        self.context.no_filtering_until = opt.post_densification_filter_delay
        self.context.prune_change_threshold = opt.prune_change_threshold
        self.context.lambda_opacity = opt.lambda_opacity

    def optimizer_step(self, render_result: RenderResult, opt: ControlGSOptimizationParams, step: int):
        self.exposure_optimizer.step()
        self.exposure_optimizer.zero_grad(set_to_none=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, cam_infos):
        GaussianModel.create_from_pcd(self, pcd, spatial_lr_scale)
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        self._split_count = torch.zeros((self._xyz.shape[0],), dtype=torch.long, device=self._xyz.device)

    @task
    def update_learning_rate(self, iteration: int):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def loss_func(self, viewpoint_cam, render_result: RenderResult, opt: ControlGSOptimizationParams) -> (torch.Tensor, LossPack):
        gt_image = viewpoint_cam.original_image
        Ll1 = l1_func(render_result.rendered_image, gt_image)

        if opt.use_fused_ssim:
            ssim_value = fused_ssim(render_result.rendered_image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim_func(render_result.rendered_image, gt_image)

        ssim_loss = (1.0 - ssim_value)

        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

        opacity = self.opacity
        opacity_l1 = opacity.sum()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) + self.context.lambda_opacity * opacity_l1

        loss_pack = LossPack(
            l1_loss=Ll1,
            ssim_loss=ssim_loss,
            loss=loss
        )

        return loss, loss_pack

    @task
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.opacity, torch.ones_like(self.opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # @task
    # def train_statis_task(self, render_result: RenderResult):
    #     self.add_densification_stats(render_result)

    # @task
    # def densify_and_prune_task(self, opt: ControlGSOptimizationParams, step: int):
    #     size_threshold = 20 if step > opt.opacity_reset_interval else None
    #     self.densify_and_prune(opt.densify_grad_threshold, 0.005, self.spatial_lr_scale, size_threshold)

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._split_count = self._split_count[valid_points_mask]  # 同步更新分裂计数
        if self._split_count.nelement() > 0:
            self._min_split_count = self._split_count.min()

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_split_count
    ):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._split_count = torch.cat([self._split_count, new_split_count], dim=0)
        self._min_split_count = self._split_count.min()

    def prune_by_opacity(self, min_opacity, extent=None):
        prune_mask = (self.opacity < min_opacity).squeeze()
        if extent is not None:
            big_points_ws = self.scaling.max(dim=1).values > 1.0 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def octree_densify(self, batchsize):
        """
        Split up to `batchsize` Gaussians from the Gaussians with the least split count,
        and maintain a `_split_count` in the attributes to record the split count.
        """

        # find all indices where the split count is equal to the minimum
        candidate_indices = (self._split_count == self._min_split_count).nonzero().squeeze(-1)

        # randomly sample up to `batchsize` from candidate_indices
        perm = torch.randperm(candidate_indices.numel(), device=candidate_indices.device)
        chosen_count = min(batchsize, candidate_indices.numel())
        batch_indices = candidate_indices[perm[:chosen_count]]

        # extract the properties of the batch of parent Gaussians
        positions_batch = self.xyz.detach()[batch_indices]
        features_dc_batch = self.features_dc.detach()[batch_indices]
        features_rest_batch = self.features_rest.detach()[batch_indices]
        rotation_raw_batch = self.rotation.detach()[batch_indices]
        scaling_raw_batch = self.scaling.detach()[batch_indices]
        opacity_raw_batch = self.opacity.detach()[batch_indices]
        split_count_batch = self.split_count[batch_indices]

        # scaling / opacity scaling for the children
        new_scaling_raw_batch = self.scaling_inverse_activation(scaling_raw_batch / (0.8 * 2.0))
        new_opacity_raw_batch = self.inverse_opacity_activation(1 - torch.sqrt(1 - opacity_raw_batch))

        # parent rotation: quaternion -> 3x3 matrix
        R_mats_batch = build_rotation(rotation_raw_batch)  # (B, 3, 3)

        # generate 8 local offsets and map them to global coordinates
        offsets_local = torch.tensor([
            [dx, dy, dz]
            for dx in [-0.25, 0.25]
            for dy in [-0.25, 0.25]
            for dz in [-0.25, 0.25]
        ], device=positions_batch.device)  # (8, 3)

        B = positions_batch.shape[0]  # batch_size
        offsets_local = offsets_local.unsqueeze(0).expand(B, -1, -1)  # (B, 8, 3)

        scaled_offsets = offsets_local * scaling_raw_batch.unsqueeze(1)  # (B, 8, 3)
        scaled_offsets_2d = scaled_offsets.reshape(B * 8, 3)  # (B*8, 3)

        R_mats_2d = R_mats_batch.unsqueeze(1) \
            .expand(-1, 8, -1, -1) \
            .reshape(B * 8, 3, 3)  # (B*8, 3, 3)

        rotated_offsets_2d = torch.bmm(
            R_mats_2d,
            scaled_offsets_2d.unsqueeze(-1)
        ).squeeze(-1)  # (B*8, 3)

        rotated_offsets = rotated_offsets_2d.view(B, 8, 3)  # (B, 8, 3)

        new_positions = positions_batch.unsqueeze(1) + rotated_offsets  # (B, 8, 3)
        new_positions = new_positions.reshape(-1, 3)  # (B*8, 3)

        # copy the rest of the properties
        new_features_dc = features_dc_batch.unsqueeze(1) \
            .expand(-1, 8, -1, -1) \
            .reshape(-1, features_dc_batch.shape[1], features_dc_batch.shape[2])

        new_features_rest = features_rest_batch.unsqueeze(1) \
            .expand(-1, 8, -1, -1) \
            .reshape(-1, features_rest_batch.shape[1], features_rest_batch.shape[2])

        new_scaling = new_scaling_raw_batch.unsqueeze(1) \
            .expand(-1, 8, -1) \
            .reshape(-1, new_scaling_raw_batch.shape[1])

        new_rotation = rotation_raw_batch.unsqueeze(1) \
            .expand(-1, 8, -1) \
            .reshape(-1, rotation_raw_batch.shape[1])

        new_opacity = new_opacity_raw_batch.unsqueeze(1) \
            .expand(-1, 8, -1) \
            .reshape(-1, new_opacity_raw_batch.shape[1])

        # split count of the children = parent + 1
        new_split_count = (split_count_batch + 1) \
            .unsqueeze(1).expand(-1, 8).reshape(-1)  # (B*8,)

        # delete the batch of parent Gaussians (as they are split)
        N = self._xyz.shape[0]
        prune_mask = torch.zeros((N,), dtype=torch.bool, device=self._xyz.device)
        prune_mask[batch_indices] = True

        # delete the selected parent Gaussians
        self.prune_points(prune_mask)

        # add the new children to the model
        self.densification_postfix(
            new_positions,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_split_count
        )

        torch.cuda.empty_cache()

    @task
    def prune_model(self, opt: ControlGSOptimizationParams, iteration: int):
        # Prune per 100
        if iteration > self._context.no_filtering_until:
            self.prune_by_opacity(opt.opacity_threshold, self.spatial_lr_scale)
            self.context.after_pruned_gaussian_count = self.xyz.shape[0]
            self.context.num_removed = self.context.before_pruned_gaussian_count - self.context.after_pruned_gaussian_count
            self.context.before_pruned_gaussian_count = self.context.after_pruned_gaussian_count
            logger.info(
                "[ITER {}] Number of Gaussians after pruning: {} (Removed: {})".format(
                    iteration, self.xyz.shape[0], self.context.num_removed)
            )
            # tqdm.write("[ITER {}] Number of Gaussians after pruning: {} (Removed: {})".format(iteration,
            #                                                                                   gaussians.get_xyz.shape[
            #                                                                                       0], num_removed))

    @task
    def densify(self, opt: ControlGSOptimizationParams, iteration: int):
        # dynamic pruning threshold
        self.context.prune_change_threshold = 2_000 if (
                self.split_count == self.min_split_count).all() else float('inf')

        # calculate the number of Gaussians after pruning
        if (self.min_split_count < opt.max_densification
                and self.context.num_removed < self.context.prune_change_threshold):
            # densify the Gaussians
            self.octree_densify(opt.densification_batch_size)

            # calculate and display the number of Gaussians after densification
            self.context.num_gaussians_after_densification = self.xyz.shape[0]

            # tqdm.write("Number of Gaussians after densification: {}".format(num_gaussians_after_densification))
            logger.info("Number of Gaussians after densification: {}".format(
                self.context.num_gaussians_after_densification
            ))

            # Display densification progress
            if (self.split_count == self.min_split_count).all():
                # tqdm.write(
                #     "[ITER {}] Completed {} rounds of densification.".format(iteration, gaussians.get_min_split_count))
                log_str = "[ITER {}] Completed {} rounds of densification.".format(iteration, self.min_split_count)
            else:
                # tqdm.write("Remaining {} Gaussians to densify in round {}.".format(
                #     (gaussians.get_split_count == gaussians.get_min_split_count).sum(),
                #     gaussians.get_min_split_count + 1))
                log_str = "Remaining {} Gaussians to densify in round {}.".format(
                    (self.split_count == self.min_split_count).sum(),
                    self.min_split_count + 1)

            logger.info(log_str)

            # Reset parameters
            self.context.before_pruned_gaussian_count = self.xyz.shape[0]
            self.context.num_removed = float('inf')

            # After densification, skip opacity filtering for 'no_filtering_until' iterations
            self.context.no_filtering_until = iteration + opt.post_densification_filter_delay

        # If the number of removed Gaussians is below the threshold after max densification rounds,
        # set opacity regularization to zero
        if (self.min_split_count == opt.max_densification
                and self.context.num_removed < opt.prune_change_threshold):
            self.context.lambda_opacity = 0




