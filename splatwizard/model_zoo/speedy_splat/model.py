import math
import torch
from tqdm import tqdm

from splatwizard.modules.densify_mixin import DensificationAndPruneMixin
from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard.modules.loss_mixin import LossMixin
from splatwizard.modules.render_mixin import RenderMixin
from splatwizard.scheduler import Scheduler, task
from splatwizard.rasterizer.speedy import SpeedyGaussianRasterizationSettings, SpeedyGaussianRasterizer


from .config import SpeedySplatModelParams, SpeedySplatOptimizationParams
from splatwizard.modules.gaussian_model import GaussianModel
from ...scene import CameraIterator
from ...utils.sh_utils import eval_sh
from ...modules.dataclass import RenderResult

from splatwizard.utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func,
)


class SpeedySplat(LossMixin, DensificationAndPruneMixin, GaussianModel):

    def __init__(self, model_param: SpeedySplatModelParams):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = model_param.sh_degree
        self.percent_dense = 0
        self.setup_functions()

    def register_pre_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: OptimizationParams):
        scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
        scheduler.register_task(range(0, opt.iterations, 1000), task=self.oneupSHdegree)

    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: SpeedySplatOptimizationParams):
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

        # --- Soft Pruning ---
        scheduler.register_task(
            range(
                opt.prune_from_iter,
                min(opt.densify_until_iter, opt.densify_until_iter),
                opt.prune_interval
            ),
            task=self.soft_prune_task, logging=True
        )

        # if iteration < opt.densify_until_iter:
        #     if iteration % opt.opacity_reset_interval == 0 or (
        #             dataset.white_background and iteration == opt.densify_from_iter):
        #         gaussians.reset_opacity()
        if ppl.white_background:
            scheduler.register_task(opt.densify_from_iter, task=self.reset_opacity, logging=True)
        scheduler.register_task(range(0, opt.densify_until_iter, opt.opacity_reset_interval),
                                task=self.reset_opacity, logging=True)

        # --- Hard Pruning ---
        scheduler.register_task(
            range(
                max(opt.densify_until_iter, opt.prune_from_iter),
                opt.prune_until_iter,
                opt.prune_interval
            ),
            task=self.hard_prune_task, logging=True
        )



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
    def densify_and_prune_task(self, opt: SpeedySplatOptimizationParams , step: int):
        size_threshold = 20 if step > opt.opacity_reset_interval else None
        self.densify_and_prune(opt.densify_grad_threshold, 0.005, self.spatial_lr_scale, size_threshold)

    def render(self, viewpoint_camera, bg_color: torch.Tensor,
               pipe: PipelineParams, opt: OptimizationParams = None, step=0, scaling_modifier=1.0, override_color=None,
               scores=None):
    # def speedy_render(self,
    #                   viewpoint_camera,
    #                   pipe: PipelineParams,
    #                   bg_color: torch.Tensor,
    #                   scores=None,
    #                   scaling_modifier=1.0,
    #                   override_color=None
    #                   ):
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

        raster_settings = SpeedyGaussianRasterizationSettings(
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

        rasterizer = SpeedyGaussianRasterizer(raster_settings=raster_settings)

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

    def score_func(self, view, pipeline, background, scores):

        img_scores = torch.zeros_like(scores)
        img_scores.requires_grad = True

        image = self.render(view, background, pipeline,
                       scores=img_scores).rendered_image

        # Backward computes and stores grad squared values
        # in img_scores's grad
        image.sum().backward()

        scores += img_scores.grad

    def prune_gaussians(self, percent, import_score: list):
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask)

    # def add_densification_stats(self, viewspace_point_tensor, update_filter):
    #     self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
    #     self.denom[update_filter] += 1

    def prune(self, cam_iter, pipe, background, prune_ratio):

        # iter_start = torch.cuda.Event(enable_timing=True)
        # iter_end = torch.cuda.Event(enable_timing=True)
        # torch.cuda.reset_peak_memory_stats()

        # iter_start.record()

        with torch.enable_grad():
            pbar = tqdm(
                total=len(cam_iter),
                desc='Computing Pruning Scores')
            scores = torch.zeros_like(self.opacity)
            for view in cam_iter:
                self.score_func(view, pipe, background,
                           scores)
                pbar.update(1)
            pbar.close()

        self.prune_gaussians(prune_ratio, scores)

        # iter_end.record()
        #
        # # Track peak memory usage (in bytes) and convert to MB
        # peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        # peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        # time_ms = iter_start.elapsed_time(iter_end)
        # time_min = time_ms / 60_000

        # return {
        #     "peak_memory_allocated": peak_memory_allocated,
        #     "peak_memory_reserved": peak_memory_reserved,
        #     "time_min": time_min
        # }

    @task
    def soft_prune_task(self, ppl: PipelineParams, opt: SpeedySplatOptimizationParams, cam_iterator: CameraIterator):
        bg_color = [1, 1, 1] if ppl.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=ppl.device)

        # prune_pkg =
        self.prune(cam_iterator, ppl, background, opt.densify_prune_ratio)

        # prune_time_min += prune_pkg['time_min']
        # prune_peak_memory_allocated = prune_pkg['peak_memory_allocated']
        # prune_peak_memory_reserved = prune_pkg['peak_memory_reserved']

    @task
    def hard_prune_task(self, ppl: PipelineParams, opt: SpeedySplatOptimizationParams, cam_iterator: CameraIterator):
        bg_color = [1, 1, 1] if ppl.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=ppl.device)

        # prune_pkg =
        self.prune(cam_iterator, ppl, background, opt.after_densify_prune_ratio)

        # prune_time_min += prune_pkg['time_min']
        # prune_peak_memory_allocated = prune_pkg['peak_memory_allocated']
        # prune_peak_memory_reserved = prune_pkg['peak_memory_reserved']