import copy
import math

import torch
from loguru import logger

from splatwizard.modules.densify_mixin import DensificationAndPruneMixin
from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard.modules.loss_mixin import LossMixin
from splatwizard.scheduler import Scheduler, task

from .config import PUP3DGSModelParams, PUP3DGSOptimizationParams, Stage
from splatwizard.modules.gaussian_model import GaussianModel

from splatwizard.utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func,
)

from ...modules.dataclass import RenderResult
from ...modules.render_mixin.compress_renderer import CompressRenderMixin, CompressRenderResult
from ...scene import CameraIterator
from splatwizard.rasterizer.pup_fisher import PUPFisherGaussianRasterizationSettings, PUPFisherGaussianRasterizer
from ...utils.sh_utils import eval_sh


class PUP3DGS(CompressRenderMixin, LossMixin, DensificationAndPruneMixin, GaussianModel):

    def __init__(self, model_param: PUP3DGSModelParams):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = model_param.sh_degree
        self.percent_dense = 0
        self.setup_functions()

    def register_pre_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: OptimizationParams):
        assert opt.current_stage == Stage.PRUNE
        # scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
        # scheduler.register_task(range(0, opt.iterations, 1000), task=self.oneup_sh_degree)

    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: PUP3DGSOptimizationParams):

        assert opt.current_stage == Stage.PRUNE
        scheduler.register_task(opt.prune_iterations, task=self.prune_task)


    @task
    def oneup_sh_degree(self):
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

    #
    # Copyright (C) 2023, Inria
    # GRAPHDECO research group, https://team.inria.fr/graphdeco
    # All rights reserved.
    #
    # This software is free for non-commercial, research and evaluation use
    # under the terms of the LICENSE.md file.
    #
    # For inquiries contact  george.drettakis@inria.fr
    #

    def fisher_render(self,
                      viewpoint_camera,
                    bg_color: torch.Tensor,
                    pipe: PipelineParams,
                    fishers,
                    scaling_modifier=1.0,
                    override_color=None):
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

        raster_settings = PUPFisherGaussianRasterizationSettings(
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

        rasterizer = PUPFisherGaussianRasterizer(raster_settings=raster_settings)

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
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            fishers=fishers,
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        # return {"render": rendered_image,
        #         "viewspace_points": screenspace_points,
        #         "visibility_filter": radii > 0,
        #         "radii": radii}

        return RenderResult(
            rendered_image=rendered_image,
            viewspace_points=screenspace_points,
            visibility_filter=radii > 0,
            radii=radii
        )

    def pool_fisher_cuda(self, view_idx, view, pipeline, background,
                         fishers, resolution):

        sym_fishers = torch.zeros(
            (fishers.shape[0], 21), dtype=fishers.dtype, device=fishers.device)
        sym_fishers.requires_grad = True

        # set patch resolution as view downsample
        view_copy = copy.deepcopy(view)
        view_copy.image_height = math.ceil(view.image_height / resolution)
        view_copy.image_width = math.ceil(view.image_width / resolution)

        render_result = self.fisher_render(view_copy, background, pipeline, sym_fishers)
        image = render_result.rendered_image

        # Backward computes and stores symetric fisher values
        # in sym_fisher's grad
        image_sum = image.sum()
        image_sum.backward()

        fishers[:, 0, 0] += sym_fishers.grad[:, 0]
        fishers[:, 0, 1] += sym_fishers.grad[:, 1]
        fishers[:, 0, 2] += sym_fishers.grad[:, 2]
        fishers[:, 0, 3] += sym_fishers.grad[:, 3]
        fishers[:, 0, 4] += sym_fishers.grad[:, 4]
        fishers[:, 0, 5] += sym_fishers.grad[:, 5]

        fishers[:, 1, 0] += sym_fishers.grad[:, 1]
        fishers[:, 1, 1] += sym_fishers.grad[:, 6]
        fishers[:, 1, 2] += sym_fishers.grad[:, 7]
        fishers[:, 1, 3] += sym_fishers.grad[:, 8]
        fishers[:, 1, 4] += sym_fishers.grad[:, 9]
        fishers[:, 1, 5] += sym_fishers.grad[:, 10]

        fishers[:, 2, 0] += sym_fishers.grad[:, 2]
        fishers[:, 2, 1] += sym_fishers.grad[:, 7]
        fishers[:, 2, 2] += sym_fishers.grad[:, 11]
        fishers[:, 2, 3] += sym_fishers.grad[:, 12]
        fishers[:, 2, 4] += sym_fishers.grad[:, 13]
        fishers[:, 2, 5] += sym_fishers.grad[:, 14]

        fishers[:, 3, 0] += sym_fishers.grad[:, 3]
        fishers[:, 3, 1] += sym_fishers.grad[:, 8]
        fishers[:, 3, 2] += sym_fishers.grad[:, 12]
        fishers[:, 3, 3] += sym_fishers.grad[:, 15]
        fishers[:, 3, 4] += sym_fishers.grad[:, 16]
        fishers[:, 3, 5] += sym_fishers.grad[:, 17]

        fishers[:, 4, 0] += sym_fishers.grad[:, 4]
        fishers[:, 4, 1] += sym_fishers.grad[:, 9]
        fishers[:, 4, 2] += sym_fishers.grad[:, 13]
        fishers[:, 4, 3] += sym_fishers.grad[:, 16]
        fishers[:, 4, 4] += sym_fishers.grad[:, 18]
        fishers[:, 4, 5] += sym_fishers.grad[:, 19]

        fishers[:, 5, 0] += sym_fishers.grad[:, 5]
        fishers[:, 5, 1] += sym_fishers.grad[:, 10]
        fishers[:, 5, 2] += sym_fishers.grad[:, 14]
        fishers[:, 5, 3] += sym_fishers.grad[:, 17]
        fishers[:, 5, 4] += sym_fishers.grad[:, 19]
        fishers[:, 5, 5] += sym_fishers.grad[:, 20]

        del sym_fishers

        return fishers

    @task
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.opacity, torch.ones_like(self.opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # @task
    # def train_statis_task(self, render_result: RenderResult):
    #     self.add_densification_stats(render_result)

    # @task
    # def densify_and_prune_task(self, opt: PUP3DGSOptimizationParams, step: int):
    #     size_threshold = 20 if step > opt.opacity_reset_interval else None
    #     self.densify_and_prune(opt.densify_grad_threshold, 0.005, self.spatial_lr_scale, size_threshold)


    @task
    def prune_task(self, ppl: PipelineParams, opt: PUP3DGSOptimizationParams, iteration: int, cam_iterator: CameraIterator):
        logger.info('Executing prune_task')
        prune_idx = opt.prune_iterations.index(iteration)
        prune_percent = opt.prune_percent[prune_idx]

        bg_color = [1, 1, 1] if ppl.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=ppl.device)

        if opt.prune_type == 'fisher':
            # Compute and save CUDA Fisher
            N = self.xyz.shape[0]
            device = self.xyz.device
            with torch.enable_grad():
                fishers = torch.zeros(N, 6, 6, device=device).float()
                # for view_idx, view in tqdm(
                #         enumerate(scene.getTrainCameras()), total=len(scene.getTrainCameras()),
                #         desc="Computing Fisher..."):
                #     pool_fisher_cuda(
                #         view_idx, view, gaussians, pipe, background,
                #         fishers, opt.fisher_resolution
                #     )
                for view_idx, viewpoint_cam in enumerate(cam_iterator):
                    self.pool_fisher_cuda(
                                view_idx, viewpoint_cam, ppl, background,
                                fishers, opt.fisher_resolution
                            )

            # torch.save(fishers, scene.model_path + f'/fisher_iter{iteration}.pt')
            # Prune using log determinant
            fishers_sv = torch.linalg.svdvals(fishers)
            fishers_log_dets = torch.log(fishers_sv).sum(dim=1)
            self.prune_gaussians(
                prune_percent,
                fishers_log_dets
            )
        # Borrowed from https://github.com/VITA-Group/LightGaussian/blob/main/prune_finetune.py#L222
        # elif opt.prune_type == 'v_important_score':
        #         gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
        #         v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
        #         gaussians.prune_gaussians(
        #             prune_percent,
        #             v_list
        #         )
        elif opt.prune_type == "v_important_score":
            gaussian_list, imp_list = self.prune_list(cam_iterator, ppl, background)
            # normalize scale
            v_list = self.calculate_v_imp_score(imp_list, opt.v_pow)

            self.prune_gaussians(
                prune_percent, v_list
            )
        else:
            raise Exception("Unsupportive pruning method")

        logger.info(f"Prune Round {prune_idx}: Number of Gaussians is {len(self.xyz)}")


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

    # def densify(self, max_grad, extent):
    #     grads = self.xyz_gradient_accum / self.denom
    #     grads[grads.isnan()] = 0.0
    #
    #     self.densify_and_clone(grads, max_grad, extent)
    #     self.densify_and_split(grads, max_grad, extent)
    #     torch.cuda.empty_cache()

    # def prune_opacity(self, percent):
    #     sorted_tensor, _ = torch.sort(self.get_opacity, dim=0)
    #     index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
    #     value_nth_percentile = sorted_tensor[index_nth_percentile]
    #     prune_mask = (self.get_opacity <= value_nth_percentile).squeeze()
    #
    #     # big_points_vs = self.max_radii2D > max_screen_size
    #     # big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
    #     # prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    #     self.prune_points(prune_mask)
    #
    #     torch.cuda.empty_cache()

    def prune_gaussians(self, percent, import_score):
        # ic(import_score.shape)
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask)

    # def add_densification_stats(self, viewspace_point_tensor, update_filter):
    #     self.xyz_gradient_accum[update_filter] += torch.norm(
    #         viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
    #     )
    #     self.denom[update_filter] += 1


