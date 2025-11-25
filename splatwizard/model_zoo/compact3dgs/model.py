import io
import math
from enum import Enum
import pickle

import numpy as np
from vector_quantize_pytorch import ResidualVQ
import torch
from torch import nn
from loguru import logger
from plyfile import PlyElement, PlyData
from dahuffman import HuffmanCodec
from dahuffman.huffmancodec import PrefixCodec
from einops import reduce
#
# torch.autograd.set_detect_anomaly(True)
from splatwizard.modules.densify_mixin import DensificationAndPruneMixin
from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard.scheduler import Scheduler, task
from splatwizard.utils.general_utils import build_rotation, build_scaling_rotation, strip_symmetric
from splatwizard.rasterizer.gaussian import GaussianRasterizationSettings, GaussianRasterizer
from splatwizard.modules.gaussian_model import GaussianModel

from splatwizard.metrics.loss_utils import l1_func, ssim_func
from splatwizard.modules.dataclass import LossPack, RenderResult
import splatwizard._cmod.tiny_cuda_nn as tcnn  # noqa
from splatwizard.utils.graphics_utils import BasicPointCloud
from splatwizard._cmod.simple_knn import distCUDA2 # noqa
from splatwizard.utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func,
)
from ...utils.misc import wrap_str
from .config import CompactGSModelParams, CompactGSOptimizationParams
from ...utils.sh_utils import RGB2SH


class GenerateMode(Enum):
    TRAINING_WITHOUT_RVQ = 0
    TRAINING_WITH_RVQ = 1
    EVALUATION = 2

class CompressType(Enum):
    PP = 0
    NPZ = 1

class CompactGSModel(DensificationAndPruneMixin, GaussianModel):

    def __init__(self, model_param: CompactGSModelParams):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self._feature = None
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.vq_scale = ResidualVQ(dim = 3, codebook_size = model_param.rvq_size, num_quantizers = model_param.rvq_num, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, learnable_codebook=True, in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
        self.vq_rot = ResidualVQ(dim = 4, codebook_size = model_param.rvq_size, num_quantizers = model_param.rvq_num, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, learnable_codebook=True, in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
        self.rvq_bit = math.log2(model_param.rvq_size)
        self.rvq_num = model_param.rvq_num
        self.recolor = tcnn.Encoding(
                 n_input_dims=3,
                 encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": model_param.max_hashmap,
                    "base_resolution": 16,
                    "per_level_scale": 1.447,
                },
        )
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 3 
            },
            )
        self.mlp_head = tcnn.Network(
                n_input_dims=(self.direction_encoding.n_output_dims+self.recolor.n_output_dims),
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

        self.prune_recolor = model_param.prune_recolor
        # TODO 如何区别Eval和Training的初始化
        self.mode = GenerateMode.EVALUATION 
        # self.mode = GenerateMode.TRAINING_WITH_RVQ
        self.compress_type = CompressType.PP

        self.register_final_eval_hook(self.final_prune_and_precompute)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._mask,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.capture_mlp(),
        )

    def capture_mlp(self):
        checkpoint = {
            "codebook_scale": self.vq_scale.state_dict(),
            "codebook_rotation": self.vq_rot.state_dict(),
            'direction_encoding': self.direction_encoding.state_dict(),
            'recolor': self.recolor.state_dict(),
            'mlp_head': self.mlp_head.state_dict(),
        }

        return checkpoint

    def restore(self, model_args, training_args=None):
        try:
            (self.active_sh_degree,
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._mask,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            mlp_checkpoint) = model_args
        except:
            logger.warning('Using deprecated compactgs checkpoint!')
            (self.active_sh_degree,
             self._xyz,
             self._scaling,
             self._rotation,
             self._opacity,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale,
             mlp_checkpoint) = model_args


        self.restore_mlp(mlp_checkpoint)
        if training_args is not None:
            self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if self.optimizer is not None:
            self.optimizer.load_state_dict(opt_dict)

    def restore_mlp(self, mlp_checkpoint):
        self.recolor.load_state_dict(mlp_checkpoint['recolor'])
        self.mlp_head.load_state_dict(mlp_checkpoint['mlp_head'])
        self.vq_scale.load_state_dict(mlp_checkpoint["codebook_scale"])
        self.vq_rot.load_state_dict(mlp_checkpoint["codebook_rotation"])
        self.direction_encoding.load_state_dict(mlp_checkpoint["direction_encoding"])

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = nn.functional.normalize


    def register_pre_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: CompactGSOptimizationParams):
        scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
        scheduler.register_task(range(0, opt.iterations, 1000), task=self.oneupSHdegree)
        scheduler.register_task(1, task=lambda:self.switch_mode(GenerateMode.TRAINING_WITHOUT_RVQ))
        scheduler.register_task(opt.rvq_iter, task=lambda:self.switch_mode(GenerateMode.TRAINING_WITH_RVQ))

    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: CompactGSOptimizationParams):
        # Densification
        # if iteration < opt.densify_until_iter:
        #     # Keep track of max radii in image-space for pruning
        #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
        #                                                          radii[visibility_filter])
        #     self.add_densification_stats(viewspace_point_tensor, visibility_filter)

        # scheduler.register_task([opt.iterations], task=self.final_prune_and_precompute)
        scheduler.register_task(range(opt.densify_until_iter), task=self.add_densification_stats_task)

        # if iteration < opt.densify_until_iter:
        #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
        #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
        #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
        scheduler.register_task(
            range(opt.densify_from_iter+opt.densification_interval, opt.densify_until_iter, opt.densification_interval),
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
        
        mask_prune_iters = [
            i for i in range(opt.densify_until_iter, opt.iterations)
            if i % opt.mask_prune_iter == 0
        ]
        scheduler.register_task(mask_prune_iters,task=self.mask_prune, logging=True)

    def switch_mode(self, mode):
        logger.info(f'switch mode {self.mode} -> {mode}')
        self.mode = mode

    def render(self, viewpoint_camera, bg_color: torch.Tensor, pipe: PipelineParams,
               opt: OptimizationParams = None, step=-1, scaling_modifier=1.0, override_color=None):
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

        raster_settings = GaussianRasterizationSettings(
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

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.xyz
        means2D = screenspace_points
        cov3D_precomp = None

        if self.mode == GenerateMode.EVALUATION: # itr == -1:
            self.precompute()
            scales = self._scaling
            rotations = self._rotation
            opacity = self._opacity

            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
            dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            shs = self.mlp_head(torch.cat([self._feature, self.direction_encoding(dir_pp)], dim=-1)).unsqueeze(1)
        else:
            mask = ((torch.sigmoid(self._mask) > 0.01).float() - torch.sigmoid(self._mask)).detach() + torch.sigmoid(
                self._mask)
            if self.mode == GenerateMode.TRAINING_WITH_RVQ:
                scales = self.vq_scale(self.scaling.unsqueeze(0))[0]
                rotations = self.vq_rot(self.rotation.unsqueeze(0))[0]
                scales = scales.squeeze() * mask
                rotations = rotations.squeeze()
                opacity = self.opacity * mask

            else:
                assert self.mode == GenerateMode.TRAINING_WITHOUT_RVQ
                scales = self.scaling * mask
                rotations = self.rotation
                opacity = self.opacity * mask

            xyz = self.contract_to_unisphere(means3D.clone().detach(),
                                           torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
            dir_pp = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            shs = self.mlp_head(torch.cat([self.recolor(xyz), self.direction_encoding(dir_pp)], dim=-1)).unsqueeze(1)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
            means3D=means3D.float(),
            means2D=means2D,
            shs=shs.float(),
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None)
        
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return RenderResult(
            rendered_image=rendered_image,
            viewspace_points=screenspace_points,
            visibility_filter=radii > 0,
            radii=radii
        )


    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        logger.info(wrap_str("Number of points at initialisation:", fused_point_cloud.shape[0]))

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device="cuda")
        
    @task
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    @property
    def covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.scaling, scaling_modifier, self._rotation)



    def training_setup(self, training_args: CompactGSOptimizationParams):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")

        other_params = []
        for params in self.recolor.parameters():
            other_params.append(params)
        for params in self.mlp_head.parameters():
            other_params.append(params)


        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._mask], 'lr': training_args.mask_lr, "name": "mask"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.optimizer_net = torch.optim.Adam(other_params, lr=training_args.net_lr, eps=1e-15)
        self.scheduler_net = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
            self.optimizer_net, start_factor=0.01, total_iters=100
        ),
            torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_net,
            milestones=training_args.net_lr_step,
            gamma=0.33,
        ),
        ]
        )
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
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
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

     
    # def load(self, path, opt=None):
    #     first_iter = None
    #     need_setup = True
    #     if os.path.isfile(path + '_pp'):
    #         path = path + '_pp.npz'
    #         print("Loading ", path)
    #         load_dict = np.load(path, allow_pickle=True)
    #         self.load_pp_from_dict(load_dict)
    #     elif os.path.isfile(path + '.npz'):
    #         path = path + '.npz'
    #         print("Loading ", path)
    #         load_dict = np.load(path, allow_pickle=True)
    #         self.load_npz_from_dict(load_dict)
    #     elif os.path.isdir(path):
    #         point_cloud_dir = os.path.join(path, 'point_cloud')
    #         point_cloud_dir = pathlib.Path(point_cloud_dir)
    #         max_iter = search_for_max_iteration(point_cloud_dir)
    #         point_cloud_dir = os.path.join(point_cloud_dir, f"iteration_{max_iter}", "point_cloud")
    #         first_iter = max_iter
    #         self.load_ply(point_cloud_dir)
    #     elif path.endswith('.ply'):
    #         self.load_ply(os.path.basename(path))
    #     else:
    #         (model_params, first_iter) = torch.load(path)
    #         self.restore(model_params, opt)
    #         need_setup = False
    
    #     return first_iter, need_setup
            
    def load_ply(self, path):
        print("Loading ", path+".ply")
        plydata = PlyData.read(path+".ply")

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        try:
            self.load_mlp_checkpoints(path.parent / "checkpoint.pth")
        except:
            print("Init from scratch Gaussian Model")
            pass
    
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._mask = optimizable_tensors["mask"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    
    def densification_postfix(self, new_xyz, new_opacities, new_scaling, new_rotation, new_mask):
        d = {"xyz": new_xyz,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "mask": new_mask}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._mask = optimizable_tensors["mask"]

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.scaling, dim=1).values > self.percent_dense*scene_extent)
        
        stds = self.scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_mask = self._mask[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_opacity, new_scaling, new_rotation, new_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_mask = self._mask[selected_pts_mask]

        self.densification_postfix(new_xyz, new_opacities, new_scaling, new_rotation, new_mask)
    
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = torch.logical_or((torch.sigmoid(self._mask) <= 0.01).squeeze(),(self.opacity < min_opacity).squeeze())
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    @task
    def mask_prune(self):
        prune_mask = (torch.sigmoid(self._mask) <= 0.01).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def post_quant(self, param, prune=False, scale=255):
        scale = float(scale)
        max_val = torch.amax(param)
        min_val = torch.amin(param)
        if prune:
            param = param*(torch.abs(param) > 0.1)
        param = (param - min_val)/(max_val - min_val)
        quant = torch.round(param * scale)
        out = (max_val - min_val) * quant / scale + min_val
        return torch.nn.Parameter(out), quant, torch.tensor([min_val, max_val])

    def dequantize_post_quant(self, param, min_val, max_val, scale=255.):
        scale = float(scale)
        param = (max_val - min_val) * param / scale + min_val
        return param

    def huffman_encode(self, param):
        input_code_list = param.view(-1).tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        codec = HuffmanCodec.from_data(input_code_list)

        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        total_mb = total_bits/8/10**6
        
        return total_mb, codec.encode(input_code_list), codec.get_code_table()
    
    @torch.no_grad()
    def save_ply(self, path):
        # mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        self.save_mlp_checkpoints(path.parent / "checkpoint.pth")

    def save_mlp_checkpoints(self, path):
        torch.save(self.capture_mlp(), path)

    def load_mlp_checkpoints(self, path):
        checkpoint = torch.load(path)
        self.restore_mlp(checkpoint)

    @torch.no_grad()
    def encode(self, path:io.BufferedWriter):
        logger.info('Start encoding...')
        if self._mask.shape[0] != self._xyz.shape[0]:
            logger.warning('Using pruned checkpoint, skip pruning!')
            self.switch_mode(GenerateMode.EVALUATION)
        else:
            self.final_prune_and_precompute()
        mb_str = self.estimate_storage()
        logger.info(mb_str)
        if self.compress_type == CompressType.PP:
            final = self.save_pp()
        else:
            assert self.compress_type == CompressType.NPZ
            final = self.save_npz()
        pickle.dump(final, path)

    def save_pp(self):
        final = dict()
        
        final["xyz"] = self._xyz.detach().cpu().half().numpy()
        final["opacity"] = np.frombuffer(self.huf_opa, dtype=np.uint8)
        final["scale"] = np.frombuffer(self.huf_sca, dtype=np.uint8)
        final["rotation"] = np.frombuffer(self.huf_rot, dtype=np.uint8)
        final["hash"] = np.frombuffer(self.huf_hash, dtype=np.uint8)
        final["mlp"] = self.mlp_head.params.cpu().half().numpy()
        final["huftable_opacity"] = self.tab_opa
        final["huftable_scale"] = self.tab_sca
        final["huftable_rotation"] = self.tab_rot
        final["huftable_hash"] = self.tab_hash
        final["codebook_scale"] = self.vq_scale.cpu().state_dict()
        final["codebook_rotation"] = self.vq_rot.cpu().state_dict()
        final["minmax_opacity"] = self.minmax_opa.numpy()
        final["minmax_hash"] = self.minmax_hash.numpy()
        final["rvq_info"] = np.array([int(self.rvq_num), int(self.rvq_bit)])
        return final

    def save_npz(self):
        final = dict()
        
        final["xyz"] = self._xyz.detach().cpu().half().numpy()
        final["opacity"] = self._opacity.detach().cpu().half().numpy()
        final["scale"] = np.packbits(np.unpackbits(self.sca_idx.unsqueeze(-1).cpu().numpy().astype(np.uint8), axis=-1, count=int(self.rvq_bit), bitorder='little').flatten(), axis=None)
        final["rotation"] = np.packbits(np.unpackbits(self.rot_idx.unsqueeze(-1).cpu().numpy().astype(np.uint8), axis=-1, count=int(self.rvq_bit), bitorder='little').flatten(), axis=None)
        final["hash"] = self.recolor.params.cpu().half().numpy()
        final["mlp"] = self.mlp_head.params.cpu().half().numpy()
        final["codebook_scale"] = self.vq_scale.cpu().state_dict()
        final["codebook_rotation"] = self.vq_rot.cpu().state_dict()
        final["rvq_info"] = np.array([int(self.rvq_num), int(self.rvq_bit)])
        return final

    @torch.no_grad()
    def decode(self, path:io.BufferedReader):
        logger.info('Start decoding...')
        decoded_pack = pickle.load(path)
        if self.compress_type == CompressType.PP:
            self.load_pp_from_dict(decoded_pack)
        else:
            assert self.compress_type == CompressType.NPZ
            self.load_npz_from_dict(decoded_pack)

    @torch.no_grad()
    def load_pp_from_dict(self, decoded_pack):
        codec = PrefixCodec(decoded_pack["huftable_opacity"])
        opacity = torch.tensor(codec.decode(decoded_pack["opacity"]))

        codec = PrefixCodec(decoded_pack["huftable_scale"])
        scale = codec.decode(decoded_pack["scale"])

        codec = PrefixCodec(decoded_pack["huftable_rotation"])
        rotation = codec.decode(decoded_pack["rotation"])

        codec = PrefixCodec(decoded_pack["huftable_hash"])
        hashgrid = torch.tensor(codec.decode(decoded_pack["hash"]))

        # opacity = (float(decoded_pack["minmax_opacity"][1]) - float(decoded_pack["minmax_opacity"][0]))*opacity/255.0 + float(decoded_pack["minmax_opacity"][0])
        # hashgrid = (float(decoded_pack["minmax_hash"][1]) - float(decoded_pack["minmax_hash"][0]))*hashgrid/255.0 + float(decoded_pack["minmax_hash"][0])

        opacity = self.dequantize_post_quant(opacity, float(decoded_pack["minmax_opacity"][0]), float(decoded_pack["minmax_opacity"][1]))
        hashgrid = self.dequantize_post_quant(
            hashgrid, float(decoded_pack["minmax_hash"][0]) , float(decoded_pack["minmax_hash"][1]))


        self.vq_scale.load_state_dict(decoded_pack["codebook_scale"])
        self.vq_rot.load_state_dict(decoded_pack["codebook_rotation"])
        scale_codes = self.vq_scale.get_codes_from_indices(torch.tensor(scale).cuda().reshape(-1,1,decoded_pack["rvq_info"][0]))
        scale = self.vq_scale.project_out(reduce(scale_codes, 'q ... -> ...', 'sum'))
        rotation_codes = self.vq_rot.get_codes_from_indices(torch.tensor(rotation).cuda().reshape(-1,1,decoded_pack["rvq_info"][0]))
        rotation = self.vq_rot.project_out(reduce(rotation_codes, 'q ... -> ...', 'sum'))

        self._xyz = nn.Parameter(torch.from_numpy(decoded_pack["xyz"]).cuda().float().requires_grad_(True))
        self._opacity = nn.Parameter(opacity.cuda().reshape(-1,1).float().requires_grad_(True))
        self._scaling = nn.Parameter(scale.squeeze(1).requires_grad_(True))
        self._rotation = nn.Parameter(rotation.squeeze(1).requires_grad_(True))
        self.recolor.params = nn.Parameter(hashgrid.cuda().half().requires_grad_(True))
        self.mlp_head.params = nn.Parameter(torch.from_numpy(decoded_pack["mlp"]).cuda().half().requires_grad_(True))

    def load_npz_from_dict(self, decoded_pack):
        scale = np.packbits(np.unpackbits(decoded_pack["scale"], axis=None)[:decoded_pack["xyz"].shape[0]*decoded_pack["rvq_info"][0]*decoded_pack["rvq_info"][1]].reshape(-1, decoded_pack["rvq_info"][1]), axis=-1, bitorder='little')
        rotation = np.packbits(np.unpackbits(decoded_pack["rotation"], axis=None)[:decoded_pack["xyz"].shape[0]*decoded_pack["rvq_info"][0]*decoded_pack["rvq_info"][1]].reshape(-1, decoded_pack["rvq_info"][1]), axis=-1, bitorder='little')

        self.vq_scale.load_state_dict(decoded_pack["codebook_scale"])
        self.vq_rot.load_state_dict(decoded_pack["codebook_rotation"])
        scale_codes = self.vq_scale.get_codes_from_indices(torch.from_numpy(scale).cuda().reshape(-1,1,decoded_pack["rvq_info"][0]).long())
        scale = self.vq_scale.project_out(reduce(scale_codes, 'q ... -> ...', 'sum'))
        rotation_codes = self.vq_rot.get_codes_from_indices(torch.from_numpy(rotation).cuda().reshape(-1,1,decoded_pack["rvq_info"][0]).long())
        rotation = self.vq_rot.project_out(reduce(rotation_codes, 'q ... -> ...', 'sum'))

        self._xyz = nn.Parameter(torch.from_numpy(decoded_pack["xyz"]).cuda().float().requires_grad_(True))
        self._opacity = nn.Parameter(torch.from_numpy(decoded_pack["opacity"]).reshape(-1,1).cuda().float().requires_grad_(True))
        self._scaling = nn.Parameter(scale.squeeze(1).requires_grad_(True))
        self._rotation = nn.Parameter(rotation.squeeze(1).requires_grad_(True))
        self.recolor.params = nn.Parameter(torch.from_numpy(decoded_pack["hash"]).cuda().half().requires_grad_(True))
        self.mlp_head.params = nn.Parameter(torch.from_numpy(decoded_pack["mlp"]).cuda().half().requires_grad_(True))

    def estimate_storage(self):
        if self.compress_type == CompressType.PP:
            self.sort_morton()

        for m in self.vq_scale.layers:
            m.training = False
        for m in self.vq_rot.layers:
            m.training = False

        self._xyz = self._xyz.clone().half().float()
        self._scaling, self.sca_idx, _ = self.vq_scale(self._scaling.unsqueeze(1))
        self._rotation, self.rot_idx, _ = self.vq_rot(self._rotation.unsqueeze(1))
        self._scaling = self._scaling.squeeze()
        self._rotation = self._rotation.squeeze()

        position_mb = self._xyz.shape[0]*3*16/8/10**6
        scale_mb = self._xyz.shape[0]*self.rvq_bit*self.rvq_num/8/10**6 + 2**self.rvq_bit*self.rvq_num*3*32/8/10**6
        rotation_mb = self._xyz.shape[0]*self.rvq_bit*self.rvq_num/8/10**6 + 2**self.rvq_bit*self.rvq_num*4*32/8/10**6
        opacity_mb = self._xyz.shape[0]*16/8/10**6
        hash_mb = self.recolor.params.shape[0]*16/8/10**6
        mlp_mb = self.mlp_head.params.shape[0]*16/8/10**6
        sum_mb = position_mb+scale_mb+rotation_mb+opacity_mb+hash_mb+mlp_mb

        mb_str = "Storage\nposition: "+str(position_mb)+"\nscale: "+str(scale_mb)+"\nrotation: "+str(rotation_mb)+"\nopacity: "+str(opacity_mb)+"\nhash: "+str(hash_mb)+"\nmlp: "+str(mlp_mb)+"\ntotal: "+str(sum_mb)+" MB"
        # self.cached_opacity = self._opacity
        # self.cached_recolor = self.recolor
        if self.compress_type == CompressType.PP:
            self._opacity, self.quant_opa, self.minmax_opa = self.post_quant(self._opacity)
            self.recolor.params, self.quant_hash, self.minmax_hash = self.post_quant(self.recolor.params, self.prune_recolor )

            scale_mb, self.huf_sca, self.tab_sca = self.huffman_encode(self.sca_idx)
            scale_mb += 2**self.rvq_bit*self.rvq_num*3*32/8/10**6
            rotation_mb, self.huf_rot, self.tab_rot = self.huffman_encode(self.rot_idx)
            rotation_mb += 2**self.rvq_bit*self.rvq_num*4*32/8/10**6
            opacity_mb, self.huf_opa, self.tab_opa = self.huffman_encode(self.quant_opa)
            hash_mb, self.huf_hash, self.tab_hash = self.huffman_encode(self.quant_hash)
            mlp_mb = self.mlp_head.params.shape[0]*16/8/10**6
            sum_mb = position_mb+scale_mb+rotation_mb+opacity_mb+hash_mb+mlp_mb

            mb_str = mb_str+"\n\nAfter PP\nposition: "+str(position_mb)+"\nscale: "+str(scale_mb)+"\nrotation: "+str(rotation_mb)+"\nopacity: "+str(opacity_mb)+"\nhash: "+str(hash_mb)+"\nmlp: "+str(mlp_mb)+"\ntotal: "+str(sum_mb)+" MB"
        else:
            self._opacity = self.opacity.clone().half().float()
        torch.cuda.empty_cache()
        return mb_str

    def final_prune(self):
        prune_mask = (torch.sigmoid(self._mask) <= 0.01).squeeze()
        self.prune_points(prune_mask)
        self._opacity = self.opacity
        self._scaling = self.scaling
        self._rotation = self.rotation
        # mb_str = self.estimate_storage()
        return None #mb_str

    def precompute(self):
        xyz = self.contract_to_unisphere(self._xyz.half(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
        self._feature = self.recolor(xyz)
        torch.cuda.empty_cache()

    # @task
    def final_prune_and_precompute(self):
        # self.compress_type = CompressType.PP
        storage = self.final_prune()
        logger.info(storage)
        self.precompute()
        self.switch_mode(GenerateMode.EVALUATION)

    @task
    def add_densification_stats_task(self, render_result: RenderResult):
        self.add_densification_stats(render_result, render_result.visibility_filter)

    def add_densification_stats(self, render_result, update_filter):
        self.max_radii2D[update_filter] = torch.max(self.max_radii2D[update_filter], render_result.radii[update_filter])
        self.xyz_gradient_accum[update_filter] += torch.norm(render_result.viewspace_points.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def optimizer_step(self, render_result: RenderResult, opt: OptimizationParams, step: int):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_net.step()
        self.optimizer_net.zero_grad(set_to_none=True)
        self.scheduler_net.step()

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x
    
    def splitBy3(self, a):
        x = a & 0x1FFFFF  # we only look at the first 21 bits
        x = (x | x << 32) & 0x1F00000000FFFF
        x = (x | x << 16) & 0x1F0000FF0000FF
        x = (x | x << 8) & 0x100F00F00F00F00F
        x = (x | x << 4) & 0x10C30C30C30C30C3
        x = (x | x << 2) & 0x1249249249249249
        return x

    def mortonEncode(self, pos: torch.Tensor) -> torch.Tensor:
        x, y, z = pos.unbind(-1)
        answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
        answer |= self.splitBy3(x) | self.splitBy3(y) << 1 | self.splitBy3(z) << 2
        return answer

    def sort_morton(self):
        with torch.no_grad():
            xyz_q = (
                (2**21 - 1)
                * (self._xyz - self._xyz.min(0).values)
                / (self._xyz.max(0).values - self._xyz.min(0).values)
            ).long()
            order = self.mortonEncode(xyz_q).sort().indices
            
            self._xyz = nn.Parameter(self._xyz[order], requires_grad=True)
            self._opacity = nn.Parameter(self._opacity[order], requires_grad=True)
            self._scaling = nn.Parameter(self._scaling[order], requires_grad=True)
            self._rotation = nn.Parameter(self._rotation[order], requires_grad=True)

    def loss_func(self, viewpoint_cam, render_result: RenderResult, opt) -> (torch.Tensor, LossPack):
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_func(render_result.rendered_image, gt_image)

        ssim_loss = (1.0 - ssim_func(render_result.rendered_image, gt_image))

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + opt.lambda_mask*torch.mean((torch.sigmoid(self._mask)))

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

    @task
    def densify_and_prune_task(self, opt: CompactGSOptimizationParams, step: int):
        size_threshold = 20 if step > opt.opacity_reset_interval else None
        self.densify_and_prune(opt.densify_grad_threshold, 0.005, self.spatial_lr_scale, size_threshold)

