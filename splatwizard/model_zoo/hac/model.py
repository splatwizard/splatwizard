import io
import pickle
from dataclasses import dataclass
from enum import Enum
from functools import reduce
import typing
import math

import numpy as np
import torch
from einops import repeat
from plyfile import PlyData, PlyElement
from torch import nn
from torch_scatter import scatter_max
from loguru import logger

from splatwizard.config import PipelineParams
from splatwizard.compression.entropy_codec import ArithmeticCodec
from splatwizard.metrics.loss_utils import l1_func, ssim_func
from splatwizard.common.constants import BIT2MB_SCALE
from splatwizard.rasterizer.gaussian import GaussianRasterizationSettings, GaussianRasterizer
from splatwizard.model_zoo.hac.config import HACModelParams, HACOptimizationParams
from splatwizard.modules.dataclass import RenderResult, LossPack
from splatwizard.modules.gaussian_model import GaussianModel
from splatwizard._cmod.simple_knn import distCUDA2    # noqa
from splatwizard.modules.grid_encoder import EntropyMix3D2DGridEncoder, EntropyGridEncoder
from splatwizard.scheduler import Scheduler, task
from splatwizard.utils.general_utils import (
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric
)
from splatwizard.utils.graphics_utils import BasicPointCloud

from splatwizard.compression.entropy_model import EntropyGaussian
from splatwizard.compression.quantizer import STE_binary, STE_multistep, Quantize_anchor, UniformQuantizer, STEQuantizer


class GenerateMode(Enum):
    TRAINING_FULL_PRECISION = 0
    TRAINING_QUANTIZED = 1
    TRAINING_ENTROPY = 2
    TRAINING_STE_ENTROPY = 3
    DECODING_AS_IS = 4
    TRAINING_STE_QUANTIZED = 5


@dataclass
class RatePack:
    bit_per_param: torch.Tensor = None
    bit_per_feat_param: torch.Tensor = None
    bit_per_scaling_param: torch.Tensor = None
    bit_per_offsets_param: torch.Tensor = None


@dataclass
class GeneratedGaussians:
    xyz: torch.Tensor
    color: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rot: torch.Tensor
    neural_opacity: torch.Tensor = None
    visible_mask: torch.Tensor = None
    mask: torch.Tensor = None
    bit_per_param: torch.Tensor = None
    bit_per_feat_param: torch.Tensor = None
    bit_per_scaling_param: torch.Tensor = None
    bit_per_offsets_param: torch.Tensor = None
    concatenated_all: torch.Tensor = None
    time_sub: float = None


@dataclass
class EntropyContext:
    mean_feat: torch.Tensor
    scale_feat: torch.Tensor
    mean_scaling: torch.Tensor
    scale_scaling: torch.Tensor
    mean_offsets: torch.Tensor
    scale_offsets: torch.Tensor
    Q_feat: torch.Tensor
    Q_scaling: torch.Tensor
    Q_offsets: torch.Tensor


@dataclass
class HACRenderResult(RatePack, RenderResult):
    time_sub: typing.Union[torch.Tensor, None] = None
    selection_mask: typing.Union[torch.Tensor, None] = None
    neural_opacity: typing.Union[torch.Tensor, None] = None
    scaling: typing.Union[torch.Tensor, None] = None
    generated_gaussians: typing.Any = None
    entropy_constrained: bool = False


anchor_round_digits = 16
Q_anchor = 1/(2 ** anchor_round_digits - 1)
use_clamp = True
use_multiprocessor = False  # Always False plz. Not yet implemented for True.


def get_binary_vxl_size(binary_vxl):
    # binary_vxl: {0, 1}
    # assert torch.unique(binary_vxl).mean() == 0.5
    ttl_num = binary_vxl.numel()

    pos_num = torch.sum(binary_vxl)
    neg_num = ttl_num - pos_num

    Pg = pos_num / ttl_num  # + 1e-6
    Pg = torch.clamp(Pg, min=1e-6, max=1-1e-6)
    pos_prob = Pg
    neg_prob = (1 - Pg)
    pos_bit = pos_num * (-torch.log2(pos_prob))
    neg_bit = neg_num * (-torch.log2(neg_prob))
    ttl_bit = pos_bit + neg_bit
    ttl_bit += 32  # Pg
    # print('binary_vxl:', Pg.item(), ttl_bit.item(), ttl_num, pos_num.item(), neg_num.item())
    return Pg, ttl_bit, ttl_bit.item()/8.0/1024/1024, ttl_num


class HAC(GaussianModel, nn.Module):

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
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, model_param: HACModelParams):
        nn.Module.__init__(self)
        GaussianModel.__init__(self)
        logger.info('hash_params:' + str(model_param))
        logger.info(model_param.resolutions_list)
        logger.info(model_param.resolutions_list_2D)

        self.model_param = model_param

        self.feat_dim = model_param.feat_dim
        self.n_offsets = model_param.n_offsets
        self.voxel_size = model_param.voxel_size
        self.update_depth = model_param.update_depth
        self.update_init_factor = model_param.update_init_factor
        self.update_hierachy_factor = model_param.update_hierachy_factor
        self.use_feat_bank = model_param.use_feat_bank
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.n_features_per_level = model_param.n_features_per_level
        self.log2_hashmap_size = model_param.log2_hashmap_size
        self.log2_hashmap_size_2D = model_param.log2_hashmap_size_2D
        self.resolutions_list = model_param.resolutions_list
        self.resolutions_list_2D = model_param.resolutions_list_2D
        self.ste_binary = model_param.ste_binary
        self.ste_multistep = model_param.ste_multistep
        self.add_noise = model_param.add_noise
        self.Q = model_param.Q
        self.use_2D = model_param.use_2D
        self.decoded_version = model_param.decoded_version

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._mask = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.mode = GenerateMode.TRAINING_FULL_PRECISION

        if self.use_2D:
            self.encoding_xyz = EntropyMix3D2DGridEncoder(
                n_features=self.model_param.n_features_per_level,
                resolutions_list=self.resolutions_list,
                log2_hashmap_size=self.model_param.log2_hashmap_size,
                resolutions_list_2D=self.resolutions_list_2D,
                log2_hashmap_size_2D=self.model_param.log2_hashmap_size_2D,
                ste_binary=self.model_param.ste_binary,
                ste_multistep=self.model_param.ste_multistep,
                add_noise=self.model_param.add_noise,
                Q=self.model_param.Q,
            ).cuda()
        else:
            self.encoding_xyz = EntropyGridEncoder(
                num_dim=3,
                n_features=self.model_param.n_features_per_level,
                resolutions_list=self.resolutions_list,
                log2_hashmap_size=self.model_param.log2_hashmap_size,
                ste_binary=self.model_param.ste_binary,
                ste_multistep=self.model_param.ste_multistep,
                add_noise=self.model_param.add_noise,
                Q=self.model_param.Q,
            ).cuda()

        encoding_params_num = 0
        for n, p in self.encoding_xyz.named_parameters():
            encoding_params_num += p.numel()
        encoding_MB = encoding_params_num / 8 / 1024 / 1024
        if not self.model_param.ste_binary: encoding_MB *= 32
        logger.info(f'encoding_param_num={encoding_params_num}, size={encoding_MB}MB.')

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        mlp_input_feat_dim = self.feat_dim

        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7*self.n_offsets),
            # nn.Linear(feat_dim, 7),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.mlp_grid = nn.Sequential(
            nn.Linear(self.encoding_xyz.output_dim, self.feat_dim*2),
            nn.ReLU(True),
            nn.Linear(self.feat_dim*2, (self.feat_dim+6+3*self.n_offsets)*2+1+1+1),
        ).cuda()

        self.mlp_deform = nn.Sequential(
            nn.Linear(self.encoding_xyz.output_dim, self.feat_dim*2),
            nn.ReLU(True),
            nn.Linear(self.feat_dim*2, 2*self.n_offsets),
        ).cuda()
        self.mlp_deform[-1].bias.data[0::2] += 10.0

        self.noise_quantizer = UniformQuantizer()
        self.ste_quantizer = STEQuantizer()
        self.entropy_gaussian = EntropyGaussian(Q=1).cuda()
        self.codec = ArithmeticCodec()
        # self.register_eval_hook(self.switch_to_decode)

    def switch_to_decode(self):
        self.decoded_version = True

    def register_pre_task(self, scheduler, ppl: PipelineParams, opt: HACOptimizationParams):
        scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
        scheduler.register_task(
            1, task=lambda: self.update_anchor_bound(),
            name='update_anchor_bound', logging=True
        )

    def register_post_task(self, scheduler: Scheduler, ppl, opt: HACOptimizationParams):
        scheduler.register_task(range(0, opt.update_until), task=self.training_statis)

        scheduler.register_task(
            range(opt.update_from, opt.pause_update_from, opt.update_interval),
            task=self.adjust_anchor, logging=True
        )

        scheduler.register_task(
            range(opt.pause_update_until, opt.update_until, opt.update_interval),
            task=self.adjust_anchor, logging=True
        )

        scheduler.register_task(
            opt.iter_switch_to_quantized, task=lambda: self.switch_mode(GenerateMode.TRAINING_QUANTIZED),
            name='switch to TRAINING_QUANTIZED', logging=True
        )

        scheduler.register_task(
            opt.iter_update_anchor_bound, task=lambda: self.update_anchor_bound(),
            name='update_anchor_bound', logging=True
        )

        scheduler.register_task(
            opt.iter_switch_to_entropy, task=lambda: self.switch_mode(GenerateMode.TRAINING_ENTROPY),
            name='switch to TRAINING_ENTROPY', logging=True
        )

    def switch_mode(self, mode):
        logger.info(f'switch mode {self.mode} -> {mode}')
        self.mode = mode

    def calc_sampled_rate(
            self,
            visible_mask,
            feat, grid_scaling, grid_offsets,
            entropy_context
    ):

        anchor = self.get_anchor[visible_mask]

        mask_anchor = self.get_mask_anchor[visible_mask]
        mask_anchor_bool = mask_anchor.to(torch.bool)
        mask_anchor_rate = (mask_anchor.sum() / mask_anchor.numel()).detach()

        binary_grid_masks = self.get_mask[visible_mask]  # differentiable mask

        choose_idx = torch.rand_like(anchor[:, 0]) <= 0.05
        choose_idx = choose_idx & mask_anchor_bool
        feat_chosen = feat[choose_idx]
        grid_scaling_chosen = grid_scaling[choose_idx]
        grid_offsets_chosen = grid_offsets[choose_idx].view(-1, 3 * self.n_offsets)
        mean = entropy_context.mean_feat[choose_idx]
        scale = entropy_context.scale_feat[choose_idx]
        mean_scaling = entropy_context.mean_scaling[choose_idx]
        scale_scaling = entropy_context.scale_scaling[choose_idx]
        mean_offsets = entropy_context.mean_offsets[choose_idx]
        scale_offsets = entropy_context.scale_offsets[choose_idx]
        Q_feat = entropy_context.Q_feat[choose_idx]
        Q_scaling = entropy_context.Q_scaling[choose_idx]
        Q_offsets = entropy_context.Q_offsets[choose_idx]
        binary_grid_masks_chosen = binary_grid_masks[choose_idx].repeat(1, 1, 3).view(-1, 3 * self.n_offsets)
        bit_feat = self.entropy_gaussian(feat_chosen, mean, scale, Q_feat, self._anchor_feat.mean())
        bit_scaling = self.entropy_gaussian(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling,
                                          self.get_scaling.mean())
        bit_offsets = self.entropy_gaussian(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets,
                                          self._offset.mean())
        bit_offsets = bit_offsets * binary_grid_masks_chosen

        # t_bit_feat = bit_feat.sum()
        bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel() * mask_anchor_rate
        bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel() * mask_anchor_rate
        bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel() * mask_anchor_rate
        bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                        (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel()) * mask_anchor_rate

        return RatePack(
            bit_per_param=bit_per_param,
            bit_per_feat_param=bit_per_feat_param,
            bit_per_scaling_param=bit_per_scaling_param,
            bit_per_offsets_param=bit_per_offsets_param,

        )

    def _calc_entropy_context(self, anchor, visible_mask):

        # TODO: 与HAC对齐

        feat_context = self.calc_interp_feat(anchor)
        feat_context = self.get_grid_mlp(feat_context)

        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(feat_context, split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets,
                                                              3 * self.n_offsets, 1, 1, 1], dim=-1)

        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
        # feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
        # grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
        # grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.unsqueeze(
        #     1)

        return EntropyContext(
            mean_feat, torch.clamp(scale_feat, 1e-9),
            mean_scaling, torch.clamp(scale_scaling, 1e-9),
            mean_offsets, torch.clamp(scale_offsets,1e-9),
            Q_feat, Q_scaling, Q_offsets
        )

    def generate_neural_gaussians(
            self,
            viewpoint_camera,
            visible_mask=None,
            mode=GenerateMode.TRAINING_FULL_PRECISION
    ):
        ## view frustum filtering for acceleration

        time_sub = 0

        if visible_mask is None:
            visible_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device=self.get_anchor.device)

        anchor = self.get_anchor[visible_mask]
        #
        feat = self._anchor_feat[visible_mask]
        grid_offsets = self._offset[visible_mask]
        grid_scaling = self.get_scaling[visible_mask]
        binary_grid_masks = self.get_mask[visible_mask]
        # mask_anchor = self.get_mask_anchor[visible_mask]
        # mask_anchor_bool = mask_anchor.to(torch.bool)
        # mask_anchor_rate = (mask_anchor.sum() / mask_anchor.numel()).detach()

        rate_pack = RatePack()

        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2
        if mode == GenerateMode.TRAINING_FULL_PRECISION or mode == GenerateMode.DECODING_AS_IS:
            # 全精度训练和解码后推断都不对feature进行处理
            pass

        elif mode == GenerateMode.TRAINING_QUANTIZED:
            # 一阶段量化，采用均匀分布模拟量化
            feat = self.noise_quantizer(feat, Q_feat)
            grid_scaling = self.noise_quantizer(grid_scaling, Q_scaling)
            grid_offsets = self.noise_quantizer(grid_offsets, Q_offsets)
                # # quantization
                # feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
                # grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
                # grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets

            # TODO: 注册任务时要考虑
            # if step == 10000:
            #     self.update_anchor_bound()

        elif mode == GenerateMode.TRAINING_ENTROPY:
            # 二阶段量化，量化的同时计算码率

            entropy_context = self.calc_entropy_params(anchor)

            # Q_feat = Q_feat * entropy_context.Q_feat
            # Q_scaling = Q_scaling * entropy_context.Q_scaling
            # Q_offsets = Q_offsets * entropy_context.Q_offsets

            feat = self.noise_quantizer(feat, entropy_context.Q_feat)
            grid_scaling = self.noise_quantizer(grid_scaling, entropy_context.Q_scaling)
            grid_offsets = self.noise_quantizer(grid_offsets, entropy_context.Q_offsets.unsqueeze(1))

            rate_pack = self.calc_sampled_rate(
                visible_mask,
                feat, grid_scaling, grid_offsets,
                # Q_feat, Q_scaling, Q_offsets,
                entropy_context
            )

        elif mode == GenerateMode.TRAINING_STE_QUANTIZED:
            entropy_context = self.calc_entropy_params(anchor)

            # Q_feat = Q_feat * entropy_context.Q_feat
            # Q_scaling = Q_scaling * entropy_context.Q_scaling
            # Q_offsets = Q_offsets * entropy_context.Q_offsets

            feat = self.ste_quantizer(feat, entropy_context.Q_feat)
            grid_scaling = self.ste_quantizer(grid_scaling, entropy_context.Q_scaling)
            grid_offsets = self.ste_quantizer(grid_offsets, entropy_context.Q_offsets.unsqueeze(1))

            rate_pack = self.calc_sampled_rate(
                visible_mask,
                feat, grid_scaling, grid_offsets,
                # Q_feat, Q_scaling, Q_offsets,
                entropy_context
            )

        else:
            raise ValueError(f'Unknown mode {mode}')


        ob_view = anchor - viewpoint_camera.camera_center
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        ob_view = ob_view / ob_dist

        ## view-adaptive feature
        if self.use_feat_bank:
            cat_view = torch.cat([ob_view, ob_dist], dim=1)  # [3+1]

            bank_weight = self.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [N_visible_anchor, 1, 3]

            feat = feat.unsqueeze(dim=-1)  # feat: [N_visible_anchor, 32]
            feat = \
                feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1] + \
                feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2] + \
                feat[:, ::1, :1] * bank_weight[:, :, 2:]
            feat = feat.squeeze(dim=-1)  # [N_visible_anchor, 32]

        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N_visible_anchor, 32+3+1]

        neural_opacity = self.get_opacity_mlp(cat_local_view)  # [N_visible_anchor, K]
        neural_opacity = neural_opacity.reshape([-1, 1])  # [N_visible_anchor*K, 1]
        neural_opacity = neural_opacity * binary_grid_masks.view(-1, 1)
        mask = (neural_opacity > 0.0)
        mask = mask.view(-1)  # [N_visible_anchor*K]

        # select opacity
        opacity = neural_opacity[mask]  # [N_opacity_pos_gaussian, 1]

        # get offset's color
        color = self.get_color_mlp(cat_local_view)  # [N_visible_anchor, K*3]
        color = color.reshape([anchor.shape[0] * self.n_offsets, 3])  # [N_visible_anchor*K, 3]

        # get offset's cov
        scale_rot = self.get_cov_mlp(cat_local_view)  # [N_visible_anchor, K*7]
        scale_rot = scale_rot.reshape([anchor.shape[0] * self.n_offsets, 7])  # [N_visible_anchor*K, 7]

        offsets = grid_offsets.view([-1, 3])  # [N_visible_anchor*K, 3]

        # combine for parallel masking
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [N_visible_anchor, 6+3]
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)  # [N_visible_anchor*K, 6+3]
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets],
                                     dim=-1)  # [N_visible_anchor*K, (6+3)+3+7+3]
        masked = concatenated_all[mask]  # [N_opacity_pos_gaussian, (6+3)+3+7+3]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

        # post-process cov
        scaling = scaling_repeat[:, 3:] * torch.sigmoid(
            scale_rot[:, :3])
        rot = self.rotation_activation(scale_rot[:, 3:7])  # [N_opacity_pos_gaussian, 4]

        offsets = offsets * scaling_repeat[:, :3]  # [N_opacity_pos_gaussian, 3]
        xyz = repeat_anchor + offsets  # [N_opacity_pos_gaussian, 3]

        gss = GeneratedGaussians(
            xyz=xyz,
            color=color,
            opacity=opacity,
            scaling=scaling,
            rot=rot,
            neural_opacity=neural_opacity,
            visible_mask=visible_mask,
            mask=mask,
            bit_per_param=rate_pack.bit_per_param,
            bit_per_feat_param=rate_pack.bit_per_feat_param,
            bit_per_scaling_param=rate_pack.bit_per_scaling_param,
            bit_per_offsets_param=rate_pack.bit_per_offsets_param,
            concatenated_all=concatenated_all,
            time_sub=time_sub
        )
        return gss


    def prefilter_voxel(self, viewpoint_camera, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                        override_color=None):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(
            self.get_anchor,
            dtype=self.get_anchor.dtype,
            requires_grad=True,
            device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

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
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        # print(viewpoint_camera.image_name)
        # print(raster_settings)

        means3D = self.get_anchor

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:  # False
            cov3D_precomp = self.get_covariance(scaling_modifier)
        else:  # into here
            scales = self.get_scaling  # requires_grad = True
            rotations = self.get_rotation  # requires_grad = True

        radii_pure = rasterizer.visible_filter(
            means3D=means3D,
            scales=scales[:, :3],
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,  # None
        )

        # visible_mask =  radii_pure > 0
        # print(visible_mask.shape, visible_mask.to(torch.int).sum())
        return radii_pure > 0


    def render(self, viewpoint_camera, background, pipe, opt=None, step=0, scaling_modifier=1.0, override_color=None):
        visible_mask = self.prefilter_voxel(viewpoint_camera, pipe, background)

        # print(visible_mask.shape, visible_mask.to(torch.int).sum())

        gss = self.generate_neural_gaussians(viewpoint_camera, visible_mask, mode=self.mode)

        screenspace_points = torch.zeros_like(gss.xyz, dtype=self.get_anchor.dtype, requires_grad=True, device="cuda") + 0

        if opt is not None:
            retain_grad = (step < opt.update_until and step >= 0)
        else:
            retain_grad = False
        # TODO: 检查实际效果
        if retain_grad:
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
            bg=background,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)


        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
            means3D=gss.xyz,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=gss.color,
            opacities=gss.opacity,
            scales=gss.scaling,
            rotations=gss.rot,
            cov3D_precomp=None)

        return HACRenderResult(
            rendered_image=rendered_image,
            viewspace_points=screenspace_points,
            visibility_filter=radii > 0,
            visible_mask=visible_mask,
            radii=radii,
            active_gaussians=(radii > 0).sum(),
            num_rendered=0,
            selection_mask=gss.mask,
            neural_opacity=gss.neural_opacity,
            scaling=gss.scaling,
            bit_per_param=gss.bit_per_param,
            bit_per_feat_param=gss.bit_per_feat_param,
            bit_per_scaling_param=gss.bit_per_scaling_param,
            bit_per_offsets_param=gss.bit_per_offsets_param,
            entropy_constrained=(gss.bit_per_param is not None),
            generated_gaussians=gss,
            # time_sub=gss.time_sub
        )


    def loss_func(self, viewpoint_cam, render_result: HACRenderResult, opt: HACOptimizationParams):
        image = render_result.rendered_image
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_func(render_result.rendered_image, gt_image)

        ssim_loss = (1.0 - ssim_func(image, gt_image))
        scaling_reg = render_result.scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg

        if render_result.bit_per_param is not None:
            _, bit_hash_grid, MB_hash_grid, _ = get_binary_vxl_size((self.get_encoding_params()+1)/2)
            denom = self._anchor.shape[0]*(self.feat_dim+6+3*self.n_offsets)
            loss = loss + opt.lmbda * (render_result.bit_per_param + bit_hash_grid / denom)

            loss = loss + 5e-4 * torch.mean(torch.sigmoid(self._mask))

        loss_pack = LossPack(
            l1_loss=Ll1,
            ssim_loss=ssim_loss,
            loss=loss
        )
        return loss, loss_pack

    def get_encoding_params(self):
        params = []
        if self.use_2D:
            params.append(self.encoding_xyz.encoding_xyz.params)
            params.append(self.encoding_xyz.encoding_xy.params)
            params.append(self.encoding_xyz.encoding_xz.params)
            params.append(self.encoding_xyz.encoding_yz.params)
        else:
            params.append(self.encoding_xyz.params)
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params

    def get_mlp_size(self, digit=32):
        mlp_size = 0
        for n, p in self.named_parameters():
            if 'mlp' in n and 'deform' not in n:
                mlp_size += p.numel()*digit
        return mlp_size, mlp_size / 8 / 1024 / 1024

    def capture(self):
        return (
            self.active_sh_degree,
            self._anchor,
            self._anchor_feat,
            self._offset,
            self._mask,
            self._scaling,
            self._rotation,
            self._opacity,
            self.x_bound_min,
            self.x_bound_max,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.mode,
            self.capture_mlp()
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._anchor,
            self._anchor_feat,
            self._offset,
            self._mask,
            self._scaling,
            self._rotation,
            self._opacity,
            x_bound_min,
            x_bound_max,
            self.max_radii2D,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.mode,
            mlp_checkpoint
        ) = model_args

        self.restore_mlp(mlp_checkpoint)
        self.training_setup(training_args)
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        # check_tensor(self._scaling)
        if self.decoded_version:
            return self._scaling
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_mask(self):
        if self.decoded_version:
            return self._mask
        mask_sig = torch.sigmoid(self._mask)
        return ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig

    @property
    def get_mask_anchor(self):
        with torch.no_grad():
            if self.decoded_version:
                mask_anchor = (torch.sum(self._mask, dim=1)[:, 0]) > 0
                return mask_anchor
            mask_sig = torch.sigmoid(self._mask)
            mask = ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig
            mask_anchor = (torch.sum(mask, dim=1)[:, 0]) > 0
            return mask_anchor

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_grid_mlp(self):
        return self.mlp_grid

    @property
    def get_deform_mlp(self):
        return self.mlp_deform

    @property
    def get_rotation(self):

        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        if self.decoded_version:
            return self._anchor
        anchor, quantized_v = Quantize_anchor.apply(self._anchor, self.x_bound_min, self.x_bound_max)
        return anchor
    
    @property
    def get_quantized_v(self):
        anchor, quantized_v = Quantize_anchor.apply(self._anchor, self.x_bound_min, self.x_bound_max)
        return quantized_v

    @torch.no_grad()
    def update_anchor_bound(self):
        x_bound_min = (torch.min(self._anchor, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self._anchor, dim=0, keepdim=True)[0]).detach()
        for c in range(x_bound_min.shape[-1]):
            x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 0.8
        for c in range(x_bound_max.shape[-1]):
            x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 0.8
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        logger.info('anchor_bound_updated')

    def calc_interp_feat(self, x):
        # x: [N, 3]
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]
        features = self.encoding_xyz(x)  # [N, 4*12]
        return features

    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def xyz(self):
        return self._anchor


    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        ratio = 1
        points = pcd.points[::ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        # print(f'Initial voxel_size: {self.voxel_size}')
        logger.info(f'Initial voxel_size: {self.voxel_size}')

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        masks = torch.ones((fused_point_cloud.shape[0], self.n_offsets, 1)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        logger.info(f"Number of points at initialisation : {fused_point_cloud.shape[0]}")

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._mask = nn.Parameter(masks.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def training_setup(self, opt_params: HACOptimizationParams):
        self.percent_dense = opt_params.percent_dense
        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        l = [
            {'params': [self._anchor], 'lr': opt_params.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': opt_params.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._mask], 'lr': opt_params.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
            {'params': [self._anchor_feat], 'lr': opt_params.feature_lr, "name": "anchor_feat"},
            {'params': [self._opacity], 'lr': opt_params.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': opt_params.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': opt_params.rotation_lr, "name": "rotation"},

            {'params': self.mlp_opacity.parameters(), 'lr': opt_params.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': opt_params.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': opt_params.mlp_color_lr_init, "name": "mlp_color"},

            {'params': self.encoding_xyz.parameters(), 'lr': opt_params.encoding_xyz_lr_init,
             "name": "encoding_xyz"},
            {'params': self.mlp_grid.parameters(), 'lr': opt_params.mlp_grid_lr_init, "name": "mlp_grid"},

            {'params': self.mlp_deform.parameters(), 'lr': opt_params.mlp_deform_lr_init, "name": "mlp_deform"},
        ]

        if self.use_feat_bank:
            l.append(
                {'params': self.mlp_feature_bank.parameters(), 'lr': opt_params.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
            )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=opt_params.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=opt_params.position_lr_final * self.spatial_lr_scale,
                                                       lr_delay_mult=opt_params.position_lr_delay_mult,
                                                       max_steps=opt_params.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=opt_params.offset_lr_init * self.spatial_lr_scale,
                                                       lr_final=opt_params.offset_lr_final * self.spatial_lr_scale,
                                                       lr_delay_mult=opt_params.offset_lr_delay_mult,
                                                       max_steps=opt_params.offset_lr_max_steps)
        self.mask_scheduler_args = get_expon_lr_func(lr_init=opt_params.mask_lr_init * self.spatial_lr_scale,
                                                     lr_final=opt_params.mask_lr_final * self.spatial_lr_scale,
                                                     lr_delay_mult=opt_params.mask_lr_delay_mult,
                                                     max_steps=opt_params.mask_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=opt_params.mlp_opacity_lr_init,
                                                            lr_final=opt_params.mlp_opacity_lr_final,
                                                            lr_delay_mult=opt_params.mlp_opacity_lr_delay_mult,
                                                            max_steps=opt_params.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=opt_params.mlp_cov_lr_init,
                                                        lr_final=opt_params.mlp_cov_lr_final,
                                                        lr_delay_mult=opt_params.mlp_cov_lr_delay_mult,
                                                        max_steps=opt_params.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=opt_params.mlp_color_lr_init,
                                                          lr_final=opt_params.mlp_color_lr_final,
                                                          lr_delay_mult=opt_params.mlp_color_lr_delay_mult,
                                                          max_steps=opt_params.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=opt_params.mlp_featurebank_lr_init,
                                                                    lr_final=opt_params.mlp_featurebank_lr_final,
                                                                    lr_delay_mult=opt_params.mlp_featurebank_lr_delay_mult,
                                                                    max_steps=opt_params.mlp_featurebank_lr_max_steps)

        self.encoding_xyz_scheduler_args = get_expon_lr_func(lr_init=opt_params.encoding_xyz_lr_init,
                                                             lr_final=opt_params.encoding_xyz_lr_final,
                                                             lr_delay_mult=opt_params.encoding_xyz_lr_delay_mult,
                                                             max_steps=opt_params.encoding_xyz_lr_max_steps,
                                                             step_sub=0 if self.ste_binary else 10000,
                                                             )
        self.mlp_grid_scheduler_args = get_expon_lr_func(lr_init=opt_params.mlp_grid_lr_init,
                                                         lr_final=opt_params.mlp_grid_lr_final,
                                                         lr_delay_mult=opt_params.mlp_grid_lr_delay_mult,
                                                         max_steps=opt_params.mlp_grid_lr_max_steps,
                                                         step_sub=0 if self.ste_binary else 10000,
                                                         )

        self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=opt_params.mlp_deform_lr_init,
                                                           lr_final=opt_params.mlp_deform_lr_final,
                                                           lr_delay_mult=opt_params.mlp_deform_lr_delay_mult,
                                                           max_steps=opt_params.mlp_deform_lr_max_steps)
    @task
    def update_learning_rate(self,  iteration: int):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mask":
                lr = self.mask_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "encoding_xyz":
                lr = self.encoding_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_grid":
                lr = self.mlp_grid_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_deform":
                lr = self.mlp_deform_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._mask.shape[1]*self._mask.shape[2]):
            l.append('f_mask_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_sparse_gaussian_ply(self, path):
        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        mask = self._mask.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, mask, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply(self, path):
        self.save_sparse_gaussian_ply(path)
        self.save_mlp_checkpoints(path.parent / "checkpoint.pth")

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        mask_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_mask")]
        mask_names = sorted(mask_names, key = lambda x: int(x.split('_')[-1]))
        masks = np.zeros((anchor.shape[0], len(mask_names)))
        for idx, attr_name in enumerate(mask_names):
            masks[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        masks = masks.reshape((masks.shape[0], 1, -1))

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._mask = nn.Parameter(torch.tensor(masks, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def load_ply(self, path):
        self.load_ply_sparse_gaussian(path )
        self.load_mlp_checkpoints(path.parent / "checkpoint.pth")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:  # Only for opacity, rotation. But seems they two are useless?
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    @task
    def training_statis(self, render_result: HACRenderResult):
        viewspace_point_tensor = render_result.viewspace_points
        opacity = render_result.neural_opacity
        update_filter = render_result.visibility_filter
        offset_selection_mask = render_result.selection_mask
        anchor_visible_mask = render_result.visible_mask

        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])

        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]


        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._mask = optimizable_tensors["mask"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):  # 3
            # for self.update_depth=3, self.update_hierachy_factor=4: 2**0, 2**1, 2**2
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            # for self.update_depth=3, self.update_hierachy_factor=4: 4**0, 4**1, 4**2
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()
                new_masks = torch.ones_like(candidate_anchor[:, 0:1]).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "mask": new_masks,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._mask = optimizable_tensors["mask"]
                self._opacity = optimizable_tensors["opacity"]

    @task
    def adjust_anchor(self, opt: HACOptimizationParams):
        # check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):

        check_interval = opt.update_interval
        success_threshold = opt.success_threshold
        grad_threshold =  opt.densify_grad_threshold
        min_opacity = opt.min_opacity
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def capture_mlp(self):
        checkpoint = {
                'opacity_mlp': self.mlp_opacity.state_dict(),
                # 'mlp_feature_bank': self.mlp_feature_bank.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                'encoding_xyz': self.encoding_xyz.state_dict(),
                'grid_mlp': self.mlp_grid.state_dict(),
                'deform_mlp': self.mlp_deform.state_dict(),
        }

        if self.use_feat_bank:
            checkpoint['mlp_feature_bank'] = self.mlp_feature_bank.state_dict()

        return checkpoint

    def restore_mlp(self, checkpoint):
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(checkpoint['mlp_feature_bank'])
        self.encoding_xyz.load_state_dict(checkpoint['encoding_xyz'])
        self.mlp_grid.load_state_dict(checkpoint['grid_mlp'])
        self.mlp_deform.load_state_dict(checkpoint['deform_mlp'])

    def save_mlp_checkpoints(self, path):
        torch.save(self.capture_mlp(), path)

    def load_mlp_checkpoints(self,path):
        checkpoint = torch.load(path)
        self.restore_mlp(checkpoint)

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
            mask = mask.unsqueeze(-1) + 0.0
            x_c = (2 - 1 / mag) * (x / mag)
            x = x_c * mask + x * (1 - mask)
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x

    @torch.no_grad()
    def estimate_final_bits(self):

        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        mask_anchor = self.get_mask_anchor

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _mask = self.get_mask[mask_anchor]
        hash_embeddings = self.get_encoding_params()

        ec = self.calc_entropy_params(_anchor)

        # Q_feat = Q_feat * entropy_context.Q_feat
        # Q_scaling = Q_scaling * entropy_context.Q_scaling
        # Q_offsets = Q_offsets * entropy_context.Q_offsets

        feat = self.ste_quantizer(_feat, ec.Q_feat)
        grid_scaling = self.ste_quantizer(_scaling, ec.Q_scaling)
        offsets = self.ste_quantizer(_grid_offsets, ec.Q_offsets.unsqueeze(1))

        # rate_pack = self.calc_sampled_rate(
        #     visible_mask,
        #     feat, grid_scaling, grid_offsets,
        #     # Q_feat, Q_scaling, Q_offsets,
        #     entropy_context
        # )


        # feat_context = self.calc_interp_feat(_anchor)  # [N_visible_anchor*0.2, 32]
        # mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
        #     torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
        # Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        # Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        # Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
        # _feat = (STE_multistep.apply(_feat, Q_feat)).detach()
        # grid_scaling = (STE_multistep.apply(_scaling, Q_scaling)).detach()
        # offsets = (STE_multistep.apply(_grid_offsets, Q_offsets.unsqueeze(1))).detach()
        offsets = offsets.view(-1, 3*self.n_offsets)
        mask_tmp = _mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets)

        bit_feat = self.entropy_gaussian.forward(feat, ec.mean_feat, ec.scale_feat, ec.Q_feat)
        bit_scaling = self.entropy_gaussian.forward(grid_scaling, ec.mean_scaling, ec.scale_scaling, ec.Q_scaling)
        bit_offsets = self.entropy_gaussian.forward(offsets, ec.mean_offsets, ec.scale_offsets, ec.Q_offsets)
        bit_offsets = bit_offsets * mask_tmp

        bit_anchor = _anchor.shape[0]*3*anchor_round_digits
        bit_feat = torch.sum(bit_feat).item()
        bit_scaling = torch.sum(bit_scaling).item()
        bit_offsets = torch.sum(bit_offsets).item()
        if self.ste_binary:
            bit_hash = get_binary_vxl_size((hash_embeddings+1)/2)[1].item()
        else:
            bit_hash = hash_embeddings.numel()*32
        bit_masks = get_binary_vxl_size(_mask)[1].item()

        # print(bit_anchor, bit_feat, bit_scaling, bit_offsets, bit_hash, bit_masks)

        log_info = f"Estimated sizes in MB: " \
                   f"anchor {round(bit_anchor / BIT2MB_SCALE, 4)}, " \
                   f"feat {round(bit_feat / BIT2MB_SCALE, 4)}, " \
                   f"scaling {round(bit_scaling / BIT2MB_SCALE, 4)}, " \
                   f"offsets {round(bit_offsets / BIT2MB_SCALE, 4)}, " \
                   f"hash {round(bit_hash / BIT2MB_SCALE, 4)}, " \
                   f"masks {round(bit_masks / BIT2MB_SCALE, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0] / BIT2MB_SCALE, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + self.get_mlp_size()[0]) / BIT2MB_SCALE, 4)}"

        logger.info(log_info)
        return log_info

    def encode_net(self):
        mlp_size = 0
        results = {}
        for n, p in self.named_parameters():
            if 'mlp' in n and 'deform' not in n:
                results[n] =  p.cpu().detach().numpy().tobytes()

        return results

    def decode_net(self, pack):
        for n, p in self.named_parameters():
            if 'mlp' in n and 'deform' not in n:
                params = torch.tensor(np.frombuffer(pack[n], dtype=np.float32)).cuda()
                p.data = params.reshape(p.shape)


    def calc_entropy_params(self, anchor_slice, flatten=False):
        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        # encode feat
        feat_context = self.calc_interp_feat(anchor_slice)  # [N_num, ?]
        # many [N_num, ?]
        mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(self.get_grid_mlp(feat_context),
                        split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets,
                                                3 * self.n_offsets, 1, 1, 1], dim=-1)

        if flatten:
            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean_feat.shape[-1]).view(-1)
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            mean_feat = mean_feat.contiguous().view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale_feat = torch.clamp(scale_feat.contiguous().view(-1), min=1e-9)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

        return EntropyContext(
            mean_feat, torch.clamp(scale_feat, min=1e-9),
            mean_scaling, torch.clamp(scale_scaling, min=1e-9),
            mean_offsets, torch.clamp(scale_offsets, min=1e-9),
            Q_feat, Q_scaling, Q_offsets
        )


    def encode_one_batch(self, _anchor,
                         _feat, _feat_mean,
                         _scaling, _scaling_mean,
                         _grid_offsets, _grid_offsets_mean,
                         _mask ):

        ec = self.calc_entropy_params(_anchor, flatten=True)

        feat = _feat.view(-1)  # [N_num*32]
        feat = STE_multistep.apply(feat, ec.Q_feat, _feat_mean)
        bs_feat = self.codec.encode(feat, ec.mean_feat, ec.scale_feat, ec.Q_feat)

        scaling = _scaling.view(-1)  # [N_num*6]
        scaling = STE_multistep.apply(scaling, ec.Q_scaling, _scaling_mean)
        bs_scaling = self.codec.encode(scaling, ec.mean_scaling, ec.scale_scaling, ec.Q_scaling)

        mask = _mask  # {0, 1}  # [N_num, K, 1]
        mask = mask.repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)  # [N_num*K*3]
        offsets = _grid_offsets.view(-1, 3 * self.n_offsets).view(-1)  # [N_num*K*3]
        offsets = STE_multistep.apply(offsets, ec.Q_offsets, _grid_offsets_mean)
        offsets[~mask] = 0.0

        bs_offset = self.codec.encode(offsets[mask], ec.mean_offsets[mask], ec.scale_offsets[mask], ec.Q_offsets[mask])

        # torch.cuda.empty_cache()
        return bs_feat, bs_scaling, bs_offset

    def decode_one_batch(self, _anchor, bs_feat, bs_scaling, bs_grid_offsets, masks_decoded):

        N_num = _anchor.shape[0]

        ec = self.calc_entropy_params(_anchor, flatten=True)

        feat_decoded = self.codec.decode_gaussian(ec.mean_feat, ec.scale_feat, ec.Q_feat, bs_feat)
        feat_decoded = feat_decoded.view(N_num, self.feat_dim)  # [N_num, 32]
        # feat_decoded_list.append(feat_decoded)

        scaling_decoded = self.codec.decode_gaussian(ec.mean_scaling, ec.scale_scaling, ec.Q_scaling, bs_scaling)
        scaling_decoded = scaling_decoded.view(N_num, 6)  # [N_num, 6]
        # scaling_decoded_list.append(scaling_decoded)

        masks_tmp = masks_decoded.repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)
        offsets_decoded_tmp = self.codec.decode_gaussian(ec.mean_offsets[masks_tmp], ec.scale_offsets[masks_tmp],
                                                     ec.Q_offsets[masks_tmp], bs_grid_offsets)
        offsets_decoded = torch.zeros_like(ec.mean_offsets)
        offsets_decoded[masks_tmp] = offsets_decoded_tmp
        offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]
        # offsets_decoded_list.append(offsets_decoded)
        #
        # xyz_decoded_list.append(anchor_sort)

        # torch.cuda.empty_cache()
        return feat_decoded, scaling_decoded, offsets_decoded

    def encode_anchor(self, ):
        # self.decoded_version = True
        mask_anchor = self.get_mask_anchor
        # self.decoded_version = False
        _anchor = self.get_anchor[mask_anchor]


        _quantized_v = self.get_quantized_v[mask_anchor]
        # print(_quantized_v[0])
        _quantized_v = _quantized_v.cpu().detach().numpy().astype(np.uint16)

        bs =  _quantized_v.tobytes()
        # byte_stream = io.BytesIO()
        # # 使用numpy.save将数组保存到BytesIO对象中
        # np.savez_compressed(byte_stream, _quantized_v)
        # bs = byte_stream.getvalue()


        # _quantized_v_decoded = torch.from_numpy(_quantized_v).cuda().to(torch.int32)
        _quantized_v_decoded = torch.from_numpy(_quantized_v.astype(np.int32)).cuda().to(torch.int32)
        interval = ((self.x_bound_max - self.x_bound_min) * Q_anchor + 1e-6)  # avoid 0, if max_v == min_v
        anchor_decoded = _quantized_v_decoded * interval + self.x_bound_min

        return anchor_decoded, bs

    def decode_anchor(self, bs):

        # new_buffer = io.BytesIO(bs)
        # with np.load(new_buffer) as data:
        #     recovered_data = data['arr_0']
        # _quantized_v_decoded = recovered_data.astype(np.int32)
        _quantized_v_decoded = np.frombuffer(bs, dtype=np.uint16).astype(np.int32)
        _quantized_v_decoded = torch.from_numpy(_quantized_v_decoded).cuda().to(torch.int32).reshape(-1, 3)
        interval = ((self.x_bound_max - self.x_bound_min) * Q_anchor + 1e-6)  # avoid 0, if max_v == min_v
        anchor_decoded = _quantized_v_decoded * interval + self.x_bound_min
        # mask_anchor = self.get_mask_anchor
        #
        # anchor_data = self._anchor.detach()
        # anchor_data[mask_anchor] = anchor_decoded
        self._anchor = nn.Parameter(anchor_decoded)

        # print(self._anchor[0])

    @torch.no_grad()
    def encode(self, path: io.BufferedWriter):

        # encoded_net = self.encode_net()
        # self.decode_net(encoded_net)
        logger.info('Start encoding')
        # self.update_anchor_bound()

        self.estimate_final_bits()
        # self.decoded_version = True
        mask_anchor = self.get_mask_anchor
        # self.decoded_version = False
        N_full = mask_anchor.shape[0]

        anchor = self.get_anchor[mask_anchor]
        feat = self._anchor_feat[mask_anchor]
        grid_offsets = self._offset[mask_anchor]
        scaling = self.get_scaling[mask_anchor]
        mask = self.get_mask[mask_anchor]

        # print('anchor', anchor.shape)

        anchor_decoded, bs_anchor = self.encode_anchor()


        N = anchor.shape[0]
        MAX_batch_size = 3_000
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)
        byte_bs_feat = 0
        byte_bs_scaling = 0
        byte_bs_offset = 0

        encoded_bs = []
        for s in range(steps):
            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)

            _anchor = anchor_decoded[N_start:N_end]
            _feat = feat[N_start:N_end]
            _scaling = scaling[N_start:N_end]
            _mask = mask[N_start:N_end]
            _grid_offsets = grid_offsets[N_start:N_end]

            encoded_bs.append(
                self.encode_one_batch(_anchor,
                                      _feat, feat.mean(),
                                      _scaling, scaling.mean(),
                                      _grid_offsets, grid_offsets.mean(),
                                      _mask)
            )

            byte_bs_feat += len(encoded_bs[-1][0])
            byte_bs_scaling += len(encoded_bs[-1][1])
            byte_bs_offset += len(encoded_bs[-1][2])


        hash_embeddings = self.get_encoding_params()  # {-1, 1}
        if self.ste_binary:
            bs_hash = self.codec.encode_bernoulli((hash_embeddings.view(-1) + 1) / 2)
        else:
            assert False
            bit_hash = hash_embeddings.numel() * 32

        bs_masks = self.codec.encode_bernoulli(mask)

        encoded_net = self.encode_net()
        byte_net = sum([len(bs) for bs in encoded_net])
        # self.decode_net(decoded_pack['net'])
        logger.info(f"anchor {len(bs_anchor)}")
        logger.info(f"bs_hash {len(bs_hash)}")
        logger.info(f"bs_masks {len(bs_masks)}")
        logger.info(f"encoded_net {byte_net}")
        logger.info(f"feat {byte_bs_feat}")
        logger.info(f"scaling {byte_bs_scaling}")
        logger.info(f"offset {byte_bs_offset}")
        final = {
            'anchor': bs_anchor,
            'data': encoded_bs,
            'hash': bs_hash,
            'mask': bs_masks,
            'net': encoded_net,
            'patched_infos': [N_full, N, MAX_batch_size],
            'bound': [self.x_bound_min, self.x_bound_max]
        }

        pickle.dump(final, path)

    def decode(self, path: io.BufferedReader):
        """
        pack structure
        {
            'anchor': bs_anchor,
            'data': encoded_bs,
            'hash': bs_hash,
            'mask': bs_masks,
            'net': encoded_net,
            'patched_infos': [N_full, N, MAX_batch_size]
        }
        Args:
            path:

        Returns:

        """
        logger.info('Start decoding...')
        self.decoded_version = True
        decoded_pack = pickle.load(path)

        self.x_bound_min, self.x_bound_max = decoded_pack['bound']



        [N_full, N, MAX_batch_size] = decoded_pack['patched_infos']
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)



        # hash_b_name = os.path.join(pre_path_name, 'hash.b')
        # masks_b_name = os.path.join(pre_path_name, 'masks.b')

        masks_decoded = self.codec.decode_bernoulli(N*self.n_offsets, decoded_pack['mask'])  # {0, 1}
        masks_decoded = masks_decoded.view(-1, self.n_offsets, 1)

        if self.ste_binary:
            N_hash = torch.zeros_like(self.get_encoding_params()).numel()
            hash_embeddings = self.codec.decode_bernoulli(N_hash, decoded_pack['hash'])  # {0, 1}
            hash_embeddings = (hash_embeddings * 2 - 1).to(torch.float32)
            hash_embeddings = hash_embeddings.view(-1, self.n_features_per_level)

        self.decode_net(decoded_pack['net'])


        if self.ste_binary:
            if self.use_2D:
                len_3D = self.encoding_xyz.encoding_xyz.params.shape[0]
                len_2D = self.encoding_xyz.encoding_xy.params.shape[0]
                # print(len_3D, len_2D, hash_embeddings.shape)
                self.encoding_xyz.encoding_xyz.params = nn.Parameter(hash_embeddings[0:len_3D])
                self.encoding_xyz.encoding_xy.params = nn.Parameter(hash_embeddings[len_3D:len_3D+len_2D])
                self.encoding_xyz.encoding_xz.params = nn.Parameter(hash_embeddings[len_3D+len_2D:len_3D+len_2D*2])
                self.encoding_xyz.encoding_yz.params = nn.Parameter(hash_embeddings[len_3D+len_2D*2:len_3D+len_2D*3])
            else:
                self.encoding_xyz.params = nn.Parameter(hash_embeddings)

        # anchor_decoded = torch.load(os.path.join(pre_path_name, 'anchor.pkl')).cuda()
        # _quantized_v_decoded = np.load(os.path.join(pre_path_name, '_quantized_v.npy')).astype(np.int32)
        # _quantized_v_decoded = torch.from_numpy(_quantized_v_decoded).cuda().to(torch.int32)
        # interval = ((self.x_bound_max - self.x_bound_min) * Q_anchor + 1e-6)  # avoid 0, if max_v == min_v
        # anchor_decoded = _quantized_v_decoded * interval + self.x_bound_min
        self.decode_anchor(decoded_pack['anchor'])

        # return
        xyz_decoded_list = []
        feat_decoded_list = []
        scaling_decoded_list = []
        offsets_decoded_list = []

        for s in range(steps):

            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)

            bs_feat, bs_scaling, bs_grid_offsets = decoded_pack['data'][s]
            anchor_sort = self._anchor[N_start:N_end]
            local_mask = masks_decoded[N_start:N_end]

            feat_decoded, scaling_decoded, offsets_decoded = self.decode_one_batch(
                anchor_sort, bs_feat, bs_scaling, bs_grid_offsets, local_mask
            )
            feat_decoded_list.append(feat_decoded)
            scaling_decoded_list.append(scaling_decoded)
            offsets_decoded_list.append(offsets_decoded)

        feat_decoded = torch.cat(feat_decoded_list, dim=0)
        scaling_decoded = torch.cat(scaling_decoded_list, dim=0)
        offsets_decoded = torch.cat(offsets_decoded_list, dim=0)

        self._anchor_feat = nn.Parameter(feat_decoded)
        self._offset = nn.Parameter(offsets_decoded)
        self._scaling = nn.Parameter(scaling_decoded)
        self._mask = nn.Parameter(masks_decoded.to(torch.float))

        rots = torch.zeros((N, 4), device="cuda")
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots.requires_grad_(False))





