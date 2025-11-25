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
import gzip
import io
import math
import pickle
import subprocess
import typing
from dataclasses import dataclass
from enum import Enum
from functools import reduce
import time
import os
import itertools

import numpy as np
from einops import repeat
from plyfile import PlyData, PlyElement
from loguru import logger
from torch_scatter import scatter_max
from sklearn.neighbors import LocalOutlierFactor
import torch
import torch.nn as nn
import torch.nn.functional as F

from splatwizard._cmod.simple_knn import distCUDA2    # noqa
from splatwizard.compression.entropy_model import EntropyGaussian
from .attribute import AttributeNetwork
from splatwizard.rasterizer.gaussian import GaussianRasterizationSettings, GaussianRasterizer
# from utils.encodings import \
#     STE_binary, STE_multistep, Quantize_anchor, \
#     anchor_round_digits, Q_anchor, \
#     encoder_anchor, decoder_anchor, \
#     encoder, decoder, \
#     encoder_gaussian, decoder_gaussian, \
#     get_binary_vxl_size

# from splatwizard.modules.triplane import *
from splatwizard.compression.cc_codec.encode import encode, RangeCoder, get_ac_max_val_latent, write_header
from splatwizard.compression.cc_codec.decode import decode, compute_offset, fast_get_neighbor
from splatwizard.modules.arm import ArmMLP, get_mu_scale, get_neighbor
from .config import CAT3DGSModelParams, CAT3DGSOptimizationParams, GenerateMode
from ...compression.entropy_codec import ArithmeticCodec
from ...compression.quantizer import Quantize_anchor, UniformQuantizer, anchor_round_digits, STEQuantizer, Q_anchor
from ...config import PipelineParams
from ...metrics.loss_utils import l1_func, ssim_func
from ...modules.dataclass import RenderResult, LossPack
from ...modules.gaussian_model import GaussianModel
from splatwizard.utils.general_utils import (
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric
)
from ...profiler import profile
from ...scene import CameraIterator
from ...scheduler import Scheduler, task
from ...utils.encodings import get_binary_vxl_size
from ...utils.graphics_utils import BasicPointCloud
from ...utils.misc import wrap_str

bit2MB_scale = 8 * 1024 * 1024
Q_EXP_SCALE = 2 ** 4





@dataclass
class RatePack:
    bit_per_param: torch.Tensor = None
    bit_per_feat_param: torch.Tensor = None
    bit_per_scaling_param: torch.Tensor = None
    bit_per_offsets_param: torch.Tensor = None
    feat_rate_per_param: torch.Tensor = None


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
    feat_rate: torch.Tensor = None
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
    feat_rate: torch.Tensor


@dataclass
class CAT3DGSRenderResult(RatePack, RenderResult):
    time_sub: typing.Union[torch.Tensor, None] = None
    selection_mask: typing.Union[torch.Tensor, None] = None
    neural_opacity: typing.Union[torch.Tensor, None] = None
    scaling: typing.Union[torch.Tensor, None] = None
    generated_gaussians: typing.Any = None
    entropy_constrained: bool = False
    step: int = 0


def set_requires_grad(module_or_param, value):
    if isinstance(module_or_param, torch.nn.Module):
        for param in module_or_param.parameters():
            if param.dtype.is_floating_point or param.dtype.is_complex:
                param.requires_grad = value
    elif isinstance(module_or_param, torch.nn.Parameter):
        if module_or_param.dtype.is_floating_point or module_or_param.dtype.is_complex:
            module_or_param.requires_grad = value
    else:
        raise TypeError("Input must be an nn.Module or nn.Parameter")


class CAT3DGS(GaussianModel, nn.Module):

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

    def __init__(self, model_param: CAT3DGSModelParams,
                 # feat_dim: int = 32,
                 # n_offsets: int = 5,
                 # voxel_size: float = 0.01,
                 # update_depth: int = 3,
                 # update_init_factor: int = 100,
                 # update_hierachy_factor: int = 4,
                 # use_feat_bank=False,
                 # n_features_per_level: int = 2,
                 # chcm_slices_list=[25, 25],
                 # chcm_for_offsets=False,
                 # chcm_for_scaling=False,
                 # ste_binary: bool = True,
                 # ste_multistep: bool = False,
                 # add_noise: bool = False,
                 # Q=1,
                 # use_2D: bool = True,
                 # decoded_version: bool = False,
                 # attribute_config: dict = None
                 ):
        nn.Module.__init__(self)
        GaussianModel.__init__(self)


        feat_dim = 50
        self.model_param = model_param
        self.feat_dim = feat_dim
        self.n_offsets =  model_param.n_offsets
        self.voxel_size = model_param.voxel_size
        self.update_depth = model_param.update_depth
        self.update_init_factor = model_param.update_init_factor
        logger.info(wrap_str("self.update_init_factor", self.update_init_factor))
        self.update_hierachy_factor = model_param.update_hierarchy_factor
        self.use_feat_bank = model_param.use_feat_bank
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.n_features_per_level = model_param.n_features_per_level
        self.ste_binary = model_param.ste_binary
        self.ste_multistep = model_param.ste_multistep
        self.add_noise = model_param.add_noise
        self.Q = model_param.Q
        self.use_2D = model_param.use_2D
        self.decoded_version = model_param.decoded_version

        self.attribute_net_config = model_param.attribute_net

        self.mode = model_param.enforce_mode if  model_param.enforce_mode is not None else GenerateMode.TRAINING_FULL_PRECISION
        # self.mode = GenerateMode.TRAINING_ENTROPY

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

        self.mask_weight = torch.ones(1, dtype=torch.float, device="cuda")
        self.p = None # For setup_triplane

        self.feature_net = None

        self.optimizer = None
        self.feature_arm_optimizer = None
        self.feature_net_optimizer = None
        self.feature_grid_optimizer = None


        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.temp_coder_list = None
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3 + 1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        mlp_input_feat_dim = feat_dim

        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_feat_dim + 3 + 1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, model_param.n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_feat_dim + 3 + 1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7 * self.n_offsets),
            # nn.Linear(feat_dim, 7),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_feat_dim + 3 + 1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3 * self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.mlp_context_from_f1 = nn.Sequential(
            nn.Linear(feat_dim // 2, 2 * feat_dim),
            nn.ReLU(True),
            nn.Linear(2 * feat_dim, feat_dim),
        ).cuda()

        self.mlp_chcm_list = nn.ModuleList()
        self.num_feat_slices = len(model_param.chcm_slices_list)
        self.chcm_slices_list = list(model_param.chcm_slices_list)
        input_dim = 0
        for i in range(self.num_feat_slices - 1):
            input_dim += self.chcm_slices_list[i]
            self.mlp_chcm_list.append(nn.Sequential(
                nn.Linear(input_dim, 2 * feat_dim),
                nn.ReLU(True),
                nn.Linear(2 * feat_dim, 2 * self.chcm_slices_list[i + 1]),
            ).cuda())

        self.chcm_for_offsets = model_param.chcm_for_offsets
        if self.chcm_for_offsets:
            self.mlp_chcm_offsets = nn.Sequential(
                nn.Linear(self.feat_dim, 2 * feat_dim),
                nn.ReLU(True),
                nn.Linear(2 * feat_dim, 6 * self.n_offsets),
            ).cuda()
        self.chcm_for_scaling = model_param.chcm_for_scaling
        if self.chcm_for_scaling:
            self.mlp_chcm_scaling = nn.Sequential(
                nn.Linear(self.feat_dim, 2 * feat_dim),
                nn.ReLU(True),
                nn.Linear(2 * feat_dim, 12),
            ).cuda()

        # self.feature_net = attribute_network(attribute_config, 'sof').cuda()
        # self.attribute_config = attribute_config
        self.noise_quantizer = UniformQuantizer()
        self.ste_quantizer = STEQuantizer()
        self.entropy_gaussian = EntropyGaussian(Q=1).cuda()

        self.codec = ArithmeticCodec()

    @property
    def xyz(self):
        return self._anchor


    def switch_to_decode(self):
        self.decoded_version = True

    def register_pre_task(self, scheduler, ppl: PipelineParams, opt: CAT3DGSOptimizationParams):
        scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
        scheduler.register_task(range(opt.iterations), task=self.update_cam_mask)
        scheduler.register_task(opt.triplane_init_fit_iter, task=self.setup_triplane, logging=True)

        scheduler.register_task(
            1, task=lambda: self.update_anchor_bound(),
            name='update_anchor_bound', logging=True
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


    def register_post_task(self, scheduler: Scheduler, ppl, opt: CAT3DGSOptimizationParams):
        scheduler.register_task(range(0, opt.update_until), task=self.training_statis)

        scheduler.register_task(
            range(opt.update_from, opt.pause_update_from, opt.update_interval),
            task=self.adjust_anchor, logging=True
        )

        scheduler.register_task(
            range(opt.pause_update_until, opt.update_until, opt.update_interval),
            task=self.adjust_anchor, logging=True
        )



    def switch_mode(self, mode):
        logger.info(f'switch mode {self.mode} -> {mode}')
        self.mode = mode

    @task
    def update_cam_mask(self, iteration: int, ppl: PipelineParams, opt: CAT3DGSOptimizationParams, cam_iterator: CameraIterator):
        # if mask_weight.shape[0] != gaussians.get_anchor.shape[0] or iteration == 0:  # 重算mask

        if self.mask_weight.shape[0] == self.get_anchor.shape[0]:
            if iteration != 0:
                return

        logger.info('update_cam_mask')
        # tmp_viewpoint_stack = scene.getTrainCameras().copy()
        anchor_visible_mask = torch.zeros(self.get_anchor.shape[0], dtype=torch.float, device="cuda")
        # n = len(tmp_viewpoint_stack)

        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        for viewpoint_cam in cam_iterator:
            # viewpoint_cam = tmp_viewpoint_stack.pop(0)

            voxel_visible_mask = self.prefilter_voxel(viewpoint_cam, ppl, background)
            anchor_visible_mask += voxel_visible_mask.to(torch.float)


        anchor_visible_mask = anchor_visible_mask.view(-1, 1)
        k = opt.cam_mask
        p = anchor_visible_mask / torch.mean(anchor_visible_mask)
        # the smaller the mask is, num of anchor is smaller
        p = p.repeat(1, 10).unsqueeze(-1)
        self.mask_weight = k * p
        self.p = p

    @task
    def setup_triplane(self, opt: CAT3DGSOptimizationParams):
        # if iteration == opt.triplane_init_fit_iter:
        self.set_feature_net() #self.p[:, 0].view(-1))
        self.training_setup_triplane(opt)

    def calc_entropy_params(self, anchor_slice, step=-1):
        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        feat_context, feat_rate = self.feature_net(anchor_slice, itr=step)
        mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(feat_context, split_size_or_sections=[
                self.feat_dim, self.feat_dim,
                6, 6,
                3 * self.n_offsets, 3 * self.n_offsets,
                1, 1, 1
            ], dim=-1)
        Q_feat = Q_feat * (1.1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1.1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1.1 + torch.tanh(Q_offsets_adj))

        return EntropyContext(
            mean_feat, torch.clamp(scale_feat, min=1e-9),
            mean_scaling, torch.clamp(scale_scaling, min=1e-9),
            mean_offsets, torch.clamp(scale_offsets, min=1e-9),
            Q_feat, Q_scaling, Q_offsets,
            feat_rate
        )


    def calc_sampled_rate(
            self,
            visible_mask, mask,
            feat, grid_scaling, grid_offsets,
            entropy_context: EntropyContext
    ):
        binary_grid_masks = self.get_mask(mask)[visible_mask]
        mask_anchor = self.get_mask_anchor(mask)[visible_mask]
        mask_anchor_bool = mask_anchor.to(torch.bool)
        mask_anchor_rate = (mask_anchor.sum() / mask_anchor.numel()).detach()

        anchor = self.get_anchor[visible_mask]
        choose_idx = torch.rand_like(anchor[:, 0]) <= 0.05
        choose_idx = choose_idx & mask_anchor_bool
        feat_chosen = feat[choose_idx]
        grid_scaling_chosen = grid_scaling[choose_idx]
        grid_offsets_chosen = grid_offsets[choose_idx].view(-1, 3 * self.n_offsets)
        means = entropy_context.mean_feat[choose_idx]
        scales = entropy_context.scale_feat[choose_idx]
        mean_scaling = entropy_context.mean_scaling[choose_idx]
        scale_scaling = entropy_context.scale_scaling[choose_idx]

        mean_offsets = entropy_context.mean_offsets[choose_idx]
        scale_offsets = entropy_context.scale_offsets[choose_idx]
        Q_feat = entropy_context.Q_feat[choose_idx]
        Q_scaling = entropy_context.Q_scaling[choose_idx]
        Q_offsets = entropy_context.Q_offsets[choose_idx]

        list_of_means = list(torch.split(means, split_size_or_sections=self.chcm_slices_list, dim=-1))
        list_of_scales = list(torch.split(scales, split_size_or_sections=self.chcm_slices_list, dim=-1))

        feat_list = list(torch.split(feat_chosen, split_size_or_sections=self.chcm_slices_list, dim=-1))

        for i in range(len(list_of_means)):
            if i == 0:
                bit_feat = self.entropy_gaussian(feat_list[i], list_of_means[i], list_of_scales[i],
                                                       Q_feat, self._anchor_feat.mean())
                bit_feats = bit_feat
                decoded_feat = feat_list[i]
            else:
                dmean, dscale = torch.split(self.get_chcm_mlp_list[i - 1](decoded_feat),
                                            split_size_or_sections=[self.chcm_slices_list[i],
                                                                    self.chcm_slices_list[i]], dim=-1)
                mean = list_of_means[i] + dmean
                scale = list_of_scales[i] + dscale
                bit_feat = self.entropy_gaussian.forward(feat_list[i], mean, scale, Q_feat,
                                                       self._anchor_feat.mean())
                bit_feats = torch.cat([bit_feats, bit_feat], dim=-1)
                decoded_feat = torch.cat([decoded_feat, feat_list[i]], dim=-1)

        if self.chcm_for_scaling:
            dmean_scaling, dscale_scaling = torch.split(self.get_chcm_mlp_scaling(decoded_feat),
                                                        split_size_or_sections=[6, 6], dim=-1)
            mean_scaling = mean_scaling + dmean_scaling
            scale_scaling = scale_scaling + dscale_scaling
        bit_scaling = self.entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling,
                                                  self.get_scaling.mean())

        if self.chcm_for_offsets:
            dmean_offsets, dscale_offsets = torch.split(self.get_chcm_mlp_offsets(decoded_feat),
                                                        split_size_or_sections=[3 * self.n_offsets,
                                                                                3 * self.n_offsets], dim=-1)
            mean_offsets = mean_offsets + dmean_offsets
            scale_offsets = scale_offsets + dscale_offsets
        bit_offsets = self.entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets,
                                                  self._offset.mean())
        binary_grid_masks_chosen = binary_grid_masks[choose_idx].repeat(1, 1, 3).view(-1, 3 * self.n_offsets)
        bit_offsets = bit_offsets * binary_grid_masks_chosen

        # bit_feat = self.entropy_gaussian.forward(feat_chosen, mean, scale, Q_feat, self._anchor_feat.mean())
        bit_per_feat_param = torch.sum(bit_feats) / bit_feats.numel() * mask_anchor_rate
        bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel() * mask_anchor_rate
        bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel() * mask_anchor_rate
        bit_per_param = (torch.sum(bit_feats) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                        (bit_feats.numel() + bit_scaling.numel() + bit_offsets.numel()) * mask_anchor_rate

        return RatePack(
            bit_per_param=bit_per_param,
            bit_per_feat_param=bit_per_feat_param,
            bit_per_scaling_param=bit_per_scaling_param,
            bit_per_offsets_param=bit_per_offsets_param,

        )

    def generate_neural_gaussians(
                self,
                viewpoint_camera,
                mask,
                visible_mask=None,
                mode=GenerateMode.TRAINING_FULL_PRECISION,
                step=-1
        ):
            # viewpoint_camera, pc: GaussianModel, visible_mask=None, is_training=False, step=0,
            #                       mask=None):
        ## view frustum filtering for acceleration

        time_sub = 0

        if visible_mask is None:
            visible_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device=self.get_anchor.device)

        anchor = self.get_anchor[visible_mask]
        #
        feat = self._anchor_feat[visible_mask]
        grid_offsets = self._offset[visible_mask]
        grid_scaling = self.get_scaling[visible_mask]
        binary_grid_masks = self.get_mask(mask)[visible_mask]
        # mask_anchor = self.get_mask_anchor(mask)[visible_mask]
        # mask_anchor_bool = mask_anchor.to(torch.bool)
        # mask_anchor_rate = (mask_anchor.sum() / mask_anchor.numel()).detach()
        # feat_rate = 0
        # bit_per_param = None
        # bit_per_feats_param = None
        # bit_per_scaling_param = None
        # bit_per_offsets_param = None
        rate_pack = RatePack()
        entropy_context = None
        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2
        if mode == GenerateMode.TRAINING_FULL_PRECISION or mode == GenerateMode.DECODING_AS_IS:
            # 全精度训练和解码后推断都不对feature进行处理
            pass
        elif mode == GenerateMode.TRAINING_QUANTIZED: #step > 3000 and step <= 10000:
                # quantization
                # feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat * 1.1
                # grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling * 1.1
                # grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets * 1.1

                feat = self.noise_quantizer(feat, Q_feat * 1.1)
                grid_scaling = self.noise_quantizer(grid_scaling, Q_scaling * 1.1)
                grid_offsets = self.noise_quantizer(grid_offsets, Q_offsets * 1.1)
        #TODO: register task
        # if step == 10000:  # update for triplane's bound
        #     pc.update_anchor_bound()
        elif mode == GenerateMode.TRAINING_ENTROPY: #if step > 10000 or step == -1:
                # feat_context, feat_rate = self.feature_net(self.get_anchor[visible_mask], itr=step)
                # scales, means, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                #     torch.split(feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3 * pc.n_offsets,
                #                                                       3 * pc.n_offsets, 1, 1, 1], dim=-1)
                # Q_feat = Q_feat * (1.1 + torch.tanh(Q_feat_adj))
                # Q_scaling = Q_scaling * (1.1 + torch.tanh(Q_scaling_adj))
                # Q_offsets = Q_offsets * (1.1 + torch.tanh(Q_offsets_adj))
                entropy_context = self.calc_entropy_params(anchor, step=step)
                # if step >= 39999:
                #     print(f"render.py, before quant, step{step}, {torch.isnan(feat).sum}")

                # feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
                # grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
                # grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.unsqueeze(
                #     1)
                feat = self.noise_quantizer(feat, entropy_context.Q_feat)
                grid_scaling = self.noise_quantizer(grid_scaling, entropy_context.Q_scaling)
                grid_offsets = self.noise_quantizer(grid_offsets, entropy_context.Q_offsets.unsqueeze(1))


                # choose_idx = torch.rand_like(anchor[:, 0]) <= 0.05
                # choose_idx = choose_idx & mask_anchor_bool
                # feat_chosen = feat[choose_idx]
                # grid_scaling_chosen = grid_scaling[choose_idx]
                # grid_offsets_chosen = grid_offsets[choose_idx].view(-1, 3 * pc.n_offsets)
                # means = means[choose_idx]
                # scales = scales[choose_idx]
                # mean_scaling = mean_scaling[choose_idx]
                # scale_scaling = scale_scaling[choose_idx]
                #
                # mean_offsets = mean_offsets[choose_idx]
                # scale_offsets = scale_offsets[choose_idx]
                # Q_feat = Q_feat[choose_idx]
                # Q_scaling = Q_scaling[choose_idx]
                # Q_offsets = Q_offsets[choose_idx]
                #
                # list_of_means = list(torch.split(means, split_size_or_sections=pc.chcm_slices_list, dim=-1))
                # list_of_scales = list(torch.split(scales, split_size_or_sections=pc.chcm_slices_list, dim=-1))
                #
                # feat_list = list(torch.split(feat_chosen, split_size_or_sections=pc.chcm_slices_list, dim=-1))
                #
                # for i in range(len(list_of_means)):
                #     if i == 0:
                #         bit_feat = pc.entropy_gaussian.forward(feat_list[i], list_of_means[i], list_of_scales[i],
                #                                                Q_feat, pc._anchor_feat.mean())
                #         bit_feats = bit_feat
                #         decoded_feat = feat_list[i]
                #     else:
                #         dmean, dscale = torch.split(pc.get_chcm_mlp_list[i - 1](decoded_feat),
                #                                     split_size_or_sections=[pc.chcm_slices_list[i],
                #                                                             pc.chcm_slices_list[i]], dim=-1)
                #         mean = list_of_means[i] + dmean
                #         scale = list_of_scales[i] + dscale
                #         bit_feat = pc.entropy_gaussian.forward(feat_list[i], mean, scale, Q_feat,
                #                                                pc._anchor_feat.mean())
                #         bit_feats = torch.cat([bit_feats, bit_feat], dim=-1)
                #         decoded_feat = torch.cat([decoded_feat, feat_list[i]], dim=-1)
                #
                # if self.chcm_for_scaling:
                #     dmean_scaling, dscale_scaling = torch.split(pc.get_chcm_mlp_scaling(decoded_feat),
                #                                                 split_size_or_sections=[6, 6], dim=-1)
                #     mean_scaling = mean_scaling + dmean_scaling
                #     scale_scaling = scale_scaling + dscale_scaling
                # bit_scaling = pc.entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling,
                #                                           pc.get_scaling.mean())
                #
                # if self.chcm_for_offsets:
                #     dmean_offsets, dscale_offsets = torch.split(pc.get_chcm_mlp_offsets(decoded_feat),
                #                                                 split_size_or_sections=[3 * pc.n_offsets,
                #                                                                         3 * pc.n_offsets], dim=-1)
                #     mean_offsets = mean_offsets + dmean_offsets
                #     scale_offsets = scale_offsets + dscale_offsets
                # bit_offsets = pc.entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets,
                #                                           pc._offset.mean())
                # binary_grid_masks_chosen = binary_grid_masks[choose_idx].repeat(1, 1, 3).view(-1, 3 * pc.n_offsets)
                # bit_offsets = bit_offsets * binary_grid_masks_chosen
                #
                # # bit_feat = pc.entropy_gaussian.forward(feat_chosen, mean, scale, Q_feat, pc._anchor_feat.mean())
                # bit_per_feats_param = torch.sum(bit_feats) / bit_feats.numel() * mask_anchor_rate
                # bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel() * mask_anchor_rate
                # bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel() * mask_anchor_rate
                # bit_per_param = (torch.sum(bit_feats) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                #                 (bit_feats.numel() + bit_scaling.numel() + bit_offsets.numel()) * mask_anchor_rate
                rate_pack = self.calc_sampled_rate(
                    visible_mask, mask,
                    feat, grid_scaling, grid_offsets,
                    # Q_feat, Q_scaling, Q_offsets,
                    entropy_context
                )

        elif mode == GenerateMode.TRAINING_STE_QUANTIZED:
        # elif not pc.decoded_version:  # test(validation)
        #     if step >= 10000 or step == -1:
        #         torch.cuda.synchronize();
        #         t1 = time.time()
        #         feat_context, feat_rate = pc.feature_net(pc.get_anchor[visible_mask], itr=step)
        #         mean1, scale1, mean2, scale2, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
        #             torch.split(feat_context,
        #                         split_size_or_sections=[pc.feat_dim // 2, pc.feat_dim // 2, pc.feat_dim // 2,
        #                                                 pc.feat_dim // 2, 6, 6, 3 * pc.n_offsets, 3 * pc.n_offsets, 1,
        #                                                 1, 1], dim=-1)
        #
        #         Q_feat = Q_feat * (1.1 + torch.tanh(Q_feat_adj))
        #         Q_scaling = Q_scaling * (1.1 + torch.tanh(Q_scaling_adj))
        #         Q_offsets = Q_offsets * (1.1 + torch.tanh(Q_offsets_adj))  # [N_visible_anchor, 1]
        #         feat = (STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())).detach()
        #         grid_scaling = (STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())).detach()
        #         grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets.unsqueeze(1), pc._offset.mean())).detach()
        #         torch.cuda.synchronize();
        #         time_sub = time.time() - t1
            entropy_context = self.calc_entropy_params(anchor)

            # Q_feat = Q_feat * entropy_context.Q_feat
            # Q_scaling = Q_scaling * entropy_context.Q_scaling
            # Q_offsets = Q_offsets * entropy_context.Q_offsets

            feat = self.ste_quantizer(feat, entropy_context.Q_feat)
            grid_scaling = self.ste_quantizer(grid_scaling, entropy_context.Q_scaling)
            grid_offsets = self.ste_quantizer(grid_offsets, entropy_context.Q_offsets.unsqueeze(1))

            rate_pack = self.calc_sampled_rate(
                visible_mask, mask,
                feat, grid_scaling, grid_offsets,
                # Q_feat, Q_scaling, Q_offsets,
                entropy_context
            )

        else:  ###test last decoding########################lastttttttttttttt
            # pass
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

        # if is_training:
        #     return xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feats_param, bit_per_scaling_param, bit_per_offsets_param, feat_rate
        # else:
        #     return xyz, color, opacity, scaling, rot, time_sub, feat_rate

        feat_rate = entropy_context.feat_rate if entropy_context is not None else None
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
            feat_rate=feat_rate,
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

        gss = self.generate_neural_gaussians(viewpoint_camera, self.mask_weight, visible_mask, mode=self.mode)

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

        return CAT3DGSRenderResult(
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
            feat_rate_per_param=gss.feat_rate,
            entropy_constrained=(gss.bit_per_param is not None),
            generated_gaussians=gss,
            # time_sub=gss.time_sub
            step=step
        )



    def get_mlp_size(self, digit=32):
        # mlp_size = 0
        # for n, p in self.named_parameters():
        #     if 'mlp' in n and 'deform' not in n:
        #         mlp_size += p.numel()*digit
        pmlp_o = self.count_parameters(self.mlp_opacity)
        pmlp_rs = self.count_parameters(self.mlp_cov)
        pmlp_color = self.count_parameters(self.mlp_color)
        parm = self.count_parameters(self.feature_net.attribute_net.grid.arm)
        parm2 = self.count_parameters(self.feature_net.attribute_net.grid.arm2)
        parm3 = self.count_parameters(self.feature_net.attribute_net.grid.arm3)
        pmlp_fa = sum(p.numel() for p in self.feature_net.get_mlp_parameters())
        pmlp_chcm = 0
        for i in range(self.num_feat_slices - 1):
            pmlp_chcm += self.count_parameters(self.mlp_chcm_list[i])
        if self.chcm_for_offsets:
            pmlp_chcm += self.count_parameters(self.mlp_chcm_offsets)
        if self.chcm_for_scaling:
            pmlp_chcm += self.count_parameters(self.mlp_chcm_scaling)
        total_mlp_p = pmlp_o + pmlp_rs + pmlp_color + parm + parm2 + parm3 + pmlp_fa + pmlp_chcm
        # print("# param of all MLP:", f"{total_mlp_p / 10 ** 6}M")
        # print("Size of all MLP(32bit):", f"{total_mlp_p * 4 / 10 ** 6}MB")
        mlp_size = total_mlp_p * digit
        return mlp_size, mlp_size / 8 / 1024 / 1024

    def loss_func(self, viewpoint_cam, render_result: CAT3DGSRenderResult, opt: CAT3DGSOptimizationParams) -> (torch.Tensor, LossPack):


        # image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg[
        #     "render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], \
        # render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        # feat_rate_per_param = render_pkg["feat_rate_per_param"]
        # bit_per_param = render_pkg["bit_per_param"]
        # bit_per_feat_param = render_pkg["bit_per_feat_param"]
        # bit_per_scaling_param = render_pkg["bit_per_scaling_param"]
        # bit_per_offsets_param = render_pkg["bit_per_offsets_param"]
        ####only consider offset and offset
        # if iteration % 2000 == 0 and bit_per_param is not None:
        #     ttl_size_feat_MB = bit_per_feat_param.item() * gaussians.get_anchor.shape[
        #         0] * gaussians.feat_dim / bit2MB_scale
        #     ttl_size_scaling_MB = bit_per_scaling_param.item() * gaussians.get_anchor.shape[0] * 6 / bit2MB_scale
        #     ttl_size_offsets_MB = bit_per_offsets_param.item() * gaussians.get_anchor.shape[
        #         0] * 3 * gaussians.n_offsets / bit2MB_scale
        #     ttl_size_MB = ttl_size_feat_MB + ttl_size_scaling_MB + ttl_size_offsets_MB
        #
        #     with torch.no_grad():
        #         grid_masks = gaussians._mask.data
        #         binary_grid_masks = (torch.sigmoid(grid_masks) > 0.01).float()
        #         mask_1_rate, mask_size_bit, mask_size_MB, mask_numel = get_binary_vxl_size(
        #             binary_grid_masks + 0.0)  # [0, 1] -> [-1, 1]
        #     a = "train"
        #     b1 = "scaling"
        #     b2 = "offset"
        #     b3 = "feat"
        #     log = {
        #         f'{a} {b1} bit_per_param': bit_per_scaling_param.item(),
        #         f'{a} {b1} ttl_size_MB': ttl_size_scaling_MB,
        #         f'{a} {b2} bit_per_param': bit_per_offsets_param.item(),
        #         f'{a} {b2} ttl_size_MB': ttl_size_offsets_MB,
        #         f'{a} {b3} bit_per_param': bit_per_feat_param,
        #         f'{a} {b1}+{b2}+{b3} bit_per_param': bit_per_param.item(),
        #         f'{a} {b1}+{b2}+{b3} ttl_size_MB': ttl_size_MB,
        #         f'{a} feature size(MB)': feat_rate_per_param / bit2MB_scale
        #     }
        #     logger.log_metrics(log, step=iteration)
        # gt_image = viewpoint_cam.original_image.cuda()
        # Ll1 = l1_loss(image, gt_image)
        image = render_result.rendered_image
        gt_image = viewpoint_cam.original_image.cuda()
        step = render_result.step

        Ll1 = l1_func(render_result.rendered_image, gt_image)


        ssim_loss = (1.0 - ssim_func(image, gt_image))
        scaling_reg = render_result.scaling.prod(dim=1).mean()
        scaffold_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg
        mask_loss = torch.mean(torch.sigmoid(self._mask))

        if step >= 3000:
            loss = scaffold_loss + max(1e-3, 0.3 * opt.lmbda) * mask_loss
        else:
            loss = scaffold_loss

        denom = self._anchor.shape[0] * (self.feat_dim + 6 + 3 * self.n_offsets)
        if opt.triplane_init_fit_iter + 5000 < step < opt.triplane_init_fit_iter + 6000:
            loss = render_result.feat_rate_per_param
        elif opt.triplane_init_fit_iter <= step and step is not None:
            loss = loss + opt.lmbda * (opt.lmbda_tri * render_result.feat_rate_per_param / denom + render_result.bit_per_param)

        loss_pack = LossPack(
            l1_loss=Ll1,
            ssim_loss=ssim_loss,
            loss=loss
        )
        return loss, loss_pack

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
            self.feature_grid_optimizer.state_dict() if self.feature_grid_optimizer is not None else None,
            self.feature_net_optimizer.state_dict() if  self.feature_net_optimizer is not None else None,
            self.feature_arm_optimizer.state_dict() if self.feature_arm_optimizer is not None else None,
            self.spatial_lr_scale,
            self.capture_mlp(),
            self.mask_weight,
            self.mode
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
                feature_grid_opt_dict,
                feature_net_opt_dict,
                feature_arm_opt_dict,
                self.spatial_lr_scale,
                mlp_checkpoint

            ) = model_args[:18]

        if len(model_args) == 19:
            self.mask_weight = model_args[18]
        elif len(model_args) == 20:
            self.mask_weight = model_args[18]
            self.mode = model_args[19]
        else:
            logger.warning('mask_weight not found, recalculate mask_weight before encoding')
            # (
            #     self.active_sh_degree,
            #     self._anchor,
            #     self._anchor_feat,
            #     self._offset,
            #     self._mask,
            #     self._scaling,
            #     self._rotation,
            #     self._opacity,
            #     x_bound_min,
            #     x_bound_max,
            #     self.max_radii2D,
            #     denom,
            #     opt_dict,
            #     feature_grid_opt_dict,
            #     feature_net_opt_dict,
            #     feature_arm_opt_dict,
            #     self.spatial_lr_scale,
            #     mlp_checkpoint,
            # ) = model_args
        self.restore_mlp(mlp_checkpoint)
        self.training_setup(training_args)


        if feature_grid_opt_dict is not None:
            self.training_setup_triplane(training_args)
        # print(f"optimizer state_dict: {opt_dict}")
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        if feature_grid_opt_dict is not None:
            self.feature_grid_optimizer.load_state_dict(feature_grid_opt_dict)
            self.feature_net_optimizer.load_state_dict(feature_net_opt_dict)
            self.feature_arm_optimizer.load_state_dict(feature_arm_opt_dict)
        torch.cuda.empty_cache()

    @property
    def get_scaling(self):
        if self.decoded_version:
            return self._scaling
        return 1.0 * self.scaling_activation(self._scaling)

    def get_mask(self, weighted_mask=None):
        if self.decoded_version:
            return self._mask
        mask_sig = torch.sigmoid(self._mask)
        if weighted_mask is not None:
            # print(mask_sig.shape, weighted_mask.shape)
            mask_sig = mask_sig * weighted_mask
        return ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig

    def get_mask_anchor(self, weighted_mask=None):
        with torch.no_grad():
            if self.decoded_version:
                mask_anchor = (torch.sum(self._mask, dim=1)[:, 0]) > 0
                return mask_anchor
            mask_sig = torch.sigmoid(self._mask)
            if weighted_mask is not None:
                mask_sig = mask_sig * weighted_mask
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
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_context_from_f1(self):
        return self.mlp_context_from_f1

    @property
    def get_chcm_mlp_list(self):
        return self.mlp_chcm_list

    @property
    def get_chcm_mlp_offsets(self):
        if self.chcm_for_offsets:
            return self.mlp_chcm_offsets
        else:
            return None

    @property
    def get_chcm_mlp_scaling(self):
        if self.chcm_for_scaling:
            return self.mlp_chcm_scaling
        else:
            return None

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
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        logger.info('anchor_bound_updated')

    @torch.no_grad()
    def set_feature_net(self, w=None):
        _anchor = self.get_anchor
        anchornp = _anchor.detach().cpu().numpy()

        x = anchornp[:, 0]
        y = anchornp[:, 1]
        z = anchornp[:, 2]
        xyz = np.vstack([x, y, z]).T
        lof = LocalOutlierFactor(n_neighbors=50, contamination=0.05)
        labels = lof.fit_predict(xyz)

        dense_points = torch.tensor(xyz[labels == 1], dtype=torch.float32).cuda()

        mean = torch.mean(dense_points, dim=0)
        centered_points = dense_points - mean
        cov_matrix = torch.cov(centered_points.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        rotated_points = torch.matmul(centered_points, sorted_eigenvectors)

        standardized_points = rotated_points / torch.sqrt(eigenvalues[sorted_indices])

        logger.info(f"mean: {mean}")
        logger.info(f"eigenvalues: {eigenvalues[sorted_indices]}")
        logger.info(f"eigenvectors: {sorted_eigenvectors}")
        w = None
        anchor_num = self.get_anchor.shape[0]
        x = round((anchor_num / 36) ** 0.5)

        # self.attribute_net_config.kplane_config.resolution = [x, x, x]

        self.attribute_net_config.kplane_config.resolution = (x, x, x)
        self.feature_net = AttributeNetwork(self.attribute_net_config).cuda()
        self.feature_net.attribute_net.grid.set_aabb(standardized_points, self.x_bound_max, self.x_bound_min)
        self.feature_net.attribute_net.grid.set_rotation_matrix(sorted_eigenvectors, mean,
                                                                torch.sqrt(eigenvalues[sorted_indices]))

    def set_feature_net_at_decode(self, resolution):


        self.attribute_net_config.kplane_config.resolution = resolution
        self.feature_net = AttributeNetwork(self.attribute_net_config).cuda()
        # self.feature_net.attribute_net.grid.set_aabb(standardized_points, self.x_bound_max, self.x_bound_min)
        # self.feature_net.attribute_net.grid.set_rotation_matrix(sorted_eigenvectors, mean,
        #                                                         torch.sqrt(eigenvalues[sorted_indices]))

    # @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data / voxel_size), axis=0) * voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        ratio = 1
        points = pcd.points[::ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0] * 0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        logger.info(f'Initial voxel_size: {self.voxel_size}')

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        masks = torch.ones((fused_point_cloud.shape[0], self.n_offsets, 1)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        logger.info(wrap_str("Number of points at initialisation : ", fused_point_cloud.shape[0]))

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

    def training_setup(self, training_args):

        # self.update_anchor_bound()  # 原版中未包含此调整，原因待查. Update:此处不应更新，因为加载时也会使用此函数。正确的时机在pre_task

        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale,
                 "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale,
                 "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init,
                 "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init,
                 "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale,
                 "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale,
                 "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init,
                 "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

            ]

        self.l_param = []
        for item in l:
            if isinstance(item['params'], list):
                for param in item['params']:
                    self.l_param.append(param)
            else:
                self.l_param.append(param)

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.offset_lr_final * self.spatial_lr_scale,
                                                       lr_delay_mult=training_args.offset_lr_delay_mult,
                                                       max_steps=training_args.offset_lr_max_steps)
        self.mask_scheduler_args = get_expon_lr_func(lr_init=training_args.mask_lr_init * self.spatial_lr_scale,
                                                     lr_final=training_args.mask_lr_final * self.spatial_lr_scale,
                                                     lr_delay_mult=training_args.mask_lr_delay_mult,
                                                     max_steps=training_args.mask_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                            lr_final=training_args.mlp_opacity_lr_final,
                                                            lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                            max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                        lr_final=training_args.mlp_cov_lr_final,
                                                        lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                        max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                          lr_final=training_args.mlp_color_lr_final,
                                                          lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                          max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                                    lr_final=training_args.mlp_featurebank_lr_final,
                                                                    lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                                    max_steps=training_args.mlp_featurebank_lr_max_steps)

        self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init,
                                                           lr_final=training_args.mlp_deform_lr_final,
                                                           lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                           max_steps=training_args.mlp_deform_lr_max_steps)

    def training_setup_triplane(self, training_args):
        feature_grid_l = [
            {'params': self.feature_net.get_grid_parameters(), 'lr': training_args.feature_lr, "name": "mlp_color"},
        ]
        self.feature_grid_optimizer = torch.optim.Adam(feature_grid_l, lr=0.0, eps=1e-15)

        feature_net_l = [
            {'params': self.feature_net.get_mlp_parameters(), 'lr': training_args.mlp_color_lr_init,
             "name": "mlp_color"},
            {'params': self.mlp_chcm_list.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
        ]
        if self.chcm_for_offsets:
            feature_net_l.append({'params': self.mlp_chcm_offsets.parameters(), 'lr': training_args.mlp_color_lr_init,
                                  "name": "mlp_color"})
        if self.chcm_for_scaling:
            feature_net_l.append({'params': self.mlp_chcm_scaling.parameters(), 'lr': training_args.mlp_color_lr_init,
                                  "name": "mlp_color"})
        self.feature_net_optimizer = torch.optim.Adam(feature_net_l, lr=0.0, eps=1e-15)

        feature_arm_l = [
            {'params': self.feature_net.get_arm_parameters(), 'lr': training_args.mlp_deform_lr_init,
             "name": "mlp_color"},
            {'params': self.feature_net.get_arm2_parameters(), 'lr': training_args.mlp_deform_lr_init,
             "name": "mlp_color"},
            {'params': self.feature_net.get_arm3_parameters(), 'lr': training_args.mlp_deform_lr_init,
             "name": "mlp_color"},
        ]
        self.feature_arm_optimizer = torch.optim.Adam(feature_arm_l, lr=0.0, eps=1e-15)

    def optimizer_step(self, render_result: RenderResult, opt: CAT3DGSOptimizationParams, step: int):
        if step < opt.triplane_init_fit_iter:
            self.optimizer.step()
        elif step >= opt.triplane_init_fit_iter and step < opt.triplane_init_fit_iter + 5000:
            if step == opt.triplane_init_fit_iter:
                set_requires_grad(self.feature_net, True)
                set_requires_grad(self.feature_net.attribute_net.grid.arm, False)
                set_requires_grad(self.feature_net.attribute_net.grid.arm2, False)
                set_requires_grad(self.feature_net.attribute_net.grid.arm3, False)
                set_requires_grad(self.mlp_context_from_f1, True)
            self.optimizer.step()
            self.feature_net_optimizer.step()
            self.feature_grid_optimizer.step()
        elif step >= opt.triplane_init_fit_iter + 5000 and step < opt.triplane_init_fit_iter + 6000:
            if step == opt.triplane_init_fit_iter + 6000:
                set_requires_grad(self.feature_net, False)
                set_requires_grad(self.feature_net.attribute_net.grid.arm, True)
                set_requires_grad(self.feature_net.attribute_net.grid.arm2, True)
                set_requires_grad(self.feature_net.attribute_net.grid.arm3, True)
                set_requires_grad(self.mlp_context_from_f1, False)
                for i in self.l_param:
                    set_requires_grad(i, False)
            self.feature_arm_optimizer.step()
        elif step >= opt.triplane_init_fit_iter + 6000 and step < opt.triplane_init_fit_iter + 9000:
            if step == opt.triplane_init_fit_iter + 6000:
                set_requires_grad(self.feature_net, True)
                set_requires_grad(self.feature_net.attribute_net.grid.grids, False)
                set_requires_grad(self.mlp_context_from_f1, True)
                for i in self.l_param:
                    set_requires_grad(i, True)
            self.feature_arm_optimizer.step()
            self.feature_net_optimizer.step()
            self.optimizer.step()
        elif step >= opt.triplane_init_fit_iter + 9000 and step < opt.triplane_init_fit_iter + 35000:
            if step == opt.triplane_init_fit_iter + 9000:
                set_requires_grad(self.feature_net.attribute_net.grid.grids, True)
            self.optimizer.step()
            self.feature_arm_optimizer.step()
            self.feature_net_optimizer.step()
            self.feature_grid_optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        if step >= opt.triplane_init_fit_iter:
            self.feature_net_optimizer.zero_grad(set_to_none=True)
            self.feature_arm_optimizer.zero_grad(set_to_none=True)
            self.feature_grid_optimizer.zero_grad(set_to_none=True)


    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters())

    @task
    def update_learning_rate(self, iteration: int):
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

        if iteration > 10000:
            for param_group in self.feature_net_optimizer.param_groups:
                if param_group["name"] == "mlp_color" and iteration > 10000:
                    lr = self.mlp_color_scheduler_args(iteration - 10000)
                    param_group['lr'] = lr

            for param_group in self.feature_grid_optimizer.param_groups:
                if param_group["name"] == "mlp_color" and iteration > 10000:
                    lr = self.mlp_color_scheduler_args(iteration - 10000)
                    param_group['lr'] = lr

            for param_group in self.feature_arm_optimizer.param_groups:
                if param_group["name"] == "mlp_color" and iteration > 15000:
                    lr = self.mlp_color_scheduler_args(iteration - 15000)
                    param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1] * self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._mask.shape[1] * self._mask.shape[2]):
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
                           np.asarray(plydata.elements[0]["z"])), axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key=lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key=lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        mask_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_mask")]
        mask_names = sorted(mask_names, key=lambda x: int(x.split('_')[-1]))
        masks = np.zeros((anchor.shape[0], len(mask_names)))
        for idx, attr_name in enumerate(mask_names):
            masks[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        masks = masks.reshape((masks.shape[0], 1, -1))

        self._anchor_feat = nn.Parameter(
            torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(
            torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._mask = nn.Parameter(
            torch.tensor(masks, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        del plydata
        torch.cuda.empty_cache()

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
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group[
                'name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:  # Only for opacity, rotation. But seems they two are useless?
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    @task
    def training_statis(self,
                        # viewspace_point_tensor, opacity, update_filter, offset_selection_mask,
                        # anchor_visible_mask
                        render_result: CAT3DGSRenderResult):

        viewspace_point_tensor = render_result.viewspace_points
        opacity = render_result.neural_opacity
        update_filter = render_result.visibility_filter
        offset_selection_mask = render_result.selection_mask
        anchor_visible_mask = render_result.visible_mask

        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity < 0] = 0
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
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group[
                'name']:
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
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_anchor(self, mask):
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
        init_length = self.get_anchor.shape[0] * self.n_offsets
        for i in range(self.update_depth):  # 3
            cur_threshold = threshold * ((self.update_hierachy_factor // 2) ** i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5 ** (i + 1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0] * self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')],
                                           dim=0)
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            size_factor = self.update_init_factor // (self.update_hierachy_factor ** i)
            cur_size = self.voxel_size * size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True,
                                                                        dim=0)

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i * chunk_size:(
                                                                                                                                i + 1) * chunk_size,
                                                                                         :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(
                    0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[
                    candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][
                    remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat(
                    [1, self.n_offsets, 1]).float().cuda()
                new_masks = torch.ones_like(candidate_anchor[:, 0:1]).unsqueeze(dim=1).repeat(
                    [1, self.n_offsets, 1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "mask": new_masks,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat(
                    [self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat(
                    [self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
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

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval * success_threshold * 0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0] * self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros(
            [self.get_anchor.shape[0] * self.n_offsets - self.offset_gradient_accum.shape[0], 1],
            dtype=torch.int32,
            device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity * self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval * success_threshold).squeeze(dim=1)  # [N, 1]
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
        if anchors_mask.sum() > 0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0] > 0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def capture_mlp(self):
        assert not self.use_feat_bank
        # mkdir_p(os.path.dirname(path))
        checkpoint = {
            'opacity_mlp': self.mlp_opacity.state_dict(),
            'cov_mlp': self.mlp_cov.state_dict(),
            'color_mlp': self.mlp_color.state_dict(),
            'context_mlp': self.mlp_chcm_list.state_dict(),
            'context_mlp_offsets': self.mlp_chcm_offsets.state_dict() if self.chcm_for_offsets else None,
            'context_mlp_scaling': self.mlp_chcm_scaling.state_dict() if self.chcm_for_scaling else None,
            'feat_base': self.feature_net.state_dict() if self.feature_net else None,
        }

        if self.use_feat_bank:
            checkpoint['mlp_feature_bank'] = self.mlp_feature_bank.state_dict()

        return checkpoint

    def restore_mlp(self, checkpoint):
        # checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(checkpoint['mlp_feature_bank'])
        self.mlp_chcm_list.load_state_dict(checkpoint['context_mlp'])
        if self.chcm_for_offsets:
            self.mlp_chcm_offsets.load_state_dict(checkpoint['context_mlp_offsets'])
        if self.chcm_for_scaling:
            self.mlp_chcm_scaling.load_state_dict(checkpoint['context_mlp_scaling'])

        if checkpoint['feat_base'] is not None:
            tri_resolution = checkpoint['feat_base']['attribute_net.resolution']
            logger.info(f'triplance resolution: {tri_resolution}')
            self.attribute_net_config.kplane_config.resolution = tri_resolution
            self.feature_net = AttributeNetwork(self.attribute_net_config).cuda()

            self.feature_net.load_state_dict(checkpoint['feat_base'])
        # print(f'feat_base: {self.feature_net}')
        # print('saving mlp checkpoints')
        # for name, param in self.feature_net.named_parameters():
        #     print(name, param.shape)
        #     print(param)
        torch.cuda.empty_cache()

    def save_mlp_checkpoints(self, path):
        torch.save(self.capture_mlp(), path)

    def load_mlp_checkpoints(self, path):
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
            dev = (2 * mag - 1) / mag ** 2 + 2 * x ** 2 * (
                    1 / mag ** 3 - (2 * mag - 1) / mag ** 4
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
    def estimate_final_bits(self, weighted_mask):
        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        mask_anchor = self.get_mask_anchor(weighted_mask)
        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]
        # print(f'1, {torch.isnan(_feat).sum()}')
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _mask = self.get_mask(weighted_mask)[mask_anchor]

        ec = self.calc_entropy_params(_anchor)


        # feat = self.noise_quantizer(_feat, entropy_context.Q_feat)
        # grid_scaling = self.noise_quantizer(_scaling, ec.Q_scaling)
        # grid_offsets = self.noise_quantizer(_grid_offsets, ec.Q_offsets.unsqueeze(1))

        # rate_pack = self.calc_sampled_rate(
        #     visible_mask, mask,
        #     feat, grid_scaling, grid_offsets,
        #     # Q_feat, Q_scaling, Q_offsets,
        #     entropy_context
        # )

        # feat_context, feat_rate = self.feature_net(_anchor, itr=-1)
        # scales, means, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
        #     torch.split(feat_context, split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets,
        #                                                       3 * self.n_offsets, 1, 1, 1], dim=-1)
        list_of_means = list(torch.split(ec.mean_feat, split_size_or_sections=self.chcm_slices_list, dim=-1))
        list_of_scales = list(torch.split(ec.scale_feat, split_size_or_sections=self.chcm_slices_list, dim=-1))

        # Q_feat = Q_feat * (1.1 + torch.tanh(Q_feat_adj))
        # Q_scaling = Q_scaling * (1.1 + torch.tanh(Q_scaling_adj))
        # Q_offsets = Q_offsets * (1.1 + torch.tanh(Q_offsets_adj))
        # _feat = (STE_multistep.apply(_feat, Q_feat)).detach()

        _feat = self.ste_quantizer(_feat, ec.Q_feat)
        # print(f'2, {torch.isnan(_feat).sum()}')
        feat_list = list(torch.split(_feat, split_size_or_sections=self.chcm_slices_list, dim=-1))
        # print(f'3, {torch.isnan(feat1).sum()}, {torch.isnan(feat2).sum()}')
        bit_feat_list = []
        for i in range(self.num_feat_slices):
            if i == 0:
                bit_feat = self.entropy_gaussian.forward(feat_list[i], list_of_means[i], list_of_scales[i], Q_feat)
                bit_feat_list.append(torch.sum(bit_feat).item())
                decoded_feat = feat_list[0]
            else:
                dmean, dscale = torch.split(self.mlp_chcm_list[i - 1](decoded_feat),
                                            split_size_or_sections=[self.chcm_slices_list[i], self.chcm_slices_list[i]],
                                            dim=-1)
                mean = list_of_means[i] + dmean
                scale = list_of_scales[i] + dscale
                bit_feat = self.entropy_gaussian.forward(feat_list[i], mean, scale, Q_feat)
                bit_feat_list.append(torch.sum(bit_feat).item())
                decoded_feat = torch.cat([decoded_feat, feat_list[i]], dim=-1)

        # dmean2, dscale2 = torch.split(self.mlp_context_from_f1(feat1), split_size_or_sections=[self.feat_dim//2, self.feat_dim//2], dim=-1)
        # scale2 = scale2 + dscale2
        # mean2 = mean2 + dmean2
        # mean = torch.cat([mean1, mean2], dim=-1)
        # scale = torch.cat([scale1, scale2], dim=-1)

        mean_scaling = ec.mean_scaling
        scale_scaling = ec.scale_scaling
        if self.chcm_for_scaling:
            dmean_scaling, dscale_scaling = torch.split(self.mlp_chcm_scaling(decoded_feat),
                                                        split_size_or_sections=[6, 6], dim=-1)
            mean_scaling = ec.mean_scaling + dmean_scaling
            scale_scaling = ec.scale_scaling + dscale_scaling

        mean_offsets = ec.mean_offsets
        scale_offsets = ec.scale_offsets
        if self.chcm_for_offsets:
            dmean_offsets, dscale_offsets = torch.split(self.mlp_chcm_offsets(decoded_feat),
                                                        split_size_or_sections=[3 * self.n_offsets, 3 * self.n_offsets],
                                                        dim=-1)
            mean_offsets = ec.mean_offsets + dmean_offsets
            scale_offsets = ec.scale_offsets + dscale_offsets
        # grid_scaling = (STE_multistep.apply(_scaling, Q_scaling)).detach()
        # offsets = (STE_multistep.apply(_grid_offsets, Q_offsets.unsqueeze(1))).detach()

        grid_scaling = self.noise_quantizer(_scaling, ec.Q_scaling).detach()
        offsets = self.noise_quantizer(_grid_offsets, ec.Q_offsets.unsqueeze(1)).detach()

        offsets = offsets.view(-1, 3 * self.n_offsets)
        mask_tmp = _mask.repeat(1, 1, 3).view(-1, 3 * self.n_offsets)
        # print(f'4, {torch.isnan(feat1).sum()}, {torch.isnan(feat2).sum()}')
        # bit_feat1 = self.entropy_gaussian.forward(feat1, mean1, scale1, Q_feat)
        # bit_feat2 = self.entropy_gaussian.forward(feat2, mean2, scale2, Q_feat)
        bit_scaling = self.entropy_gaussian.forward(grid_scaling, mean_scaling, scale_scaling, Q_scaling)
        bit_offsets = self.entropy_gaussian.forward(offsets, mean_offsets, scale_offsets, Q_offsets)
        bit_offsets = bit_offsets * mask_tmp
        bit_anchor = _anchor.shape[0] * 3 * anchor_round_digits
        # bit_feat1 = torch.sum(bit_feat1).item()
        # bit_feat2 = torch.sum(bit_feat2).item()
        bit_scaling = torch.sum(bit_scaling).item()
        bit_offsets = torch.sum(bit_offsets).item()

        bit_masks = get_binary_vxl_size(_mask)[1].item()
        ftrirate = ec.feat_rate / bit2MB_scale
        log_info = f"\nEstimated sizes in MB: " \
                   f"anchor {round(bit_anchor / bit2MB_scale, 4)}, " \
                   f"total feat {round(sum(bit_feat_list) / bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling / bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets / bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks / bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0] / bit2MB_scale, 4)}, " \
                   f"Triplane_f {round(ftrirate.item(), 4)}," \
                   f"Total {round((bit_anchor + sum(bit_feat_list) + bit_scaling + bit_offsets + bit_masks + self.get_mlp_size()[0]) / bit2MB_scale + ftrirate.item(), 4)}"
        return log_info

    def encode_net(self):
        mlp_size = 0
        results = {}
        for n, p in self.named_parameters():
            if 'mlp' in n and 'deform' not in n:
                results[n] =  p.cpu().detach().numpy().tobytes()
        for n, p in self.feature_net.named_parameters():

            if 'grid.grids.' in n:
                continue
            # print(n, p.shape, p.dtype)
            results['feature_net.' + n] = p.cpu().detach().numpy().tobytes()

        return results

    def decode_net(self, pack):
        for n, p in self.named_parameters():
            if 'mlp' in n and 'deform' not in n:
                # print(n)
                params = torch.tensor(np.frombuffer(pack[n], dtype=np.float32)).cuda()
                p.data = params.reshape(p.shape)
        for n, p in self.feature_net.named_parameters():
            if 'grid.grids.' in n:
                continue
            if 'resolution' in n or 'grid.aabb' in n:
                # print(n, '64')
                params = torch.tensor(np.frombuffer(pack['feature_net.' + n], dtype=np.int64)).cuda()
            else:
                params = torch.tensor(np.frombuffer(pack['feature_net.' + n], dtype=np.float32)).cuda()

            p.data = params.reshape(p.shape)


    def encode_anchor(self, weighted_mask):
        # self.decoded_version = True
        mask_anchor = self.get_mask_anchor(weighted_mask)
        # self.decoded_version = False
        _anchor = self.get_anchor[mask_anchor]


        _quantized_v = self.get_quantized_v[mask_anchor]
        # print(_quantized_v[0])
        _quantized_v = _quantized_v.cpu().detach().numpy().astype(np.uint16)
        bs =  _quantized_v.tobytes()

        # _quantized_v_decoded = torch.from_numpy(_quantized_v).cuda().to(torch.int32)
        _quantized_v_decoded = torch.from_numpy(_quantized_v.astype(np.int32)).cuda().to(torch.int32)
        interval = ((self.x_bound_max - self.x_bound_min) * Q_anchor + 1e-6)  # avoid 0, if max_v == min_v
        anchor_decoded = _quantized_v_decoded * interval + self.x_bound_min

        return anchor_decoded, bs

    def decode_anchor(self, bs):
        # anchor_decoded = torch.load(os.path.join(pre_path_name, 'anchor.pkl')).cuda()
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


    def encode_one_batch(self, _anchor,
                         _feat, _feat_mean,
                         _scaling, _scaling_mean,
                         _grid_offsets, _grid_offsets_mean,
                         _mask ):
        N_num = _anchor.shape[0]
        ec = self.calc_entropy_params(_anchor)

        # feat = _feat.view(-1)  # [N_num*32]
        # feat = self.ste_quantizer(feat, ec.Q_feat, _feat_mean)
        # bs_feat = self.codec.encode(feat, ec.mean_feat, ec.scale_feat, ec.Q_feat)

        anchor_sort = _anchor # anchor[N_start:N_end] #[indices]  # [N_num, 3]
        # feat_context, feat_rate = self.feature_net(anchor_sort, itr=-1)

        means = list(torch.split(ec.mean_feat, split_size_or_sections=self.chcm_slices_list, dim=-1))
        scales = list(torch.split(ec.scale_feat, split_size_or_sections=self.chcm_slices_list, dim=-1))

        # Q_feat_list = [Q_feat * (1.1 + torch.tanh(Q_feat_adj.contiguous().repeat(1, ch).view(-1))) for ch in
        #                self.chcm_slices_list]
        Q_feat_list = [ec.Q_feat.contiguous().repeat(1, ch).view(-1) for ch in self.chcm_slices_list]
        # Q_feat_adj2 = Q_feat_adj.contiguous().repeat(1, self.feat_dim//2)

        # feat = _feat[N_start:N_end][indices]  # [N_num*32]

        _feat_list = list(torch.split(_feat, split_size_or_sections=self.chcm_slices_list, dim=-1))
        bit_feat_lists = []
        feat_lists = []
        decoded_feat = None
        for i in range(self.num_feat_slices):
            temp_feat = _feat_list[i].reshape(-1)
            temp_feat = self.ste_quantizer(temp_feat, Q_feat_list[i], _feat.mean())
            if decoded_feat is None: # i == 0
                mean = means[i].reshape(-1)
                scale = torch.clamp(scales[i].reshape(-1), min=1e-9)
            else:
                temp_feat = _feat_list[i].reshape(-1)
                temp_feat = self.ste_quantizer(temp_feat, Q_feat_list[i], _feat.mean())

                dmean, dscale = torch.split(self.mlp_chcm_list[i - 1](decoded_feat),
                                            split_size_or_sections=[self.chcm_slices_list[i],
                                                                    self.chcm_slices_list[i]], dim=-1)
                mean = (means[i] + dmean).reshape(-1)
                scale = torch.clamp((scales[i] + dscale).reshape(-1), min=1e-9)

            bs_feat = self.codec.encode(temp_feat, mean, scale, Q_feat_list[i])
            bit_feat_lists.append(bs_feat)

            temp_feat = temp_feat.reshape(N_num, -1)
            feat_lists.append(temp_feat)
            decoded_feat = torch.cat([decoded_feat, temp_feat], dim=-1) if decoded_feat is not None else temp_feat


        if self.chcm_for_scaling:
            dmean_scaling, dscale_scaling = torch.split(self.mlp_chcm_scaling(decoded_feat),
                                                        split_size_or_sections=[6, 6], dim=-1)
            mean_scaling = ec.mean_scaling + dmean_scaling
            scale_scaling = ec.scale_scaling + dscale_scaling
        else:
            mean_scaling = ec.mean_scaling
            scale_scaling = ec.scale_scaling
        if self.chcm_for_offsets:
            dmean_offsets, dscale_offsets = torch.split(self.mlp_chcm_offsets(decoded_feat),
                                                        split_size_or_sections=[3 * self.n_offsets, 3 * self.n_offsets],
                                                        dim=-1)
            mean_offsets = ec.mean_offsets + dmean_offsets
            scale_offsets = ec.scale_offsets + dscale_offsets
        else:
            mean_offsets = ec.mean_offsets
            scale_offsets = ec.scale_offsets


        Q_scaling = ec.Q_scaling.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
        # Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
        mean_scaling = mean_scaling.contiguous().view(-1)
        # mean_offsets = mean_offsets.contiguous().view(-1)
        scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
        scale_scaling = scale_scaling.contiguous().view(-1)
        # scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
        # # Q_feat1 = Q_feat * (1.1 + torch.tanh(Q_feat_adj1))
        # # Q_feat2 = Q_feat * (1.1 + torch.tanh(Q_feat_adj2))
        # Q_scaling = Q_scaling * (1.1 + torch.tanh(Q_scaling_adj))
        # Q_offsets = Q_offsets * (1.1 + torch.tanh(Q_offsets_adj))

        scaling = _scaling.view(-1)  # [N_num*6]
        scaling = self.ste_quantizer(scaling, Q_scaling, self.get_scaling.mean())
        # torch.cuda.synchronize();
        # t0 = time.time()
        bs_scaling = self.codec.encode(scaling, mean_scaling, scale_scaling, Q_scaling)
        # torch.cuda.synchronize();
        # t_codec += time.time() - t0
        # bit_scaling_list.append(bit_scaling)
        # min_scaling_list.append(min_scaling)
        # max_scaling_list.append(max_scaling)
        # scaling_list.append(scaling)
        #

        Q_offsets = ec.Q_offsets.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
        mask = _mask #[N_start:N_end]#[indices]  # {0, 1}  # [N_num, K, 1]
        mask = mask.repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)  # [N_num*K*3]
        offsets = _grid_offsets.view(-1, 3 * self.n_offsets).view(-1)  # [N_num*K*3]
        offsets = self.ste_quantizer(offsets, Q_offsets, self._offset.mean())
        offsets[~mask] = 0.0
        # torch.cuda.synchronize();
        # t0 = time.time()
        bs_offset = self.codec.encode(offsets[mask], mean_offsets.reshape(-1)[mask], scale_offsets.reshape(-1)[mask], Q_offsets[mask])

        return bit_feat_lists, bs_scaling, bs_offset

    def decode_one_batch(self, _anchor, bs_feat, bs_scaling, bs_grid_offsets, masks_decoded):

        N_num = _anchor.shape[0]

        ec = self.calc_entropy_params(_anchor)

        # feat_decoded = self.codec.decode_gaussian(ec.mean_feat, ec.scale_feat, ec.Q_feat, bs_feat)
        # feat_decoded = feat_decoded.view(N_num, self.feat_dim)  # [N_num, 32]
        # # feat_decoded_list.append(feat_decoded)
        #
        # scaling_decoded = self.codec.decode_gaussian(ec.mean_scaling, ec.scale_scaling, ec.Q_scaling, bs_scaling)
        # scaling_decoded = scaling_decoded.view(N_num, 6)  # [N_num, 6]
        # # scaling_decoded_list.append(scaling_decoded)
        #
        # masks_tmp = masks_decoded.repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)
        # offsets_decoded_tmp = self.codec.decode_gaussian(ec.mean_offsets[masks_tmp], ec.scale_offsets[masks_tmp],
        #                                              ec.Q_offsets[masks_tmp], bs_grid_offsets)
        # offsets_decoded = torch.zeros_like(ec.mean_offsets)
        # offsets_decoded[masks_tmp] = offsets_decoded_tmp
        # offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]
        #
        #         # encode feat
        #         feat_context, feat_rate = self.feature_net(anchor_decoded[N_start:N_end], itr=-1)
        #         scales, means, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
        #             torch.split(feat_context,
        #                         split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3 * self.n_offsets,
        #                                                 3 * self.n_offsets, 1, 1, 1], dim=-1)
        means = list(torch.split(ec.mean_feat, split_size_or_sections=self.chcm_slices_list, dim=-1))
        scales = list(torch.split(ec.scale_feat, split_size_or_sections=self.chcm_slices_list, dim=-1))
        # Q_feat_list = [Q_feat * (1.1 + torch.tanh(Q_feat_adj.contiguous().repeat(1, ch).view(-1))) for ch in
        #                        self.chcm_slices_list]
        Q_feat_list = [ec.Q_feat.contiguous().repeat(1, ch).view(-1) for ch in self.chcm_slices_list]
        #         # feat = _feat[N_start:N_end][indices]  # [N_num*32]
        #
        #         # _feat_list = list(torch.split(feat, split_size_or_sections=self.chcm_slices_list, dim=-1))
        #
        #         # Q_feat_list.append(Q_feat * (1 + torch.tanh(Q_feat_adj.contiguous())))
        #         # Q_scaling_list.append(Q_scaling * (1 + torch.tanh(Q_scaling_adj.contiguous())))
        #         # Q_offsets_list.append(Q_offsets * (1 + torch.tanh(Q_offsets_adj.contiguous())))
        #
        for i in range(self.num_feat_slices):
            if i == 0:
                mean = means[i].contiguous().view(-1)
                scale = torch.clamp(scales[i].contiguous().view(-1), min=1e-9)
                feat = self.codec.decode_gaussian(mean, scale, Q_feat_list[i], bs_feat[i])
                feat = feat.view(N_num, self.chcm_slices_list[i])
                # temp_feat = _feat_list[i].reshape(-1)
                # temp_feat = STE_multistep.apply(temp_feat, Q_feat_list[i], _feat.mean())
                # temp_feat = temp_feat.reshape(N_num, -1)
                decoded_feat = feat


            else:
                dmean, dscale = torch.split(self.mlp_chcm_list[i - 1](decoded_feat),
                                            split_size_or_sections=[self.chcm_slices_list[i],
                                                                    self.chcm_slices_list[i]], dim=-1)
                mean = means[i] + dmean
                scale = scales[i] + dscale
                # if s == 5:
                #     print(mean[0], scale[0])
                mean = mean.contiguous().view(-1)
                scale = torch.clamp(scale.contiguous().view(-1), min=1e-9)
                feat = self.codec.decode_gaussian(mean, scale, Q_feat_list[i], bs_feat[i])
                feat = feat.view(N_num, self.chcm_slices_list[i])
                # temp_feat = _feat_list[i].reshape(-1)
                # temp_feat = STE_multistep.apply(temp_feat, Q_feat_list[i], _feat.mean())
                # temp_feat = temp_feat.reshape(N_num, -1)
                decoded_feat = torch.cat([decoded_feat, feat], dim=-1)
                # print(feat_b_name_list[i])
                # print(torch.mean(feat-temp_feat))
        #         feat_decoded_list.append(decoded_feat)
        if self.chcm_for_scaling:
            dmean_scaling, dscale_scaling = torch.split(self.mlp_chcm_scaling(decoded_feat),
                                                        split_size_or_sections=[6, 6], dim=-1)
            mean_scaling = ec.mean_scaling + dmean_scaling
            scale_scaling = ec.scale_scaling + dscale_scaling
        else:
            mean_scaling = ec.mean_scaling
            scale_scaling = ec.scale_scaling
        if self.chcm_for_offsets:
            dmean_offsets, dscale_offsets = torch.split(self.mlp_chcm_offsets(decoded_feat),
                                                        split_size_or_sections=[3 * self.n_offsets, 3 * self.n_offsets],
                                                        dim=-1)
            mean_offsets = ec.mean_offsets + dmean_offsets
            scale_offsets = ec.scale_offsets + dscale_offsets
        else:
            mean_offsets = ec.mean_offsets
            scale_offsets = ec.scale_offsets

        Q_scaling = ec.Q_scaling.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
        # Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
        mean_scaling = mean_scaling.contiguous().view(-1)
        # mean_offsets = mean_offsets.contiguous().view(-1)
        scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
        scale_scaling = scale_scaling.contiguous().view(-1)
        # scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
        # # Q_feat1 = Q_feat * (1.1 + torch.tanh(Q_feat_adj1))
        # # Q_feat2 = Q_feat * (1.1 + torch.tanh(Q_feat_adj2))
        # Q_scaling = Q_scaling * (1.1 + torch.tanh(Q_scaling_adj))
        # Q_offsets = Q_offsets * (1.1 + torch.tanh(Q_offsets_adj))

        # scaling = _scaling.view(-1)  # [N_num*6]
        # scaling = self.ste_quantizer(scaling, Q_scaling, self.get_scaling.mean())
        # torch.cuda.synchronize();
        # t0 = time.time()
        scaling_decoded = self.codec.decode_gaussian( mean_scaling, scale_scaling, Q_scaling, bs_scaling)
        scaling_decoded = scaling_decoded.view(N_num, 6)

        Q_offsets = ec.Q_offsets.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
        masks_tmp = masks_decoded.repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)
        offsets_decoded_tmp = self.codec.decode_gaussian(mean_offsets.reshape(-1)[masks_tmp], scale_offsets.reshape(-1)[masks_tmp],
                                                     Q_offsets[masks_tmp], bs_grid_offsets)

        offsets_decoded = torch.zeros_like(mean_offsets).reshape(-1)
        offsets_decoded[masks_tmp] = offsets_decoded_tmp
        offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]



        return decoded_feat, scaling_decoded, offsets_decoded



    @torch.no_grad()
    def encode(self, path: io.BufferedWriter):
        logger.info('Start encoding')
        # torch.backends.cudnn.enabled = False
        # torch.backends.cudnn.allow_tf32 = False
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        # print(self.get_mlp_size())

        self.estimate_final_bits(self.mask_weight)

        mask_anchor = self.get_mask_anchor(self.mask_weight)

        anchor = self.get_anchor[mask_anchor]
        feat = self._anchor_feat[mask_anchor]
        grid_offsets = self._offset[mask_anchor]
        scaling = self.get_scaling[mask_anchor]
        mask = self.get_mask(self.mask_weight)[mask_anchor]

        N = anchor.shape[0]
        MAX_batch_size = 5000
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)
        # print('steps', steps)

        anchor_decoded, bs_anchor = self.encode_anchor(self.mask_weight)
        anchor_decoded = anchor

        triplane_pack, total_triplane_bytes = self.encode_triplane()
        self.feature_net = self.feature_net.cuda()
        self.feature_net.attribute_net.grid.non_zero_pixel_ctx_index = self.feature_net.attribute_net.grid.non_zero_pixel_ctx_index.cuda()
        bit_feat_lists = [[] for _ in range(self.num_feat_slices)]

        # bit_feat2_list = []
        # bit_scaling_list = []
        # bit_offsets_list = []
        # anchor_infos_list = []
        # indices_list = []
        # min_feat_lists = [[] for _ in range(self.num_feat_slices)]
        # max_feat_lists = [[] for _ in range(self.num_feat_slices)]

        byte_bs_feat = 0
        byte_bs_scaling = 0
        byte_bs_offset = 0

        encoded_bs = []
        for s in range(steps):
            # print('encode', s)
            N_num = min(MAX_batch_size, N - s * MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s + 1) * MAX_batch_size, N)

            indices = torch.tensor(data=range(N_num), device='cuda', dtype=torch.long)  # [N_num]

            # indices_list.append(indices + N_start)

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

            # current_bs_pack =
            byte_bs_feat += sum([len(b) for b in encoded_bs[-1][0]])
            byte_bs_scaling += len(encoded_bs[-1][1])
            byte_bs_offset += len(encoded_bs[-1][2])

        bs_masks = self.codec.encode_bernoulli(mask)


        encoded_net = self.encode_net()
        byte_net = sum([len(bs) for bs in encoded_net.values()])
        # self.decode_net(decoded_pack['net'])
        logger.info(f"anchor {len(bs_anchor)}")
        logger.info(f"triplane {total_triplane_bytes}")
        logger.info(f"bs_masks {len(bs_masks)}")
        logger.info(f"encoded_net {byte_net}")
        logger.info(f"feat {byte_bs_feat}")
        logger.info(f"scaling {byte_bs_scaling}")
        logger.info(f"offset {byte_bs_offset}")

        final = {
            'anchor': bs_anchor,
            'data': encoded_bs,
            'mask': bs_masks,
            'net': encoded_net,
            'triplane': triplane_pack,
            'bound': [self.x_bound_min, self.x_bound_max],
            'patched_infos': [ N, MAX_batch_size],
        }

        pickle.dump(final, path)
        # pickle_data = pickle.dumps(final, protocol=pickle.HIGHEST_PROTOCOL)
        # compressed = gzip.compress(pickle_data, compresslevel=9)
        # path.write(compressed)


    @torch.no_grad()
    def decode(self, path: io.BufferedReader):
        # torch.backends.cudnn.enabled = False
        # torch.backends.cudnn.allow_tf32 = False
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        # torch.cuda.synchronize();
        # t1 = time.time()
        # print('Start decoding ...')
        # [N_full, N, MAX_batch_size, anchor_infos_list, min_feat_lists, max_feat_lists, min_scaling_list,
        #  max_scaling_list, min_offsets_list, max_offsets_list, prob_masks, triplane_coder_list] = patched_infos
        # [N, MAX_batch_size]
        # steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        logger.info('Start decoding...')
        self.decoded_version = True
        decoded_pack = pickle.load(path)

        # decompressed = gzip.decompress(path.read())
        # decoded_pack =  pickle.loads(decompressed)


        self.x_bound_min, self.x_bound_max = decoded_pack['bound']



        [N, MAX_batch_size] = decoded_pack['patched_infos']
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)
        self.decode_anchor(decoded_pack['anchor'])
        resolution  = torch.tensor(np.frombuffer(decoded_pack['net']['feature_net.attribute_net.resolution'], dtype=np.int64)).cuda()
        self.set_feature_net_at_decode(resolution)
        self.decode_net(decoded_pack['net'])

        feat_decoded_list = []
        scaling_decoded_list = []
        offsets_decoded_list = []
        self.decode_triplane(decoded_pack['triplane'])
        #
        # masks_b_name = os.path.join(pre_path_name, 'masks.b')
        #
        # p = torch.zeros(size=[N, self.n_offsets, 1], device='cuda').to(torch.float32)
        # p[...] = prob_masks
        # masks_decoded = decoder(p.view(-1), masks_b_name)  # {-1, 1}
        # masks_decoded = (masks_decoded + 1) / 2  # {0, 1}
        # masks_decoded = masks_decoded.view(-1, self.n_offsets, 1)
        #
        # mask_anchor = self.get_mask_anchor(weighted_mask)
        masks_decoded = self.codec.decode_bernoulli(N*self.n_offsets, decoded_pack['mask'])  # {0, 1}
        masks_decoded = masks_decoded.view(-1, self.n_offsets, 1)



        Q_feat_list = []
        Q_scaling_list = []
        Q_offsets_list = []
    #
    #     anchor_decoded = torch.load(os.path.join(pre_path_name, 'anchor.pkl')).cuda()
        for s in range(steps):
            N_num = min(MAX_batch_size, N - s * MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s + 1) * MAX_batch_size, N)

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
        # self._anchor_feat = nn.Parameter(self._anchor_feat[mask_anchor])
        # assert self._offset.shape == _offset.shape
        self._offset = nn.Parameter(offsets_decoded)
        # self._offset = nn.Parameter(self._offset[mask_anchor])
        # If change the following attributes, decoded_version must be set True
        # self.decoded_version = True
        # assert self.get_anchor.shape == _anchor.shape
        # self._anchor = nn.Parameter(_anchor)
        # assert self._scaling.shape == _scaling.shape
        self._scaling = nn.Parameter(scaling_decoded)
        # self._scaling = nn.Parameter(self._scaling[mask_anchor])
        # assert self._mask.shape == _mask.shape
        self._mask = nn.Parameter(masks_decoded.to(torch.float))
        # self._mask = nn.Parameter(self._mask[mask_anchor])

        rots = torch.zeros((N, 4), device="cuda")
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots.requires_grad_(False))

    #
    def encode_triplane(self):
        # encode_start_time = time.time()
        self.feature_net.attribute_net.grid.non_zero_pixel_ctx_index = self.feature_net.attribute_net.grid.non_zero_pixel_ctx_index.to(
            'cpu')
        self.feature_net.attribute_net.grid = self.feature_net.attribute_net.grid.to('cpu')
        # temp_output = self.feature_net.attribute_net.grid.grid_encode_forward(get_proba_param=True)

        range_coder_latent_list = []
        ac_max_val_list = []
        # for latent in temp_output.get('latent'):
        #     ac_max_val_latent = get_ac_max_val_latent(latent)
        #     range_coder_latent = RangeCoder(self.feature_net.attribute_net.grid.n_ctx_rowcol, ac_max_val_latent,
        #                                     Q_EXP_SCALE)
        #     range_coder_latent_list.append(range_coder_latent)
        #     ac_max_val_list.append(range_coder_latent.AC_MAX_VAL)

        range_coder_latent_list2 = []
        ac_max_val_list2 = []
        # for latent in temp_output.get('latent2'):
        #     ac_max_val_latent = get_ac_max_val_latent(latent)
        #     range_coder_latent = RangeCoder(self.feature_net.attribute_net.grid.n_ctx_rowcol, ac_max_val_latent,
        #                                     Q_EXP_SCALE)
        #     range_coder_latent_list2.append(range_coder_latent)
        #     ac_max_val_list2.append(range_coder_latent.AC_MAX_VAL)

        range_coder_latent_list3 = []
        ac_max_val_list3 = []
        # for latent in temp_output.get('latent3'):
        #     ac_max_val_latent = get_ac_max_val_latent(latent)
        #     range_coder_latent = RangeCoder(self.feature_net.attribute_net.grid.n_ctx_rowcol, ac_max_val_latent,
        #                                     Q_EXP_SCALE)
        #     range_coder_latent_list3.append(range_coder_latent)
        #     ac_max_val_list3.append(range_coder_latent.AC_MAX_VAL)

        encoder_output = self.feature_net.attribute_net.grid.grid_encode_forward(get_proba_param=True,
                                                                                 # AC_MAX_VAL=ac_max_val_list,
                                                                                 # AC_MAX_VAL2=ac_max_val_list2,
                                                                                 # AC_MAX_VAL3=ac_max_val_list3,
                                                                                 replace=False)

        bitstream_dict = {
            'xy': [],
            'yz': [],
            'zx': []
        }

        total_size = 0
        # xy_bitstream_path = os.path.join(pre_path_name, 'xy_bitstream')
        n_bytes_per_latent = []
        for j in range(len(encoder_output.get('latent'))):
            current_mu = encoder_output.get('mu')[j]
            current_scale = encoder_output.get('scale')[j]
            current_scale = torch.round(current_scale * Q_EXP_SCALE) / Q_EXP_SCALE
            current_scale = torch.clamp(current_scale, min=1 / Q_EXP_SCALE)
            current_y = encoder_output.get('latent')[j]
            # cur_latent_bitstream = f'{xy_bitstream_path}_{j}.bin'
            ac_max_val_latent = get_ac_max_val_latent(current_y)
            # print('latent1', 'idx', j, ac_max_val_latent)
            range_coder_latent = RangeCoder(self.feature_net.attribute_net.grid.n_ctx_rowcol, ac_max_val_latent,
                                            Q_EXP_SCALE)
            # range_coder_latent_list3.append(range_coder_latent)
            ac_max_val_list.append(range_coder_latent.AC_MAX_VAL)

            bs = range_coder_latent.encode(
                # cur_latent_bitstream,
                current_y.flatten().cpu(),
                current_mu.flatten().cpu(),
                current_scale.flatten().cpu(),
                (self.feature_net.attribute_net.grid.output_dim, current_y.size()[-2], current_y.size()[-1]),
            )
            bitstream_dict['xy'].append(bs)
            total_size += len(bs) * 4

        for j in range(len(encoder_output.get('latent2'))):
            current_mu = encoder_output.get('mu2')[j]
            current_scale = encoder_output.get('scale2')[j]
            current_scale = torch.round(current_scale * Q_EXP_SCALE) / Q_EXP_SCALE
            current_scale = torch.clamp(current_scale, min=1 / Q_EXP_SCALE)
            current_y = encoder_output.get('latent2')[j]

            ac_max_val_latent = get_ac_max_val_latent(current_y)
            # print('latent2', 'idx', j, ac_max_val_latent)
            range_coder_latent = RangeCoder(self.feature_net.attribute_net.grid.n_ctx_rowcol, ac_max_val_latent,
                                            Q_EXP_SCALE)
            # range_coder_latent_list3.append(range_coder_latent)
            ac_max_val_list2.append(range_coder_latent.AC_MAX_VAL)

            # cur_latent_bitstream = f'{yz_bitstream_path}_{j}.bin'
            bs = range_coder_latent.encode(
                # cur_latent_bitstream,
                current_y.flatten().cpu(),
                current_mu.flatten().cpu(),
                current_scale.flatten().cpu(),
                (self.feature_net.attribute_net.grid.output_dim, current_y.size()[-2], current_y.size()[-1]),
            )
            bitstream_dict['yz'].append(bs)
            total_size += len(bs) * 4


        # self.debug_pack = {}
        for j in range(len(encoder_output.get('latent3'))):
            current_mu = encoder_output.get('mu3')[j]
            current_scale = encoder_output.get('scale3')[j]
            current_scale = torch.round(current_scale * Q_EXP_SCALE) / Q_EXP_SCALE
            current_scale = torch.clamp(current_scale, min=1 / Q_EXP_SCALE)
            current_y = encoder_output.get('latent3')[j]

            ac_max_val_latent = get_ac_max_val_latent(current_y)
            # print('latent3', 'idx', j, ac_max_val_latent)
            range_coder_latent = RangeCoder(self.feature_net.attribute_net.grid.n_ctx_rowcol, ac_max_val_latent,
                                            Q_EXP_SCALE)
            # range_coder_latent_list3.append(range_coder_latent)
            ac_max_val_list3.append(range_coder_latent.AC_MAX_VAL)

            bs = range_coder_latent.encode(
                # cur_latent_bitstream,
                current_y.flatten().cpu(),
                current_mu.flatten().cpu(),
                current_scale.flatten().cpu(),
                (self.feature_net.attribute_net.grid.output_dim, current_y.size()[-2], current_y.size()[-1]),
            )
            bitstream_dict['zx'].append(bs)
            total_size += len(bs) * 4

        bitstream_dict['meta'] = [ac_max_val_list, ac_max_val_list2, ac_max_val_list3]



        return bitstream_dict, total_size

    def triplane_decode_one_plane(self, range_coder, bitstream, j, arm_net):
        decoded_y = []
        coo_combs = list(itertools.combinations(range(3), 2))
        # print(coo_combs)
        hw = [self.feature_net.attribute_net.grid.resolutions[cc] for cc in coo_combs[1][::-1]]

        h_grid, w_grid = hw[0] * self.feature_net.attribute_net.grid.multiscale_res_multipliers[j], hw[1] * \
                         self.feature_net.attribute_net.grid.multiscale_res_multipliers[j]

        n_ctx_rowcol = 2
        mask_size = 2 * n_ctx_rowcol + 1
        pad = n_ctx_rowcol

        # range_coder = range_coder_latent_list[j]
        # xy_bitstream_path = os.path.join(pre_path_name, 'xy_bitstream')

        range_coder.load_bitstream(bitstream)
        coding_order = range_coder.generate_coding_order(
            (self.feature_net.attribute_net.grid.output_dim, h_grid, w_grid), n_ctx_rowcol
        )
        flat_coding_order = coding_order.flatten().contiguous()
        flat_coding_order_np = flat_coding_order.detach().cpu().numpy()
        flat_index_coding_order_np = np.argsort(flat_coding_order_np, kind='stable')
        flat_index_coding_order = torch.from_numpy(flat_index_coding_order_np).long().to(flat_coding_order.device)
        _, occurrence_coding_order = torch.unique(flat_coding_order, return_counts=True)
        current_y = torch.zeros((1, self.feature_net.attribute_net.grid.output_dim, h_grid, w_grid), device='cpu')

        # Compute the 1d offset of indices to form the context
        offset_index_arm = compute_offset(current_y, mask_size)

        # pad and flatten and current y to perform everything in 1d
        current_y = F.pad(current_y, (pad, pad, pad, pad), mode='constant', value=0.)
        current_y = current_y.flatten().contiguous()

        # Pad the coding order with unreachable values (< 0) to mimic the current_y padding
        # then flatten it as we want to index current_y which is a 1d tensor
        coding_order = F.pad(coding_order, (pad, pad, pad, pad), mode='constant', value=-1)
        coding_order = coding_order.flatten().contiguous()

        # Count the number of decoded values to iterate within the
        # flat_index_coding_order array.
        cnt = 0
        n_ctx_row_col = n_ctx_rowcol
        for index_coding in range(flat_coding_order.max() + 1):
            cur_context = fast_get_neighbor(
                current_y, mask_size, offset_index_arm,
                flat_index_coding_order[cnt: cnt + occurrence_coding_order[index_coding]],
                w_grid, self.feature_net.attribute_net.grid.output_dim
            )

            # ----- From now: run on CPU
            # Compute proba param from context

            # cur_raw_proba_param = self.feature_net.attribute_net.grid.arm(cur_context.cuda())
            cur_raw_proba_param = arm_net(cur_context.cpu())
            cur_raw_proba_param = cur_raw_proba_param.cpu()
            cur_mu, cur_scale = get_mu_scale(cur_raw_proba_param)
            cur_scale = torch.round(cur_scale * Q_EXP_SCALE) / Q_EXP_SCALE
            cur_scale = torch.clamp(cur_scale, min=1 / Q_EXP_SCALE)
            # Decode and store the value at the proper location within current_y
            x_delta = n_ctx_row_col + 1
            if index_coding < w_grid:
                start_y = 0
                start_x = index_coding
            else:
                start_y = (index_coding - w_grid) // x_delta + 1
                start_x = w_grid - x_delta + (index_coding - w_grid) % x_delta

            x = range_coder.decode(cur_mu, cur_scale)
            current_y[
                [
                    coding_order == index_coding
                ]
            ] = x
            # Increment the counter of loaded value
            cnt += occurrence_coding_order[index_coding]

        # Reshape y as a 4D grid, and remove padding
        current_y = current_y.reshape(1, self.feature_net.attribute_net.grid.output_dim, h_grid + 2 * pad,
                                      w_grid + 2 * pad)
        current_y = current_y[:, :, pad:-pad, pad:-pad]
        current_y = current_y / 2 ** 4

        # decoded_latents[j].append(nn.Parameter(current_y))
        return nn.Parameter(current_y)


    #
    def decode_triplane(self, triplane_pack):
        decode_start_time = time.time()

        # ac_max_val_list, ac_max_val_list2, ac_max_val_list3, range_coder_latent_list, range_coder_latent_list2, range_coder_latent_list3 = \
        # ax_max_val_lists[0], ax_max_val_lists[1], ax_max_val_lists[2], ax_max_val_lists[3], ax_max_val_lists[4], \
        # ax_max_val_lists[5]

        meta_info = triplane_pack['meta']

        decoded_latents = [nn.ParameterList() for _ in
                           range(len(self.feature_net.attribute_net.grid.multiscale_res_multipliers))]
        self.feature_net.attribute_net.grid.arm = self.feature_net.attribute_net.grid.arm.to('cpu')
        self.feature_net.attribute_net.grid.arm2 = self.feature_net.attribute_net.grid.arm2.to('cpu')
        self.feature_net.attribute_net.grid.arm3 = self.feature_net.attribute_net.grid.arm3.to('cpu')

        decoded_triplane = nn.ModuleList()

        # for i in self.debug_pack.keys():
        #     range_coder_params, bitstream, i, arm3 = self.debug_pack[i]
        #     range_coder = RangeCoder(*range_coder_params)
        #     latent = self.triplane_decode_one_plane(range_coder, bitstream, i, arm3)



        for i, ac_max_val_latent in enumerate(meta_info[0]):
            # print('decode', 'xy')
            # ac_max_val_latent = get_ac_max_val_latent(latent)
            range_coder = RangeCoder(self.feature_net.attribute_net.grid.n_ctx_rowcol, ac_max_val_latent,
                                            Q_EXP_SCALE)
            bitstream = triplane_pack['xy'][i]

            latent = self.triplane_decode_one_plane(range_coder, bitstream, i, self.feature_net.attribute_net.grid.arm)
            decoded_latents[i].append(latent)
            # range_coder_latent_list.append(range_coder_latent)
            # ac_max_val_list.append(range_coder_latent.AC_MAX_VAL)

        for i, ac_max_val_latent in enumerate(meta_info[1]):
            # print('decode', 'yz')
            # ac_max_val_latent = get_ac_max_val_latent(latent)
            range_coder = RangeCoder(self.feature_net.attribute_net.grid.n_ctx_rowcol, ac_max_val_latent,
                                            Q_EXP_SCALE)
            bitstream = triplane_pack['yz'][i]

            latent = self.triplane_decode_one_plane(range_coder, bitstream, i, self.feature_net.attribute_net.grid.arm2)
            decoded_latents[i].append(latent)
            # range_coder_latent_list.append(range_coder_latent)
            # ac_max_val_list.append(range_coder_latent.AC_MAX_VAL)

        for i, ac_max_val_latent in enumerate(meta_info[2]):
            range_coder = RangeCoder(self.feature_net.attribute_net.grid.n_ctx_rowcol, ac_max_val_latent, Q_EXP_SCALE)
            bitstream = triplane_pack['zx'][i]
            latent = self.triplane_decode_one_plane(range_coder, bitstream, i,
                                                    self.feature_net.attribute_net.grid.arm3)
            decoded_latents[i].append(latent)

        for decoded_latent in decoded_latents:
            decoded_triplane.append(decoded_latent)

        # decode_end_time = time.time()

        # print('Triplane decoding time:', decode_end_time - decode_start_time)

        self.feature_net.attribute_net.grid.grids = decoded_triplane
        self.feature_net.attribute_net.grid = self.feature_net.attribute_net.grid.cuda()
        self.feature_net.attribute_net.grid.grids = self.feature_net.attribute_net.grid.grids.cuda()
        self.feature_net.attribute_net.grid.non_zero_pixel_ctx_index = self.feature_net.attribute_net.grid.non_zero_pixel_ctx_index.cuda()





