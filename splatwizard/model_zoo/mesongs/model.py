import io
import os
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
from tqdm import tqdm
import tempfile
import zipfile
import gc
import yaml 

from splatwizard.compression.entropy_codec import ArithmeticCodec
from splatwizard.modules.densify_mixin import DensificationAndPruneMixin
from splatwizard.metrics.loss_utils import l1_func, ssim_func
from splatwizard.common.constants import BIT2MB_SCALE
from splatwizard.rasterizer.meson_gs import GaussianRasterizationSettings, GaussianRasterizer, GaussianRasterizerIndexed
from splatwizard.model_zoo.mesongs.config import MesonGSModelParams, MesonGSOptimizationParams
from splatwizard.config import PipelineParams
from splatwizard.model_zoo.mesongs.meson_utils import VanillaQuan, vq_features, split_length
from splatwizard.model_zoo.mesongs.raht_torch import haar3D_param, inv_haar3D_param, transform_batched_torch, itransform_batched_torch, copyAsort
from splatwizard.modules.dataclass import RenderResult, LossPack
from splatwizard.modules.gaussian_model import GaussianModel
from splatwizard._cmod.simple_knn import distCUDA2    # noqa
from splatwizard.scheduler import Scheduler, task
from splatwizard.utils.general_utils import (
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric
)
from splatwizard.utils.graphics_utils import BasicPointCloud
from splatwizard.utils.sh_utils import eval_sh
from splatwizard.compression.entropy_model import EntropyGaussian
from splatwizard.compression.quantizer import STE_binary, STE_multistep, Quantize_anchor, UniformQuantizer, STEQuantizer
from splatwizard.modules.loss_mixin import LossMixin
from splatwizard.modules.dataclass import RenderResult

from ...scene import CameraIterator

@dataclass
class MesonGSRenderResult(RenderResult):
    imp: typing.Any = None
    
def d1halfing_fast(pmin,pmax,pdepht):
    return np.linspace(pmin,pmax,2**int(pdepht)+1)
                       
def octreecodes(ppoints, pdepht, merge_type='mean',imps=None):
    minx=np.amin(ppoints[:,0])
    maxx=np.amax(ppoints[:,0])
    miny=np.amin(ppoints[:,1])
    maxy=np.amax(ppoints[:,1])
    minz=np.amin(ppoints[:,2])
    maxz=np.amax(ppoints[:,2])
    xletra=d1halfing_fast(minx,maxx,pdepht)
    yletra=d1halfing_fast(miny,maxy,pdepht)
    zletra=d1halfing_fast(minz,maxz,pdepht)
    otcodex=np.searchsorted(xletra,ppoints[:,0],side='right')-1
    otcodey=np.searchsorted(yletra,ppoints[:,1],side='right')-1
    otcodez=np.searchsorted(zletra,ppoints[:,2],side='right')-1
    ki=otcodex*(2**(pdepht*2))+otcodey*(2**pdepht)+otcodez
    
    ki_ranks = np.argsort(ki)
    ppoints = ppoints[ki_ranks]
    ki = ki[ki_ranks]

    ppoints = np.concatenate([ki.reshape(-1, 1), ppoints], -1)
    # print('here 4', ppoints.shape)
    dedup_points = np.split(ppoints[:, 1:], np.unique(ki, return_index=True)[1][1:])
    
    # print('ki.shape', ki.shape)
    
    # print('ki.shape', ki.shape)
    final_feature = []
    if merge_type == 'mean':
        for dedup_point in dedup_points:
            # print(np.mean(dedup_point, 0).shape)
            final_feature.append(np.mean(dedup_point, 0).reshape(1, -1))
    elif merge_type == 'imp':
        dedup_imps = np.split(imps, np.unique(ki, return_index=True)[1][1:])
        for dedup_point, dedup_imp in zip(dedup_points, dedup_imps):
            dedup_imp = dedup_imp.reshape(1, -1)
            if dedup_imp.shape[-1] == 1:
                # print('dedup_point.shape', dedup_point.shape)
                final_feature.append(dedup_point)
            else:
                # print('dedup_point.shape, dedup_imp.shape', dedup_point.shape, dedup_imp.shape)
                fdp = (dedup_imp / np.sum(dedup_imp)) @ dedup_point
                # print('fdp.shape', fdp.shape)
                final_feature.append(fdp)
    elif merge_type == 'rand':
        for dedup_point in dedup_points:
            ld = len(dedup_point)
            id = torch.randint(0, ld, (1,))[0]
            final_feature.append(dedup_point[id].reshape(1, -1))
    else:
        raise NotImplementedError
    ki = np.unique(ki)
    final_feature = np.concatenate(final_feature, 0)
    # print('final_feature.shape', final_feature.shape)
    return (ki,minx,maxx,miny,maxy,minz,maxz, final_feature)


def create_octree_overall(ppoints, pfeatures, imp, depth, oct_merge):
    ori_points_num = ppoints.shape[0]
    ppoints = np.concatenate([ppoints, pfeatures], -1)
    occ=octreecodes(ppoints, depth, oct_merge, imp)
    final_points_num = occ[0].shape[0]
    occodex=(occ[0]/(2**(depth*2))).astype(int)
    occodey=((occ[0]-occodex*(2**(depth*2)))/(2**depth)).astype(int)
    occodez=(occ[0]-occodex*(2**(depth*2))-occodey*(2**depth)).astype(int)
    voxel_xyz = np.array([occodex,occodey,occodez], dtype=int).T
    features = occ[-1][:, 3:]
    paramarr=np.asarray([occ[1],occ[2],occ[3],occ[4],occ[5],occ[6]]) # boundary
    # print('oct[0]', type(oct[0]))
    return voxel_xyz, features, occ[0], paramarr, ori_points_num, final_points_num

def decode_oct(paramarr, oct, depth):
    minx=(paramarr[0])
    maxx=(paramarr[1])
    miny=(paramarr[2])
    maxy=(paramarr[3])
    minz=(paramarr[4])
    maxz=(paramarr[5])
    xletra=d1halfing_fast(minx,maxx,depth)
    yletra=d1halfing_fast(miny,maxy,depth)
    zletra=d1halfing_fast(minz,maxz,depth)
    occodex=(oct/(2**(depth*2))).astype(int)
    occodey=((oct-occodex*(2**(depth*2)))/(2**depth)).astype(int)
    occodez=(oct-occodex*(2**(depth*2))-occodey*(2**depth)).astype(int)  
    V = np.array([occodex,occodey,occodez], dtype=int).T
    koorx=xletra[occodex]
    koory=yletra[occodey]
    koorz=zletra[occodez]
    ori_points=np.array([koorx,koory,koorz]).T

    return ori_points, V

def ToEulerAngles_FT(q):

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = torch.sqrt(1 + 2 * (w * y - x * z))
    cosp = torch.sqrt(1 - 2 * (w * y - x * z))
    pitch = 2 * torch.arctan2(sinp, cosp) - torch.pi / 2
    
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.arctan2(siny_cosp, cosy_cosp)

    roll = roll.reshape(-1, 1)
    pitch = pitch.reshape(-1, 1)
    yaw = yaw.reshape(-1, 1)

    return torch.concat([roll, pitch, yaw], -1)

def seg_quant_ave(x, split, qas):
    start = 0
    cnt = 0
    outs = []
    for length in split:
        outs.append(qas[cnt](x[start:start+length]))
        cnt += 1
        start += length
    return torch.concat(outs, dim=0)

def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2. ** num_bits - 1.
 
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    
    return q_x


def safe_euler(euler):
    # 把 NaN 转为 0，但保持 requires_grad
    euler = torch.where(torch.isnan(euler), torch.zeros_like(euler), euler)

    # 限制角度范围，防止 sin/cos 溢出造成 inf
    euler = torch.clamp(euler, min=-1e6, max=1e6)
    return euler


def torch_vanilla_quant_ave(x, split, qas):
    start = 0
    cnt = 0
    outs = []
    trans = []
    for length in split:
        i_scale = qas[cnt].scale
        i_zp = qas[cnt].zero_point
        i_bit = qas[cnt].bit
        outs.append(quantize_tensor(
            x[start:start+length], 
            scale=i_scale,
            zero_point=i_zp,
            num_bits=i_bit).cpu().numpy()) 
        trans.extend([i_scale.item(), i_zp.item()])
        cnt += 1
        start += length
    return np.concatenate(outs, axis=0), trans

def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x - zero_point)

def torch_vanilla_dequant_ave(x, split, sz):
    cnt = 0 
    start = 0
    outs = []
    for length in split:
        i_scale = sz[cnt]
        i_zp = sz[cnt+1]
        outs.append(
            dequantize_tensor(
                x[start:start+length],
                scale=i_scale,
                zero_point=i_zp
            )
        )
        cnt+=2
        start += length
    return torch.concat(outs, axis=0)


class MesonGS(LossMixin, DensificationAndPruneMixin, GaussianModel):

    def setup_functions(self):
        def build_rotation_from_euler(roll, pitch, yaw):
            R = torch.zeros((roll.size(0), 3, 3), device='cuda')

            R[:, 0, 0] = torch.cos(pitch) * torch.cos(roll)
            R[:, 0, 1] = -torch.cos(yaw) * torch.sin(roll) + torch.sin(yaw) * torch.sin(pitch) * torch.cos(roll)
            R[:, 0, 2] = torch.sin(yaw) * torch.sin(roll) + torch.cos(yaw) * torch.sin(pitch) * torch.cos(roll)
            R[:, 1, 0] = torch.cos(pitch) * torch.sin(roll)
            R[:, 1, 1] = torch.cos(yaw) * torch.cos(roll) + torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll)
            R[:, 1, 2] = -torch.sin(yaw) * torch.cos(roll) + torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll)
            R[:, 2, 0] = -torch.sin(pitch)
            R[:, 2, 1] = torch.sin(yaw) * torch.cos(pitch)
            R[:, 2, 2] = torch.cos(yaw) * torch.cos(pitch)

            return R


        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        def safe_euler(euler):
            # 把 NaN 转为 0，但保持 requires_grad
            euler = torch.where(torch.isnan(euler), torch.zeros_like(euler), euler)

            # 限制角度范围，防止 sin/cos 溢出造成 inf
            euler = torch.clamp(euler, min=-1e6, max=1e6)
            return euler

        def build_covariance_from_scaling_euler(scaling, scaling_modifier, euler, return_symm=True):
            euler = safe_euler(euler)
            s = scaling_modifier * scaling
            L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
            R = build_rotation_from_euler(euler[:, 2], euler[:, 1], euler[:, 0])

            L[:,0,0] = s[:,0]
            L[:,1,1] = s[:,1]
            L[:,2,2] = s[:,2]

            L = R @ L
            actual_covariance = L @ L.transpose(1, 2)
            if return_symm:
                symm = strip_symmetric(actual_covariance)
                return symm
            else:
                return actual_covariance

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.covariance_activation_for_euler = build_covariance_from_scaling_euler
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
    
    def __init__(self, model_param: MesonGSModelParams):
        logger.info('mesongs start init here')
        super().__init__()
        self.finetune_lr_scale = model_param.finetune_lr_scale
        self.num_bits = model_param.num_bits
        self.depth = model_param.depth
        self.percent = model_param.percent
        self.raht = model_param.raht
        self.merge_type = model_param.oct_merge
        self.debug = model_param.debug
        self.clamp_color = model_param.clamp_color
        self.per_channel_quant = model_param.per_channel_quant
        self.per_block_quant = model_param.per_block_quant
        self.use_indexed = model_param.use_indexed
        self.scene_imp = model_param.scene_imp
        self.n_block = model_param.n_block
        self.codebook_size = model_param.codebook_size
        self.batch_size = model_param.batch_size
        self.steps = model_param.steps
        if model_param.yaml_path is not "":
            with open(model_param.yaml_path, "r") as f:
                config_dict = yaml.safe_load(f)
        self.depth = config_dict["depth"]
        self.percent = config_dict["prune"]
        logger.info(f"yaml prune {self.percent}")
        self.codebook_size = config_dict["cb"]
        self.n_block = config_dict["n_block"]
        self.finetune_lr_scale = config_dict["finetune_lr_scale"]
        logger.info("use config yaml")
        
        
        self.active_sh_degree = 0
        self.max_sh_degree = 3
        self._cov = torch.empty(0)
        self._euler = torch.empty(0)
        self._feature_indices = torch.empty(0)
        self.qas = nn.ModuleList([])
        self._V = None
        self.optimizer = None
        self.w = None
        self.val = None
        self.TMP = None
        self.res_tree = None
        self.ret_features = None
        self.setup_functions()
        self.n_sh = (self.max_sh_degree + 1) ** 2 
        logger.info(f'self.n_sh {self.n_sh}')
        
    
    def pre_volume(self, volume, beta):
        # volume = torch.tensor(volume)
        index = int(volume.shape[0] * 0.9)
        sorted_volume, _ = torch.sort(volume, descending=True)
        kth_percent_largest = sorted_volume[index]
        # Calculate v_list
        v_list = torch.pow(volume / kth_percent_largest, beta)
        return v_list

    def training_setup(self, training_args: MesonGSOptimizationParams):
        logger.info(self.spatial_lr_scale)
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        
        if self.finetune_lr_scale < 1.0 - 0.001:
            logger.info('training setup: finetune')
            training_args.position_lr_init = training_args.position_lr_init * self.finetune_lr_scale
            training_args.feature_lr = training_args.feature_lr * self.finetune_lr_scale
            training_args.opacity_lr = training_args.opacity_lr * self.finetune_lr_scale
            training_args.scaling_lr = training_args.scaling_lr * self.finetune_lr_scale
            training_args.rotation_lr = training_args.rotation_lr * self.finetune_lr_scale
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init*self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr*self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
    @task
    def cal_imp(
        self, 
        cam_iterator: CameraIterator, 
        pipe: PipelineParams,
        opt: MesonGSOptimizationParams):
        beta_list = {
            'chair': 0.03,
            'drums': 0.05,
            'ficus': 0.03,
            'hotdog': 0.03,
            'lego': 0.05,
            'materials': 0.03,
            'mic': 0.03,
            'ship': 0.03,
            'bicycle': 0.03,
            'bonsai': 0.1,
            'counter': 0.1,
            'garden': 0.1,
            'kitchen': 0.1,
            'room': 0.1,
            'stump': 0.01,
        }   
        
        full_opa_imp = None
        bg_color = [1, 1, 1] if pipe.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=pipe.device)
        with torch.no_grad():
            for idx, view in enumerate(tqdm(cam_iterator, desc="count imp")):
                render_results = self.vanilla_render(
                    view, 
                    background,
                    pipe, 
                    opt, 
                    render_type='imp'
                )
                if full_opa_imp is None:
                    full_opa_imp = torch.zeros_like(render_results.imp).cuda()
                full_opa_imp.add_(render_results.imp)
                    
                del render_results
                gc.collect()
                torch.cuda.empty_cache()
            
        volume = torch.prod(self.scaling, dim=1)

        v_list = self.pre_volume(volume, beta_list.get(self.scene_imp, 0.1))
        imp = v_list * full_opa_imp
        
        self.imp = imp.detach()
        self.prune_mask()
        
    @torch.no_grad
    def prune_mask(self):
        sorted_tensor, _ = torch.sort(self.imp, dim=0)
        index_nth_percentile = int(self.percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (self.imp <= value_nth_percentile).squeeze()
        self.imp = self.imp[torch.logical_not(prune_mask)]
        self.prune_points(prune_mask)
        logger.info(f'self xyz {self.xyz.shape}')
    
    @task
    @torch.no_grad
    def octree_coding(self):
        features = torch.concat([
            self._opacity.detach(), 
            self._features_dc.detach().flatten(-2).contiguous(), 
            self._features_rest.detach().flatten(-2).contiguous(), 
            self._scaling.detach(), 
            self._rotation.detach()], -1).cpu().numpy()

        V, features, oct, paramarr, _, _ = create_octree_overall(
            self._xyz.detach().cpu().numpy(), 
            features,
            self.imp,
            depth=self.depth,
            oct_merge=self.merge_type)
        dxyz, _ = decode_oct(paramarr, oct, self.depth)
        
        if self.raht:
            # morton sort
            logger.info("here raht?")
            w, val, reorder = copyAsort(V)
            self.reorder = reorder
            self.res = haar3D_param(self.depth, w, val)
            self.res_inv = inv_haar3D_param(V, self.depth)
            self.scale_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        
        opacities = features[:, :1]
        features_dc = features[:, 1:4].reshape(-1, 1, 3)
        features_extra = features[:, 4:4 + 3 * (self.n_sh-1)].reshape(-1, self.n_sh - 1, 3)
        scales=features[:,49:52]
        rots=features[:,52:56]
        
        self.oct = oct
        self.oct_param = paramarr
        self._xyz = nn.Parameter(torch.tensor(dxyz, dtype=torch.float, device="cuda").requires_grad_(False))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    @task
    def init_qas(self):
        n_qs = 10 * self.n_block
        for i in range(n_qs): 
            self.qas.append(VanillaQuan(bit=self.num_bits).cuda())
        logger.info(f'Init qa, length: {n_qs}')
    
    @task
    @torch.no_grad
    def vq_fe(self):
        features_extra = self._features_rest.detach().flatten(-2)
        codebook, vq_indices = vq_features(
            features_extra,
            self.imp,
            self.codebook_size,
            self.batch_size,
            self.steps,
        )

        self._feature_indices = nn.Parameter(vq_indices.detach().contiguous(), requires_grad=False)
        self._features_rest = nn.Parameter(codebook.detach().contiguous(), requires_grad=True)
    
    @property
    def original_rotation(self):
        return self._rotation
    
    @property
    def original_opacity(self):
        return self._opacity
    
    @property
    def original_scales(self):
        return self._scaling
    
    @property
    def get_features_extra(self):
        features_extra = self._features_rest.reshape((-1, 3, (self.max_sh_degree + 1) ** 2 - 1))
        return features_extra
    
    @property
    def feature_indices(self):
        return self._feature_indices
    
    @property
    def get_indexed_feature_extra(self):
        n_sh = (self.active_sh_degree + 1) ** 2
        num_points = self.xyz.shape[0]
        fi = self._feature_indices.detach().cpu()
        fr = self._features_rest.detach().cpu()
        ret = torch.zeros([num_points, 3 * (n_sh - 1)])
        for i in range(num_points):
            ret[i] = self._features_rest[int(fi[i])]
        return ret.reshape(-1, n_sh - 1, 3)
        # return torch.matmul(F.one_hot(self._feature_indices).float(), self._features_rest).reshape(-1, self.n_sh - 1, 3)

    @property
    def get_cov(self):
        return self._cov

    @property
    def get_euler(self):
        return self._euler
    
    def get_covariance(self, scaling_modifier=1):
        if self.get_euler.shape[0] > 0:
            # print('go with euler')
            return self.covariance_activation_for_euler(self.scaling, scaling_modifier, self._euler)
        elif self.get_cov.shape[0] > 0:
            return self.get_cov
        else:
            # print('gaussian model: get cov from scaling and rotations.')
            return self.covariance_activation(self.scaling, scaling_modifier, self._rotation)
        
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._cov,
            self._euler,
            self._feature_indices,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.qas.state_dict(),
            self.reorder, 
            self.res,
            self.oct,
            self.oct_param
        )

    def restore(self, model_args, training_args=None):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._cov,
            self._euler,
            self._feature_indices,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            qas_state_dict,
            self.reorder,
            self.res,
            self.oct,
            self.oct_param
        ) = model_args
        if training_args is not None:
            self.training_setup(training_args)

        # Since training_setup will reset these parameters, we assign values to them manually
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if self.optimizer is not None:
            self.optimizer.load_state_dict(opt_dict)
        n_qs = 10 * self.n_block
        for i in range(n_qs): 
            self.qas.append(VanillaQuan(bit=self.num_bits).cuda())
        self.qas.load_state_dict(qas_state_dict)


    def vanilla_render(
        self,
        viewpoint_camera, 
        background,
        pipe: PipelineParams,
        opt: MesonGSOptimizationParams,
        scaling_modifier: float=1.0,
        override_color=None,
        render_type: str = 'imp'
    ):
        meson_count = False
        if render_type == 'imp':
            meson_count = True
        
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
            bg=background,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.debug,
            clamp_color=self.clamp_color,
            meson_count=meson_count,
            f_count=False,
            depth_count=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    
        means3D = self.xyz
        means2D = screenspace_points

        opacity = self.opacity
        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python or self.get_cov.shape[0] > 0 or self.get_euler.shape[0] > 0:
            cov3D_precomp = self.get_covariance(scaling_modifier)
            # print('gaussian_renderer __init__', cov3D_precomp.shape)
        else:
            scales = self.scaling
            rotations = self.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if pipe.convert_SHs_python:
                shs_view = self.features.transpose(1, 2).view(-1, 3, (self.active_sh_degree+1)**2)
                dir_pp = (self.xyz - viewpoint_camera.camera_center.repeat(self.features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                # print('shs_view', shs_view.max(), shs_view.min())
                # print('active_sh_degree', pc.active_sh_degree)
                sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                # print('sh2rgb.max(), sh2rgb.min()', sh2rgb.max(), sh2rgb.min())
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                # colors_precomp = colors_precomp.nan_to_num(0)
                # colors_precomp = 
                # print('colors_precomp', colors_precomp.max(), colors_precomp.min(), colors_precomp[:5])
            else:
                shs = self.features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        if meson_count:
            rendered_image, radii, imp = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)

            return MesonGSRenderResult(
                    rendered_image=rendered_image,
                    viewspace_points=screenspace_points,
                    visibility_filter=radii > 0,
                    radii=radii,
                    imp=imp
                )
        else:
            rendered_image, radii = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)

            return MesonGSRenderResult(
                    rendered_image=rendered_image,
                    viewspace_points=screenspace_points,
                    visibility_filter=radii > 0,
                    radii=radii
                )
        # elif f_count:
        #     rendered_image, radii, imp, gaussians_count, opa_imp = rasterizer(
        #         means3D = means3D,
        #         means2D = means2D,
        #         shs = shs,
        #         colors_precomp = colors_precomp,
        #         opacities = opacity,
        #         scales = scales,
        #         rotations = rotations,
        #         cov3D_precomp = cov3D_precomp)
        #     return {"render": rendered_image,
        #         "viewspace_points": screenspace_points,
        #         "visibility_filter" : radii > 0,
        #         "radii": radii,
        #         "imp": imp,
        #         "gaussians_count": gaussians_count,
        #         "opa_imp": opa_imp}

        # elif depth_count:
        #     rendered_image, radii, out_depth = rasterizer(
        #         means3D = means3D,
        #         means2D = means2D,
        #         shs = shs,
        #         colors_precomp = colors_precomp,
        #         opacities = opacity,
        #         scales = scales,
        #         rotations = rotations,
        #         cov3D_precomp = cov3D_precomp)
        #     return {"render": rendered_image,
        #         "viewspace_points": screenspace_points,
        #         "visibility_filter" : radii > 0,
        #         "radii": radii,
        #         "depth": out_depth}
        
        
        
    def render(
        self, 
        viewpoint_camera, 
        background,
        pipe: PipelineParams,
        opt: MesonGSOptimizationParams = None,
        step: int = None,
        scaling_modifier: float=1.0,
        override_color=None,
        render_type: str ='ft'): 
        # logger.info(f'recent step {step}')
        if (render_type == 'vanilla') or (pipe.eval_mode is not None):
            
            return self.vanilla_render(
                viewpoint_camera,
                background,
                pipe,
                opt,
                scaling_modifier,
                override_color,
                render_type
            )
        
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
            bg=background,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.debug,
            clamp_color=self.clamp_color,
            meson_count=False,
            f_count=False,
            depth_count=False
        )
        rasterizer = GaussianRasterizerIndexed(raster_settings=raster_settings)
        
        re_range = [1, 4]
        shzero_range = [4, 7]
        
        means3D = self.xyz
        means2D = screenspace_points
        
        if self.raht:
            r = self.original_rotation
            norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
            q = r / norm[:, None]
            eulers = ToEulerAngles_FT(q)
            rf = torch.concat([self.original_opacity, eulers, self.features_dc.contiguous().squeeze()], -1)

            C = rf[self.reorder]
            iW1 = self.res['iW1']
            iW2 = self.res['iW2']
            iLeft_idx = self.res['iLeft_idx']
            iRight_idx = self.res['iRight_idx']

            for d in range(self.depth * 3):
                w1 = iW1[d]
                w2 = iW2[d]
                left_idx = iLeft_idx[d]
                right_idx = iRight_idx[d]
                C[left_idx], C[right_idx] = transform_batched_torch(w1, 
                                                    w2, 
                                                    C[left_idx], 
                                                    C[right_idx])
            
            quantC = torch.zeros_like(C)
            quantC[0] = C[0]
            if self.per_channel_quant:
                for i in range(C.shape[-1]):
                    quantC[1:, i] = self.qas[i](C[1:, i])
            elif self.per_block_quant:
                qa_cnt = 0
                lc1 = C.shape[0] - 1
                split_ac = split_length(lc1, self.n_block)
                for i in range(C.shape[-1]):
                    quantC[1:, i] = seg_quant_ave(C[1:, i], split_ac, self.qas[qa_cnt : qa_cnt + self.n_block])
                    qa_cnt += self.n_block
                
            else:
                quantC[1:] = self.qa(C[1:])

            res_inv = self.res_inv
            pos = res_inv['pos']
            iW1 = res_inv['iW1']
            iW2 = res_inv['iW2']
            iS = res_inv['iS']
            
            iLeft_idx = res_inv['iLeft_idx']
            iRight_idx = res_inv['iRight_idx']
        
            iLeft_idx_CT = res_inv['iLeft_idx_CT']
            iRight_idx_CT = res_inv['iRight_idx_CT']
            iTrans_idx = res_inv['iTrans_idx']
            iTrans_idx_CT = res_inv['iTrans_idx_CT'] 

            CT_yuv_q_temp = quantC[pos.astype(int)]
            raht_features = torch.zeros(quantC.shape).cuda()
            OC = torch.zeros(quantC.shape).cuda()
            
            for i in range(self.depth*3):
                w1 = iW1[i]
                w2 = iW2[i]
                S = iS[i]
                
                left_idx, right_idx = iLeft_idx[i], iRight_idx[i]
                left_idx_CT, right_idx_CT = iLeft_idx_CT[i], iRight_idx_CT[i]
                
                trans_idx, trans_idx_CT = iTrans_idx[i], iTrans_idx_CT[i]
                
                
                OC[trans_idx] = CT_yuv_q_temp[trans_idx_CT]
                OC[left_idx], OC[right_idx] = itransform_batched_torch(w1, 
                                                        w2, 
                                                        CT_yuv_q_temp[left_idx_CT], 
                                                        CT_yuv_q_temp[right_idx_CT])  
                CT_yuv_q_temp[:S] = OC[:S]

            raht_features[self.reorder] = OC
            
            scales = self.original_scales
            
            if self.per_channel_quant:
                scalesq = torch.zeros_like(scales).cuda()
                scaleqa_offset = 7
                for i in range(scaleqa_offset, scaleqa_offset + 3):
                    scalesq[:, i-scaleqa_offset] = self.qas[i](scales[:, i-scaleqa_offset])

            elif self.per_block_quant:
                scalesq = torch.zeros_like(scales).cuda()
                split_scale = split_length(scales.shape[0], self.n_block)
                for i in range(scales.shape[-1]):
                    # scalesq[:, i] = seg_quant(scales[:, i], pc.lseg, pc.qas[qa_cnt : qa_cnt + blocks_in_channel])
                    scalesq[:, i] = seg_quant_ave(scales[:, i], split_scale, self.qas[qa_cnt: qa_cnt + self.n_block])
                    qa_cnt += self.n_block
            else:
                scalesq = self.scale_qa(scales)
                    
            scaling = torch.exp(scalesq)
            
            # if re_mode == 'rot':
            #     rotations = raht_features[:, 1:5]
            #     cov3D_precomp = pc.covariance_activation(scaling, 1.0, rotations)
            # elif re_mode == 'euler':
            eulers = raht_features[:, 1:4]
            cov3D_precomp = self.covariance_activation_for_euler(scaling, 1.0, eulers)

            assert cov3D_precomp is not None
            
            opacity = raht_features[:, :1]
            opacity = torch.sigmoid(opacity)    
            
            scales = None
            rotations = None
            eulers = None
            colors_precomp = None
            
            if self.use_indexed:
                sh_zero = raht_features[:, shzero_range[0]:].unsqueeze(1).contiguous()
                sh_ones = self.get_features_extra.reshape(-1, (self.active_sh_degree+1)**2 - 1, 3)
                sh_indices = self.feature_indices
            else:
                features_dc = raht_features[:, shzero_range[0]:].unsqueeze(1)
                feature_extra = self.get_indexed_feature_extra
                features = torch.cat((features_dc, feature_extra), dim=1)
                shs_view = features.transpose(1, 2).view(-1, 3, (self.active_sh_degree+1)**2)
                dir_pp = (self.xyz - viewpoint_camera.camera_center.repeat(features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            raise Exception("Sorry, w/o raht version is unimplemented.")
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            opacities = opacity,
            sh_indices = sh_indices,
            sh_zero = sh_zero,
            sh_ones = sh_ones,
            colors_precomp = colors_precomp,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        return MesonGSRenderResult(
                rendered_image=rendered_image,
                viewspace_points=screenspace_points,
                visibility_filter=radii > 0,
                radii=radii
            )
    
    @task
    def finetuning_setup(self, training_args: MesonGSOptimizationParams):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")
    
        print('finetuning setup: finetune')
        training_args.position_lr_init = training_args.position_lr_init * self.finetune_lr_scale
        training_args.feature_lr = training_args.feature_lr * self.finetune_lr_scale
        training_args.opacity_lr = training_args.opacity_lr * self.finetune_lr_scale
        training_args.scaling_lr = training_args.scaling_lr * self.finetune_lr_scale
        training_args.rotation_lr = training_args.rotation_lr * self.finetune_lr_scale * 0.2
        
        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init*self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr*self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    @task
    def update_learning_rate(self, iteration: int):
        iteration += 30000 # meson customize
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    
    def register_pre_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: MesonGSOptimizationParams):

        # scheduler.register_task(range(0, opt.iterations, 1000), task=self.oneupSHdegree)
        # scheduler.register_task(1, task=self.calc_importance_task)
        # scheduler.register_task(1, task=self.training_setup)
        scheduler.register_task(1, task=self.cal_imp)
        scheduler.register_task(1, task=self.octree_coding)
        scheduler.register_task(1, task=self.init_qas)
        scheduler.register_task(1, task=self.vq_fe)
        scheduler.register_task(1, task=self.finetuning_setup)
        scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
        # scheduler.register_task(1, task=self.re_exec_training_setup, priority=1) # exec after vq_compress
        # pass
    
    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: MesonGSOptimizationParams):    
        pass
    
    @torch.no_grad
    def encode(self, tmp_file: io.BufferedWriter):
        # path = tmp_file.name
        logger.info(f"xyz shape {self.xyz.shape}")
        with tempfile.TemporaryDirectory() as exp_dir:
            os.makedirs(exp_dir, exist_ok=True)
            bin_dir = os.path.join(exp_dir, 'bins')
            os.makedirs(bin_dir, exist_ok=True)
            trans_array = []
            trans_array.append(self.depth)
            trans_array.append(self.n_block)
            
            scale_offset = 7

            with torch.no_grad():
                np.savez_compressed(os.path.join(bin_dir , 'oct'), points=self.oct, params=self.oct_param)
                ntk = self._feature_indices.detach().contiguous().cpu().int().numpy()
                cb = self._features_rest.detach().contiguous().cpu().numpy()
                np.savez_compressed(os.path.join(bin_dir , 'ntk.npz'), ntk=ntk)
                np.savez_compressed(os.path.join(bin_dir , 'um.npz'), umap=cb)
                
                r = self.original_rotation
                norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
                q = r / norm[:, None]
                eulers = ToEulerAngles_FT(q)
                
                rf = torch.concat([self.original_opacity.detach(), eulers.detach(), self.features_dc.detach().contiguous().squeeze()], axis=-1)
                
                # '''ckpt'''
                # rf_cpu = rf.cpu().numpy()
                # np.save('duipai/rf_cpu.npy', rf_cpu)
                # ''''''
                
                C = rf[self.reorder]
                iW1 = self.res['iW1']
                iW2 = self.res['iW2']
                iLeft_idx = self.res['iLeft_idx']
                iRight_idx = self.res['iRight_idx']

                for d in range(self.depth * 3):
                    w1 = iW1[d]
                    w2 = iW2[d]
                    left_idx = iLeft_idx[d]
                    right_idx = iRight_idx[d]
                    C[left_idx], C[right_idx] = transform_batched_torch(w1, 
                                                        w2, 
                                                        C[left_idx], 
                                                        C[right_idx])

                cf = C[0].cpu().numpy()
                
                qa_cnt = 0
                lc1 = C.shape[0] - 1
                qci = [] 
                split = split_length(lc1, self.n_block)
                for i in range(C.shape[-1]):
                    t1, trans1 = torch_vanilla_quant_ave(C[1:, i], split, self.qas[qa_cnt : qa_cnt + self.n_block])
                    qci.append(t1)
                    # .reshape(-1, 1)
                    trans_array.extend(trans1)
                    qa_cnt += self.n_block
                qci = np.concatenate(qci, axis=-1)
                    
                np.savez_compressed(os.path.join(bin_dir,'orgb.npz'), f=cf, i=qci.astype(np.uint8))
                
                scaling = self.original_scales.detach()
                lc1 = scaling.shape[0]
                scaling_q = []
                split_scale = split_length(lc1, self.n_block)
                for i in range(scaling.shape[-1]):
                    t1, trans1 = torch_vanilla_quant_ave(scaling[:, i], split_scale, self.qas[qa_cnt : qa_cnt + self.n_block])
                    scaling_q.append(t1)
                    # .reshape(-1, 1)
                    trans_array.extend(trans1)
                    qa_cnt += self.n_block
                scaling_q = np.concatenate(scaling_q, axis=-1)

                np.savez_compressed(os.path.join(bin_dir,'ct.npz'), i=scaling_q.astype(np.uint8))
                
                trans_array = np.array(trans_array)
                np.savez_compressed(os.path.join(bin_dir, 't.npz'), t=trans_array)

                bin_zip_name = bin_dir.split('/')[-1]
                bin_zip_path = os.path.join(exp_dir, f'{bin_zip_name}.zip')
                os.system(f'zip -j {bin_zip_path} {bin_dir}/*')

                zip_file_size = os.path.getsize(bin_zip_path)

                print('final sum:', zip_file_size , 'B')
                print('final sum:', zip_file_size / 1024, 'KB')
                print('final sum:', zip_file_size / 1024 / 1024, 'MB')
                
                with open(bin_zip_path, "rb") as f_in:
                    # 分块读取，避免一次性读入大文件
                    while True:
                        chunk = f_in.read(8192)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                tmp_file.flush()     # 确保写入落盘
                return zip_file_size

    @torch.no_grad
    def decode(self, tmp_file: io.BufferedReader):
        path = tmp_file.name
        print(path)
        with tempfile.TemporaryDirectory() as exp_dir:
            bin_dir = os.path.join(exp_dir, 'bins')
            print('bin_dir', bin_dir)
            os.makedirs(bin_dir, exist_ok=True)
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(bin_dir)
            trans_array = np.load(os.path.join(bin_dir, 't.npz'))["t"]
            
            depth = int(trans_array[0])
            self.depth = depth
            
            oct_vals = np.load(os.path.join(bin_dir , 'oct.npz'))
            
            octree = oct_vals["points"]
            oct_param = oct_vals["params"]
            self.og_number_points = octree.shape[0]

            dxyz, V = decode_oct(oct_param, octree, depth)

            self._xyz = nn.Parameter(torch.tensor(dxyz, dtype=torch.float, device="cuda").requires_grad_(False))
            n_points = dxyz.shape[0]
            
            ntk = np.load(os.path.join(bin_dir , 'ntk.npz'))["ntk"]
            cb = torch.tensor(np.load(os.path.join(bin_dir , 'um.npz'))["umap"])
            # print('ntk.shape', ntk.shape)
            # print('cb.shape', cb.shape)
            
            features_rest = torch.zeros([ntk.shape[0], cb.shape[1]])
            for i in range(ntk.shape[0]):
                features_rest[i] = cb[int(ntk[i])]
            self.n_sh = (self.max_sh_degree + 1) ** 2
            self._features_rest = nn.Parameter(features_rest.to("cuda")).contiguous().reshape(-1, self.n_sh - 1, 3).requires_grad_(False)
            
            # self._features_rest = nn.Parameter(
            #     torch.matmul(
            #         F.one_hot(torch.tensor(ntk, dtype=torch.long, device="cuda")).float(), 
            #         torch.tensor(cb, dtype=torch.float, device="cuda")
            #     ).contiguous().reshape(-1, self.n_sh - 1, 3).requires_grad_(False))
            
            # print('gaussian model, line 1027, trans_array', trans_array.shape, trans_array)
            
            oef_vals = np.load(os.path.join(bin_dir,'orgb.npz'))
            orgb_f = torch.tensor(oef_vals["f"], dtype=torch.float, device="cuda")
            q_orgb_i = torch.tensor(oef_vals["i"].astype(np.float32), dtype=torch.float, device="cuda").reshape(7, -1).contiguous().transpose(0, 1)
            q_scale_i = torch.tensor(np.load(os.path.join(bin_dir, 'ct.npz'))["i"], dtype=torch.float, device="cuda").reshape(3, -1).contiguous().transpose(0, 1)
            # q_orgb_i = torch.tensor(oef_vals["i"].astype(np.float32), dtype=torch.float, device="cuda").reshape(-1, 7)
            # q_scale_i = torch.tensor(np.load(os.path.join(bin_dir, 'ct.npz'))["i"], dtype=torch.float, device="cuda").reshape(-1, 3)
            
            print('rf_orgb_f size', orgb_f.shape)
            print('q_rf_orgb_i.shape', q_orgb_i.shape)
            print('q_scale_i.shape', q_scale_i.shape)
            
            # lseg = int(trans_array[1])
            # self.lseg = lseg
            n_block = int(trans_array[1])
            self.n_block = n_block
            
            '''dequant'''
            qa_cnt = 2
            rf_orgb = []
            rf_len = q_orgb_i.shape[0]
            # print('rf_len, n_points', rf_len, n_points)
            assert rf_len + 1 == n_points
            # if rf_len % self.lseg == 0:
            #     n_rf = rf_len // self.lseg
            # else:
            #     n_rf = rf_len // self.lseg + 1
            split = split_length(rf_len, n_block)
            for i in range(7):
                rf_i = torch_vanilla_dequant_ave(q_orgb_i[:, i], split, trans_array[qa_cnt:qa_cnt+2*n_block])
                # print('rf_i.shape', rf_i.shape)
                rf_orgb.append(rf_i.reshape(-1, 1))
                qa_cnt += 2*n_block
            rf_orgb = torch.concat(rf_orgb, dim=-1)
            
            
            de_scale = []
            scale_len = q_scale_i.shape[0]
            assert scale_len == n_points
            # if scale_len % self.lseg == 0:
            #     n_scale = scale_len // self.lseg
            # else:
            #     n_scale = scale_len // self.lseg + 1
            scale_split = split_length(scale_len, n_block)
            for i in range(3):
                scale_i = torch_vanilla_dequant_ave(q_scale_i[:, i], scale_split, trans_array[qa_cnt:qa_cnt+2*n_block])
                de_scale.append(scale_i.reshape(-1, 1))
                qa_cnt += 2*n_block
            de_scale = torch.concat(de_scale, axis=-1).to("cuda")
            self._scaling = nn.Parameter(de_scale.requires_grad_(False))
            
            print('qa_cnt', qa_cnt, 'trans', trans_array.shape)
            print('rf_orgb.shape, de_scale.shape', rf_orgb.shape, de_scale.shape)
            
            C = torch.concat([orgb_f.reshape(1, -1), rf_orgb], 0)
            
            w, val, reorder = copyAsort(V)

            # print('saving 2')
            self.reorder = reorder  
            res_inv = inv_haar3D_param(V, depth)
            pos = res_inv['pos']
            iW1 = res_inv['iW1']
            iW2 = res_inv['iW2']
            iS = res_inv['iS']
            
            iLeft_idx = res_inv['iLeft_idx']
            iRight_idx = res_inv['iRight_idx']
        
            iLeft_idx_CT = res_inv['iLeft_idx_CT']
            iRight_idx_CT = res_inv['iRight_idx_CT']
            iTrans_idx = res_inv['iTrans_idx']
            iTrans_idx_CT = res_inv['iTrans_idx_CT']

            CT_yuv_q_temp = C[pos.astype(int)]
            raht_features = torch.zeros(C.shape).cuda()
            OC = torch.zeros(C.shape).cuda()

            for i in range(depth*3):
                w1 = iW1[i]
                w2 = iW2[i]
                S = iS[i]
                
                left_idx, right_idx = iLeft_idx[i], iRight_idx[i]
                left_idx_CT, right_idx_CT = iLeft_idx_CT[i], iRight_idx_CT[i]
                
                trans_idx, trans_idx_CT = iTrans_idx[i], iTrans_idx_CT[i]
                
                
                OC[trans_idx] = CT_yuv_q_temp[trans_idx_CT]
                OC[left_idx], OC[right_idx] = itransform_batched_torch(w1, 
                                                        w2, 
                                                        CT_yuv_q_temp[left_idx_CT], 
                                                        CT_yuv_q_temp[right_idx_CT])  
                CT_yuv_q_temp[:S] = OC[:S]

            raht_features[reorder] = OC
            
            
            self._opacity = nn.Parameter(raht_features[:, :1].requires_grad_(False))
            self._euler = nn.Parameter(raht_features[:, 1:4].nan_to_num_(0).requires_grad_(False))
            self._features_dc = nn.Parameter(raht_features[:, 4:].unsqueeze(1).requires_grad_(False))
            # print('max euler', torch.max(self._euler))
            
            self.active_sh_degree = self.max_sh_degree