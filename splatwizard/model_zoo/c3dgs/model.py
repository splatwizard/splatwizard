

import torch
import time
import math

from loguru import logger
from torch.distributed.checkpoint import save_state_dict
from tqdm import tqdm, trange

from splatwizard.modules.densify_mixin import DensificationAndPruneMixin
from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard.modules.loss_mixin import LossMixin
from splatwizard.rasterizer.indexed_gs import GaussianRasterizationSettings, GaussianRasterizer, GaussianRasterizerIndexed
from splatwizard.scheduler import Scheduler, task

from .config import C3DGSModelParams, C3DGSOptimizationParams, CompressionSettings
from splatwizard.modules.gaussian_model import GaussianModel

from splatwizard.utils.general_utils import (
    inverse_sigmoid, get_expon_lr_func,
    build_scaling_rotation, strip_symmetric
)
from splatwizard.utils.system_utils import mkdir_p
from splatwizard.utils.splats import to_full_cov, extract_rot_scale
from ...modules.dataclass import RenderResult

from ...scene import CameraIterator

from enum import Enum
import numpy as np
from plyfile import PlyElement, PlyData
import gc
import os
from os import makedirs, path
from typing import Dict, Tuple, Optional
from torch import nn
from splatwizard._cmod.weighted_distance import weighted_distance
from torch_scatter import scatter
from torch.nn.functional import normalize

from ...utils.misc import check_tensor
from ...utils.sh_utils import eval_sh


class ColorMode(Enum):
    NOT_INDEXED = 0
    ALL_INDEXED = 1


class FakeQuantizationHalf(torch.autograd.Function):
    """performs fake quantization for half precision"""

    @staticmethod
    def forward(_, x: torch.Tensor) -> torch.Tensor:
        return x.half().float()

    @staticmethod
    def backward(_, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


# def build_rotation(r):
#     norm = torch.sqrt(
#         r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
#     )
#
#     q = r / norm[:, None]
#
#     R = torch.zeros((q.size(0), 3, 3), device=r.device)
#
#     r = q[:, 0]
#     x = q[:, 1]
#     y = q[:, 2]
#     z = q[:, 3]
#
#     R[:, 0, 0] = 1 - 2 * (y * y + z * z)
#     R[:, 0, 1] = 2 * (x * y - r * z)
#     R[:, 0, 2] = 2 * (x * z + r * y)
#     R[:, 1, 0] = 2 * (x * y + r * z)
#     R[:, 1, 1] = 1 - 2 * (x * x + z * z)
#     R[:, 1, 2] = 2 * (y * z - r * x)
#     R[:, 2, 0] = 2 * (x * z - r * y)
#     R[:, 2, 1] = 2 * (y * z + r * x)
#     R[:, 2, 2] = 1 - 2 * (x * x + y * y)
#     return R
#
# def build_scaling_rotation(s, r):
#     L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
#     R = build_rotation(r)
#
#     L[:, 0, 0] = s[:, 0]
#     L[:, 1, 1] = s[:, 1]
#     L[:, 2, 2] = s[:, 2]
#
#     L = R @ L
#     return L

# def strip_lowerdiag(L):
#     uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
#
#     uncertainty[:, 0] = L[:, 0, 0]
#     uncertainty[:, 1] = L[:, 0, 1]
#     uncertainty[:, 2] = L[:, 0, 2]
#     uncertainty[:, 3] = L[:, 1, 1]
#     uncertainty[:, 4] = L[:, 1, 2]
#     uncertainty[:, 5] = L[:, 2, 2]
#     return uncertainty
#
# def strip_symmetric(sym):
#     return strip_lowerdiag(sym)

# def mkdir_p(folder_path):
#     # Creates a directory. equivalent to using mkdir -p on the command line
#     try:
#         makedirs(folder_path)
#     except OSError as exc: # Python >2.5
#         if exc.errno == EEXIST and path.isdir(folder_path):
#             pass
#         else:
#             raise


def splitBy3(a):
    x = a & 0x1FFFFF  # we only look at the first 21 bits
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


def mortonEncode(pos: torch.Tensor) -> torch.Tensor:
    x, y, z = pos.unbind(-1)
    answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2
    return answer


class VectorQuantize(nn.Module):
    def __init__(
            self,
            channels: int,
            codebook_size: int = 2 ** 12,
            decay: float = 0.5,
    ) -> None:
        super().__init__()
        self.decay = decay
        self.codebook = nn.Parameter(
            torch.empty(codebook_size, channels), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.codebook)
        self.entry_importance = nn.Parameter(
            torch.zeros(codebook_size), requires_grad=False
        )
        self.eps = 1e-5

    def uniform_init(self, x: torch.Tensor):
        amin, amax = x.aminmax()
        self.codebook.data = torch.rand_like(self.codebook) * (amax - amin) + amin

    def update(self, x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            min_dists, idx = weighted_distance(x.detach(), self.codebook.detach())
            acc_importance = scatter(
                importance, idx, 0, reduce="sum", dim_size=self.codebook.shape[0]
            )

            ema_inplace(self.entry_importance, acc_importance, self.decay)

            codebook = scatter(
                x * importance[:, None],
                idx,
                0,
                reduce="sum",
                dim_size=self.codebook.shape[0],
            )

            ema_inplace(
                self.codebook,
                codebook / (acc_importance[:, None] + self.eps),
                self.decay,
            )

            return min_dists

    def forward(
            self,
            x: torch.Tensor,
            return_dists: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        min_dists, idx = weighted_distance(x.detach(), self.codebook.detach())
        if return_dists:
            return self.codebook[idx], idx, min_dists
        else:
            return self.codebook[idx], idx


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


class C3DGS(LossMixin, DensificationAndPruneMixin, GaussianModel):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(
                scaling, scaling_modifier, rotation, strip_sym=True
        ):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if strip_sym:
                return strip_symmetric(actual_covariance)
            else:
                return actual_covariance

        self.scaling_activation = lambda x: torch.nn.functional.normalize(
            torch.nn.functional.relu(x)
        )
        self.scaling_inverse_activation = lambda x: x
        self.scaling_factor_activation = torch.exp
        self.scaling_factor_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, model_param: C3DGSModelParams):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = model_param.sh_degree
        self.percent_dense = 0
        self._scaling_factor = torch.empty(0)
        self.setup_functions()

        # quantization related stuff
        self._feature_indices = None
        self._gaussian_indices = None

        self.quantization = not model_param.not_quantization_aware
        self.color_index_mode = ColorMode.NOT_INDEXED

        self.features_dc_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.features_rest_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.opacity_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.scaling_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.scaling_factor_qa = torch.ao.quantization.FakeQuantize(
            dtype=torch.qint8
        ).cuda()
        self.rotation_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.xyz_qa = FakeQuantizationHalf.apply

        if not self.quantization:
            self.features_dc_qa.disable_fake_quant()
            self.features_dc_qa.disable_observer()
            self.features_rest_qa.disable_fake_quant()
            self.features_rest_qa.disable_observer()

            self.scaling_qa.disable_fake_quant()
            self.scaling_qa.disable_observer()
            self.scaling_factor_qa.disable_fake_quant()
            self.scaling_factor_qa.disable_observer()

            self.opacity_qa.disable_fake_quant()
            self.opacity_qa.disable_observer()

            self.rotation_qa.disable_fake_quant()
            self.rotation_qa.disable_observer()
            self.xyz_qa = lambda x: x

    def register_pre_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: OptimizationParams):

        # scheduler.register_task(range(0, opt.iterations, 1000), task=self.oneupSHdegree)
        # scheduler.register_task(1, task=self.calc_importance_task)
        scheduler.register_task(1, task=self.vq_compress)
        scheduler.register_task(range(opt.iterations), task=self.update_learning_rate)
        scheduler.register_task(1, task=self.re_exec_training_setup, priority=1) # exec after vq_compress
        # pass

    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: C3DGSOptimizationParams):
        pass

    @task
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def render(self, viewpoint_camera, background, pipe, opt=None, step=0, scaling_modifier=1.0, override_color=None,
               clamp_color: bool = True,
               cov3d: torch.Tensor = None
               ):
        """
            Render the scene.

            Background tensor (bg_color) must be on GPU!
            """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
                torch.zeros_like(
                    self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device="cuda"
                )
                + 0
        )
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
            debug=pipe.debug,
            clamp_color=clamp_color,
        )

        if self.color_index_mode == ColorMode.ALL_INDEXED and self.is_gaussian_indexed:
            rasterizer = GaussianRasterizerIndexed(raster_settings=raster_settings)

            means3D = self.get_xyz
            means2D = screenspace_points
            opacity = self.get_opacity
            shs = self._get_features_raw
            scales = self.get_scaling_normalized
            scale_factors = self.get_scaling_factor
            rotations = self._rotation_post_activation

            # Rasterize visible Gaussians to image, obtain their radii (on screen).
            rendered_image, radii = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                sh_indices=self._feature_indices,
                g_indices=self._gaussian_indices,
                colors_precomp=None,
                opacities=opacity,
                scales=scales,
                scale_factors=scale_factors,
                rotations=rotations,
                cov3D_precomp=None,
            )

            # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
            # They will be excluded from value updates used in the splitting criteria.
            # return {
            #     "render": rendered_image,
            #     "viewspace_points": screenspace_points,
            #     "visibility_filter": radii > 0,
            #     "radii": radii,
            # }
            return RenderResult(
                rendered_image=rendered_image,
                viewspace_points=screenspace_points,
                visibility_filter=radii > 0,
                radii=radii
            )
        else:
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            means3D = self.get_xyz
            means2D = screenspace_points
            opacity = self.get_opacity

            # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
            # scaling / rotation by the rasterizer.
            scales = None
            rotations = None
            cov3D_precomp = cov3d
            if cov3D_precomp is None:
                if pipe.compute_cov3D_python:
                    cov3D_precomp = self.get_covariance(scaling_modifier)
                else:
                    scales = self.get_scaling
                    rotations = self.get_rotation

            # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
            # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
            shs = None
            colors_precomp = None
            if override_color is None:
                if pipe.convert_SHs_python:
                    shs_view = self.get_features.transpose(1, 2).view(
                        -1, 3, (self.max_sh_degree + 1) ** 2
                    )
                    dir_pp = self.get_xyz - viewpoint_camera.camera_center.repeat(
                        self.get_features.shape[0], 1
                    )
                    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                else:
                    shs = self.get_features
            else:
                colors_precomp = override_color

            #
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
            )

            # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
            # They will be excluded from value updates used in the splitting criteria.
            # return {
            #     "render": rendered_image,
            #     "viewspace_points": screenspace_points,
            #     "visibility_filter": radii > 0,
            #     "radii": radii,
            # }
            return RenderResult(
                rendered_image=rendered_image,
                viewspace_points=screenspace_points,
                visibility_filter=radii > 0,
                radii=radii
            )

    @property
    def get_scaling(self):
        scaling_n = self.scaling_qa(self.scaling_activation(self._scaling))
        scaling_factor = self.scaling_factor_activation(
            self.scaling_factor_qa(self._scaling_factor)
        )
        if self.is_gaussian_indexed:
            return scaling_factor * scaling_n[self._gaussian_indices]
        else:
            return scaling_factor * scaling_n

    @property
    def get_scaling_normalized(self):
        return self.scaling_qa(self.scaling_activation(self._scaling))

    @property
    def get_scaling_factor(self):
        return self.scaling_factor_activation(
            self.scaling_factor_qa(self._scaling_factor)
        )

    @property
    def get_rotation(self):
        rotation = self.rotation_activation(self.rotation_qa(self._rotation))
        if self.is_gaussian_indexed:
            return rotation[self._gaussian_indices]
        else:
            return rotation

    @property
    def _rotation_post_activation(self):
        return self.rotation_activation(self.rotation_qa(self._rotation))

    @property
    def get_xyz(self):
        return self.xyz_qa(self._xyz)

    @property
    def get_features(self):
        features_dc = self.features_dc_qa(self._features_dc)
        features_rest = self.features_rest_qa(self._features_rest)

        if self.color_index_mode == ColorMode.ALL_INDEXED:
            return torch.cat((features_dc, features_rest), dim=1)[self._feature_indices]
        else:
            return torch.cat((features_dc, features_rest), dim=1)

    @property
    def _get_features_raw(self):
        features_dc = self.features_dc_qa(self._features_dc)

        # check_tensor(features_dc)
        features_rest = self.features_rest_qa(self._features_rest)
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_qa(self.opacity_activation(self._opacity))

    @property
    def get_covariance(self, scaling_modifier=1, strip_sym: bool = True):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.get_rotation, strip_sym
        )

    def get_normalized_covariance(self, scaling_modifier=1, strip_sym: bool = True):
        scaling_n = self.scaling_qa(self.scaling_activation(self._scaling))
        return self.covariance_activation(
            scaling_n, scaling_modifier, self.get_rotation, strip_sym
        )

    @property
    def is_color_indexed(self):
        return self._feature_indices is not None

    @property
    def is_gaussian_indexed(self):
        return self._gaussian_indices is not None

    def training_setup(self, training_args: C3DGSOptimizationParams):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._scaling_factor],
                "lr": training_args.scaling_lr,
                "name": "scaling_factor",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    @task
    def re_exec_training_setup(self, opt: C3DGSOptimizationParams):
        self.training_setup(opt)

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

    # def construct_list_of_attributes(self):
    #     l = ["x", "y", "z", "nx", "ny", "nz"]
    #     # All channels except the 3 DC
    #     for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
    #         l.append("f_dc_{}".format(i))
    #     for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
    #         l.append("f_rest_{}".format(i))
    #     l.append("opacity")
    #     for i in range(self._scaling.shape[1]):
    #         l.append("scale_{}".format(i))
    #     # l.append("scale_factor")
    #     for i in range(self._rotation.shape[1]):
    #         l.append("rot_{}".format(i))
    #     return l

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._scaling_factor,
            self._feature_indices,
            self._gaussian_indices,
            self.quantization,
            self.color_index_mode,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.features_dc_qa.state_dict(),
            self.features_rest_qa.state_dict(),
            self.opacity_qa.state_dict(),
            self.scaling_qa.state_dict(),
            self.scaling_factor_qa.state_dict(),
            self.rotation_qa.state_dict(),
            self._context
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
            self._scaling_factor,
            self._feature_indices,
            self._gaussian_indices,
            self.quantization,
            self.color_index_mode,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            features_dc_qa_state_dict,
            features_rest_qa_state_dict,
            opacity_qa_state_dict,
            scaling_qa_state_dict,
            scaling_factor_qa_state_dict,
            rotation_qa_state_dict,
            context
        ) = model_args
        if training_args is not None:
            self.training_setup(training_args)

        # Since training_setup will reset these parameters, we assign values to them manually
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if self.optimizer is not None:
            self.optimizer.load_state_dict(opt_dict)
        self._context = context  # type(self._context)(**context)

        self.features_dc_qa.load_state_dict(features_dc_qa_state_dict)
        self.features_rest_qa.load_state_dict(features_rest_qa_state_dict)
        self.opacity_qa.load_state_dict(opacity_qa_state_dict)
        self.scaling_qa.load_state_dict(scaling_qa_state_dict)
        self.scaling_factor_qa.load_state_dict(scaling_factor_qa_state_dict)
        self.rotation_qa.load_state_dict(rotation_qa_state_dict)

    def save_ply(self, path):
        # mkdir_p(os.path.dirname(path))

        if self.is_gaussian_indexed or self.is_color_indexed:
            logger.warning(
                "indexed colors/gaussians are not supported for ply files and are converted to dense attributes"
            )

        color_features = self.get_features.detach()

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            color_features[:, :1]
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            color_features[:, 1:]
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = (
            self.scaling_factor_inverse_activation(self.get_scaling.detach())
            .cpu()
            .numpy()
        )

        rotation = self.get_rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)


    # def load(self, path: str,override_quantization=False):
    #     ext = os.path.splitext(path)[1]
    #     if ext == ".ply":
    #         self.load_ply(path)
    #     elif ext == ".npz":
    #         self.load_npz(path,override_quantization)
    #     else:
    #         raise NotImplementedError(f"file ending '{ext}' not supported")

    def load_ply(self, path, data=None):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_") and not p.name.startswith("scale_factor")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        scaling = self.scaling_factor_activation(torch.tensor(scales, dtype=torch.float, device="cuda"))
        scaling_norm = scaling.norm(2, -1, keepdim=True)
        self._scaling = nn.Parameter(
            self.scaling_inverse_activation(scaling / scaling_norm).requires_grad_(True)
        )
        self._scaling_factor = nn.Parameter(
            self.scaling_factor_inverse_activation(scaling_norm)
            .detach()
            .requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def quantize_param(self, param, qa, key):
        # save_dict = {}
        # rotation = self.rotation_activation(self._rotation).detach()
        param_q = torch.quantize_per_tensor(
            param,
            scale=qa.scale,
            zero_point=qa.zero_point,
            dtype=qa.dtype,
        ).int_repr()
        return {
            key: param_q.cpu().numpy(),  # value
            key + '_scale': qa.scale.cpu().numpy(),  # scale
            key + '_zp': qa.zero_point.cpu().numpy()  # zero point
        }
        #     param_q.cpu().numpy(),       # value
        #     qa.scale.cpu().numpy(),      # scale
        #     qa.zero_point.cpu().numpy()  # zero point
        # )

    def dequantize_param(self, pack, qa, key):
        value, scale, zero_point = pack[key], pack[key+'_scale'], pack[key + '_zp']
        features_dc_q = torch.from_numpy(value).int().cuda()
        features_dc_scale = torch.from_numpy(scale).cuda()
        features_dc_zero_point = torch.from_numpy(
            zero_point
        ).cuda()
        features_dc = (features_dc_q - features_dc_zero_point) * features_dc_scale
        # self._features_dc = nn.Parameter(features_dc, requires_grad=True)

        qa.scale = features_dc_scale
        qa.zero_point = features_dc_zero_point
        qa.activation_post_process.min_val = features_dc.min()
        qa.activation_post_process.max_val = features_dc.max()

        return features_dc

    @torch.no_grad()
    def save_npz(
            self,
            path,
            compress: bool = True,
            half_precision: bool = False,
            sort_morton=False,
    ):
            if sort_morton:
                self._sort_morton()
            # if isinstance(path, str):
            #     mkdir_p(os.path.dirname(os.path.abspath(path)))

            dtype = torch.half if half_precision else torch.float32

            save_dict = dict()

            save_dict["quantization"] = self.quantization

            # save position
            if self.quantization:
                save_dict["xyz"] = self.get_xyz.detach().half().cpu().numpy()
            else:
                save_dict["xyz"] = self._xyz.detach().cpu().numpy()

            # # save color features
            if self.quantization:
                # features_dc_q = torch.quantize_per_tensor(
                #     self._features_dc.detach(),
                #     self.features_dc_qa.scale,
                #     self.features_dc_qa.zero_point,
                #     self.features_dc_qa.dtype,
                # ).int_repr()
                #
                # save_dict["features_dc"] = features_dc_q.cpu().numpy()
                # save_dict["features_dc_scale"] = self.features_dc_qa.scale.cpu().numpy()
                # save_dict[
                #     "features_dc_zero_point"
                # ] = self.features_dc_qa.zero_point.cpu().numpy()

                # value, scale, zero_point = self.quantize_param(self._features_dc, self.features_dc_qa)
                # save_dict["features_dc"] = value
                # save_dict["features_dc_scale"] = scale
                # save_dict["features_dc_zero_point"] = zero_point
                save_dict.update(self.quantize_param(self._features_dc, self.features_dc_qa, 'features_dc'))
            else:
                save_dict["features_dc"] = self._features_dc.detach().cpu().numpy()
            #
            if self.quantization:
                # features_rest_q = torch.quantize_per_tensor(
                #     self._features_rest.detach(),
                #     self.features_rest_qa.scale,
                #     self.features_rest_qa.zero_point,
                #     self.features_rest_qa.dtype,
                # ).int_repr()
                # save_dict["features_rest"] = features_rest_q.cpu().numpy()
                # save_dict["features_rest_scale"] = self.features_rest_qa.scale.cpu().numpy()
                # save_dict[
                #     "features_rest_zero_point"
                # ] = self.features_rest_qa.zero_point.cpu().numpy()
                save_dict.update(self.quantize_param(self._features_rest, self.features_rest_qa, 'features_rest'))
            else:

                save_dict["features_rest"] = self._features_rest.detach().cpu().numpy()
            #
            # # save opacity
            if self.quantization:
                opacity = self.opacity_activation(self._opacity).detach()
                # opacity_q = torch.quantize_per_tensor(
                #     opacity,
                #     scale=self.opacity_qa.scale,
                #     zero_point=self.opacity_qa.zero_point,
                #     dtype=self.opacity_qa.dtype,
                # ).int_repr()
                # save_dict["opacity"] = opacity_q.cpu().numpy()
                # save_dict["opacity_scale"] = self.opacity_qa.scale.cpu().numpy()
                # save_dict[
                #     "opacity_zero_point"
                # ] = self.opacity_qa.zero_point.cpu().numpy()
                save_dict.update(self.quantize_param(opacity, self.opacity_qa, 'opacity'))
            else:
                save_dict["opacity"] = self._opacity.detach().to(dtype).cpu().numpy()
            #
            # # save indices
            if self.is_color_indexed:
                save_dict["feature_indices"] = (
                    self._feature_indices.detach().contiguous().cpu().int().numpy()
                )
            if self.is_gaussian_indexed:
                save_dict["gaussian_indices"] = (
                    self._gaussian_indices.detach().contiguous().cpu().int().numpy()
                )
            #
            # # save scaling
            if self.quantization:
                scaling = self.scaling_activation(self._scaling.detach())
                # scaling_q = torch.quantize_per_tensor(
                #     scaling,
                #     scale=self.scaling_qa.scale,
                #     zero_point=self.scaling_qa.zero_point,
                #     dtype=self.scaling_qa.dtype,
                # ).int_repr()
                # save_dict["scaling"] = scaling_q.cpu().numpy()
                # save_dict["scaling_scale"] = self.scaling_qa.scale.cpu().numpy()
                # save_dict[
                #     "scaling_zero_point"
                # ] = self.scaling_qa.zero_point.cpu().numpy()
                save_dict.update(self.quantize_param(scaling, self.scaling_qa, 'scaling'))

                scaling_factor = self._scaling_factor.detach()
                # scaling_factor_q = torch.quantize_per_tensor(
                #     scaling_factor,
                #     scale=self.scaling_factor_qa.scale,
                #     zero_point=self.scaling_factor_qa.zero_point,
                #     dtype=self.scaling_factor_qa.dtype,
                # ).int_repr()
                # save_dict["scaling_factor"] = scaling_factor_q.cpu().numpy()
                # save_dict[
                #     "scaling_factor_scale"
                # ] = self.scaling_factor_qa.scale.cpu().numpy()
                # save_dict[
                #     "scaling_factor_zero_point"
                # ] = self.scaling_factor_qa.zero_point.cpu().numpy()
                save_dict.update(self.quantize_param(scaling_factor, self.scaling_factor_qa, 'scaling_factor'))
            else:
                save_dict["scaling"] = self._scaling.detach().to(dtype).cpu().numpy()
                save_dict["scaling_factor"] = (
                    self._scaling_factor.detach().to(dtype).cpu().numpy()
                )
            #
            # # save rotation
            if self.quantization:
                rotation = self.rotation_activation(self._rotation).detach()
                # rotation_q = torch.quantize_per_tensor(
                #     rotation,
                #     scale=self.rotation_qa.scale,
                #     zero_point=self.rotation_qa.zero_point,
                #     dtype=self.rotation_qa.dtype,
                # ).int_repr()
                # save_dict["rotation"] = rotation_q.cpu().numpy()
                # save_dict["rotation_scale"] = self.rotation_qa.scale.cpu().numpy()
                # save_dict[
                #     "rotation_zero_point"
                # ] = self.rotation_qa.zero_point.cpu().numpy()
                save_dict.update(self.quantize_param(rotation, self.rotation_qa, 'rotation'))

            else:
                save_dict["rotation"] = self._rotation.detach().to(dtype).cpu().numpy()

            save_fn = np.savez_compressed if compress else np.savez
            save_fn(path, **save_dict)

    def load_npz(self, path, override_quantization=False):
        state_dict = np.load(path)

        quantization = state_dict["quantization"]
        if not override_quantization and self.quantization != quantization:
            print("WARNING: model is not quantisation aware but loaded model is")
        if override_quantization:
            self.quantization = quantization

        # load position
        self._xyz = nn.Parameter(
            torch.from_numpy(state_dict["xyz"]).float().cuda(), requires_grad=True
        )

        # load color
        if quantization:
            # features_dc_q = torch.from_numpy(state_dict["features_dc"]).int().cuda()
            # features_dc_scale = torch.from_numpy(state_dict["features_dc_scale"]).cuda()
            # features_dc_zero_point = torch.from_numpy(
            #     state_dict["features_dc_zero_point"]
            # ).cuda()
            # features_dc = (features_dc_q - features_dc_zero_point) * features_dc_scale
            # self._features_dc = nn.Parameter(features_dc, requires_grad=True)
            #
            # self.features_dc_qa.scale = features_dc_scale
            # self.features_dc_qa.zero_point = features_dc_zero_point
            # self.features_dc_qa.activation_post_process.min_val = features_dc.min()
            # self.features_dc_qa.activation_post_process.max_val = features_dc.max()

            self._features_dc = nn.Parameter(
                self.dequantize_param(state_dict, self.features_dc_qa, 'features_dc'),
                requires_grad=True
            )

        else:
            features_dc = torch.from_numpy(state_dict["features_dc"]).float().cuda()
            self._features_dc = nn.Parameter(features_dc, requires_grad=True)
        #
        if quantization:
            # features_rest_q = torch.from_numpy(state_dict["features_rest"]).int().cuda()
            # features_rest_scale = torch.from_numpy(
            #     state_dict["features_rest_scale"]
            # ).cuda()
            # features_rest_zero_point = torch.from_numpy(
            #     state_dict["features_rest_zero_point"]
            # ).cuda()
            # features_rest = (
            #                         features_rest_q - features_rest_zero_point
            #                 ) * features_rest_scale
            # self._features_rest = nn.Parameter(features_rest, requires_grad=True)
            # self.features_rest_qa.scale = features_rest_scale
            # self.features_rest_qa.zero_point = features_rest_zero_point
            # self.features_rest_qa.activation_post_process.min_val = features_rest.min()
            # self.features_rest_qa.activation_post_process.max_val = features_rest.max()

            self._features_rest = nn.Parameter(
                self.dequantize_param(state_dict, self.features_rest_qa, 'features_rest'),
                requires_grad=True
            )
        else:

            features_rest = torch.from_numpy(state_dict["features_rest"]).float().cuda()
            self._features_rest = nn.Parameter(features_rest, requires_grad=True)
        #
        #
        # # load opacity
        if quantization:
            # opacity_q = torch.from_numpy(state_dict["opacity"]).int().cuda()
            # opacity_scale = torch.from_numpy(state_dict["opacity_scale"]).cuda()
            # opacity_zero_point = torch.from_numpy(
            #     state_dict["opacity_zero_point"]
            # ).cuda()
            # opacity = (opacity_q - opacity_zero_point) * opacity_scale
            # self._opacity = nn.Parameter(
            #     self.inverse_opacity_activation(opacity), requires_grad=True
            # )
            # self.opacity_qa.scale = opacity_scale
            # self.opacity_qa.zero_point = opacity_zero_point
            # self.opacity_qa.activation_post_process.min_val = opacity.min()
            # self.opacity_qa.activation_post_process.max_val = opacity.max()

            opacity = self.dequantize_param(state_dict, self.opacity_qa, 'opacity')
            self._opacity = nn.Parameter(
                self.inverse_opacity_activation(opacity), requires_grad=True
            )
        else:
            self._opacity = nn.Parameter(
                torch.from_numpy(state_dict["opacity"]).float().cuda(),
                requires_grad=True,
            )
        #
        # # load scaling
        if quantization:
            # scaling_q = torch.from_numpy(state_dict["scaling"]).int().cuda()
            # scaling_scale = torch.from_numpy(state_dict["scaling_scale"]).cuda()
            # scaling_zero_point = torch.from_numpy(
            #     state_dict["scaling_zero_point"]
            # ).cuda()
            # scaling = (scaling_q - scaling_zero_point) * scaling_scale
            # self._scaling = nn.Parameter(
            #     self.scaling_inverse_activation(scaling), requires_grad=True
            # )
            # self.scaling_qa.scale = scaling_scale
            # self.scaling_qa.zero_point = scaling_zero_point
            # self.scaling_qa.activation_post_process.min_val = scaling.min()
            # self.scaling_qa.activation_post_process.max_val = scaling.max()

            scaling = self.dequantize_param(state_dict, self.scaling_qa, 'scaling')
            self._scaling = nn.Parameter(
                self.scaling_inverse_activation(scaling), requires_grad=True
            )


            # scaling_factor_q = (
            #     torch.from_numpy(state_dict["scaling_factor"]).int().cuda()
            # )
            # scaling_factor_scale = torch.from_numpy(
            #     state_dict["scaling_factor_scale"]
            # ).cuda()
            # scaling_factor_zero_point = torch.from_numpy(
            #     state_dict["scaling_factor_zero_point"]
            # ).cuda()
            # scaling_factor = (
            #                          scaling_factor_q - scaling_factor_zero_point
            #                  ) * scaling_factor_scale
            # self._scaling_factor = nn.Parameter(
            #     scaling_factor,
            #     requires_grad=True,
            # )
            # self.scaling_factor_qa.scale = scaling_factor_scale
            # self.scaling_factor_qa.zero_point = scaling_factor_zero_point
            # self.scaling_factor_qa.activation_post_process.min_val = (
            #     scaling_factor.min()
            # )
            # self.scaling_factor_qa.activation_post_process.max_val = (
            #     scaling_factor.max()
            # )

            scaling_factor = self.dequantize_param(state_dict, self.scaling_factor_qa, 'scaling_factor')
            self._scaling_factor = nn.Parameter(
                scaling_factor,
                requires_grad=True,
            )
        else:
            self._scaling_factor = nn.Parameter(
                torch.from_numpy(state_dict["scaling_factor"]).float().cuda(),
                requires_grad=True,
            )
            self._scaling = nn.Parameter(
                torch.from_numpy(state_dict["scaling"]).float().cuda(),
                requires_grad=True,
            )

        # load rotation
        if quantization:
            # rotation_q = torch.from_numpy(state_dict["rotation"]).int().cuda()
            # rotation_scale = torch.from_numpy(state_dict["rotation_scale"]).cuda()
            # rotation_zero_point = torch.from_numpy(
            #     state_dict["rotation_zero_point"]
            # ).cuda()
            # rotation = (rotation_q - rotation_zero_point) * rotation_scale
            # self._rotation = nn.Parameter(rotation, requires_grad=True)
            # self.rotation_qa.scale = rotation_scale
            # self.rotation_qa.zero_point = rotation_zero_point
            # self.rotation_qa.activation_post_process.min_val = rotation.min()
            # self.rotation_qa.activation_post_process.max_val = rotation.max()

            rotation = self.dequantize_param(state_dict, self.rotation_qa, 'rotation')
            self._rotation = nn.Parameter(rotation, requires_grad=True)
        else:
            self._rotation = nn.Parameter(
                torch.from_numpy(state_dict["rotation"]).float().cuda(),
                requires_grad=True,
            )

        if "gaussian_indices" in list(state_dict.keys()):
            self._gaussian_indices = nn.Parameter(
                torch.from_numpy(state_dict["gaussian_indices"]).long().to("cuda"),
                requires_grad=False,
            )

        self.color_index_mode = ColorMode.NOT_INDEXED
        if "feature_indices" in list(state_dict.keys()):
            self._feature_indices = nn.Parameter(
                torch.from_numpy(state_dict["feature_indices"]).long().to("cuda"),
                requires_grad=False,
            )
            self.color_index_mode = ColorMode.ALL_INDEXED

        self.active_sh_degree = self.max_sh_degree

    def _sort_morton(self):
        with torch.no_grad():
            xyz_q = (
                    (2 ** 21 - 1)
                    * (self._xyz - self._xyz.min(0).values)
                    / (self._xyz.max(0).values - self._xyz.min(0).values)
            ).long()
            order = mortonEncode(xyz_q).sort().indices
            self._xyz = nn.Parameter(self._xyz[order], requires_grad=True)
            self._opacity = nn.Parameter(self._opacity[order], requires_grad=True)
            self._scaling_factor = nn.Parameter(
                self._scaling_factor[order], requires_grad=True
            )

            if self.is_color_indexed:
                self._feature_indices = nn.Parameter(
                    self._feature_indices[order], requires_grad=False
                )
            else:
                self._features_rest = nn.Parameter(
                    self._features_rest[order], requires_grad=True
                )
                self._features_dc = nn.Parameter(
                    self._features_dc[order], requires_grad=True
                )

            if self.is_gaussian_indexed:
                self._gaussian_indices = nn.Parameter(
                    self._gaussian_indices[order], requires_grad=False
                )
            else:
                self._scaling = nn.Parameter(self._scaling[order], requires_grad=True)
                self._rotation = nn.Parameter(self._rotation[order], requires_grad=True)

    def mask_splats(self, mask: torch.Tensor):
        with torch.no_grad():
            self._xyz = nn.Parameter(self._xyz[mask], requires_grad=True)
            self._opacity = nn.Parameter(self._opacity[mask], requires_grad=True)
            self._scaling_factor = nn.Parameter(
                self._scaling_factor[mask], requires_grad=True
            )

            if self.is_color_indexed:
                self._feature_indices = nn.Parameter(
                    self._feature_indices[mask], requires_grad=False
                )
            else:
                self._features_dc = nn.Parameter(
                    self._features_dc[mask], requires_grad=True
                )
                self._features_rest = nn.Parameter(
                    self._features_rest[mask], requires_grad=True
                )
            if self.is_gaussian_indexed:
                self._gaussian_indices = nn.Parameter(
                    self._gaussian_indices[mask], requires_grad=False
                )
            else:
                self._scaling = nn.Parameter(self._scaling[mask], requires_grad=True)
                self._rotation = nn.Parameter(self._rotation[mask], requires_grad=True)

    def set_color_indexed(self, features: torch.Tensor, indices: torch.Tensor):
        self._feature_indices = nn.Parameter(indices, requires_grad=False)
        self._features_dc = nn.Parameter(features[:, :1].detach(), requires_grad=True)
        self._features_rest = nn.Parameter(features[:, 1:].detach(), requires_grad=True)
        self.color_index_mode = ColorMode.ALL_INDEXED

    def set_gaussian_indexed(
            self, rotation: torch.Tensor, scaling: torch.Tensor, indices: torch.Tensor
    ):
        self._gaussian_indices = nn.Parameter(indices.detach(), requires_grad=False)
        self._rotation = nn.Parameter(rotation.detach(), requires_grad=True)
        self._scaling = nn.Parameter(scaling.detach(), requires_grad=True)

    # @task
    # def reset_opacity(self):
    #     opacities_new = inverse_sigmoid(torch.min(self.opacity, torch.ones_like(self.opacity) * 0.01))
    #     optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
    #     self._opacity = optimizable_tensors["opacity"]

    # @task
    # def train_statis_task(self, render_result: RenderResult):
    #     self.add_densification_stats(render_result)

    # @task
    # def densify_and_prune_task(self, opt: C3DGSOptimizationParams, step: int):
    #     size_threshold = 20 if step > opt.opacity_reset_interval else None
    #     self.densify_and_prune(opt.densify_grad_threshold, 0.005, self.spatial_lr_scale, size_threshold)

    # @task
    # def calc_importance_task(self, ppl: PipelineParams, cam_iterator: CameraIterator):
    #     # if self.scene is None:
    #     #     raise ValueError("Scene is required for calc_importance_task")
    #     logger.info("Calculating color and gaussian sensitivity...")
    #     logger.info(f"cam_iterator in task: {cam_iterator}, type={type(cam_iterator)}")
    #
    #     color_importance, gaussian_sensitivity = self.calc_importance(
    #         cam_iterator, ppl
    #     )
    #     logger.info("Importance calculation done")
    #     self.context.color_importance = color_importance
    #     self.context.gaussian_sensitivity = gaussian_sensitivity

    def calc_importance(
            self, cam_iter, pipe
            # ) -> Tuple[torch.Tensor, torch.Tensor]:
    ):
        scaling = self.scaling_qa(
            self.scaling_activation(self._scaling.detach())
        )
        cov3d = self.covariance_activation(
            scaling, 1.0, self.rotation.detach(), True
        ).requires_grad_(True)
        scaling_factor = self.scaling_factor_activation(
            self.scaling_factor_qa(self._scaling_factor.detach())
        )

        h1 = self._features_dc.register_hook(lambda grad: grad.abs())
        h2 = self._features_rest.register_hook(lambda grad: grad.abs())
        h3 = cov3d.register_hook(lambda grad: grad.abs())
        background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

        self._features_dc.grad = None
        self._features_rest.grad = None
        num_pixels = 0

        # logger.info(f"cam_iterator in task: {cam_iter}, type={type(cam_iter)}")

        # for camera in tqdm(cam_iter, desc="Calculating sensitivity"):
        # for camera in cam_iter:
        for iteration, viewpoint_cam in enumerate(cam_iter):
            cov3d_scaled = cov3d * scaling_factor.square()

            render_result = self.render(
                viewpoint_cam,
                background,
                pipe,
                opt=None,
                step=0,
                clamp_color=False,
                cov3d=cov3d_scaled,
            )
            rendering = render_result.rendered_image

            loss = rendering.sum()
            loss.backward()
            num_pixels += rendering.shape[1] * rendering.shape[2]

        importance = torch.cat(
            [self._features_dc.grad, self._features_rest.grad],
            1,
        ).flatten(-2) / num_pixels
        cov_grad = cov3d.grad / num_pixels
        h1.remove()
        h2.remove()
        h3.remove()
        torch.cuda.empty_cache()
        return importance.detach(), cov_grad.detach()

    @task
    def vq_compress(self, ppl: PipelineParams, opt: C3DGSOptimizationParams, cam_iterator: CameraIterator):
        logger.info("vq compression..")
        # color_importance_n = self.context.color_importance.amax(-1)
        #
        # gaussian_importance_n = self.context.gaussian_sensitivity.amax(-1)
        color_importance, gaussian_sensitivity = self.calc_importance(
            cam_iterator, ppl
        )
        with torch.no_grad():

            color_importance_n = color_importance.amax(-1)
            gaussian_importance_n = gaussian_sensitivity.amax(-1)

            torch.cuda.empty_cache()

            color_compression_settings = CompressionSettings(
                codebook_size=opt.color_codebook_size,
                importance_prune=opt.color_importance_prune,
                importance_include=opt.color_importance_include,
                steps=int(opt.color_cluster_iterations),
                decay=opt.color_decay,
                batch_size=opt.color_batch_size,
            )

            gaussian_compression_settings = CompressionSettings(
                codebook_size=opt.gaussian_codebook_size,
                importance_prune=None,
                importance_include=opt.gaussian_importance_include,
                steps=int(opt.gaussian_cluster_iterations),
                decay=opt.gaussian_decay,
                batch_size=opt.gaussian_batch_size,
            )

            self.compress_gaussians(
                color_importance_n,
                gaussian_importance_n,
                color_compression_settings if not opt.not_compress_color else None,
                gaussian_compression_settings
                if not opt.not_compress_gaussians
                else None,
                opt.color_compress_non_dir,
                prune_threshold=opt.prune_threshold,
            )

    def vq_features(
            self,
            features: torch.Tensor,
            importance: torch.Tensor,
            codebook_size: int,
            vq_chunk: int = 2 ** 16,
            steps: int = 1000,
            decay: float = 0.8,
            scale_normalize: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        importance_n = importance / importance.max()
        vq_model = VectorQuantize(
            channels=features.shape[-1],
            codebook_size=codebook_size,
            decay=decay,
        ).to(device=features.device)

        vq_model.uniform_init(features)

        errors = []
        for i in trange(steps):
            batch = torch.randint(low=0, high=features.shape[0], size=[vq_chunk])
            vq_feature = features[batch]
            error = vq_model.update(vq_feature, importance=importance_n[batch]).mean().item()
            errors.append(error)
            if scale_normalize:
                # this computes the trace of the codebook covariance matrices
                # we devide by the trace to ensure that matrices have normalized eigenvalues / scales
                tr = vq_model.codebook[:, [0, 3, 5]].sum(-1)
                vq_model.codebook /= tr[:, None]

        gc.collect()
        torch.cuda.empty_cache()

        start = time.time()
        _, vq_indices = vq_model(features)
        torch.cuda.synchronize(device=vq_indices.device)
        end = time.time()
        print(f"calculating indices took {end - start} seconds ")
        return vq_model.codebook.data.detach(), vq_indices.detach()

    def join_features(
            self,
            all_features: torch.Tensor,
            keep_mask: torch.Tensor,
            codebook: torch.Tensor,
            codebook_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        keep_features = all_features[keep_mask]
        compressed_features = torch.cat([codebook, keep_features], 0)

        indices = torch.zeros(
            len(all_features), dtype=torch.long, device=all_features.device
        )
        indices[~keep_mask] = codebook_indices
        indices[keep_mask] = torch.arange(len(keep_features), device=indices.device) + len(
            codebook
        )

        return compressed_features, indices

    def compress_color(
            self,
            color_importance: torch.Tensor,
            color_comp: CompressionSettings,
            color_compress_non_dir: bool,
    ):
        keep_mask = color_importance > color_comp.importance_include

        print(
            f"color keep: {keep_mask.float().mean() * 100:.2f}%"
        )

        vq_mask_c = ~keep_mask

        # remove zero sh component
        if color_compress_non_dir:
            n_sh_coefs = self.get_features.shape[1]
            color_features = self.get_features.detach().flatten(-2)
        else:
            n_sh_coefs = self.get_features.shape[1] - 1
            color_features = self.get_features[:, 1:].detach().flatten(-2)
        if vq_mask_c.any():
            print("compressing color...")
            color_codebook, color_vq_indices = self.vq_features(
                color_features[vq_mask_c],
                color_importance[vq_mask_c],
                color_comp.codebook_size,
                color_comp.batch_size,
                color_comp.steps,
            )
        else:
            color_codebook = torch.empty(
                (0, color_features.shape[-1]), device=color_features.device
            )
            color_vq_indices = torch.empty(
                (0,), device=color_features.device, dtype=torch.long
            )

        all_features = color_features
        compressed_features, indices = self.join_features(
            all_features, keep_mask, color_codebook, color_vq_indices
        )

        self.set_color_indexed(compressed_features.reshape(-1, n_sh_coefs, 3), indices)

    def compress_covariance(
            self,
            gaussian_importance: torch.Tensor,
            gaussian_comp: CompressionSettings,
    ):

        keep_mask_g = gaussian_importance > gaussian_comp.importance_include

        vq_mask_g = ~keep_mask_g

        print(f"gaussians keep: {keep_mask_g.float().mean() * 100:.2f}%")

        covariance = self.get_normalized_covariance(strip_sym=True).detach()

        if vq_mask_g.any():
            print("compressing gaussian splats...")
            cov_codebook, cov_vq_indices = self.vq_features(
                covariance[vq_mask_g],
                gaussian_importance[vq_mask_g],
                gaussian_comp.codebook_size,
                gaussian_comp.batch_size,
                gaussian_comp.steps,
                scale_normalize=True,
            )
        else:
            cov_codebook = torch.empty(
                (0, covariance.shape[1], 1), device=covariance.device
            )
            cov_vq_indices = torch.empty((0,), device=covariance.device, dtype=torch.long)

        compressed_cov, cov_indices = self.join_features(
            covariance,
            keep_mask_g,
            cov_codebook,
            cov_vq_indices,
        )

        rot_vq, scale_vq = extract_rot_scale(to_full_cov(compressed_cov))

        self.set_gaussian_indexed(
            rot_vq.to(compressed_cov.device),
            scale_vq.to(compressed_cov.device),
            cov_indices,
        )

    def compress_gaussians(
            self,
            color_importance: torch.Tensor,
            gaussian_importance: torch.Tensor,
            color_comp: Optional[CompressionSettings],
            gaussian_comp: Optional[CompressionSettings],
            color_compress_non_dir: bool,
            prune_threshold: float = 0.,
    ):
        with torch.no_grad():
            if prune_threshold >= 0:
                non_prune_mask = color_importance > prune_threshold
                print(f"prune: {(1 - non_prune_mask.float().mean()) * 100:.2f}%")
                self.mask_splats(non_prune_mask)
                gaussian_importance = gaussian_importance[non_prune_mask]
                color_importance = color_importance[non_prune_mask]

            if color_comp is not None:
                self.compress_color(
                    color_importance,
                    color_comp,
                    color_compress_non_dir,
                )
            if gaussian_comp is not None:
                self.compress_covariance(
                    gaussian_importance,
                    gaussian_comp,
                )


    def check_quant(self):
        save_dict = {}

        features_dc_ref = self.features_dc_qa(self._features_dc)
        #
        #
        # features_dc_q = torch.quantize_per_tensor(
        #     self._features_dc.detach(),
        #     self.features_dc_qa.scale,
        #     self.features_dc_qa.zero_point,
        #     self.features_dc_qa.dtype,
        # ).int_repr()
        #
        # save_dict["features_dc"] = features_dc_q.cpu().numpy()
        # save_dict["features_dc_scale"] = self.features_dc_qa.scale.cpu().numpy()
        # save_dict[
        #     "features_dc_zero_point"
        # ] = self.features_dc_qa.zero_point.cpu().numpy()

        save_dict.update(self.quantize_param(self._features_dc, self.features_dc_qa, 'features_dc'))

        # features_dc_q = torch.from_numpy(save_dict["features_dc"]).int().cuda()
        # features_dc_scale = torch.from_numpy(save_dict["features_dc_scale"]).cuda()
        # features_dc_zero_point = torch.from_numpy(
        #     save_dict["features_dc_zero_point"]
        # ).cuda()
        # features_dc = (features_dc_q - features_dc_zero_point) * features_dc_scale
        # self._features_dc = nn.Parameter(features_dc, requires_grad=True)
        #
        # self.features_dc_qa.scale = features_dc_scale
        # self.features_dc_qa.zero_point = features_dc_zero_point
        # self.features_dc_qa.activation_post_process.min_val = features_dc.min()
        # self.features_dc_qa.activation_post_process.max_val = features_dc.max()
        #

        self.features_dc_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self._features_dc = nn.Parameter(
            self.dequantize_param(save_dict, self.features_dc_qa, 'features_dc'),
            requires_grad=True
        )

        features_dc = self.features_dc_qa(self._features_dc)



        check_tensor(features_dc_ref - features_dc)





    def encode(self, path):
        # self.check_quant()
        # exit()
        # self.quantization = False
        self.save_npz(path, sort_morton=True)
        # pass

    def decode(self, path):
        self.load_npz(path, override_quantization=True)
        # self.quantization = False

        # pass