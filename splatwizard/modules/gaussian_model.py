import abc
import io
import pathlib

import torch
from plyfile import PlyElement, PlyData
from torch import nn
from loguru import logger
import numpy as np



from splatwizard.scheduler import Scheduler
from splatwizard.utils.system_utils import mkdir_p, search_for_max_iteration
from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard._cmod.simple_knn import distCUDA2
from splatwizard.utils.sh_utils import eval_sh, RGB2SH
from splatwizard.utils.general_utils import build_scaling_rotation, strip_symmetric, inverse_sigmoid
from splatwizard.utils.graphics_utils import BasicPointCloud
from splatwizard.modules.dataclass import RenderResult, LossPack, EvalPack, ModelContext
from splatwizard.utils.misc import wrap_str


class BaseGaussianModel:
    """
    Abstract Gaussian Splatting Base class
    """
    eval_pack_cls: type = EvalPack

    def __init__(self):
        self.optimizer = None
        self._training = False

        # self._called_at_switch_to_train = []
        self._train_hooks = []
        self._eval_hooks = []
        self._final_eval_hooks = []
        self._encode_hooks = []
        self._decode_hooks = []

    @abc.abstractmethod
    def render(self, viewpoint_camera, background, pipe, opt=None, step=None, scaling_modifier=1.0, override_color=None) -> RenderResult:
        raise NotImplementedError()

    def training_setup(self, *args, **kwargs):
        pass

    def register_pre_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: OptimizationParams):
        pass

    def register_post_task(self, scheduler: Scheduler, ppl: PipelineParams, opt: OptimizationParams):
        pass

    def after_setup_hook(self, ppl: PipelineParams, opt: OptimizationParams):
        pass

    def register_train_hook(self, func):
        self._train_hooks.append(func)

    def register_eval_hook(self, func):
        self._eval_hooks.append(func)

    def register_final_eval_hook(self, func):
        self._final_eval_hooks.append(func)

    def register_encode_hook(self, func):
        self._encode_hooks.append(func)

    def register_decode_hook(self, func):
        self._decode_hooks.append(func)

    @abc.abstractmethod
    def loss_func(self, *args, **kwargs) -> (torch.Tensor, LossPack):
        raise NotImplementedError()

    def train_report(self, loss_pack: LossPack, iteration, tb_writer):  # noqa
        if tb_writer is None:
            return
        tb_writer.add_scalar(f'train/l1_loss', loss_pack.l1_loss.item(), iteration)
        tb_writer.add_scalar(f'train/total_loss', loss_pack.loss.item(), iteration)
        tb_writer.add_scalar(f'train/train_elapsed_time', loss_pack.train_elapsed_time, iteration)
        tb_writer.add_scalar(f'train/task_elapsed_time', loss_pack.task_elapsed_time, iteration)
        tb_writer.add_scalar(f'train/peak_memory_allocated (MB)', loss_pack.peak_memory_allocated / (1024 ** 2), iteration)
        tb_writer.add_scalar(f'train/peak_memory_reserved (MB)', loss_pack.peak_memory_reserved / (1024 ** 2), iteration)

    def eval_report(self, eval_pack: EvalPack, iteration, tb_writer=None, ):   # noqa
        logger.info(wrap_str(
            f'[ITER {iteration}]', 'Evaluating test',
            'L1:', eval_pack.l1_val,
            'PSNR', eval_pack.psnr_val,
            'SSIM', eval_pack.ssim_val,
            'LPIPS', eval_pack.lpips_val,
            'FPS', 1 / eval_pack.frame_time

        ))

        if eval_pack.total_bytes != 0 :
            logger.info(wrap_str(
                f'total {eval_pack.total_bytes} bytes ({eval_pack.total_bytes/1024/1024} MB)',
                f'encode {eval_pack.encode_time} decode {eval_pack.decode_time}'
            ))
        if tb_writer is not None:
            tb_writer.add_scalar(f'eval/PSNR', eval_pack.psnr_val, iteration)
            tb_writer.add_scalar(f'eval/SSIM', eval_pack.ssim_val, iteration)

    def test_report(self, eval_pack: EvalPack, iteration, ):  # noqa

        logger.info(wrap_str(
            f'[ITER {iteration}]', 'Evaluating test',
            'L1:', eval_pack.l1_val,
            'PSNR', eval_pack.psnr_val,
            'SSIM', eval_pack.ssim_val,
            'LPIPS', eval_pack.lpips_val,
            'Visible Gaus', eval_pack.avg_gaussians,
            'Total Gaus', eval_pack.total_gaussian,
            'FPS', 1 / eval_pack.frame_time

        ))
        result_dict = {
            'L1': eval_pack.l1_val,
            'PSNR': eval_pack.psnr_val,
            'SSIM': eval_pack.ssim_val,
            'LPIPS':  eval_pack.lpips_val,
            'Visible_Gaus' : eval_pack.avg_gaussians,
            'total_gaus': eval_pack.total_gaussian,
            'frame_time': eval_pack.frame_time,
            'FPS':  1 / eval_pack.frame_time,
        }


        if eval_pack.total_bytes != 0:
            logger.info(wrap_str(
                f'total {eval_pack.total_bytes} bytes ({eval_pack.total_bytes / 1024 / 1024} MB)',
                f'encode {eval_pack.encode_time} decode {eval_pack.decode_time}'
            ))

            result_dict.update({
                'total_bytes': eval_pack.total_bytes,
                'encode_time': eval_pack.encode_time,
                'decode_time': eval_pack.decode_time,
            })


        logger.info(wrap_str(
            'peak memory allocated', eval_pack.peak_memory_allocated / (1024 ** 2), 'MB,',
            'peak memory reserved', eval_pack.peak_memory_reserved / (1024 ** 2), 'MB'
        ))
        result_dict.update({
            'peak_memory_allocated_bytes': eval_pack.peak_memory_allocated,
            'peak_memory_reserved_bytes': eval_pack.peak_memory_reserved,
        })
        return result_dict




    def capture(self):
        pass

    def restore(self, *args, **kwargs):
        pass

    def save(self, path, iteration, type_='checkpoint'):
        pass

    def load(self, path):
        pass

    def post_eval(self, eval_pack):  # noqa
        return eval_pack

    @abc.abstractmethod
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        raise NotImplementedError()

    def encode(self, path: io.BufferedWriter):
        pass

    def decode(self, path: io.BufferedReader):
        pass

    def train(self):
        self._training = True
        for k, v in self.__dict__.items():
            if isinstance(v, nn.Module):
                v.train()
        # for func in self._called_at_switch_to_train:
        #     func()
        for func in self._train_hooks:
            func()

    def eval(self):
        self._training = False
        for k, v in self.__dict__.items():
            if isinstance(v, nn.Module):
                v.eval()
        for func in self._eval_hooks:
            func()

    def final_eval(self):
        for func in self._final_eval_hooks:
            logger.info(f'executing final eval hook {func.__name__}')
            func()

    def set_state(self, *args, **kwargs):
        pass

    def optimizer_step(self, render_result: RenderResult, opt: OptimizationParams, step: int):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)


class GaussianModel(BaseGaussianModel, abc.ABC):
    def __init__(self):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = 0

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._covariance = torch.empty(0)

        self._context: ModelContext = ModelContext()

        self.xyz_gradient_accum = None
        self.denom = None
        self.max_radii2D = None

        self.spatial_lr_scale = 0

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = None
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.relu = torch.relu
        self.log = torch.log



    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.covariance_activation = build_covariance_from_scaling_rotation

    @property
    def xyz(self):
        return self._xyz

    @property
    def scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def features_dc(self):
        return self._features_dc

    @property
    def features_rest(self):
        return self._features_rest

    @property
    def context(self):
        return self._context

    @property
    def features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.scaling, scaling_modifier, self._rotation)

    def training_setup(self, *args, **kwargs):
        pass

    def register_post_task(self, scheduler, ppl: PipelineParams, opt: OptimizationParams):
        pass

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict() if self.optimizer is not None else None,
            self.spatial_lr_scale,
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
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            context
        ) = model_args
        if training_args is not None:
            self.training_setup(training_args)

        # Since training_setup will reset these parameters, we assign values to them manually
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if self.optimizer is not None:
            self.optimizer.load_state_dict(opt_dict)
        self._context = context # type(self._context)(**context)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, cam_infos=None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        logger.info(wrap_str("Number of points at initialisation : ", fused_point_cloud.shape[0]))

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device="cuda")

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def load_ply(self, path, data=None):
        if data is None:
            plydata = PlyData.read(path)
        else:
            plydata = data

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def save_ply(self, path):
        # mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save(self, checkpoint_dir, iteration, type_='pth'):
        if type_ == 'pth':
            logger.info("[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((self.capture(), iteration), checkpoint_dir / f"ckpt{iteration}.pth")
        elif type_ == 'ply':
            logger.info("[ITER {}] Saving PLY file".format(iteration))
            # checkpoint_dir.parent point to output_dir,
            point_cloud_path = checkpoint_dir.parent / "point_cloud/iteration_{}".format(iteration)
            point_cloud_path.mkdir(parents=True, exist_ok=True)
            # mkdir_p(os.path.dirname(point_cloud_path))
            self.save_ply(point_cloud_path / "point_cloud.ply")
        else:
            raise NotImplementedError(f'Unsupported checkpoint type: {type_}')

    def load(self, path, opt=None):
        path = pathlib.Path(path)
        need_setup = True
        if path.is_dir():
            point_cloud_dir = path / 'point_cloud'
            max_iter = search_for_max_iteration(point_cloud_dir)
            point_cloud_dir = point_cloud_dir / f"iteration_{max_iter}" / "point_cloud.ply"
            first_iter = max_iter
            self.load_ply(point_cloud_dir)

        elif path.suffix == '.ply':
            self.load_ply(path)
            first_iter = None
        else:
            (model_params, first_iter) = torch.load(path, weights_only=False)
            self.restore(model_params, opt)
            need_setup = False

        return first_iter, need_setup

    def encode(self, path: io.BufferedWriter):
        self.save_ply(path)

    def decode(self, path: io.BufferedReader):
        self.load_ply(path)



