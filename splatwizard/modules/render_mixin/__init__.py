import abc
import json
from functools import lru_cache
from typing import Union

import math
# import os
# import time
# from functools import reduce

# import numpy as np
import torch
import typing


from splatwizard.config import PipelineParams, OptimizationParams
from splatwizard.modules.gaussian_model import BaseGaussianModel, GaussianModel
from splatwizard.scene import camera_to_JSON
from splatwizard.utils.sh_utils import eval_sh
from splatwizard.rasterizer.gaussian import GaussianRasterizationSettings, GaussianRasterizer
from splatwizard.rasterizer.accel_gs import AccelGaussianRasterizationSettings
from splatwizard.rasterizer.accel_gs import AccelGaussianRasterizer
from splatwizard.rasterizer.flashgs import FlashGSRasterizer
from splatwizard.modules.dataclass import RenderResult
from loguru import logger


try:
    _Base: typing.TypeAlias = GaussianModel
except AttributeError:
    _Base = object

# _Base = object


class BaseRenderMixin(_Base):
    def render(self, viewpoint_camera, bg_color: torch.Tensor, pipe, opt=None, step=0, scaling_modifier=1.0,
               override_color=None):
        ...


class RenderMixin(_Base):
    """doc"""
    def render(self, viewpoint_camera, bg_color: torch.Tensor,
               pipe: PipelineParams, opt: OptimizationParams=None, step=0, scaling_modifier=1.0, override_color=None):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(
            self.xyz, dtype=self.xyz.dtype,
            requires_grad=True, device="cuda") + 0
        if self._training:
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
            cov3D_precomp=cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        # return {"render": rendered_image,
        #         "viewspace_points": screenspace_points,
        #         "visibility_filter": radii > 0,
        #         "radii": radii}
        if opt is not None and opt.use_trained_exposure:
            exposure = self.get_exposure_from_name(viewpoint_camera.image_name)
            rendered_image = torch.matmul(
                rendered_image.permute(1, 2, 0),
                exposure[:3, :3]
            ).permute(2, 0, 1) + exposure[:3, 3, None, None]

        return RenderResult(
            rendered_image=rendered_image,
            viewspace_points=screenspace_points,
            visibility_filter=radii > 0,
            radii=radii
        )


class AccelRenderMixin(_Base):
    """doc"""
    def render(self, viewpoint_camera, bg_color: torch.Tensor, pipe, opt=None, step=0, scaling_modifier=1.0,
               override_color=None):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """
        pixel_weights = None
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self.xyz, dtype=self.xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = AccelGaussianRasterizationSettings(
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
            debug=pipe.debug,
            # pixel_weights=pixel_weights
        )

        rasterizer = AccelGaussianRasterizer(raster_settings=raster_settings)

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
                shs_view = self.features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
                dir_pp = (self.xyz - viewpoint_camera.camera_center.repeat(self.features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                dc, shs = self.features_dc, self.features_rest
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
                means3D=means3D,
                means2D=means2D,
                dc=dc,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp)


        return RenderResult(
            rendered_image=rendered_image,
            viewspace_points=screenspace_points,
            visibility_filter=radii > 0,
            radii=radii
        )


class FlashGSRenderMixin(RenderMixin):

    def __init__(self):  # noqa
        MAX_NUM_RENDERED = 2 ** 26
        MAX_NUM_TILES = 2 ** 20

        self.rasterizer = FlashGSRasterizer(1, 'cuda', MAX_NUM_RENDERED, MAX_NUM_TILES)
        self.register_eval_hook(lambda: self.cached_get_covariance.cache_clear())

    @lru_cache(1)
    def cached_get_covariance(self, scaling_modifier):
        return self.get_covariance(scaling_modifier)

    def render(self, viewpoint_camera, bg_color: torch.Tensor, pipe: PipelineParams, opt=None, step=0, scaling_modifier=1.0, override_color=None):
        if self._training:
            return RenderMixin.render(self,
                viewpoint_camera,
                bg_color, pipe, opt, step, scaling_modifier, override_color
            )

        # ref_result =  RenderMixin.render(self,
        #         viewpoint_camera,
        #         bg_color, pipe, opt, step, scaling_modifier, override_color
        # )



        # class Camera:
        #     def __init__(self, camera_json):
        #         self.id = camera_json['id']
        #         self.img_name = camera_json['img_name']
        #         self.width = camera_json['width']
        #         self.height = camera_json['height']
        #         self.position = torch.tensor(camera_json['position'])
        #         self.rotation = torch.tensor(camera_json['rotation'])
        #         self.focal_x = camera_json['fx']
        #         self.focal_y = camera_json['fy']
        #         self.zFar = 100.0
        #         self.zNear = 0.01
        #
        #         # print(camera_json)
        #
        # camera = Camera(camera_to_JSON(0, viewpoint_camera.info))
        # # scene_path = '/dev/shm/test.ply'
        # # self.save_ply(scene_path)
        # camera.width = viewpoint_camera.image_width
        # camera.height = viewpoint_camera.image_height

        # camera_path = "/data1/lizhuo/gaussian-splatting/bicycle_model/cameras.json"
        # print(camera_path)
        # device = torch.device('cuda:0')
        # assert not pipe.white_background
        # bg_color = torch.zeros(3, dtype=torch.float32)  # black, note this must on cpu!
        bg_color = bg_color.cpu()
        #
        #
        # with open(camera_path, 'r') as camera_file:
        #     cameras_json = json.loads(camera_file.read())
        #
        # ref_cam = Camera(cameras_json[0])

        # MAX_NUM_RENDERED = 2 ** 26
        # MAX_NUM_TILES = 2 ** 20
        # scene_path = "/data1/lizhuo/gaussian-splatting/bicycle_model/point_cloud/iteration_30000/point_cloud.ply"
        # f_num_vertex, f_position, f_shs, f_opacity, f_cov3d = flash_gaussian_splatting.ops.loadPly(
        #     scene_path)
        #
        # self.load_ply(scene_path)
        # rasterizer = FlashGSRasterizer(self.xyz.shape[0], self.xyz.device, MAX_NUM_RENDERED, MAX_NUM_TILES)
        # rasterizer = FlashGSRasterizer(f_num_vertex, self.xyz.device, MAX_NUM_RENDERED, MAX_NUM_TILES)

        position = self.xyz
        shs = self.features.reshape(-1, 48)
        opacity = self.opacity.squeeze(1)
        cov3d_precomp = self.cached_get_covariance(scaling_modifier)

        # print((f_position.cuda() - position).abs().max())
        # print((f_shs.cuda() - shs).abs().max())
        # print((f_opacity.cuda() - opacity).abs().max())
        # print((f_cov3d.cuda() - cov3d_precomp).abs().max())


        # rendered_image = rasterizer.forward(f_position.cuda(), f_shs.cuda(), f_opacity.cuda(), f_cov3d.cuda(), viewpoint_camera, bg_color)
        rendered_image = self.rasterizer.forward(position, shs, opacity, cov3d_precomp, viewpoint_camera, bg_color)
        rendered_image = rendered_image.permute(2, 0, 1).float() / 255
        # plt.imshow(rendered_image1.cpu().numpy())
        # plt.show()
        #
        # plt.imshow(rendered_image2.cpu().numpy())
        # plt.show()
        # print((rendered_image1.float() - rendered_image2.float()).abs().max())
        # exit()

        # print((ref_result.rendered_image.float() - rendered_image).abs().max())
        return RenderResult(
            rendered_image=rendered_image,

        )



# hjx debug
# from torchvision.utils import save_image


# hjx debug end
# class CompactGSRenderMixin(RenderMixin):
