import abc
import json
from dataclasses import dataclass
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
from splatwizard.rasterizer.compress import CompressGaussianRasterizationSettings, CompressGaussianRasterizer
from splatwizard.modules.dataclass import RenderResult


try:
    _Base: typing.TypeAlias = GaussianModel
except AttributeError:
    _Base = object


@dataclass
class CompressRenderResult(RenderResult):
    gaussians_count: typing.Any = None
    important_score: typing.Any = None


class CompressRenderMixin(_Base):
    """
    Reimplemented render function for "LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS"
    """
    def render(self,
               viewpoint_camera,
               bg_color: torch.Tensor,
               pipe: PipelineParams,
               opt: OptimizationParams=None,
               step=0,
               scaling_modifier=1.0,
               override_color=None):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means

        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
                torch.zeros_like(
                    self.xyz, dtype=self.xyz.dtype, requires_grad=True, device="cuda"
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

        raster_settings = CompressGaussianRasterizationSettings(
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
            f_count=False,
        )

        rasterizer = CompressGaussianRasterizer(raster_settings=raster_settings)

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
                shs_view = self.features.transpose(1, 2).view(
                    -1, 3, (self.max_sh_degree + 1) ** 2
                )
                dir_pp = self.xyz - viewpoint_camera.camera_center.repeat(
                    self.features.shape[0], 1
                )
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
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        # return {
        #     "render": rendered_image,
        #     "viewspace_points": screenspace_points,
        #     "visibility_filter": radii > 0,
        #     "radii": radii,
        # }

        return CompressRenderResult(
            rendered_image=rendered_image,
            viewspace_points=screenspace_points,
            visibility_filter=radii > 0,
            radii=radii
        )

    def count_render(self,
            viewpoint_camera,
            pipe,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            override_color=None,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
                torch.zeros_like(
                    self.xyz, dtype=self.xyz.dtype, requires_grad=True, device="cuda"
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

        raster_settings = CompressGaussianRasterizationSettings(
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
            f_count=True,
        )

        rasterizer = CompressGaussianRasterizer(raster_settings=raster_settings)
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
                shs_view = self.features.transpose(1, 2).view(
                    -1, 3, (self.max_sh_degree + 1) ** 2
                )
                dir_pp = self.xyz - viewpoint_camera.camera_center.repeat(
                    self.features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        gaussians_count, important_score, rendered_image, radii = rasterizer(
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
        #     "gaussians_count": gaussians_count,
        #     "important_score": important_score,
        # }

        return CompressRenderResult(
            rendered_image=rendered_image,
            viewspace_points=screenspace_points,
            visibility_filter=radii > 0,
            radii=radii,
            gaussians_count=gaussians_count,
            important_score=important_score,
        )