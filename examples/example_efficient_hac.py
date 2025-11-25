import sys
import math

import torch

from splatwizard.main import main
from splatwizard.model_zoo import register_model
from splatwizard.model_zoo.gs.model import GSModel, GSModelParams, GSOptimizationParams
from splatwizard.model_zoo.hac.config import HACModelParams, HACOptimizationParams
from splatwizard.model_zoo.hac.model import HAC, HACRenderResult
from splatwizard.modules.render_mixin import FlashGSRenderMixin

from splatwizard.rasterizer.accel_gs import AccelGaussianRasterizationSettings, AccelGaussianRasterizer

class EfficientHAC(HAC):
    def __init__(self, model_params):
        super().__init__(model_params)

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

        raster_settings = AccelGaussianRasterizationSettings(
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

        rasterizer = AccelGaussianRasterizer(raster_settings=raster_settings)
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

        raster_settings = AccelGaussianRasterizationSettings(
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

        rasterizer = AccelGaussianRasterizer(raster_settings=raster_settings)


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


if __name__ == "__main__":
    register_model('ehac', HACModelParams, HACOptimizationParams, EfficientHAC)
    sys.exit(main())
