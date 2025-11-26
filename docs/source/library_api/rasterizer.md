# Rasterizer

Splatwizard integrates multiple rasterizers from prior work, which can be directly imported and used. Each rasterizer closely follows the papi design in the original codebase to minimize migration and application efforts.

## Standard Gaussian rasterizer

***class* splatwizard.rasterizer.gaussian.<font color="#e83e8c">GaussianRasterizer</font>(raster_settings: GaussianRasterizationSettings)**

Standard Gaussian Splatting rasterizer from [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

The raster_setting class is defined as 

```python
class splatwizard.rasterizer.gaussian.GaussianRasterizationSettings(
    image_height: int,
    image_width: int,
    tanfovx: float,
    tanfovy: float,
    bg: torch.Tensor,
    scale_modifier: float,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    sh_degree: int,
    campos: torch.Tensor,
    prefiltered: bool,
    debug: bool
)
```

Rasterizer function is defined as 
```python
class GaussianRasterizer(nn.Module):
    ...
    def rasterizer(self,
        means3D: Tensor, 
        means2D: Tensor, 
        opacities: Tensor, 
        shs: Tensor | None, 
        colors_precomp: Tensor | None, 
        scales: Tensor | None, 
        rotations: Tensor| None, 
        cov3D_precomp: Tensor | None
    ):
        ...
        return rendered_image, radii

```

## Standard Gaussian rasterizer with depth

***class* splatwizard.rasterizer.gs_dr_aa.<font color="#e83e8c">GSDRAAGaussianRasterizer</font>(raster_settings: GSDRAARasterizationSettings)**

Standard Gaussian Splatting rasterizer with inv_depth and anti-aliasing.


The raster_setting class is defined as 

```python
class splatwizard.rasterizer.gs_dr_aa.GSDRAARasterizationSettings(
    image_height: int
    image_width: int 
    tanfovx: float
    tanfovy: float
    bg: Tensor
    scale_modifier: float
    viewmatrix: Tensor
    projmatrix: Tensor
    sh_degree: int
    campos: Tensor
    prefiltered: bool
    debug: bool
    antialiasing: bool
)
```

Rasterizer function is defined as 
```python
class GSDRAAGaussianRasterizer(nn.Module):
    ...
    def rasterizer(self,
        means3D: Tensor, 
        means2D: Tensor, 
        opacities: Tensor, 
        shs: Tensor | None, 
        colors_precomp: Tensor | None, 
        scales: Tensor | None, 
        rotations: Tensor | None, 
        cov3D_precomp: Tensor | None
    ):
        ...
        return rendered_image, radii, invdepths

```



## Accelerated Gaussian rasterizer



***class* splatwizard.rasterizer.accel_gs.<font color="#e83e8c">AccelGaussianRasterizer</font>(raster_settings: AccelGaussianRasterizationSettings)**

Gaussian Splatting rasterizer from [Taming 3DGS: High-Quality Radiance Fields with Limited Resources](https://humansensinglab.github.io/taming-3dgs)

The raster_setting class is defined as 

```python
class splatwizard.rasterizer.accel_gs.AccelGaussianRasterizationSettings(
    image_height: int,
    image_width: int,
    tanfovx: float,
    tanfovy: float,
    bg: torch.Tensor,
    scale_modifier: float,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    sh_degree: int,
    campos: torch.Tensor,
    prefiltered: bool,
    debug: bool
)
```

Rasterizer function is defined as 
```python
class AccelGaussianRasterizer(nn.Module):
    ...
    def rasterizer(self,
        means3D: Tensor, 
        means2D: Tensor, 
        opacities: Tensor, 
        shs: Tensor | None, 
        colors_precomp: Tensor | None, 
        scales: Tensor | None, 
        rotations: Tensor| None, 
        cov3D_precomp: Tensor | None
    ):
        ...
        return rendered_image, radii

```


## Speedy-Splat rasterizer



***class* splatwizard.rasterizer.speedy.<font color="#e83e8c">SpeedyGaussianRasterizer</font>(raster_settings: SpeedyGaussianRasterizationSettings)**

Gaussian Splatting rasterizer from [Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives](https://speedysplat.github.io/)

The raster_setting class is defined as 

```python
class splatwizard.rasterizer.speedy.SpeedyGaussianRasterizationSettings(
    image_height: int,
    image_width: int,
    tanfovx: float,
    tanfovy: float,
    bg: torch.Tensor,
    scale_modifier: float,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    sh_degree: int,
    campos: torch.Tensor,
    prefiltered: bool,
    debug: bool
)
```

Rasterizer function is defined as 
```python
class SpeedyGaussianRasterizer(nn.Module):
    ...
    def rasterizer(self,
        means3D: Tensor, 
        means2D: Tensor, 
        opacities: Tensor,
        scores: Tensor,
        shs: Tensor | None, 
        colors_precomp: Tensor | None, 
        scales: Tensor | None, 
        rotations: Tensor| None, 
        cov3D_precomp: Tensor | None
    ):
        ...
        return rendered_image, radii

```

## Speedy-Splat Tensor Core rasterizer



***class* splatwizard.rasterizer.speedy_tcgs.<font color="#e83e8c">SpeedyTCGaussianRasterizer</font>(raster_settings: SpeedyTCGaussianRasterizationSettings)**

Gaussian Splatting rasterizer from [TC-GS: A Faster Gaussian Splatting Module Utilizing Tensor Cores](https://arxiv.org/abs/2505.24796)

The raster_setting class is defined as 

```python
class splatwizard.rasterizer.speedy_tcgs.SpeedyTCGaussianRasterizationSettings(
    image_height: int,
    image_width: int,
    tanfovx: float,
    tanfovy: float,
    bg: torch.Tensor,
    scale_modifier: float,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    sh_degree: int,
    campos: torch.Tensor,
    prefiltered: bool,
    debug: bool
)
```

Rasterizer function is defined as 
```python
class SpeedyTCGaussianRasterizer(nn.Module):
    ...
    def rasterizer(self,
        means3D: Tensor, 
        means2D: Tensor, 
        opacities: Tensor,
        scores: Tensor,
        shs: Tensor | None, 
        colors_precomp: Tensor | None, 
        scales: Tensor | None, 
        rotations: Tensor| None, 
        cov3D_precomp: Tensor | None
    ):
        ...
        return rendered_image, radii

```


## FlashGS

***class* splatwizard.rasterizer.flashgs.<font color="#e83e8c">FlashGSRasterizer</font>(num_vertex, device, MAX_NUM_RENDERED, MAX_NUM_TILES)**

Gaussian Splatting rasterizer from [FlashGS: Efficient 3D Gaussian Splatting for Large-scale and High-resolution Rendering](https://github.com/InternLandMark/FlashGS)

**Note**: For FlashGSRasterizer, we recommand to directly use `splatwizard.modules.render_mixin.FlashGSRenderMixin`, which includes both `forward` and `backward` method (using standard GS rasterizer in stages where FlashGS is not applicable).

Rasterizer function is defined as 
```python
class FlashGSRasterizer(nn.Module):
    ...
    def forward(self, 
        position: Tensor, 
        shs: Tensor, 
        opacity: Tensor, 
        cov3d: Tensor, 
        camera, 
        bg_color: Tensor
    ):
        ...
        return out_color
```

## 2D Gaussian Splatting rasterizer

***class* splatwizard.rasterizer.surfel_gs.<font color="#e83e8c">SurfelGaussianRasterizer</font>(raster_settings: SurfelGaussianRasterizationSettings)**

Gaussian Splatting rasterizer from [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surfsplatting.github.io/)

The raster_setting class is defined as 

```python
class splatwizard.rasterizer.surfel_gs.SurfelGaussianRasterizer(
    image_height: int,
    image_width: int,
    tanfovx: float,
    tanfovy: float,
    bg: torch.Tensor,
    scale_modifier: float,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    sh_degree: int,
    campos: torch.Tensor,
    prefiltered: bool,
    debug: bool
)
```

Rasterizer function is defined as 
```python
class SurfelGaussianRasterizer(nn.Module):
    ...
    def rasterizer(self,
        means3D: Tensor, 
        means2D: Tensor, 
        opacities: Tensor,
        scores: Tensor,
        shs: Tensor | None, 
        colors_precomp: Tensor | None, 
        scales: Tensor | None, 
        rotations: Tensor| None, 
        cov3D_precomp: Tensor | None
    ):
        ...
        return rendered_image, radii, allmap

```

For detailed usage of `allmap`, please refer to the implemetation of 2DGS in `splatwizard/model_zoo/surfel_gs/model.py`
