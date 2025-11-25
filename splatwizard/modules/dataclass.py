from dataclasses import dataclass
from pathlib import Path

import torch
import typing


@dataclass
class RenderResult:
    rendered_image: torch.Tensor = None
    viewspace_points: torch.Tensor = None
    visible_mask: torch.Tensor = None
    visibility_filter: torch.Tensor = None
    radii: torch.Tensor = None
    active_gaussians: int = 0
    num_rendered: int = 0
    selection_mask: typing.Union[torch.Tensor, None] = None
    scaling: typing.Union[torch.Tensor, None] = None
    depth: typing.Union[torch.Tensor, None] = None
    # other_params: typing.Union[typing.Dict, None] = None


@dataclass
class LossPack:
    l1_loss: torch.Tensor = None
    l2_loss: torch.Tensor = None
    ssim_loss: torch.Tensor = None
    loss: torch.Tensor = None
    total_elapsed_time: float = 0
    task_elapsed_time: float = 0
    train_elapsed_time: float = 0
    peak_memory_allocated: int = 0
    peak_memory_reserved: int = 0


@dataclass
class EvalPack:
    l1_val: float = 0
    l2_val: float = 0
    psnr_val: float = 0
    ssim_val: float = 0
    ms_ssim_val: float = 0
    lpips_val: float = 0
    frame_time: float = 0
    rendered_images: typing.List[torch.Tensor] = None
    encode_time: float = 0
    decode_time: float = 0
    peak_memory_allocated: int = 0
    peak_memory_reserved: int = 0
    total_bytes: int = 0
    avg_gaussians: int = 0
    total_gaussian: int = 0


@dataclass
class TrainContext:
    model: str = None
    base_output_dir: Path = None
    output_dir: Path = None
    render_result_dir = None
    checkpoint_dir: Path = None
    tb_writer: typing.Any = None


@dataclass
class ModelContext:
    pass





