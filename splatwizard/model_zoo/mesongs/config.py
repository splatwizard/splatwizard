from dataclasses import dataclass, field
from enum import Enum
from typing import List

from splatwizard.config import ModelParams, OptimizationParams, PipelineParams


class Stage(Enum):
    TRAIN = 1
    PRUNE = 2
    DISTILL = 3
    ENCODE = 4


@dataclass
class MesonGSModelParams(ModelParams):
    sh_degree: int = 3
    save_imp: bool = False
    depth_count: bool = False
    save_mode: str = 'euler'
    not_update_rot: bool = False
    skip_quant_rot: bool = False
    hyper_config: str = "universal"
    save_ft_type: str = ""
    n_block: int = 66
    eval: bool = False
    codeft: bool = False
    no_simulate: bool = False
    oct_merge: str = "mean"
    codebook_size: int = 2048
    batch_size: int = 262144
    steps: int = 2000
    raht: bool = True
    percent: float = 0.66
    meson_count: bool = True
    f_count: bool = False
    debug: bool = False
    lseg: int = -1
    csv_path: str = ''
    depth: int = 12
    num_bits: int = 8
    clamp_color: bool = True
    per_channel_quant: bool = False
    per_block_quant: bool = True
    use_indexed: bool = True
    scene_imp: str = "" # scene_name
    yaml_path: str = "" # REQUIRE, config path
    finetune_lr_scale: float = 1.0


@dataclass
class MesonGSOptimizationParams(OptimizationParams):
    require_pretrained = False
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 10_000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.001
    rotation_lr = 0.001
    percent_dense = 0.01
    
    lambda_dssim = 0.2
    densification_interval = 100
    opacity_reset_interval = 3000
    densify_from_iter = 500
    densify_until_iter = 15_000
    densify_grad_threshold = 0.0002

    iterations: int = 3_000

    # current_stage: Stage = None
