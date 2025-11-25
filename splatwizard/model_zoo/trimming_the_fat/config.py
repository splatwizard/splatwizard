from dataclasses import dataclass
from enum import Enum
from typing import List

from splatwizard.config import ModelParams, OptimizationParams


class Stage(Enum):
    TRAIN = 1
    PRUNE = 2

@dataclass
class TTFModelParams(ModelParams):
    sh_degree: int = 3
    opacity_qbit: int = 24
    scale_qbit: int = 24
    rotation_qbit: int = 24
    xyz_qbit: int = 24


@dataclass
class TTFOptimizationParams(OptimizationParams):
    require_pretrained: bool = True
    iterations: int = 15_000
    position_lr_init: float = 0.0000016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: float = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    # opacity_reset_interval: int = 3000
    # densify_from_iter: int = 0
    # densify_until_iter: int = 5_000
    # densify_grad_threshold: int = 0.0002
    # random_background: bool = False

    # for post-process method, reset iter when load a pre-trained model
    gradient_aware_prune_start = 100
    gradient_aware_prune_end = 5_000
    gradient_aware_prune_interval = 500
    pruning_level: float = 0.425

    # stages: List[Stage] = None
    current_stage: Stage = Stage.PRUNE
