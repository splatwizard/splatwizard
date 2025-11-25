from dataclasses import dataclass, field
from enum import Enum
from typing import List

from splatwizard.config import ModelParams, OptimizationParams


class Stage(Enum):
    TRAIN = 1
    PRUNE = 2


@dataclass
class PUP3DGSModelParams(ModelParams):
    sh_degree: int = 3


@dataclass
class PUP3DGSOptimizationParams(OptimizationParams):
    require_pretrained: bool = False
    position_lr_init: float = 0.0000016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01

    # densification_interval: int = 100
    # opacity_reset_interval: int = 3000
    # densify_from_iter: int = 500
    # densify_until_iter: int = 15_000
    # densify_grad_threshold: int = 0.0002
    # random_background: int = False

    iterations: int = 10_000
    prune_iterations: List[int] = field(default_factory=lambda: [1, 5_001])

    prune_percent: List[float] = field(default_factory=lambda: [0.8, 0.5])
    prune_type: str = 'fisher'
    v_pow: float = 0.1
    fisher_resolution: int = 4

    current_stage: Stage = Stage.PRUNE


