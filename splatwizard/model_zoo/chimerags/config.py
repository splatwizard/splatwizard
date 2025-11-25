from dataclasses import dataclass, field
from enum import Enum
from typing import List

from splatwizard.config import ModelParams, OptimizationParams


class Stage(Enum):
    TRAIN = 1
    PRUNE = 2
    DISTILL = 3
    ENCODE = 4


@dataclass
class ChimeraGSModelParams(ModelParams):
    sh_degree: int = 3
    render_type: str = 'flash'
    sh2_vq_ratio: float = 0.8
    sh3_vq_ratio: float = 0.6


@dataclass
class ChimeraGSOptimizationParams(OptimizationParams):
    require_pretrained = False
    position_lr_init = 0.0000016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30_000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    percent_dense = 0.01

    current_stage: Stage = None


@dataclass
class ChimeraGSPruneOptimizationParams(ChimeraGSOptimizationParams):
    require_pretrained = False
    position_lr_init = 0.0000016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30_000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001


    densification_interval = 100
    opacity_reset_interval = 3000
    densify_from_iter = 500
    densify_until_iter = 15_000
    densify_grad_threshold = 0.0002
    random_background = False

    iterations: int = 5_000
    prune_iterations: List[int] = field(default_factory=lambda: [1])

    prune_percent: float = 0.66
    prune_decay: float = 1
    prune_type: str = 'v_important_score'
    v_pow: float = 0.1
    densify_iteration: List[int] = field(default_factory=lambda: [-1])

    current_stage: Stage = Stage.PRUNE


@dataclass
class ChimeraGSDistillOptimizationParams(ChimeraGSOptimizationParams):
    require_pretrained = True
    position_lr_init = 0.0000016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30_000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    percent_dense = 0.01

    iterations: int = 5_000

    v_pow: float = 0.1

    new_max_sh_degree: int = 2

    teacher_checkpoint: str = None
    lmbda_mask: float = 1e-2

    augmented_view: bool = True
    current_stage: Stage = Stage.DISTILL


@dataclass
class ChimeraGSEncodeOptimizationParams(ChimeraGSOptimizationParams):
    new_max_sh_degree: int = 2
    current_stage: Stage = Stage.ENCODE

