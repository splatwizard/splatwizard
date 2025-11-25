from dataclasses import dataclass
from splatwizard.config import ModelParams, OptimizationParams


@dataclass
class Trim3DGSModelParams(ModelParams):
    sh_degree: int = 3


@dataclass
class Trim3DGSOptimizationParams(OptimizationParams):
    require_pretrained: bool = False
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01

    densification_interval: float = 100
    opacity_reset_interval: float = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    random_background: bool = False


@dataclass
class Trim3DGSFinetuneOptimizationParams(OptimizationParams):
    require_pretrained: bool = False
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01

    densification_interval: float = 100
    opacity_reset_interval: float = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    random_background: bool = False

