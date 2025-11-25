from dataclasses import dataclass
from splatwizard.config import ModelParams, OptimizationParams


@dataclass
class GSDRAAModelParams(ModelParams):
    sh_degree: int = 3


@dataclass
class GSDRAAOptimizationParams(OptimizationParams):
    require_pretrained = False
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30_000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    percent_dense = 0.01

    densification_interval = 100
    opacity_reset_interval = 3000
    densify_from_iter = 500
    densify_until_iter = 15_000
    densify_grad_threshold = 0.0002
    random_background = False

