from dataclasses import dataclass
from splatwizard.config import ModelParams, OptimizationParams


@dataclass
class SurfelGSModelParams(ModelParams):
    sh_degree: int = 3
    depth_ratio: float = 0.0

@dataclass
class SurfelGSOptimizationParams(OptimizationParams):
    require_pretrained: bool = False
    iterations: int = 30_000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2

    # in original paper, use lambda_dist=1000 for DTU dataset
    lambda_dist: float = 0.0
    lambda_normal: float = 0.05
    opacity_cull: float = 0.05


    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002

