from dataclasses import dataclass
from typing import List

from splatwizard.config import ModelParams, OptimizationParams


@dataclass
class CompactGSModelParams(ModelParams):
    sh_degree: int = 3
    rvq_size: int = 64
    rvq_num: int = 6
    prune_recolor: bool = True
    prune_quant_opa: bool = False
    prune_quant_hash: bool = False
    max_hashmap: int = 16


@dataclass
class CompactGSOptimizationParams(OptimizationParams):
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
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    mask_prune_iter: int = 1_000
    rvq_iter: int = 29_000
    mask_lr: float = 1e-3
    net_lr: float = 1e-3
    net_lr_step: List[int] = (25_000, )
    lambda_mask: float = 4e-3
    