from dataclasses import dataclass
from splatwizard.config import ModelParams, OptimizationParams


@dataclass
class ControlGSModelParams(ModelParams):
    sh_degree: int = 3
    require_cam_infos: bool = True


@dataclass
class ControlGSOptimizationParams(OptimizationParams):
    require_pretrained = False
    iterations: int = 100_000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01

    exposure_lr_init: float = 0.001
    exposure_lr_final: float = 0.0001
    exposure_lr_delay_steps: float = 5000
    exposure_lr_delay_mult: float = 0.001

    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: int = 0.0002

    # ControlGS Hyperparameters:
    lambda_opacity: float = 3e-7  # Opacity regularization strength; ↑ = compact, ↓ = quality (range: [1e-7, 1e-6])
    max_densification: int = 6  # Max number of densification steps
    densification_batch_size: int = 100_000  # Batch size used for densification
    prune_change_threshold: int = 2_000  # Min Gaussian count change to trigger pruning
    opacity_threshold: float = 0.005  # Opacity threshold below which pruning occurs
    post_densification_filter_delay: int = 100  # Delay (iterations) before applying opacity filtering after densification

    use_trained_exposure: bool = True


