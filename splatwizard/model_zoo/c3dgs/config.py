from dataclasses import dataclass
from splatwizard.config import ModelParams, OptimizationParams


@dataclass
class C3DGSModelParams(ModelParams):
    sh_degree: int = 3
    not_quantization_aware = False


@dataclass
class C3DGSOptimizationParams(OptimizationParams):
    require_pretrained: bool= False
    position_lr_init: float = 0.0000016 # 0.00016
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
    # densify_from_iter = 500
    # densify_until_iter = 15_000
    # densify_grad_threshold = 0.0002
    # random_background = False
    #
    # gradient_aware_prune_start = 30_000
    # gradient_aware_prune_end = 35_000
    # gradient_aware_prune_interval = 500
    # pruning_level: float = 0.025
    iterations:int = 5000  # Used as finetune iteration

    # load_iteration = -1
    # finetune_iterations = 5000

    # not_quantization_aware = False
    color_codebook_size: int = 2 ** 12
    color_importance_include: float = 0.6 * 1e-6
    color_importance_prune: float = 0.0
    color_cluster_iterations: int = 100
    color_decay: float = 0.8
    color_batch_size: int = 2 ** 18
    color_weights_per_param: bool = False
    color_compress_non_dir: bool = True
    not_compress_color: bool = False

    gaussian_codebook_size: int = 2 ** 12
    gaussian_importance_include: float = 0.3 * 1e-5
    gaussian_cluster_iterations: int = 800
    gaussian_decay: float = 0.8
    gaussian_batch_size: int = 2 ** 20
    not_compress_gaussians: bool = False
    not_sort_morton: bool = False

    prune_threshold: float = 0.

    # output_vq = "./eval_vq"
    # start_checkpoint = ""


@dataclass
class CompressionSettings:
    codebook_size: int
    importance_prune: float
    importance_include: float
    steps: int
    decay: float
    batch_size: int