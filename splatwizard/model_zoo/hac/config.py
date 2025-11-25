from dataclasses import dataclass
from typing import Tuple

from splatwizard.config import OptimizationParams, ModelParams


@dataclass
class HACModelParams(ModelParams):
    # 基本参数
    sh_degree: int = 3
    feat_dim: int = 50
    n_offsets: int = 10
    voxel_size: float = 0.001  # if voxel_size<=0, using 1nn dist
    update_depth: int = 3
    update_init_factor: int = 16
    update_hierachy_factor: int = 4

    n_features_per_level: int = 4
    log2_hashmap_size: int = 13
    log2_hashmap_size_2D: int = 15

    ste_binary: bool = True
    ste_multistep: bool = False
    add_noise: bool = False
    Q: float = 1
    use_2D: bool = True

    # 特征和路径参数
    use_feat_bank: bool = False

    resolutions_list: Tuple[int] = (18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514)
    resolutions_list_2D: Tuple[int] = (130, 258, 514, 1026)

    decoded_version: bool = False



@dataclass
class HACOptimizationParams(OptimizationParams):
    # total iterations
    iterations: int = 30_000

    # position learning rate
    position_lr_init: float = 0.0
    position_lr_final: float = 0.0
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000

    # offset learning rate
    offset_lr_init: float = 0.01
    offset_lr_final: float = 0.0001
    offset_lr_delay_mult: float = 0.01
    offset_lr_max_steps: int = 30_000

    # mask learning rate
    mask_lr_init: float = 0.01
    mask_lr_final: float = 0.0001
    mask_lr_delay_mult: float = 0.01
    mask_lr_max_steps: int = 30_000

    # 基本学习率参数
    feature_lr: float = 0.0075
    opacity_lr: float = 0.02
    scaling_lr: float = 0.007
    rotation_lr: float = 0.002

    # MLP不透明度参数
    mlp_opacity_lr_init: float = 0.002
    mlp_opacity_lr_final: float = 0.00002
    mlp_opacity_lr_delay_mult: float = 0.01
    mlp_opacity_lr_max_steps: int = 30_000

    # MLP协方差参数
    mlp_cov_lr_init: float = 0.004
    mlp_cov_lr_final: float = 0.004
    mlp_cov_lr_delay_mult: float = 0.01
    mlp_cov_lr_max_steps: int = 30_000

    # MLP颜色参数
    mlp_color_lr_init: float = 0.008
    mlp_color_lr_final: float = 0.00005
    mlp_color_lr_delay_mult: float = 0.01
    mlp_color_lr_max_steps: int = 30_000

    # MLP特征库参数
    mlp_featurebank_lr_init: float = 0.01
    mlp_featurebank_lr_final: float = 0.00001
    mlp_featurebank_lr_delay_mult: float = 0.01
    mlp_featurebank_lr_max_steps: int = 30_000

    # 编码参数
    encoding_xyz_lr_init: float = 0.005
    encoding_xyz_lr_final: float = 0.00001
    encoding_xyz_lr_delay_mult: float = 0.33
    encoding_xyz_lr_max_steps: int = 30_000

    # MLP网格参数
    mlp_grid_lr_init: float = 0.005
    mlp_grid_lr_final: float = 0.00001
    mlp_grid_lr_delay_mult: float = 0.01
    mlp_grid_lr_max_steps: int = 30_000

    # MLP变形参数
    mlp_deform_lr_init: float = 0.005
    mlp_deform_lr_final: float = 0.0005
    mlp_deform_lr_delay_mult: float = 0.01
    mlp_deform_lr_max_steps: int = 30_000

    # 其他优化参数
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2

    # 锚点密集化参数
    start_stat: int = 500
    update_from: int = 1500
    update_interval: int = 100
    update_until: int = 15_000
    pause_update_from: int = 3000
    pause_update_until: int = 4000

    min_opacity: float = 0.005
    success_threshold: float = 0.8
    densify_grad_threshold: float = 0.0002

    iter_switch_to_quantized: int = 3_000
    iter_update_anchor_bound: int = 10_000
    iter_switch_to_entropy: int = 10_001

    lmbda: float = 0.001
