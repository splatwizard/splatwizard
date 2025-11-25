from dataclasses import dataclass
from enum import Enum
from typing import List

from splatwizard.config import OptimizationParams, ModelParams
from splatwizard.modules.triplane import TriPlaneFieldConfig

class GenerateMode(Enum):
    TRAINING_FULL_PRECISION = 0
    TRAINING_QUANTIZED = 1
    TRAINING_ENTROPY = 2
    TRAINING_STE_ENTROPY = 3
    DECODING_AS_IS = 4
    TRAINING_STE_QUANTIZED = 5


@dataclass
class AttributeNetParams:
    # attribute_config
    net_width: int = 100
    net_depth: int = 0
    final_dim: int = (6 + 30 + 50) * 2 + 1 + 1 + 1
    bounds: int = 1

    kplane_config = TriPlaneFieldConfig()


    if_contract: bool = True
    comp_iter: int = 15000
    multires: List[int] = (1, 2)


@dataclass
class CAT3DGSModelParams(ModelParams):
    sh_degree: int = 3
    feat_dim: int = 50
    n_offsets: int = 10
    voxel_size: float = 0.001  # if voxel_size<=0, using 1nn dist
    update_depth: int = 3
    update_init_factor: int = 16
    update_hierarchy_factor: int = 4

    n_features_per_level: int = 4
    use_feat_bank: bool = False

    # # attribute_config
    # net_width: int = 100
    # net_depth: int = 0
    # final_dim: int = (6 + 30 + 50) * 2 + 1 + 1 + 1
    # bounds: int = 1
    #
    # kplane_grid_dimensions: int = 2
    # kplane_input_coordinate_dim: int = 3
    # kplane_output_coordinate_dim: int = 72
    # kplane_resolution: Tuple[int] = (0, 0, 0)

    chcm_slices_list: List[int] = (12, 12, 13, 13)
    chcm_for_offsets: bool = False
    chcm_for_scaling: bool = False

    attribute_net: AttributeNetParams = AttributeNetParams()

    enforce_mode: GenerateMode = None #TRAINING_FULL_PRECISION



    if_contract: bool = True
    comp_iter: int = 15000
    multires: List[int] = (1, 2)

    ste_binary: bool = True
    ste_multistep: bool = False
    add_noise: bool = False
    Q: float = 1
    use_2D: bool = True

    decoded_version: bool = False




    # 'net_width': 100,
    # 'net_depth': 0,
    # 'final_dim': (6 + 30 + 50) * 2 + 1 + 1 + 1,
    # 'bounds': 1,
    # 'kplanes_config': {
    #     'grid_dimensions': 2,
    #     'input_coordinate_dim': 3,
    #     'output_coordinate_dim': 72,
    #     'resolution': [0, 0, 0]
    # },
    # 'if_contract': True,
    # 'comp_iter': 15000,
    # 'multires': [1, 2]


@dataclass
class CAT3DGSOptimizationParams(OptimizationParams):
    iterations: int = 40_000
    lr_max_iteration: int = 40000

    # for anchor densification
    start_stat: int = 500
    update_from: int = 1500
    update_interval: int = 100
    update_until: int = 15_000

    pause_update_from: int = 3_000
    pause_update_until: int = 4_000
    iter_switch_to_quantized: int = 3_000
    iter_update_anchor_bound: int = 10_000
    iter_switch_to_entropy: int = 10_000

    triplane_init_fit_iter: int = 10_000

    position_lr_init: float = 0.0
    position_lr_final: float = 0.0
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = lr_max_iteration

    offset_lr_init: float = 0.01
    offset_lr_final: float = 0.0001
    offset_lr_delay_mult: float = 0.01
    offset_lr_max_steps: int = lr_max_iteration

    mask_lr_init: float = 0.01
    mask_lr_final: float = 0.0001
    mask_lr_delay_mult: float = 0.01
    mask_lr_max_steps: int = lr_max_iteration

    feature_lr: float = 0.0075
    opacity_lr: float = 0.02
    scaling_lr: float = 0.007
    rotation_lr: float = 0.002

    mlp_opacity_lr_init: float = 0.002
    mlp_opacity_lr_final: float = 0.00002
    mlp_opacity_lr_delay_mult: float = 0.01
    mlp_opacity_lr_max_steps: int = lr_max_iteration

    mlp_cov_lr_init: float = 0.004
    mlp_cov_lr_final: float = 0.004
    mlp_cov_lr_delay_mult: float = 0.01
    mlp_cov_lr_max_steps: int = lr_max_iteration

    mlp_color_lr_init: float = 0.008
    mlp_color_lr_final: float = 0.00005
    mlp_color_lr_delay_mult: float = 0.01
    mlp_color_lr_max_steps: int = lr_max_iteration

    mlp_featurebank_lr_init: float = 0.01
    mlp_featurebank_lr_final: float = 0.00001
    mlp_featurebank_lr_delay_mult: float = 0.01
    mlp_featurebank_lr_max_steps: int = lr_max_iteration

    encoding_xyz_lr_init: float = 0.005
    encoding_xyz_lr_final: float = 0.00001
    encoding_xyz_lr_delay_mult: float = 0.33
    encoding_xyz_lr_max_steps: int = lr_max_iteration

    mlp_grid_lr_init: float = 0.005
    mlp_grid_lr_final: float = 0.00001
    mlp_grid_lr_delay_mult: float = 0.01
    mlp_grid_lr_max_steps: int = lr_max_iteration

    mlp_deform_lr_init: float = 0.005
    mlp_deform_lr_final: float = 0.0005
    mlp_deform_lr_delay_mult: float = 0.01
    mlp_deform_lr_max_steps: int = 50_000

    percent_dense: float = 0.01
    lambda_dssim: float = 0.2

    min_opacity: float = 0.005  # 0.2
    success_threshold: float = 0.8
    densify_grad_threshold: float = 0.0002

    lmbda: float = 0.002
    lmbda_tri: float = 10.0

    cam_mask: float = 1
    camera_dependent_task: bool = True
