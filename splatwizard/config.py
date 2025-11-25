from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Annotated
import os
import sys
from argparse import ArgumentParser, Namespace

from typing_extensions import Doc


class DataMode(Enum):
    FULL = "full"
    SPLIT = "split"
    SPLIT_LOD = "split_lod"


class EvalMode(Enum):
    NORMAL = 1
    ENCODE_DECODE = 2
    DECODE = 3


class ReconEvalMode(Enum):
    MESH = 1
    MESH_PCD = 2
    PCD = 3

class CullingMode(Enum):
    VIEW_CUT = 1
    VIEW_KEEP = 2

@dataclass
class ModelParams:
    require_cam_infos: bool = False

    # 基本参数
    # sh_degree: int = 3
    # feat_dim: int = 50
    # n_offsets: int = 10
    # voxel_size: float = 0.001  # if voxel_size<=0, using 1nn dist
    # update_depth: int = 3
    # update_init_factor: int = 16
    # update_hierachy_factor: int = 4
    #
    # # 特征和路径参数
    # use_feat_bank: bool = False
    # source_path: str = ""
    # model_path: str = ""
    # images: str = "images"
    # resolution: int = -1
    # white_background: bool = False
    # data_device: str = "cuda"
    # eval: bool = True
    # lod: int = 0

    # def __post_init__(self):
    #     if self.source_path:
    #         self.source_path = os.path.abspath(self.source_path)

@dataclass
class DatasetParams:
    # Path to the root directory of the dataset
    source_path: str = None
    # Path to the PLY file containing point cloud data (optional)
    ply_data_path: str = None

    # Mask dir, you can use outer mask files instead of alpha channel in original image
    mask_dir: str = None
    # Reference ply file, for evaluating chamfer distance
    ref_ply_path: str = None
    # ObsMask file
    obs_mask_path: str = None
    # plane file
    plane_path: str = None
    mask_background: bool = True


    # Subdirectory name containing images in COLMAP format scene dataset
    images: str = "images"
    # Dataset identifier/name
    dataset: str = None
    # Sample every Nth view for testing
    test_sample_freq: int = 8
    # lod value for DataMode.SPLIT_LOD, common for BungeeNerf dataset
    lod: int = 30
    # Dataset partitioning strategy (SPLIT/FULL)
    data_mode: DataMode = DataMode.SPLIT


@dataclass
class OutputParams:
    # Output directory for results and artifacts
    output_dir: str = None

    # Iteration numbers to save model checkpoints
    checkpoint_iterations: List[int] = field(default_factory=list)
    #  Path to load a pre-trained model checkpoint
    checkpoint: str = None
    # File format for checkpoints (pth/ply)
    checkpoint_type: str = 'pth'
    # Initial checkpoint for post-processing models (e.g. LightGaussian)
    init_checkpoint: str = None
    # Path for compressed model binary (bitstream) output
    bitstream: str = None
    # File format for checkpoint at the end of training. Both pth and ply checkpoint will be saved by default
    final_checkpoint: str = ('pth', 'ply')



@dataclass
class PipelineParams(OutputParams, DatasetParams):

    # Whether to compute spherical harmonics (SHs) colors using Python instead of CUDA
    convert_SHs_python: bool = False
    # Whether to compute 3D covariance matrices using Python instead of CUDA
    compute_cov3D_python: bool = False
    # Enable debug mode for additional logging and checks
    debug: bool = False

    # Use Lanczos resampling for image downscaling (higher quality, NOT SUPPORTED YET)
    lanczos_resample: bool = False

    # Target resolution for processing (-1 for auto rescaling to smaller than 1.6K)
    resolution: int = -1
    # Whether to background as pure white during rendering
    white_background: bool = False
    # Device for data loading ('cuda' or 'cpu')
    data_device: str = "cuda"

    # Level of Detail (LOD) for multi-resolution processing (NOT SUPPORTED YET)
    lod: int = 0


    # Evaluation methodology. options: None (Training mode),
    # NORMAL (normal evaluation),
    # ENCODE_DECODE (encode and decode model before evaluation),
    # DECODE (decode model from bitstream before evaluation)
    eval_mode: EvalMode = None

    save_bitstream: bool = True

    # Perform warmup iterations before evaluation for more accurate FPS
    eval_warmup: bool = True

    # save rendered image
    save_rendered_image: bool = False





    device: str = 'cuda'
    # Run evaluation every N training iterations
    eval_freq: int = 200
    eval_start_iter: int = 0

    # Random seed for reproducibility (None for random initialization)
    seed: int = None

    # Enable performance profiling during training
    profile_train: bool = True
    # Number of parallel workers for data loading
    num_workers: int = 0
    # Cache entire dataset into memory for faster access
    cache_dataset: bool = True



@dataclass
class OptimizationParams:
    require_pretrained: bool = False
    # Whether Structure-from-Motion (SfM) data is required
    require_sfm: bool = True
    # Whether to use fused SSIM (Structural Similarity) loss
    use_fused_ssim: bool = True
    # Whether to use trained exposure parameters
    use_trained_exposure: bool = False

    # Total number of optimization iterations
    iterations: int = 30_000
    # Whether to use random background during training
    random_background: bool = False
    # Weight factor for SSIM (Structural Dissimilarity) loss component
    lambda_dssim: float = 0.2

    # Whether to use camera dependent task
    camera_dependent_task: bool = True
    # Some camera dependent task only require cam intrinsic
    camera_dependent_task_require_images: bool = False


    # # List of optimization stages (defined by specific model)
    # stages: List[Enum] = None

    # Current optimization stage (defined by specific model)
    current_stage: Enum = None

    force_setup: bool = False


@dataclass
class MeshExtractorParams:
    # Mesh: voxel size for TSDF
    voxel_size: float =-1.0
    # Mesh: Max depth range for TSDF
    depth_trunc: float =-1.0
    # Mesh: truncation value for TSDF
    sdf_trunc: float = -1.0
    # Mesh: number of connected clusters to export
    num_cluster: int = 50
    # Mesh: using unbounded mode for meshing
    unbounded: bool = False
    # Mesh: resolution for unbounded mesh extraction
    mesh_res: int = 1024
    # Shrinkage factor during culling, larger than 1 means to shrink the reconstruct point cloud / mesh
    shrink_factor: float = 1.0
    # Used in evaluation of chamfer distance
    downsample_density: float =0.2
    patch_size: float = 60
    max_dist: float = 20
    visualize_threshold: float = 10
    # Reconstruction evaluation mode
    recon_eval_mode: ReconEvalMode = ReconEvalMode.MESH
    skip_culling: bool = False
    culling_mode: CullingMode = CullingMode.VIEW_KEEP



