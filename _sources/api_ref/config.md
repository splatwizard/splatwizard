# Configuration


### Arguments
- `-h`, `--help`  show this help message and exit
- `--config_path` *Path*  Path to a config file containing default values to use. (default: None)

### PipelineParams

- `--source_path` *str*  Path to the root directory of the dataset (default: None)
- `--ply_data_path` *str*  Path to the PLY file containing point cloud data (optional) (default: None)
- `--images` *str*  Subdirectory name containing images in COLMAP format scene dataset (default: images)
- `--dataset` *str*  Dataset identifier/name (default: None)
- `--test_sample_freq` *int*  Sample every Nth view for testing (default: 8)
- `--data_mode` *DataMode*  Dataset partitioning strategy (SPLIT/FULL) (default: SPLIT)
- `--output_dir` *str*  Output directory for results and artifacts (default: None)
- `--checkpoint_iterations` *int*  Iteration numbers to save model checkpoints (default: [])
- `--checkpoint` *str*  Path to load a pre-trained model checkpoint (default: None)
- `--checkpoint_type` *str*  File format for checkpoints (pth/ply) (default: pth)
- `--init_checkpoint` *str*  Initial checkpoint for post-processing models (e.g. LightGaussian) (default: None)
- `--bitstream` *str*  Path for compressed model binary (bitstream) output (default: None)
- `--final_checkpoint` *str*  File format for checkpoint at the end of training. Both pth and ply checkpoint will be saved by default (default: ('pth', 'ply'))
- `--convert_SHs_python` *bool*  Whether to compute spherical harmonics (SHs) colors using Python instead of CUDA (default: False)
- `--compute_cov3D_python` *bool*  Whether to compute 3D covariance matrices using Python instead of CUDA (default: False)
- `--debug` *bool*  Enable debug mode for additional logging and checks (default: False)
- `--lanczos_resample` *bool*  Use Lanczos resampling for image downscaling (higher quality, NOT SUPPORTED YET) (default: False)
- `--resolution` *int*  Target resolution for processing (-1 for auto rescaling to smaller than 1.6K) (default: -1)
- `--white_background` *bool*  Whether to background as pure white during rendering (default: False)
- `--data_device` *str*  Device for data loading ('cuda' or 'cpu') (default: cuda)
- `--lod` *int*  Level of Detail (LOD) for multi-resolution processing (NOT SUPPORTED YET) (default: 0)
- `--eval_mode` *EvalMode*  Evaluation methodology. options: None (Training mode), NORMAL (normal evaluation), ENCODE_DECODE (encode and decode model before evaluation), DECODE (decode model from bitstream before evaluation) (default: None)
- `--eval_warmup` *bool*  Perform warmup iterations before evaluation for more accurate FPS (default: True)
- `--device` *str*  (default: cuda)
- `--eval_freq` *int*  Run evaluation every N training iterations (default: 200)
- `--seed` *int*  Random seed for reproducibility (None for random initialization) (default: None)
- `--profile_train` *bool*  Enable performance profiling during training (default: True)
- `--num_workers` *int*  Number of parallel workers for data loading (default: 0)
- `--cache_dataset` *bool*  Cache entire dataset into memory for faster access (default: True)

### GroupConfig
- `--model` *{3dgs,hac,trimming_the_fat,compactgs,c3dgs,controlgs,contextgs,lightgaussian,pup3dgs}*  (default: 3dgs)
- `--optim` *{3dgs,hac,trimming_the_fat,compactgs,c3dgs,controlgs,contextgs,lightgaussian_prune,lightgaussian_distill,lightgaussian_encode,pup3dgs}*  (default: 3dgs)

### ModelParams
- `--require_cam_infos` *bool*  (default: False)
- `--sh_degree` *int*  (default: 3)

### OptimizationParams
- `--require_pretrained` *bool*  (default: False)
- `--require_sfm` *bool*  Whether Structure-from-Motion (SfM) data is required (default: True)
- `--use_fused_ssim` *bool*  Whether to use fused SSIM (Structural Similarity) loss (default: True)
- `--use_trained_exposure` *bool*  Whether to use trained exposure parameters (default: False)
- `--iterations` *int*  Total number of optimization iterations (default: 30000)
- `--random_background` *bool*  Whether to use random background during training (default: False)
- `--lambda_dssim` *float*  Weight factor for SSIM (Structural Dissimilarity) loss component (default: 0.2)
- `--camera_dependent_task` *bool*  Whether to use camera dependent task (default: True)
- `--current_stage` *Enum*  Current optimization stage (defined by specific model) (default: None)
- `--force_setup` *bool*  (default: False)
