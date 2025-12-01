# Extract mesh from 3DGS
The reconstruction pipeline is consistent with that proposed in [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surfsplatting.github.io/), 
using the TSDF to extract a mesh from depth maps.

:::{tip}
:class: myclass1 myclass2
:name: a-tip-reference
Reconstruction requires additional dependencies, please install the deps using
```shell
pip install splatwizard[recon]
```

:::

Here, we use 2DGS model as an example to illustrate how to reconstruct mesh from multiview images.

First, train a 2DGS model
```shell
sw-train \
  --source_path /data/DTU/scan24 \
  --depth_ratio 1 \
  --lambda_dist 1000 \
  --model 2dgs \
  --optim 2dgs \
  --data_mode FULL \
  --mask_background False \
  --output_dir /output/2dgs
```
By default, the background will be masked if an alpha channel exists. Using `--mask_background False`  disables the behaviour.
`--data_mode FULL` means using the entire dataset as the training set, without splitting it into a training set and a test set.

Second, check fitting result
```shell
sw-eval \
  --source_path /data/DTU/scan24 \
  --output_dir /output/2dgs \
  --model 2dgs \
  --optim 2dgs \
  --data_mode FULL \
  --mask_background False \
  --checkpoint /output/2dgs/checkpoints/ckpt30000.pth
```
Make sure to use consistent settings for --mask_background and --data_mode as you did during training.

Third, extract the mesh from trained model
```shell
sw-recon
  --source_path /data/DTU/scan24 \
  --output_dir /output/2dgs \
  --mask_dir /data/DTU/scan24/scan63/mask \
  --model 2dgs \
  --optim 2dgs \
  --depth_ratio 1 \
  --num_cluster 1 \
  --voxel_size 0.004 \
  --sdf_trunc 0.016 \
  --depth_trunc 3.0 \
  --data_mode FULL \
  --mask_background False \
  --skip_culling False \
  --checkpoint /output/2dgs/checkpoints/ckpt30000.pth
```
`--voxel_size`, `--sdf_trunc` and `--depth_trunc` are data-dependent hyperparameters.
`--skip_culling` controls whether to perform culling operations after extraction. 
If set to False, background points/mesh will be culled based on the alpha channel. 
You can also use a custom mask by specifying the `--mask_dir`.

Final mesh file is saved in `--output_dir /output/2dgs/fuse_culled.ply`


