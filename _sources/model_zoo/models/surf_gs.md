# 2D Gaussian Splatting
Before using the model, please read [Extract mesh from 3DGS](/tutorial/reconstruction) first.
## Train model
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

## Evaluate model
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

## Extract mesh
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
  --shrink_factor 1.03 \
  --data_mode FULL \
  --mask_background False \
  --checkpoint /output/2dgs/checkpoints/ckpt30000.pth
```

