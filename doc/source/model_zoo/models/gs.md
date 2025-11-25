# 3D Gaussian Splatting
Train model
```shell
sw-train \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/gs \                  
    --model 3dgs \
    --optim 3dgs
```

Evaluate model
```shell
sw-eval \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/gs \                  
    --model 3dgs \
    --optim 3dgs \
    --checkpoint /output/gs/checkpoints/ckpt30000.pth
```