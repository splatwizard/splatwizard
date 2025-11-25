# 3D Gaussian Splatting
Train model
```shell
sw-train \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/speedy_splat \                  
    --model speedy_splat \
    --optim speedy_splat
```

Evaluate model
```shell
sw-eval \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/gs \                  
    --model speedy_splat \
    --optim speedy_splat \
    --checkpoint /output/speedy_splat/checkpoints/ckpt30000.pth
```