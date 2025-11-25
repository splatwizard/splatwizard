# Train Your First 3DGS model

SplatWizard provides both the original 3DGS model and a series of compression-related model implementations. 
Taking the original 3DGS as an example, you can train it using default parameters:

```shell
sw-train \
  --source_path /data/MipNeRF-360/bicycle \   # specify the directory of dataset
  --output_dir /output/gs \                   # specify output directory
  --model 3dgs \                              # specify model
  --optim 3dgs                                # specify optimization parameters

```

To evaluate the model, using
```shell
sw-eval \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/gs \                  
    --model 3dgs \
    --optim 3dgs \
    --checkpoint /output/gs/checkpoints/ckpt30000.pth
```
