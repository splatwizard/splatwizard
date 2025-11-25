# ControlGS
Train Model
```shell
python -m splatwizard.main \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/controlgs/ \
    --model controlgs \
    --optim controlgs \
    --lambda_opacity 1e-7 # [1e-7, 2e-7, 3e-7, 5e-7, 1e-6]
```

Eval model
```shell
python -m splatwizard.main \
    --source_path /data/MipNeRF-360/bicycle \
    --model controlgs \
    --optim controlgs \
    --checkpoint /output/controlgs/checkpoints/chkpnt30000.pth
```

