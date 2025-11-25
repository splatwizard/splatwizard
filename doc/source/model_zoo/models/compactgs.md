# Compact 3DGS
Train Model

For Real-world scenes (eg., 360, T&T, and DB )
```shell
python -m splatwizard.main \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/compactgs/ \
    --model compactgs \
    --optim compactgs \
    --max_hashmap 19 \
    --lambda_mask 0.0005 \
    --mask_lr 0.01 \
    --net_lr 0.01 \
    --net_lr_step 5000 15000 25000
```
 
For Nerf-synthetic, DTU scenes
```shell
python -m splatwizard.main \
    --source_path /data/Nerf-synthetic/chair \
    --output_dir /output/compactgs/ \
    --model compactgs \
    --optim compactgs \
    --max_hashmap 16 \
    --lambda_mask 4e-3 \
    --mask_lr 1e-3 \
    --net_lr 1e-3 \
    --net_lr_step 25000 
```

Eval model
```shell
python -m splatwizard.main \
    --source_path /data/MipNeRF-360/bicycle \
    --model compactgs \
    --optim compactgs \
    --max_hashmap 19 \   # use same max_hashmap as used in training
    --eval_mode ENCODE_DECODE \
    --checkpoint /output/compactgs/checkpoints/ckpt30000.pth
```

:::{tip}
:class: myclass1 myclass2
:name: a-tip-reference
Currently, model parameters are not saved in checkpoint. When evaluate a model, please specify corresponding model
parameters. In Compact 3DGS, make sure using same `--max_hashmap` in training and evaluation.

:::