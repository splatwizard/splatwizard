# CAT-3DGS
Train model
```shell
sw-train \
--source_path /data/MipNeRF-360/bicycle \
--output_dir /output/cat3dgs/ \
--model cat3dgs \
--optim cat3dgs \
--lmbda 0.002
```

Eval model
```shell
sw-eval \
--source_path /data/MipNeRF-360/bicycle \
--model cat3dgs \
--optim cat3dgs \
--data_mode SPLIT \
--eval_mode ENCODE_DECODE \
--checkpoint /output/cat3dgs/checkpoints/ckpt40000.pth
```