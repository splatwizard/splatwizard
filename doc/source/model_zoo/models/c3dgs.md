# Compressed 3DGS
Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis

:::{tip}
:class: myclass1 myclass2
:name: a-tip-reference
C3DGS requires a pre-trained 3DGS model as `--init_checkpoint`.

:::

Compress and finetune
```shell
sw-train \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/c3dgs/ \
    --model c3dgs \
    --optim c3dgs \
    --init_checkpoint /path/to/point_cloud.ply
```

Evaluate
```shell
sw-eval \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/c3dgs/ \
    --model c3dgs \
    --optim c3dgs \
    --eval_mode ENCODE_DECODE \
    --checkpoint /output/c3dgs//checkpoints/ckpt5000.pth
```