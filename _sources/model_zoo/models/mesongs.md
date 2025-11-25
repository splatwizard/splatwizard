# MesonGS
MesonGS: Post-training Compression of 3D Gaussians via Efficient Attribute Transformation

:::{tip}
:class: myclass1 myclass2
:name: a-tip-reference
MesonGS requires a pre-trained 3DGS model as `--init_checkpoint`.

:::
Install
```shell
Todo
```

Compress and finetune
```shell
sw-train \
    --source_path /mnt/storage/users/szxie_data/nerf_data/360_v2/bicycle \
    --output_dir outputs/mesongs/ \
    --model mesongs \
    --optim mesongs \
    --init_checkpoint /mnt/storage/users/szxie_data/MesonGS/outputs/360_v2/bicycle/baseline/3dgs/point_cloud/iteration_30000/point_cloud.ply
```

Evaluate
```shell
sw-eval \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/mesongs/ \
    --model mesongs \
    --optim mesongs \
    --eval_mode ENCODE_DECODE \
    --checkpoint /output/mesongs//checkpoints/ckpt5000.pth
```