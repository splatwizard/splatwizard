# PUP-3DGS


:::{tip}
:class: myclass1 myclass2
:name: a-tip-reference
PUP-3DGS requires a pre-trained 3DGS model as `--init_checkpoint`.

:::

Prune model
```shell
sw-train \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/pup3dgs/ \
    --model pup3dgs \
    --optim pup3dgs \
    --init_checkpoint /path/to/point_cloud.ply
```

Evaluate
```shell
sw-eval \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/pup3dgs/ \
    --model pup3dgs \
    --optim pup3dgs \
    --checkpoint /output/pup3dgs/PRUNE/point_cloud/iteration_10000/point_cloud.ply
```