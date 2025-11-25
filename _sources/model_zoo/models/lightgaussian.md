# LightGaussian


:::{tip}
:class: myclass1 myclass2
:name: a-tip-reference
LightGaussian requires a pre-trained 3DGS model as `--init_checkpoint`.

:::


Prune model
```shell
sw-train \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/lightgs/ \
    --model lightgaussian \
    --optim lightgaussian_prune \
    --init_checkpoint /path/to/point_cloud.ply
```

Distillate model
```shell
sw-train \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/lightgs/ \
    --model lightgaussian \
    --optim lightgaussian_distill \
    --init_checkpoint /output/lightgs/PRUNE/point_cloud/iteration_5000/point_cloud.ply \
    --teacher_checkpoint /path/to/point_cloud.ply
```

Evaluate model 
```shell
sw-eval \
--source_path /data/MipNeRF-360/bicycle \
--model lightgaussian \
--optim lightgaussian_encode \
--eval_mode ENCODE_DECODE \
--sh_degree 2 \
--checkpoint /output/lightgs/DISTILL/checkpoints/ckpt5000.pth
```
Note encoding stage of lightgaussian requires importance score. 
Please use checkpoint instead of .ply file. In default setting, distillation stage will automatically save checkpoint file .
