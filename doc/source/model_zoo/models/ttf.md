# TTF
Prune model
```shell
sw-main \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/ttf \
    --model trimming_the_fat \
    --optim trimming_the_fat \
    --init_checkpoint /output/gs/point_cloud.ply \
    --pruning_level 0.425  # [0.225-0.6]
```
:::{tip}
:class: myclass1 myclass2
:name: a-tip-reference
The TTF model performs pruning based on a pre-trained model, 
thus it only involves the `PRUNE` phase and requires a pre-trained 3DGS model as `--init_checkpoint`.

:::


Evaluate model
```shell
sw-eval \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/ttf \                  
    --model trimming_the_fat \
    --optim trimming_the_fat \
    --checkpoint /output/ttf/PRUNE/checkpoints/ckpt30000.pth
```