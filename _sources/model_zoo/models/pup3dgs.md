# pup3dgs
Train model
```shell
python -m splatwizard.main \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/pup3dgs/ \
    --model pup3dgs \
    --optim pup3dgs 
```

Train mutilple model
```shell
#!/bin/bash

NAMES=("chair"  "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")  
TYPE="pup3dgs"

for NAME in "${NAMES[@]}"
do
  CUDA_VISIBLE_DEVICES=0 python -m splatwizard.main \ 
    --source_path /data/MipNeRF-360/bicycle/${NAME} \ 
    --output_dir ./outputs/${TYPE}/${NAME} \ 
    --init_checkpoint ./outputs/gs/${NAME}/ \ 
    --model ${TYPE} \
    --optim ${TYPE} 
done
``` 

Eval model
```shell
CUDA_VISIBLE_DEVICES=0 python -m splatwizard.scripts.eval \
    --source_path /_dataset/nerf_synthetic/${NAME} \
    --checkpoint ./outputs/pup3dgs/${NAME}/PRUNE \
    --model pup3dgs \
    --optim pup3dgs
```

