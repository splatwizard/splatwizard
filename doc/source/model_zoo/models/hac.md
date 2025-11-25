# HAC
Train model
```shell
python -m splatwizard.main \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/hac/ \
    --model hac \
    --optim hac 
```

Eval model
```shell
python -m splatwizard.main \
    --source_path /data/MipNeRF-360/bicycle \
    --model hac \
    --optim hac \
    --eval_mode ENCODE_DECODE \
    --checkpoint /output/hac/checkpoints/chkpnt30000.pth
```

