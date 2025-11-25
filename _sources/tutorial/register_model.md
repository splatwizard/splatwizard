# Use new model

To simplify the use of numerous models in the framework, Splatwizard designed a registration mechanism to enable parameterized model invocation. 
For newly added models, they can be registered in the following format

```python
# train_flashgs.py
import sys

from splatwizard.main import main
from splatwizard.model_zoo import register_model
from splatwizard.model_zoo.gs.model import GSModel, GSModelParams, GSOptimizationParams
from splatwizard.modules.render_mixin import FlashGSRenderMixin


class FlashGS(FlashGSRenderMixin, GSModel):
    def __init__(self, model_params):
        GSModel.__init__(self, model_params)
        FlashGSRenderMixin.__init__(self)


if __name__ == "__main__":
    register_model('flashgs', GSModelParams, GSOptimizationParams, FlashGS)
    sys.exit(main())
```


In the example, we defined a FlashGS, which replace default rasterizer by FlashGS rasterizer. 
Then we can use the script to train
```shell
python  train_flashgs.py \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/flashgs \                  
    --model flashgs \     # Using register key
    --optim flashgs
```

and evaluate the model
```shell
python  train_flashgs.py \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/flashgs \                  
    --model flashgs \
    --optim flashgs \
    --eval_mode NORMAL \   # Using evaluation mode
    --checkpoint /output/flashgs/checkpoints/ckpt30000.pth
```