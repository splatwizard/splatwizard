# Use images without SfM
When specifying a dataset using `--source_path` option, 
the path can contain only image files without including SfM. 
In this case, Splatwizard will automatically use Colmap to build the SfM. 
For example, when using the following command to train the model
```bash

sw-train \
  --source_path /data/dataset/mipnerf/bicycles/images \
  --model 3dgs \
  --optim 3dgs \
  --output_dir /data/output/splatwizard/dev/
```
Splatwizard will automatically create `/data/dataset/mipnerf/bicycles/images_colmap` and 
generate corresponding SfM data to the folder. 

:::{tip}
:class: myclass1 myclass2
:name: a-tip-reference
Due to limitations of pycolmap, this reconstruction process can only be executed on the CPU. 
If CUDA acceleration is required for the reconstruction process, please refer to [link](https://pypi.org/project/pycolmap/)

:::