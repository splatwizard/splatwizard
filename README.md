# Splatwizard: Gaussian Splatting Compression Toolkit

[![License](https://img.shields.io/github/license/splatwizard/splatwizard?color=blue)](https://github.com/splatwizard/splatwizard/blob/main/LICENSE.md)
[![PyPI](https://img.shields.io/pypi/v/splatwizard?color=brightgreen)](https://pypi.org/project/splatwizard/)


Splatwizard is a one-stop toolkit designed for researching 3DGS compression, dedicated to accelerating community exploration in this field.
With flexible API design, you can easily combine advanced components from state-of-the-art models to build customized models.
Key features of the project include:

- **Easy-to-adapt evaluation framework** for new models
- **Comprehensive evaluation metrics**
- **Multiple baseline models** included for comparison

## Latest News
- 2025/11/25: Release first version.

## Installation
Splatwizard can be installed via pip
```bash
pip install splatwizard
```

Since the installation process requires significant time to precompile all components, you can monitor the installation progress using the `--verbose` option.
```bash
pip install splatwizard --verbose
```

Additionally, pip compiles components in an isolated environment by default. 
You can use `--no-build-isolation` to perform the compilation directly in the current environment.
Make sure `torch` and `ninja` are installed before running the command.
```bash
pip install splatwizard --verbose --no-build-isolation
```


To speed up compilation, the installation process uses all available CPU cores by default. 
If you need to limit the number of cores used for compilation, you can specify the maximum cores used via environment variable `MAX_BUILD_JOBS`.
The following command will use up to 8 CPU cores during compilation.
```bash
MAX_BUILD_JOBS=8 pip install splatwizard --verbose --no-build-isolation
```


## Quick start

Train your first 3DGS model in splatwizard
```shell
sw-train \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/gs \                  
    --model 3dgs \
    --optim 3dgs
```

Evaluate model

```shell
sw-eval \
    --source_path /data/MipNeRF-360/bicycle \
    --output_dir /output/gs \                  
    --model 3dgs \
    --optim 3dgs \
    --checkpoint /output/gs/checkpoints/ckpt30000.pth
```

For more models. please check [Model List](splatwizard.github.io/splatwizard/model_zoo/model_list.html)


## Documentation
* [Installation](https://splatwizard.github.io/splatwizard/getting_start/installation.html)
* [Training model](https://splatwizard.github.io/splatwizard/getting_start/first_model.html)
* [How to develop your own model](https://splatwizard.github.io/splatwizard/tutorial/concept.html)
* [Model zoo](https://splatwizard.github.io/splatwizard/model_zoo/model_list.html)


## License

Splatwizard is licensed under the MIT License. The project incorporates code from other projects, which remains under their original licenses.