# Splatwizard: Gaussian Splatting Compression Toolkit


[![arXiv](https://img.shields.io/badge/arXiv-2512.24742-b31b1b.svg)](https://arxiv.org/abs/2512.24742)
[![License](https://img.shields.io/github/license/splatwizard/splatwizard?color=blue)](https://github.com/splatwizard/splatwizard/blob/main/LICENSE.md)
[![PyPI](https://img.shields.io/pypi/v/splatwizard?color=brightgreen)](https://pypi.org/project/splatwizard/)


Splatwizard is a one-stop toolkit designed for researching 3DGS compression, dedicated to accelerating community exploration in this field.
With flexible API design, you can easily combine advanced components from state-of-the-art models to build customized models.
Key features of the project include:

- **Easy-to-adapt evaluation framework** for new models
- **Comprehensive evaluation metrics**
- **Multiple baseline models** included for comparison

## Latest News
- 2025/12/31: Preprint available on [arXiv](https://arxiv.org/abs/2512.24742).
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

Additionally, pip compiles components in an isolated environment by default. You can use `--no-build-isolation` to perform the compilation directly in the current environment.
```bash
pip install splatwizard --verbose --no-build-isolation
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
* [Model zoo](https://interdigitalinc.github.io/CompressAI/zoo.html)


## Citation
If you find our work helpful, please consider citing:

```
@article{liu2025splatwizard,
  title={Splatwizard: A Benchmark Toolkit for 3D Gaussian Splatting Compression},
  author={Liu, Xiang and Zhou, Yimin and Wang, Jinxiang and Huang, Yujun and Xie, Shuzhao and Qin, Shiyu and Hong, Mingyao and Li, Jiawei and Wang, Yaowei and Wang, Zhi and others},
  journal={arXiv preprint arXiv:2512.24742},
  year={2025}
}
```

## License

Splatwiard is licensed under the MIT License. The project incorporates code from other projects, which remains under their original licenses.