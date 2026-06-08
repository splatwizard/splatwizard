# Splatwizard: Gaussian Splatting Compression Toolkit


[![arXiv](https://img.shields.io/badge/arXiv-2512.24742-b31b1b.svg)](https://arxiv.org/abs/2512.24742)
[![License](https://img.shields.io/github/license/splatwizard/splatwizard?color=blue)](https://github.com/splatwizard/splatwizard/blob/main/LICENSE.md)
[![PyPI](https://img.shields.io/pypi/v/splatwizard?color=brightgreen)](https://pypi.org/project/splatwizard/)


Splatwizard is a one-stop toolkit designed for research on 3DGS compression, dedicated to accelerating exploration and development in this field.
With flexible API design, you can easily combine advanced components from state-of-the-art models to build customized models.
Key features of the project include:

- **Easy-to-adapt evaluation framework** for new models
- **Comprehensive evaluation metrics**
- **Multiple baseline models** included for comparison

## Latest News
- 2026/06/07: Benchmark paper published at [CVPRF'26](https://openaccess.thecvf.com/content/CVPR2026F/html/Liu_Splatwizard_A_Benchmark_Toolkit_for_3D_Gaussian_Splatting_Compression_CVPRF_2026_paper.html)
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


## Citation
If you find our work helpful, please consider citing:

```
@InProceedings{Liu_2026_CVPR,
    author    = {Liu, Xiang and Zhou, Yimin and Wang, Jinxiang and Huang, Yujun and Xie, Shuzhao and Qin, Shiyu and Hong, Mingyao and Li, Jiawei and Wang, Yaowei and Wang, Zhi and Xia, Shu-Tao and Chen, Bin},
    title     = {Splatwizard: A Benchmark Toolkit for 3D Gaussian Splatting Compression},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {2261-2271}
}
```

## License

Splatwizard is licensed under the MIT License. The project incorporates code from other projects, which remains under their original licenses.