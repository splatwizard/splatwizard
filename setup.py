import pathlib
from pathlib import Path
from setuptools import setup, find_packages, find_namespace_packages
import os

from torch.utils.cpp_extension import CUDAExtension, BuildExtension

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"


def map_path_str(source: Path):
    s = str(Path(source).relative_to(Path(__file__).resolve().parent))
    return './' + s

def get_ext():

    return BuildExtension


def export_extension(mod_dir):
    mod_dir = Path(__file__).parent / Path(mod_dir)
    base_path = Path(__file__).parent
    relative_path = mod_dir.relative_to(base_path)

    mod_import_path = str(relative_path).replace('/', '.')
    config_file = mod_dir / 'config.py'
    context = {'__file__': str(config_file.absolute())}
    with open(config_file) as f:
        exec(f.read(), context)

    source = [map_path_str(s) for s in context['SOURCES']]

    extension = CUDAExtension(
                name=mod_import_path + '.' + context['NAME'],
                sources=source,
                include_dirs=[Path(s).resolve() for s in context['EXTRA_INCLUDE_PATHS']],
                extra_compile_args={'cxx': context['CXX_FLAGS'],
                                    'nvcc': context['NVCC_FLAGS']},
                libraries=context['LIBRARIES'],
                library_dirs=context['LD_DIRS'],
            )

    return extension


setup(
    name="splatwizard",
    version="0.0.1",
    packages=find_packages(),
    # packages=['mypkg', 'mypkg.subpkg1', 'mypkg.subpkg2'],
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "einops",
        "lpips",
        "tqdm",
        "plyfile>=0.8.1",
        "loguru",
        "numpy",
        "simple-parsing",
        "pycolmap",
        "ninja",
        "matplotlib",
        "colorama",
        "vector-quantize-pytorch==1.22.0",
        "dahuffman",
        "torchac"
    ],
    extras_require={
            # dev dependencies. Install them by `pip install gsplat[dev]`
            "dev": [
                # "black[jupyter]==22.3.0",
                # "isort==5.10.1",
                # "pylint==2.13.4",
                "pytest",
                "tox-current-env",
                "tox"
                # "pytest-xdist==2.5.0",
                # "typeguard>=2.13.3",
                # "pyyaml==6.0",
                # "build",
                # "twine",
            ],
        },
    python_requires=">=3.7.13",
    author="actcwlf",
    description="Gaussian Splatting compression toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=False,
    entry_points={
        "console_scripts": [
            # "sw-main=splatwizard.main:main",
            "sw-train=splatwizard.scripts.train:main",
            "sw-eval=splatwizard.scripts.eval:main",
            "sw-encode=splatwizard.scripts.encode:main",
        ],
    },
    ext_modules=[
        export_extension('splatwizard/_cmod/arithmetic'),
        export_extension('splatwizard/_cmod/fused_ssim'),
        export_extension('splatwizard/_cmod/gridencoder'),
        export_extension('splatwizard/_cmod/lanczos_resampling'),
        export_extension('splatwizard/_cmod/simple_knn'),
        export_extension('splatwizard/_cmod/rasterizer/accel_gs'),
        export_extension('splatwizard/_cmod/rasterizer/diff_gaussian_rasterization'),
        export_extension('splatwizard/_cmod/rasterizer/flashgs'),
        export_extension('splatwizard/_cmod/rasterizer/pup_fisher'),
        export_extension('splatwizard/_cmod/rasterizer/compress'),
        export_extension('splatwizard/_cmod/rasterizer/speedy_splat'),
        export_extension('splatwizard/_cmod/rasterizer/indexed_gs'),
        export_extension('splatwizard/_cmod/rasterizer/gs_dr_aa'),
        export_extension('splatwizard/_cmod/rasterizer/speedy_tcgs'),
        export_extension('splatwizard/_cmod/rasterizer/surfel_gs'),
        export_extension('splatwizard/_cmod/rasterizer/meson_gs'),
        export_extension('splatwizard/_cmod/weighted_distance'),
        export_extension('splatwizard/_cmod/tiny_cuda_nn'),
    ],
    cmdclass={
        'build_ext': get_ext()
    }
)
