import os
from pathlib import Path


module_path = Path(__file__)
module_path_str = str(module_path.parent.absolute())
glm_dir = module_path.parent.parent.parent / "third_party" / "glm"


NAME = '_flash_gaussian_splatting'
CXX_FLAGS = ["-g", "-O1", "-std=c++17"]
NVCC_FLAGS = ["-O1", '-Xptxas="-O1"']

module_path = os.path.dirname(os.path.abspath(__file__))

SOURCES = [
    module_path_str + "/csrc/cuda_rasterizer/preprocess.cu",
    module_path_str + "/csrc/cuda_rasterizer/render.cu",
    module_path_str + "/csrc/cuda_rasterizer/sort.cu",
    module_path_str + "/csrc/pybind.cpp",
]
EXTRA_INCLUDE_PATHS = [str(glm_dir.absolute())]
LD_DIRS = []
LIBRARIES = []

