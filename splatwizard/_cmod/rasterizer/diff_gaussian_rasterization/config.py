import os
from pathlib import Path




module_path = Path(__file__)
module_path_str = str(module_path.parent.absolute())
glm_dir = module_path.parent.parent.parent / "third_party" / "glm"


NAME = '_diff_gaussian_rasterization'
CXX_FLAGS = []
NVCC_FLAGS = []

module_path = os.path.dirname(os.path.abspath(__file__))

SOURCES = [
    module_path_str + "/cuda_rasterizer/rasterizer_impl.cu",
    module_path_str + "/cuda_rasterizer/forward.cu",
    module_path_str + "/cuda_rasterizer/backward.cu",
    module_path_str + "/rasterize_points.cu",
    module_path_str + "/ext.cpp"
]
EXTRA_INCLUDE_PATHS = [str(glm_dir.absolute())]
LD_DIRS = []
LIBRARIES = []

