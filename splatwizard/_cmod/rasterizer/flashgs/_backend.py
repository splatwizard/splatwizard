# import os
# from pathlib import Path
# from torch.utils.cpp_extension import load
#
#
# module_path = Path(__file__)
# module_path_str = str(module_path.parent.absolute())
# glm_dir = module_path.parent.parent.parent / "third_party" / "glm"
#
# # print(glm_dir)
# # ["-I" + str(glm_dir.absolute())] +
# CXX_FLAGS = ["-g", "-O1", "-std=c++17"]
# NVCC_FLAGS = ["-O1", '-Xptxas="-O1"']
#
#
# _backend = load(name='flash_gaussian_splatting',
#                 # extra_cflags=c_flags,
#                 extra_cuda_cflags=["-I" + str(glm_dir.absolute())] +NVCC_FLAGS,
#                 sources=[
#                     module_path_str + "/csrc/cuda_rasterizer/preprocess.cu",
#                     module_path_str + "/csrc/cuda_rasterizer/render.cu",
#                     module_path_str + "/csrc/cuda_rasterizer/sort.cu",
#                     module_path_str + "/csrc/pybind.cpp",
#                 ],
#                 extra_cflags=["-I" + str(glm_dir.absolute())] + CXX_FLAGS,
#                 )
#
#
#
#
# __all__ = ['_backend']
#


from splatwizard._cmod.common import load
from .config import NAME, NVCC_FLAGS, CXX_FLAGS, EXTRA_INCLUDE_PATHS, SOURCES


_C = load(name=NAME,
                extra_cflags=CXX_FLAGS,
                extra_cuda_cflags=NVCC_FLAGS,
                sources=SOURCES,
                extra_include_paths=EXTRA_INCLUDE_PATHS,
                )

__all__ = ['_C']