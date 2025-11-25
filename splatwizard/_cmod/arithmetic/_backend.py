
# import os.path as osp
# from torch.utils.cpp_extension import load
# import os

from ..common import load

from .config import NAME, NVCC_FLAGS, CXX_FLAGS, EXTRA_INCLUDE_PATHS, SOURCES
# module_path = os.path.dirname(os.path.abspath(__file__))
# include_dirs = [osp.join(module_path, "include")]


_C = load(name=NAME,
          extra_cflags=CXX_FLAGS,
          extra_cuda_cflags=NVCC_FLAGS,
          sources=SOURCES,
          extra_include_paths=EXTRA_INCLUDE_PATHS,
)

__all__ = ['_C']
