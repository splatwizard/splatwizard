from splatwizard._cmod.common import load
from .config import NAME, NVCC_FLAGS, CXX_FLAGS, EXTRA_INCLUDE_PATHS, SOURCES



_C = load(name=NAME,
          extra_cflags=CXX_FLAGS,
          extra_cuda_cflags=NVCC_FLAGS,
          sources=SOURCES,
          extra_include_paths=EXTRA_INCLUDE_PATHS,
)

__all__ = ['_C']
