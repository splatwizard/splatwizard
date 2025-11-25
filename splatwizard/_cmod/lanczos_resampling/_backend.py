#
# import os
# from torch.utils.cpp_extension import load
# module_path = os.path.dirname(os.path.abspath(__file__))
#
#
# _backend = load(name='_fast_lanczos',
#                 sources=[
#                     module_path + "/cuda_lanczos/lanczos.cu",
#                     module_path + "/ext.cpp"]
#                 )
#
# __all__ = ['_backend']


from splatwizard._cmod.common import load
from .config import NAME, NVCC_FLAGS, CXX_FLAGS, EXTRA_INCLUDE_PATHS, SOURCES


_C = load(name=NAME,
                extra_cflags=CXX_FLAGS,
                extra_cuda_cflags=NVCC_FLAGS,
                sources=SOURCES,
                extra_include_paths=EXTRA_INCLUDE_PATHS,
                )

__all__ = ['_C']