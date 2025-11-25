#
# from torch.utils.cpp_extension import load
# import os
#
# module_path = os.path.dirname(os.path.abspath(__file__))
# cxx_compiler_flags = []
#
# if os.name == 'nt':
#     cxx_compiler_flags.append("/wd4624")
#
#
# _backend = load(name='_simple_knn',
#                 extra_cflags=cxx_compiler_flags,
#                 # extra_cuda_cflags=["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")],
#                 sources=[
#                     module_path + "/spatial.cu",
#                     module_path + "/simple_knn.cu",
#                     module_path + "/ext.cpp"
#                 ],
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
