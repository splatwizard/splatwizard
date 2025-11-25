# import os
# from pathlib import Path
# from torch.utils.cpp_extension import load
#
#
# module_path = Path(__file__)
# module_path_str = str(module_path.parent.absolute())
# glm_dir = module_path.parent.parent.parent / "third_party" / "glm"
#
# _backend = load(name='_taming_rasterizer',
#                 # extra_cflags=c_flags,
#                 extra_cuda_cflags=["-I" + str(glm_dir.absolute())],
#                 sources=[
#                     module_path_str + "/cuda_rasterizer/rasterizer_impl.cu",
#                     module_path_str + "/cuda_rasterizer/forward.cu",
#                     module_path_str + "/cuda_rasterizer/backward.cu",
#                     module_path_str + "/cuda_rasterizer/adam.cu",
#                     module_path_str + "/rasterize_points.cu",
#                     module_path_str + "/conv.cu",
#                     module_path_str + "/ext.cpp"
#                 ],
#                 )
#
# __all__ = ['_backend']
#
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