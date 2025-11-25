
import os.path as osp
import os

module_path = os.path.dirname(os.path.abspath(__file__))
include_dirs = [osp.join(module_path, "rans"), osp.join(module_path, "ryg_rans")]


NAME = '_rans_codec'
CXX_FLAGS = ['-O3']
NVCC_FLAGS = ['-O2']
SOURCES = [
    module_path + '/rans/rans_interface.cpp',
]
EXTRA_INCLUDE_PATHS = include_dirs
LD_DIRS = []
LIBRARIES = []

