
import os.path as osp
import os

module_path = os.path.dirname(os.path.abspath(__file__))
# include_dirs = [osp.join(module_path, "include")]


NAME = '_rans_ops'
CXX_FLAGS = ['-O2']
NVCC_FLAGS = ['-O2']
SOURCES = [
    module_path + '/ops.cpp',
]
EXTRA_INCLUDE_PATHS = []
LD_DIRS = []
LIBRARIES = []

