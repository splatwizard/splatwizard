
import os.path as osp
import os

module_path = os.path.dirname(os.path.abspath(__file__))
include_dirs = [osp.join(module_path, "include")]


NAME = '_arithmetic'
CXX_FLAGS = ['-O2']
NVCC_FLAGS = ['-O2']
SOURCES = [
    module_path + '/arithmetic.cpp',
    module_path + '/arithmetic_kernel.cu'
]
EXTRA_INCLUDE_PATHS = include_dirs
LD_DIRS = []
LIBRARIES = []

