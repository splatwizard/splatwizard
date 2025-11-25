import os

module_path = os.path.dirname(os.path.abspath(__file__))


NAME = '_fast_lanczos'
CXX_FLAGS = []
NVCC_FLAGS = []
SOURCES = [
    module_path + "/cuda_lanczos/lanczos.cu",
    module_path + "/ext.cpp"
]
EXTRA_INCLUDE_PATHS = []
LD_DIRS = []
LIBRARIES = []

