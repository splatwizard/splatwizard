import os

module_path = os.path.dirname(os.path.abspath(__file__))


NAME = '_fused_ssim_cuda'
CXX_FLAGS = []
NVCC_FLAGS = []
SOURCES = [
    module_path + "/ssim.cu",
    module_path + "/ext.cpp"
]
EXTRA_INCLUDE_PATHS = []
LD_DIRS = []
LIBRARIES = []

