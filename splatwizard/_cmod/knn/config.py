import os

module_path = os.path.dirname(os.path.abspath(__file__))



NAME = '_pytorch3d_knn'
CXX_FLAGS = []
NVCC_FLAGS = []

SOURCES = [
    module_path + "/knn/knn.cu",
    module_path + "/knn/knn_cpu.cpp",
    module_path + "/ext.cpp"
]
EXTRA_INCLUDE_PATHS = [module_path , module_path+ "/utils"]
LD_DIRS = []
LIBRARIES = []

