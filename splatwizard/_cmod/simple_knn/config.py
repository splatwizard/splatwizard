import os

module_path = os.path.dirname(os.path.abspath(__file__))



NAME = '_simple_knn'
CXX_FLAGS = []
NVCC_FLAGS = []

if os.name == 'nt':
    NVCC_FLAGS.append("/wd4624")

SOURCES = [
    module_path + "/spatial.cu",
    module_path + "/simple_knn.cu",
    module_path + "/ext.cpp"
]
EXTRA_INCLUDE_PATHS = []
LD_DIRS = []
LIBRARIES = []

