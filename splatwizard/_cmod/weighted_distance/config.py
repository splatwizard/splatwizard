import os

module_path = os.path.dirname(os.path.abspath(__file__))



NAME = '_weighted_distance'
CXX_FLAGS = []
NVCC_FLAGS = []

if os.name == 'nt':
    NVCC_FLAGS.append("/wd4624")

SOURCES = [
    module_path + "/weighted_distance.cu",
    module_path + "/ext.cpp"
]
EXTRA_INCLUDE_PATHS = []
LD_DIRS = []
LIBRARIES = []

