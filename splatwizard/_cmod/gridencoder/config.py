import os


_src_path = os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    '-O3', '-std=c++17',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
]

if os.name == "posix":
    c_flags = ['-O3', '-std=c++17']
elif os.name == "nt":
    c_flags = ['/O2', '/std:c++17']

    # find cl.exe
    def find_cl_path():
        import glob
        for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            paths = sorted(glob.glob(r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition), reverse=True)
            if paths:
                return paths[0]

    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
        os.environ["PATH"] += ";" + cl_path

else:
    c_flags = []

module_path = os.path.dirname(os.path.abspath(__file__))

NAME = '_gridencoder'
CXX_FLAGS = c_flags
NVCC_FLAGS = nvcc_flags
SOURCES = [
    module_path + "/src/gridencoder.cu",
    module_path + "/src/bindings.cpp"
]
EXTRA_INCLUDE_PATHS = []
LD_DIRS = []
LIBRARIES = []

