import torch

try:
    from . import _simple_knn as _C
except ImportError:
    from ._backend import  _C



def distCUDA2(points: torch.Tensor):
    return _C.distCUDA2(points)

