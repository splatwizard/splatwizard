
try:
    from . import _weighted_distance as _C
except ImportError:
    from ._backend import  _C


def weighted_distance(*args, **kwargs):
    return _C.weightedDistance(*args, **kwargs)


