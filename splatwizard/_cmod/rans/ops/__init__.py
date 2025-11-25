import torch


try:
    from . import _rans_ops
except ImportError:
    from ._backend import _C as _rans_ops



