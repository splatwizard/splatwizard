import torch


try:
    from . import _arithmetic
except ImportError:
    from ._backend import _C as _arithmetic


def calculate_cdf(
    mean: torch.Tensor ,
    scale: torch.Tensor ,
    Q: torch.Tensor,
    min_value,
    max_value
):
    return _arithmetic.calculate_cdf(mean, scale, Q, min_value, max_value)


def arithmetic_encode(
    sym: torch.Tensor,
    cdf: torch.Tensor,
    chunk_size,
    N,
    Lp
):
    return _arithmetic.arithmetic_encode(sym, cdf, chunk_size, N, Lp)


def arithmetic_decode(
    cdf: torch.Tensor,
    in_cache_all: torch.Tensor,
    in_cnt_all: torch.Tensor,
    chunk_size,
    N,
    Lp):

    return _arithmetic.arithmetic_decode(cdf, in_cache_all, in_cnt_all, chunk_size, N, Lp)

