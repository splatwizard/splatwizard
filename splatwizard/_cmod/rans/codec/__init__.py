try:
    from . import _rans_codec
except ImportError:
    from ._backend import _C as _rans_codec

