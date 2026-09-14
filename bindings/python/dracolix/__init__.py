"""
DracoLIX Python API - an outgoing-only binding to the native C++ core.

Zero-NumPy runtime: the core never depends on numpy. Arrays interoperate with
numpy / memoryview through the PEP 3118 buffer protocol only.
"""

import sys as _sys

from . import _dracolix_nb as _native
from ._dracolix_nb import (
    Array,
    CscMatrix,
    CsrMatrix,
    DType,
    arange,
    array,
    asarray,
    empty,
    f32,
    f64,
    full,
    i32,
    i64,
    ones,
    zeros,
)

# ``dlx.bool_`` is the bool dtype singleton (``bool`` is a Python builtin).
# Note: the compiled module keeps its ``bool`` attribute too - the core
# resolves dtypes through it.
bool_ = _native.bool

# Single source of truth for the version: it lives in the compiled extension.
__version__ = _native.__version__

__all__ = [
    "Array",
    "CscMatrix",
    "CsrMatrix",
    "DType",
    "arange",
    "array",
    "asarray",
    "bool_",
    "empty",
    "f32",
    "f64",
    "full",
    "i32",
    "i64",
    "ones",
    "zeros",
]

_dtype_names = {"f32": f32, "f64": f64, "i32": i32, "i64": i64, "bool_": bool_}
_dtype_by_name = {k: v for k, v in _dtype_names.items()}
del _dtype_names


def dtype(x) -> DType:
    """Return the dtype of an Array, or resolve a dtype name/singleton."""
    if isinstance(x, Array):
        return x.dtype
    if isinstance(x, DType):
        return x
    if isinstance(x, str):
        try:
            return _dtype_by_name[x]
        except KeyError:
            raise ValueError(f"unknown dtype name: {x!r}")
    raise TypeError("expected Array, DType, or a dtype name string")


def _version_info() -> tuple:
    return tuple(int(p) for p in __version__.split("."))


__version_info__ = _version_info()
del _version_info
del _sys
