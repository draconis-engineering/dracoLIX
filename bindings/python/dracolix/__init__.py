"""
DracoLIX Python API - an outgoing-only binding to the native C++ core.

Arrays interoperate with numpy / memoryview through the PEP 3118 buffer protocol.
"""

import sys as _sys

from . import _dracolix_nb as _native
from ._dracolix_nb import (  # type: ignore
    Array,
    CscMatrix,
    CsrMatrix,
    DType,
    abs,
    absolute,
    acos,
    angle,
    arange,
    array,
    asarray,
    asin,
    atan,
    c64,
    c128,
    ceil,
    cg_solve,
    cholesky,
    conj,
    conjugate,
    cos,
    cosh,
    det,
    determinant,
    diag,
    diagonal,
    dot,
    eig,
    empty,
    exp,
    expm1,
    f32,
    f64,
    floor,
    full,
    i8,
    i16,
    i32,
    i64,
    imag,
    inv,
    inverse,
    isfinite,
    isinf,
    isnan,
    linspace,
    log,
    log1p,
    log2,
    log10,
    lu,
    matmul,
    matvec,
    norm,
    ones,
    qr,
    rank,
    real,
    sin,
    sinh,
    solve,
    square,
    sqrt,
    svd,
    tan,
    tanh,
    trunc,
    u8,
    u16,
    u32,
    u64,
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
    "abs",
    "absolute",
    "acos",
    "angle",
    "arange",
    "array",
    "asarray",
    "asin",
    "atan",
    "bool_",
    "c64",
    "c128",
    "ceil",
    "cg_solve",
    "cholesky",
    "conj",
    "conjugate",
    "cos",
    "cosh",
    "det",
    "determinant",
    "diag",
    "diagonal",
    "dot",
    "eig",
    "empty",
    "exp",
    "expm1",
    "f32",
    "f64",
    "floor",
    "full",
    "i8",
    "i16",
    "i32",
    "i64",
    "imag",
    "inv",
    "inverse",
    "isfinite",
    "isinf",
    "isnan",
    "linspace",
    "log",
    "log1p",
    "log2",
    "log10",
    "lu",
    "matmul",
    "matvec",
    "norm",
    "ones",
    "qr",
    "rank",
    "real",
    "sin",
    "sinh",
    "solve",
    "square",
    "sqrt",
    "svd",
    "tan",
    "tanh",
    "trunc",
    "u8",
    "u16",
    "u32",
    "u64",
    "zeros",
]

_dtype_names = {
    "f32": f32,
    "f64": f64,
    "i8": i8,
    "i16": i16,
    "i32": i32,
    "i64": i64,
    "u8": u8,
    "u16": u16,
    "u32": u32,
    "u64": u64,
    "c64": c64,
    "c128": c128,
    "bool_": bool_,
}
_dtype_by_name = {k: v for k, v in _dtype_names.items()}
# NumPy-style aliases.
_dtype_by_name.update(
    {
        "int8": i8,
        "int16": i16,
        "int32": i32,
        "int64": i64,
        "uint8": u8,
        "uint16": u16,
        "uint32": u32,
        "uint64": u64,
        "float32": f32,
        "float64": f64,
        "complex64": c64,
        "complex128": c128,
        "complex": c128,
        "bool": bool_,
    }
)
del _dtype_names


def dtype(x) -> DType:
    """Return the dtype of an Array, or resolve a dtype name/singleton."""
    if isinstance(x, Array):
        return x.dtype
    if isinstance(x, DType):
        return x
    if x is complex:
        return c128
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
