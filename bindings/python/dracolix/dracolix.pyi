"""Type stubs for the DracoLIX Python API (generated alongside the extension)."""

from collections.abc import Sequence
from typing import Any, TypeAlias

# --------------------------------------------------------------------------
# Dtypes
# --------------------------------------------------------------------------
class DType:
    """A DracoLIX data type. Instances are the module singletons
    ``dlx.f32``, ``dlx.f64``, ``dlx.i32``, ``dlx.i64``, ``dlx.bool_``."""

    name: str
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

f32: DType
f64: DType
i32: DType
i64: DType
bool_: DType

Scalar: TypeAlias = bool | int | float

# --------------------------------------------------------------------------
# Array
# --------------------------------------------------------------------------
class Array:
    """A dense N-dimensional DracoLIX array.

    Promotion rules for binary operations (``+ - *``).
      * bool and integer mix -> the integer type (i64 if either side is i64,
        otherwise i32); ``True`` behaves as ``1``.
      * any int/float mix -> float (f64 if either side is f64 or i64, else f32).
      * ``/`` (true division) always returns f64 (or f32 when both sides are
        f32); integer division therefore returns floats, like NumPy.

    Bool arrays are promoted to i32 before arithmetic (``True`` == ``1``);
    comparisons always return bool arrays. ``@`` (matmul) rejects bool arrays.
    """

    dtype: DType
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    ndim: int
    size: int
    itemsize: int
    layout: str  # "row-major" | "col-major"
    T: Array  # 2-D transpose

    def __len__(self) -> int: ...
    def __bool__(self) -> bool:
        """Single-element arrays convert to the element; multi-element raises."""

    def __iter__(self) -> Any: ...  # yields scalars over the flattened array
    def __getitem__(
        self, key: int | slice | tuple[int | slice, ...]
    ) -> Scalar | Array: ...
    def __setitem__(self, key: int | tuple[int, ...], value: Scalar) -> None:
        """Integer assignment only; slice assignment is not supported yet."""

    # reductions
    def sum(self) -> Scalar: ...
    def sum(self, axis: int) -> Array: ...
    def min(self) -> Scalar: ...
    def min(self, axis: int) -> Array: ...
    def max(self) -> Scalar: ...
    def max(self, axis: int) -> Array: ...
    def mean(self) -> float: ...
    def mean(self, axis: int) -> Array: ...

    # arithmetic dunders
    def __add__(self, other: Array | Scalar) -> Array: ...
    def __radd__(self, other: Array | Scalar) -> Array: ...
    def __sub__(self, other: Array | Scalar) -> Array: ...
    def __rsub__(self, other: Array | Scalar) -> Array: ...
    def __mul__(self, other: Array | Scalar) -> Array: ...
    def __rmul__(self, other: Array | Scalar) -> Array: ...
    def __truediv__(self, other: Array | Scalar) -> Array: ...
    def __rtruediv__(self, other: Array | Scalar) -> Array: ...
    def __neg__(self) -> Array: ...
    def __matmul__(self, other: Array) -> Array | Scalar: ...
    def __eq__(self, other: Array | Scalar) -> Array: ...  # bool array
    def __ne__(self, other: Array | Scalar) -> Array: ...  # bool array

    # methods
    def copy(self) -> Array: ...
    def astype(self, dtype: DType) -> Array: ...
    def reshape(self, shape: int | Sequence[int]) -> Array: ...
    def transpose(self) -> Array: ...
    def tolist(self) -> Any: ...

# --------------------------------------------------------------------------
# Constructors
# --------------------------------------------------------------------------
def array(data: Array | Sequence[Scalar] | Any, dtype: DType | None = None) -> Array:
    """Create an array, always copying input data.

    ``dtype`` defaults to f64 (any float), i64 (all ints), or bool (all bools).
    Accepts lists/tuples, buffer objects (memoryview, bytes, numpy arrays via
    the buffer protocol), and existing dracolix arrays.
    """

def asarray(data: Array | Sequence[Scalar] | Any, dtype: DType | None = None) -> Array:
    """Like ``array()`` but does not copy an existing dracolix Array."""

def zeros(shape: int | Sequence[int], dtype: DType | None = None) -> Array:
    """Zero-filled array; default dtype is f64."""

def ones(shape: int | Sequence[int], dtype: DType | None = None) -> Array:
    """One-filled array; default dtype is f64."""

def empty(shape: int | Sequence[int], dtype: DType | None = None) -> Array:
    """Uninitialized-memory array (currently == zeros, the core zero-fills)."""

def full(
    shape: int | Sequence[int], fill_value: Scalar, dtype: DType | None = None
) -> Array:
    """Array filled with ``fill_value``. dtype defaults from the fill value."""

def arange(
    start: Scalar, stop: Scalar | None = None, step: Scalar | None = None
) -> Array:
    """Evenly spaced values; int args -> i64, any float arg -> f64."""
