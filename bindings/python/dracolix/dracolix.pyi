"""Type stubs for the DracoLIX Python API (generated alongside the extension)."""

from collections.abc import Sequence
from typing import Any, TypeAlias, overload

# --------------------------------------------------------------------------
# Dtypes
# --------------------------------------------------------------------------
class DType:
    """A DracoLIX data type. Instances are the module singletons
    ``dlx.f32``, ``dlx.f64``, ``dlx.i32``, ``dlx.i64``, ``dlx.c64``,
    ``dlx.c128``, ``dlx.bool_``."""

    name: str
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

f32: DType
f64: DType
i32: DType
i64: DType
c64: DType  # complex64: two float32 components
c128: DType  # complex128: two float64 components
bool_: DType

Scalar: TypeAlias = bool | int | float | complex
# ``complex`` (the builtin type) is accepted as an alias for ``c128``.
DTypeLike: TypeAlias = DType | type[complex]

# --------------------------------------------------------------------------
# Array
# --------------------------------------------------------------------------
class Array:
    """A dense N-dimensional DracoLIX array.

    Promotion rules for binary operations (``+ - *``).
      * bool and integer mix -> the integer type (i64 if either side is i64,
        otherwise i32); ``True`` behaves as ``1``.
      * any int/float mix -> float (f64 if either side is f64 or i64, else f32).
      * any complex mix -> c64, widening to c128 when combined with f64, i64,
        or another c128 operand (NumPy-compatible).
      * ``/`` (true division) always returns f64 (or f32 when both sides are
        f32); integer division therefore returns floats, like NumPy. Complex
        operands keep their promoted complex dtype.

    Bool arrays are promoted to i32 before arithmetic (``True`` == ``1``);
    comparisons always return bool arrays. ``@`` (matmul) rejects bool arrays.

    Complex notes: ``min``/``max`` (and their ``axis`` forms) are undefined
    because complex numbers have no ordering; ``mean`` returns a complex scalar
    and ``norm`` uses the modulus. ``real``/``imag``/``conj``/``abs``/``angle``
    work for every dtype (real arrays report a zero imaginary part and a phase
    of 0 or pi). ``eig``/``svd`` and sparse matrices do not support complex yet.
    """

    dtype: DType
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    ndim: int
    size: int
    itemsize: int
    layout: str  # "row-major" | "col-major" || switch to boolean flag for this?
    T: Array  # 2-D transpose
    real: Array  # real part (a copy)
    imag: Array  # imaginary part (zeros for real dtypes)

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
    @overload
    def sum(self) -> Scalar: ...
    @overload
    def sum(self, axis: int) -> Array: ...
    @overload
    def min(self) -> Scalar: ...
    @overload
    def min(self, axis: int) -> Array: ...
    @overload
    def max(self) -> Scalar: ...
    @overload
    def max(self, axis: int) -> Array: ...
    @overload
    def mean(self) -> Scalar: ...
    @overload
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
    def __eq__(self, other: Array | Scalar) -> Array: ...  # bool array # type: ignore
    def __ne__(self, other: Array | Scalar) -> Array: ...  # bool array # type: ignore

    # methods
    def copy(self) -> Array: ...
    def astype(self, dtype: DTypeLike) -> Array: ...
    def reshape(self, shape: int | Sequence[int]) -> Array: ...
    def transpose(self) -> Array: ...
    def tolist(self) -> Any: ...
    def conj(self) -> Array:
        """Element-wise complex conjugate (identity for real dtypes)."""
    def conjugate(self) -> Array:
        """Alias of :meth:`conj`."""
    def abs(self) -> Array:
        """Element-wise modulus; dtype is the underlying real type."""
    def __abs__(self) -> Array:
        """Same as :meth:`abs`."""
    def angle(self) -> Array:
        """Element-wise phase in radians, in (-pi, pi]."""

# --------------------------------------------------------------------------
# Constructors
# --------------------------------------------------------------------------
def array(data: Array | Sequence[Scalar] | Any, dtype: DTypeLike | None = None) -> Array:
    """Create an array, always copying input data.

    ``dtype`` defaults to c128 (any complex), f64 (any float), i64 (all ints), or bool (all bools). The builtin ``complex`` type is accepted as an alias for c128.
    Accepts lists/tuples, buffer objects (memoryview, bytes, numpy arrays via
    the buffer protocol), and existing dracolix arrays.
    """

def asarray(data: Array | Sequence[Scalar] | Any, dtype: DTypeLike | None = None) -> Array:
    """Like ``array()`` but does not copy an existing dracolix Array."""

def zeros(shape: int | Sequence[int], dtype: DTypeLike | None = None) -> Array:
    """Zero-filled array; default dtype is f64."""

def ones(shape: int | Sequence[int], dtype: DTypeLike | None = None) -> Array:
    """One-filled array; default dtype is f64."""

def empty(shape: int | Sequence[int], dtype: DTypeLike | None = None) -> Array:
    """Uninitialized-memory array (currently == zeros, the core zero-fills)."""

def full(
    shape: int | Sequence[int], fill_value: Scalar, dtype: DTypeLike | None = None
) -> Array:
    """Array filled with ``fill_value``. dtype defaults from the fill value."""

def arange(
    start: Scalar, stop: Scalar | None = None, step: Scalar | None = None
) -> Array:
    """Evenly spaced values; int args -> i64, any float arg -> f64."""

# --------------------------------------------------------------------------
# Sparse matrices
# --------------------------------------------------------------------------
class CsrMatrix:
    """Compressed sparse row (CSR) matrix.

    Only sparse-compatible dtypes (f32/f64/i32/i64) are allowed; bool is not
    supported. Matmul follows the dense operands' dtype (both operands must match).
    ``A @ B`` accepts another sparse matrix, a dense 1-D/2-D array (numpy or
    dracolix Array, lists, or any buffer), and honors ``numpy.ndarray @ A``.
    """

    dtype: DType
    shape: tuple[int, int]
    rows: int
    cols: int
    nnz: int
    row_ptr: Array  # length rows + 1, dtype i64
    col_ind: Array  # column of each stored entry, dtype i64
    values: Array  # stored entries
    T: CscMatrix  # CSC of A^T

    def __init__(
        self, data: Array | Sequence[Scalar] | Any, dtype: DType | None = None
    ) -> None: ...
    def to_dense(self) -> Array: ...
    def to_csr(self) -> CsrMatrix: ...
    def to_csc(self) -> CscMatrix: ...
    def at(self, row: int, col: int) -> Scalar: ...
    def __matmul__(
        self, other: CsrMatrix | CscMatrix | Array | Any
    ) -> CsrMatrix | Array: ...
    def solve(self, b: Array | Sequence[Scalar] | Any) -> Array:
        """Solve A x = b with the conjugate-gradient for SPD matrices
        (``A`` must be square symmetric positive-definite)."""
    @staticmethod
    def from_coo(
        rows: int,
        cols: int,
        row_ind: Array | Sequence[int] | Any,
        col_ind: Array | Sequence[int] | Any,
        values: Array | Sequence[Scalar] | Any,
        drop_zeros: bool = True,
    ) -> CsrMatrix:
        """Build from coordinates; duplicate entries are summed, and zero
        entries are dropped when ``drop_zeros`` is True (default)."""

class CscMatrix:
    """Compressed sparse column (CSC) matrix.

    Identical matmul story to :class:`CsrMatrix`; the CSC layout is kept by
    converting to CSR internally for the multiply.
    """

    dtype: DType
    shape: tuple[int, int]
    rows: int
    cols: int
    nnz: int
    col_ptr: Array  # length cols + 1, dtype i64
    row_ind: Array  # row of each stored entry, dtype i64
    values: Array  # stored entries
    T: CsrMatrix  # CSR of A^T

    def __init__(
        self, data: Array | Sequence[Scalar] | Any, dtype: DType | None = None
    ) -> None: ...
    def to_dense(self) -> Array: ...
    def to_csr(self) -> CsrMatrix: ...
    def to_csc(self) -> CscMatrix: ...
    def at(self, row: int, col: int) -> Scalar: ...
    def __matmul__(
        self, other: CsrMatrix | CscMatrix | Array | Any
    ) -> CsrMatrix | Array: ...
    def solve(self, b: Array | Sequence[Scalar] | Any) -> Array:
        """Solve A x = b with the conjugate-gradient for SPD matrices
        (``A`` must be square symmetric positive-definite)."""
    @staticmethod
    def from_coo(
        rows: int,
        cols: int,
        row_ind: Array | Sequence[int] | Any,
        col_ind: Array | Sequence[int] | Any,
        values: Array | Sequence[Scalar] | Any,
        drop_zeros: bool = True,
    ) -> CscMatrix:
        """Build from coordinates; duplicate entries are summed, and zero
        entries are dropped when ``drop_zeros`` is True (default)."""

# --------------------------------------------------------------------------
# Linear algebra
# --------------------------------------------------------------------------
def matmul(A: Array | Any, B: Array | Any) -> Array | Scalar:
    """Matrix product. 2-D and batched N-D; ``A @ B`` with 1-D vectors
    returns a dot product scalar. Requires matching non-bool dtypes."""

def matvec(A: Array | Any, x: Array | Any) -> Array:
    """Matrix-vector product: A (..., M,N) @ x (..., N) or (N,) -> (..., M)."""

def dot(a: Array | Any, b: Array | Any) -> Scalar:
    """Inner product of two 1-D arrays; requires matching non-bool dtypes."""

def norm(a: Array, p: int = 2) -> float:
    """Vector norm: p=0 (max-abs / inf), p=1, p=2. Requires 1-D input."""

# --------------------------------------------------------------------------
# Complex helpers (defined for every dtype; see the Array notes)
# --------------------------------------------------------------------------
def real(a: Array) -> Array:
    """Element-wise real part. Complex input returns the underlying float dtype."""

def imag(a: Array) -> Array:
    """Element-wise imaginary part (all zeros for real/integer input)."""

def conj(a: Array) -> Array:
    """Element-wise complex conjugate (identity for real/integer input)."""

def conjugate(a: Array) -> Array:
    """Alias of :func:`conj`."""

def abs(a: Array) -> Array:
    """Element-wise modulus / absolute value."""

def absolute(a: Array) -> Array:
    """Alias of :func:`abs`."""

def angle(a: Array) -> Array:
    """Element-wise phase in radians, in (-pi, pi]; 0 or pi for real input."""

def diagonal(A: Array) -> Array:
    """Extract the diagonal of a 2-D array."""

def diag(d: Array) -> Array:
    """Build a square diagonal matrix from a 1-D array."""

def lu(A: Array) -> tuple[Array, Array, Array]:
    """LU with partial pivoting: returns ``(L, U, piv)`` where ``piv`` is an
    i64 index array such that row ``i`` of ``P@A`` is row ``piv[i]`` of ``A``
    (``L @ U == P @ A``)."""

def solve(A: Array, b: Array | Any) -> Array:
    """Solve ``A x = b`` via LU. ``b`` may be 1-D or 2-D (multiple RHS)."""

def inv(A: Array) -> Array:
    """Matrix inverse via LU. Raises if ``A`` is singular."""

def inverse(A: Array) -> Array:
    """Alias of :func:`inv`."""

def det(A: Array) -> Scalar:
    """Determinant of a square matrix."""

def determinant(A: Array) -> Scalar:
    """Alias of :func:`det`."""

def rank(A: Array, tol: float = 1e-9) -> int:
    """Numerical rank: number of pivots with |U[i,i]| > tol."""

def qr(A: Array) -> tuple[Array, Array]:
    """Reduced QR (Modified Gram-Schmidt): ``A m x n (m>=n) -> (Q m x n, R n x n)``
    with ``Q`` orthonormal and ``A == Q @ R``. For complex input the inner products use the conjugate (unitary Q)."""

def cholesky(A: Array) -> Array:
    """Lower-triangular Cholesky factor ``L`` with ``A == L @ L.T`` for real
    symmetric, and ``A == L @ L.H`` for complex Hermitian, positive-definite
    ``A``."""

def eig(A: Array) -> tuple[Array, Array]:
    """Symmetric eigendecomposition (Jacobi): ``(values, vectors)`` with
    orthonormal columns. Requires real float input (f32 is promoted to f64);
    complex input raises ``TypeError``. Returns f64 arrays."""

def svd(A: Array) -> tuple[Array, Array, Array]:
    """SVD via ``A^T A`` eigens: ``(U, S, Vt)``, ``A == U @ diag(S) @ Vt``.
    Requires real float input (f32 promoted to f64); complex input raises
    ``TypeError``. Returns f64 arrays."""

def cg_solve(A: CsrMatrix, b: Array | Sequence[Scalar] | Any) -> Array:
    """Conjugate-gradient solver for ``A x = b`` on a sparse square SPD CSR
    matrix. See :meth:`CsrMatrix.solve`."""


