"""Python binding tests for the dracolix nanobind extension.

The core must work with zero NumPy in the runtime; numpy is only used as an
*optional* cross-check of the PEP 3118 buffer protocol interop.
"""

import math
import struct

import pytest

import dracolix as dlx


# ---------------------------------------------------------------------------
# dtype singletons
# ---------------------------------------------------------------------------
def test_dtype_singletons_are_identical():
    assert dlx.f32 is dlx.f32
    assert dlx.array([1.0]).dtype is dlx.f64
    assert dlx.array([1.0]).dtype == dlx.f64
    assert dlx.f32 != dlx.f64
    assert dlx.f32 == dlx.f32
    assert isinstance(dlx.f64, dlx.DType)
    assert repr(dlx.f64) == "dtype('f64')"
    assert hash(dlx.f64) == hash(dlx.f64)
    assert dlx.f64.name == "f64"


def test_dtype_resolver():
    assert dlx.dtype(dlx.f32) is dlx.f32
    assert dlx.dtype(dlx.array([3])) is dlx.i64
    assert dlx.dtype("bool_") is dlx.bool_
    with pytest.raises(ValueError):
        dlx.dtype("nope")


# ---------------------------------------------------------------------------
# construction from Python sequences
# ---------------------------------------------------------------------------
def test_array_from_list_dtype_inference():
    a = dlx.array([1, 2, 3])
    assert a.dtype is dlx.i64
    a = dlx.array([1.0, 2.0])
    assert a.dtype is dlx.f64
    a = dlx.array([True, False])
    assert a.dtype is dlx.bool_
    a = dlx.array([1, 2.5])
    assert a.dtype is dlx.f64


def test_array_shape_and_metadata():
    a = dlx.array([[1, 2, 3], [4, 5, 6]])
    assert a.shape == (2, 3)
    assert a.ndim == 2
    assert a.size == 6
    assert a.strides == (3, 1)
    assert a.itemsize == 8
    assert a.layout == "row-major"
    assert a.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_array_ragged_rejected():
    with pytest.raises(ValueError):
        dlx.array([[1, 2], [3]])


def test_array_explicit_dtype():
    a = dlx.array([1, 2, 3], dtype=dlx.f32)
    assert a.dtype is dlx.f32
    b = dlx.array([1.0, 2.0], dtype=dlx.i32)  # integral floats ok
    assert b.dtype is dlx.i32
    assert b.tolist() == [1, 2]
    with pytest.raises(TypeError):
        dlx.array([1.5, 2.5], dtype=dlx.i32)  # non-integral float -> error
    with pytest.raises(TypeError):
        dlx.array([1, 2], dtype="f64")


def test_asarray_no_copy():
    a = dlx.array([1, 2, 3])
    assert dlx.asarray(a) is a
    b = dlx.asarray(a, dtype=dlx.f64)
    assert b is not a
    assert b.dtype is dlx.f64
    c = dlx.array(a)
    assert c is not a
    assert c.tolist() == a.tolist()


def test_constructors():
    assert dlx.zeros((2, 3)).tolist() == [[0, 0, 0], [0, 0, 0]]
    assert dlx.zeros(3).dtype is dlx.f64
    assert dlx.ones((2, 2), dtype=dlx.i64).tolist() == [[1, 1], [1, 1]]
    assert dlx.full((2, 2), 7, dtype=dlx.i32).tolist() == [[7, 7], [7, 7]]
    assert dlx.full((3,), 2.5).dtype is dlx.f64
    assert dlx.empty((2, 2)).tolist() == [[0, 0], [0, 0]]  # zero-filled for now
    assert dlx.arange(5).tolist() == [0, 1, 2, 3, 4]
    assert dlx.arange(2, 8, 2).tolist() == [2, 4, 6]
    assert dlx.arange(5, 0, -2).tolist() == [5, 3, 1]
    assert dlx.arange(0.0, 1.0, 0.25).dtype is dlx.f64
    assert dlx.arange(3).dtype is dlx.i64


# ---------------------------------------------------------------------------
# buffer protocol: export (memoryview / numpy)
# ---------------------------------------------------------------------------
def test_buffer_export_memoryview():
    a = dlx.array([1.0, 2.0, 3.0])
    mv = memoryview(a)
    assert mv.format == "d"
    assert mv.shape == (3,)
    assert bytes(mv) == struct.pack("3d", 1.0, 2.0, 3.0)


def test_buffer_export_multiindex_int():
    a = dlx.array([[1, 2, 3], [4, 5, 6]], dtype=dlx.i32)
    mv = memoryview(a)
    assert mv.ndim == 2
    assert mv.shape == (2, 3)
    assert mv.format == "i"
    assert mv[0, 1] == 2
    assert mv[1, 2] == 6


def test_buffer_export_bool():
    mv = memoryview(dlx.array([True, True, False], dtype=dlx.bool_))
    assert list(mv) == [True, True, False]


def test_buffer_import_memoryview():
    mv = memoryview(struct.pack("3d", 9.0, 8.0, 7.0)).cast("d")
    a = dlx.array(mv)
    assert a.dtype is dlx.f64
    assert a.tolist() == [9.0, 8.0, 7.0]


def test_buffer_import_bytes():
    a = dlx.array(b"\x01\x02\x03")
    assert a.dtype is dlx.i32
    assert a.tolist() == [1, 2, 3]


def test_numpy_interop():
    np = pytest.importorskip("numpy")

    a = dlx.array([[1.0, 2.0], [3.0, 4.0]])
    n = np.asarray(a)
    assert n.shape == (2, 2)
    assert n.dtype == np.float64
    np.testing.assert_array_equal(n, [[1.0, 2.0], [3.0, 4.0]])

    src = np.arange(6, dtype=np.int64).reshape(2, 3)
    b = dlx.array(src)
    assert b.dtype is dlx.i64
    assert b.tolist() == src.tolist()

    c = dlx.array(np.float32([1.0, 2.0]))
    assert c.dtype is dlx.f32


# ---------------------------------------------------------------------------
# arithmetic + broadcasting + promotion
# ---------------------------------------------------------------------------
def test_arithmetic():
    a = dlx.array([1, 2, 3], dtype=dlx.i64)
    assert (a + 1).tolist() == [2, 3, 4]
    assert (a - 1).tolist() == [0, 1, 2]
    assert (a * 2).tolist() == [2, 4, 6]
    assert (a - dlx.array([1, 1, 1])).tolist() == [0, 1, 2]
    assert (1 + a).tolist() == [2, 3, 4]
    assert (6 - a).tolist() == [5, 4, 3]
    assert (-a).tolist() == [-1, -2, -3]


def test_truediv_is_float():
    a = dlx.array([1, 2, 3], dtype=dlx.i64)
    r = a / 2
    assert r.dtype is dlx.f64
    assert r.tolist() == [0.5, 1.0, 1.5]
    f = dlx.array([1.0, 2.0], dtype=dlx.f32) / dlx.array([4.0, 5.0], dtype=dlx.f32)
    assert f.dtype is dlx.f32


def test_promotion():
    i = dlx.array([1, 2], dtype=dlx.i64)
    f = dlx.array([0.5, 1.5], dtype=dlx.f64)
    assert (i + f).dtype is dlx.f64
    small = dlx.array([1, 2], dtype=dlx.i32)
    assert (small + i).dtype is dlx.i64
    assert (small + dlx.array([1.0, 2.0], dtype=dlx.f32)).dtype is dlx.f32
    assert (dlx.array([True]) + dlx.array([1])).dtype is dlx.i64


def test_broadcasting():
    a = dlx.array([[1, 2, 3], [4, 5, 6]])
    b = dlx.array([10, 20, 30])
    assert (a + b).tolist() == [[11, 22, 33], [14, 25, 36]]
    assert (a * 2).shape == (2, 3)


def test_broadcast_mismatch_raises():
    a = dlx.array([1, 2, 3])
    b = dlx.array([1, 2])
    with pytest.raises(ValueError):
        a + b


def test_unsupported_operand_returns_operators_notimplemented():
    a = dlx.array([1, 2, 3])
    with pytest.raises(TypeError):
        a + "nope"
    assert (a == "nope") is False


def test_matmul_and_dot():
    A = dlx.array([[1.0, 2.0], [3.0, 4.0]])
    B = dlx.array([[5.0, 6.0], [7.0, 8.0]])
    C = A @ B
    assert C.tolist() == [[19.0, 22.0], [43.0, 50.0]]
    v = dlx.array([1.0, 2.0])
    w = dlx.array([3.0, 4.0])
    assert v @ w == 11.0
    with pytest.raises(TypeError):
        dlx.array([True, False]) @ dlx.array([True, False])


# ---------------------------------------------------------------------------
# comparisons
# ---------------------------------------------------------------------------
def test_comparisons():
    a = dlx.array([1, 2, 3], dtype=dlx.i64)
    eq = a == dlx.array([1, 9, 3])
    assert eq.dtype is dlx.bool_
    assert eq.tolist() == [True, False, True]
    assert (a == 2).tolist() == [False, True, False]
    assert (a != 2).tolist() == [True, False, True]
    assert (dlx.array([True]) == dlx.array([True])).tolist()[0]


# ---------------------------------------------------------------------------
# indexing & iteration
# ---------------------------------------------------------------------------
def test_indexing_int():
    a = dlx.array([[1, 2, 3], [4, 5, 6]])
    assert a[0, 1] == 2
    assert a[1, 2] == 6
    assert a[-1, -1] == 6
    with pytest.raises(IndexError):
        a[0, 3]
    with pytest.raises(IndexError):
        a[2, 0]


def test_indexing_slice():
    a = dlx.array([0, 1, 2, 3, 4, 5])
    assert a[1:4].tolist() == [1, 2, 3]
    assert a[::2].tolist() == [0, 2, 4]
    assert a[::-1].tolist() == [5, 4, 3, 2, 1, 0]
    assert a[5:1:-2].tolist() == [5, 3]
    assert a[:].tolist() == a.tolist()
    assert a[3] == 3  # scalar return, not array
    assert not isinstance(a[3], dlx.Array)


def test_indexing_mixed_2d():
    a = dlx.array([[1, 2, 3], [4, 5, 6]])
    assert a[0].tolist() == [1, 2, 3]  # row selection
    assert a[:, 1].tolist() == [2, 5]
    assert a[1, :].tolist() == [4, 5, 6]
    with pytest.raises(IndexError):
        a[0, 0, 0]  # too many indices


def test_setitem():
    a = dlx.array([1, 2, 3])
    a[0] = 9
    a[-1] = 7
    assert a.tolist() == [9, 2, 7]
    b = dlx.array([[0, 0], [0, 0]], dtype=dlx.f64)
    b[1, 0] = 2.5
    assert b[1, 0] == 2.5
    with pytest.raises(TypeError):
        a[0:2] = [1, 2]


def test_iteration():
    a = dlx.array([[1, 2], [3, 4]])
    assert list(a) == [1, 2, 3, 4]
    assert [x for x in dlx.arange(3)] == [0, 1, 2]


def test_len_and_truth():
    a = dlx.array([1, 2, 3])
    assert len(a) == 3
    with pytest.raises(TypeError):
        bool(a)
    assert bool(dlx.array([1])) is True
    assert bool(dlx.array([0])) is False


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------
def test_reductions_global():
    a = dlx.array([[1.0, 2.0], [3.0, 4.0]])
    assert a.sum() == 10.0
    assert a.min() == 1.0
    assert a.max() == 4.0
    assert a.mean() == 2.5
    assert dlx.array([True, True, False]).sum() == 2
    assert dlx.array([True, False]).mean() == 0.5
    assert dlx.array([3, 1, 2], dtype=dlx.i32).min() == 1


def test_reductions_axis():
    a = dlx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert a.sum(axis=0).tolist() == [5.0, 7.0, 9.0]
    assert a.sum(axis=1).tolist() == [6.0, 15.0]
    assert a.sum(axis=-1).tolist() == [6.0, 15.0]
    assert a.min(axis=0).tolist() == [1.0, 2.0, 3.0]
    assert a.max(axis=1).tolist() == [3.0, 6.0]
    assert a.mean(axis=0).tolist() == [2.5, 3.5, 4.5]
    with pytest.raises(IndexError):
        a.sum(axis=5)


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------
def test_reshape_transpose_astype_copy():
    a = dlx.arange(6).reshape((2, 3))
    assert a.shape == (2, 3)
    assert a.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert a.reshape(6).tolist() == dlx.arange(6).tolist()
    assert a.transpose().tolist() == [[0, 3], [1, 4], [2, 5]]
    assert a.T.tolist() == a.transpose().tolist()
    c = dlx.array([1, 2, 3]).astype(dlx.f32)
    assert c.dtype is dlx.f32
    assert c.tolist() == [1.0, 2.0, 3.0]
    d = a.copy()
    assert d is not a
    assert d.tolist() == a.tolist()


def test_repr():
    r = repr(dlx.array([1, 2, 3]))
    assert "array(" in r and "dtype=i64" in r


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------
def test_invalid_inputs():
    with pytest.raises(TypeError):
        dlx.array(5)
    with pytest.raises(ValueError):
        dlx.array([])
    with pytest.raises(ValueError):
        dlx.zeros(0)
    with pytest.raises(ValueError):
        dlx.zeros((2, -1))
    with pytest.raises(ValueError):
        dlx.arange(0, 5, 0)