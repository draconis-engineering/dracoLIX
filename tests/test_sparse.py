"""Python binding tests for dracolix sparse matrices (CSR / CSC).

The core must work with zero NumPy; numpy is only used as an optional
cross-check of the PEP 3118 buffer protocol interop.
"""

import dracolix as dlx
import pytest

np = pytest.importorskip("numpy")


A = [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]


# ---------------------------------------------------------------------------
# construction from dense
# ---------------------------------------------------------------------------
def test_csr_from_dense_preserves_entries():
    s = dlx.CsrMatrix(A)
    assert s.shape == (3, 3)
    assert s.rows == 3
    assert s.cols == 3
    assert s.nnz == 4
    assert s.dtype is dlx.f64
    assert s.to_dense().tolist() == A


def test_csc_from_dense_preserves_entries():
    c = dlx.CscMatrix(A)
    assert c.shape == (3, 3)
    assert c.nnz == 4
    assert c.to_dense().tolist() == A


def test_sparse_dtype_variants():
    for np_dtype, dlx_dtype in [
        (np.float32, dlx.f32),
        (np.float64, dlx.f64),
        (np.int32, dlx.i32),
        (np.int64, dlx.i64),
    ]:
        data = np.array(A, dtype=np_dtype)
        s = dlx.CsrMatrix(data)
        assert s.dtype is dlx_dtype
        c = dlx.CscMatrix(data)
        assert c.dtype is dlx_dtype


def test_sparse_rejects_bool():
    with pytest.raises(TypeError):
        dlx.CsrMatrix(np.array([[True, False], [False, True]]))
    with pytest.raises(TypeError):
        dlx.CscMatrix(np.array([[True, False], [False, True]]))


# ---------------------------------------------------------------------------
# from_coo
# ---------------------------------------------------------------------------
def test_from_coo_csr():
    s = dlx.CsrMatrix.from_coo(3, 3, [0, 1, 2, 0], [1, 2, 0, 2], [5.0, 7.0, 9.0, 3.0])
    assert s.to_dense().tolist() == [[0.0, 5.0, 3.0], [0.0, 0.0, 7.0], [9.0, 0.0, 0.0]]
    assert s.nnz == 4


def test_from_coo_csc():
    c = dlx.CscMatrix.from_coo(3, 3, [0, 1, 2, 0], [1, 2, 0, 2], [5.0, 7.0, 9.0, 3.0])
    assert c.to_dense().tolist() == [[0.0, 5.0, 3.0], [0.0, 0.0, 7.0], [9.0, 0.0, 0.0]]
    assert c.nnz == 4


def test_from_coo_merges_duplicates():
    s = dlx.CsrMatrix.from_coo(2, 2, [0, 0, 1], [1, 1, 1], [2.0, 3.0, 4.0])
    assert s.to_dense().tolist() == [[0.0, 5.0], [0.0, 4.0]]
    assert s.nnz == 2


def test_from_coo_drops_zeros_by_default():
    s = dlx.CsrMatrix.from_coo(2, 2, [0, 1, 1], [0, 1, 1], [1.0, 5.0, -5.0])
    assert s.to_dense().tolist() == [[1.0, 0.0], [0.0, 0.0]]
    assert s.nnz == 1
    c = dlx.CscMatrix.from_coo(2, 2, [0, 1, 1], [0, 1, 1], [1.0, 5.0, -5.0], False)
    assert c.nnz == 2
    assert c.at(1, 1) == 0.0


def test_from_coo_out_of_bounds(monkeypatch):
    with pytest.raises(IndexError):
        dlx.CsrMatrix.from_coo(2, 2, [3], [0], [1.0])


# ---------------------------------------------------------------------------
# access
# ---------------------------------------------------------------------------
def test_at():
    s = dlx.CsrMatrix(A)
    assert s.at(0, 0) == 1.0
    assert s.at(0, 2) == 2.0
    assert s.at(1, 1) == 3.0
    assert s.at(0, 1) == 0.0


def test_storage_arrays():
    s = dlx.CsrMatrix([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    assert s.row_ptr.tolist() == [0, 2, 3, 4]
    assert s.col_ind.tolist() == [0, 2, 1, 2]
    assert s.values.tolist() == [1.0, 2.0, 3.0, 4.0]
    c = s.to_csc()
    assert c.col_ptr.tolist() == [0, 1, 2, 4]
    assert c.row_ind.tolist() == [0, 1, 0, 2]
    assert c.values.tolist() == [1.0, 3.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# conversions and transpose
# ---------------------------------------------------------------------------
def test_to_csc_is_same_matrix():
    s = dlx.CsrMatrix(A)
    c = s.to_csc()
    assert isinstance(c, dlx.CscMatrix)
    assert c.to_dense().tolist() == A
    assert c.to_csr().to_dense().tolist() == A


def test_to_csr_is_same_matrix():
    c = dlx.CscMatrix(A)
    s = c.to_csr()
    assert isinstance(s, dlx.CsrMatrix)
    assert s.to_dense().tolist() == A


def test_transpose():
    s = dlx.CsrMatrix(A)
    t = s.T
    assert t.to_dense().tolist() == np.array(A).T.tolist()
    c = s.to_csc()
    assert c.T.to_dense().tolist() == np.array(A).T.tolist()


def test_roundtrip_dense_sparse_dense():
    rng = np.random.default_rng(7)
    m = rng.random((5, 6))
    m[m < 0.5] = 0.0
    from_csr = dlx.CsrMatrix(m).to_dense()
    from_csc = dlx.CscMatrix(m).to_dense()
    np.testing.assert_allclose(from_csr.tolist(), m)
    np.testing.assert_allclose(from_csc.tolist(), m)


# ---------------------------------------------------------------------------
# matmul
# ---------------------------------------------------------------------------
def test_sparse_sparse_matmul():
    s = dlx.CsrMatrix(A)
    r = s @ s
    assert isinstance(r, dlx.CsrMatrix)
    expected = np.array(A) @ np.array(A)
    np.testing.assert_allclose(r.to_dense().tolist(), expected)


def test_csc_sparse_matmul():
    c = dlx.CscMatrix(A)
    r = c @ c
    np.testing.assert_allclose(r.to_dense().tolist(), np.array(A) @ np.array(A))
    r2 = c @ dlx.CsrMatrix(A)
    np.testing.assert_allclose(r2.to_dense().tolist(), np.array(A) @ np.array(A))


def test_sparse_dense_matmul():
    s = dlx.CsrMatrix(A)
    r = s @ np.array(A)
    np.testing.assert_allclose(r.tolist(), np.array(A) @ np.array(A))
    r2 = np.array(A) @ s
    np.testing.assert_allclose(r2.tolist(), np.array(A) @ np.array(A))
    r3 = dlx.CscMatrix(A) @ np.array(A)
    np.testing.assert_allclose(r3.tolist(), np.array(A) @ np.array(A))


def test_sparse_vec_matmul():
    s = dlx.CsrMatrix(A)
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose((s @ v).tolist(), np.array(A) @ v)
    np.testing.assert_allclose((s.to_csc() @ v).tolist(), np.array(A) @ v)


def test_matmul_validation():
    s = dlx.CsrMatrix(A)
    v = np.array([1.0, 2.0, 3.0])
    with pytest.raises((ValueError, RuntimeError)):
        _ = s @ v[:2]  # wrong length
    with pytest.raises((ValueError, RuntimeError)):
        _ = dlx.CsrMatrix(A) @ dlx.CsrMatrix([[1.0, 0.0], [0.0, 1.0]])  # shape mismatch


def test_matmul_dtype_mismatch():
    s = dlx.CsrMatrix(np.array(A, dtype=np.float32))
    with pytest.raises(TypeError):
        _ = s @ dlx.CsrMatrix(np.array(A, dtype=np.float64))


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
def test_repr():
    assert "csr_matrix(3 x 3" in repr(dlx.CsrMatrix(A))
    assert "csc_matrix(3 x 3" in repr(dlx.CscMatrix(A))


def test_numpy_interop_dense():
    data = np.array(A, dtype=np.float32)
    s = dlx.CsrMatrix(data)
    back = np.asarray(s.to_dense())
    assert back.dtype == np.float32
    np.testing.assert_allclose(back, data)
