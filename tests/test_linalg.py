"""Python binding tests for the dracolix linear algebra + decomposition API.

Zero-NumPy runtime; numpy is only used as an optional PEP-3118 interop
cross-check (importorskip), matching the other test modules.
"""

import dracolix as dlx
import pytest

np = pytest.importorskip("numpy")

TOL = 1e-9


def close(x, y):
    assert abs(x - y) < TOL


def as_list(x):
    return x.tolist() if hasattr(x, "tolist") else x


def close_all(a, b):
    a = as_list(a)
    b = as_list(b)
    assert isinstance(a, list) and isinstance(b, list)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if isinstance(x, list):
            close_all_rec(x, y)
        else:
            close(x, y)


def close_all_rec(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if isinstance(x, list):
            close_all_rec(x, y)
        else:
            close(x, y)


A = [[4.0, 3.0], [6.0, 3.0]]
B = [[2.0, 1.0], [1.0, 3.0]]


# ---------------------------------------------------------------------------
# basic linalg
# ---------------------------------------------------------------------------
def test_matmul_module_function_matches_op():
    Am = dlx.array(A)
    Bm = dlx.array(B)
    assert dlx.matmul(Am, Bm).tolist() == (Am @ Bm).tolist()


def test_matvec():
    Am = dlx.array(A)
    x = dlx.array([1.0, 2.0])
    assert dlx.matvec(Am, x).tolist() == [10.0, 12.0]


def test_dot_float_and_int():
    close(dlx.dot(dlx.array([1.0, 2.0]), dlx.array([3.0, 4.0])), 11.0)
    assert dlx.dot(dlx.array([1, 2, 3]), dlx.array([4, 5, 6])) == 32
    close(dlx.dot(dlx.array([1.5]), dlx.array([2.0])), 3.0)


def test_norm_modes():
    a = dlx.array([3.0, 4.0])
    close(dlx.norm(a), 5.0)
    close(dlx.norm(a, p=1), 7.0)
    close(dlx.norm(a, p=0), 4.0)


def test_diagonal_and_diag():
    Am = dlx.array(A)
    assert dlx.diagonal(Am).tolist() == [4.0, 3.0]
    d = dlx.diag(dlx.array([1.0, 2.0, 3.0]))
    assert d.tolist() == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]


# ---------------------------------------------------------------------------
# LU / solve / det / inv / rank
# ---------------------------------------------------------------------------
def test_lu_decomposition():
    Am = dlx.array(A)
    L, U, piv = dlx.lu(Am)
    assert piv.dtype is dlx.i64
    # pivot: row i of P@A is original row piv[i]
    perm = piv.tolist()
    permuted = [A[i] for i in perm]
    assert dlx.matmul(L, U).tolist() == permuted


def test_solve_recovers_rhs():
    Am = dlx.array(A)
    b = dlx.array([1.0, 1.0])
    x = dlx.solve(Am, b)
    close_all(dlx.matvec(Am, x), b)


def test_solve_multi_rhs():
    Am = dlx.array([[2.0, 1.0], [1.0, 3.0]])
    Bm = dlx.array([[1.0, 2.0], [3.0, 4.0]])
    X = dlx.solve(Am, Bm)
    close_all(Am @ X, Bm)


def test_det():
    close(dlx.det(dlx.array(A)), -6.0)
    close(dlx.determinant(dlx.array(A)), -6.0)


def test_inverse():
    Am = dlx.array(A)
    inv = dlx.inv(Am)
    close_all(Am @ inv, [[1.0, 0.0], [0.0, 1.0]])
    close_all(dlx.inverse(Am).tolist(), inv.tolist())


def test_rank():
    assert dlx.rank(dlx.array(A)) == 2
    rank1 = dlx.array([[1.0, 2.0], [2.0, 4.0]])
    assert dlx.rank(rank1) == 1
    assert dlx.rank(dlx.array([[0.0, 0.0], [0.0, 0.0]])) == 0


# ---------------------------------------------------------------------------
# QR / Cholesky
# ---------------------------------------------------------------------------
def test_qr_orthonormal_and_reconstruct():
    Q, R = dlx.qr(dlx.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))
    QtQ = Q.T @ Q
    close_all(QtQ, [[1.0, 0.0], [0.0, 1.0]])
    close_all(Q @ R, [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])


def test_cholesky_and_solve():
    S = dlx.array([[4.0, 2.0], [2.0, 3.0]])
    L = dlx.cholesky(S)
    close_all(L @ L.T, S)
    x = dlx.solve(S, dlx.array([1.0, 2.0]))
    close_all(dlx.matvec(S, x), [1.0, 2.0])


# ---------------------------------------------------------------------------
# eig / svd (float-only, promote f32 -> f64)
# ---------------------------------------------------------------------------
def test_eig_symmetric():
    S = dlx.array([[4.0, 1.0], [1.0, 3.0]])
    vals, vecs = dlx.eig(S)
    assert vals.dtype is dlx.f64 and vecs.dtype is dlx.f64
    # eigenvalues descending, and A v = lambda v
    assert vals.tolist()[0] >= vals.tolist()[1]
    for i in range(2):
        v = [float(vecs[(r, i)]) for r in range(2)]
        close_all(dlx.matvec(S, dlx.array(v)), [vals[i] * c for c in v])


def test_eig_promotes_f32():
    vals, vecs = dlx.eig(dlx.array([[4.0, 1.0], [1.0, 3.0]], dtype=dlx.f32))
    assert vals.dtype is dlx.f64
    close(vals.tolist()[0], 4.618033988749894)


def test_svd_reconstruct():
    M = dlx.array([[1.0, 2.0], [3.0, 4.0]])
    U, S, Vt = dlx.svd(M)
    assert U.dtype is dlx.f64 and S.dtype is dlx.f64 and Vt.dtype is dlx.f64
    assert S.tolist()[0] >= S.tolist()[-1]
    close_all(U @ dlx.diag(S) @ Vt, [[1.0, 2.0], [3.0, 4.0]])


def test_svd_numpy_crosscheck():
    M = dlx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _, s, _ = dlx.svd(M)
    _, ns, _ = np.linalg.svd(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    for a, b in zip(s.tolist(), ns.tolist()):
        close(a, b)


# ---------------------------------------------------------------------------
# sparse solvers
# ---------------------------------------------------------------------------
def test_csr_solve_cg():
    s = dlx.CsrMatrix([[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]])
    y = s.solve(dlx.array([1.0, 2.0, 3.0]))
    close_all(s @ y, [1.0, 2.0, 3.0])


def test_csr_solve_matches_dense():
    A_dense = [[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]]
    b = dlx.array([1.0, 2.0, 3.0])
    x_cg = dlx.cg_solve(dlx.CsrMatrix(A_dense), b)
    x_dense = dlx.solve(dlx.array(A_dense), b)
    close_all(x_cg, x_dense)


def test_csc_solve_matches_csr():
    A_dense = [[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]]
    b = dlx.array([1.0, 2.0, 3.0])
    x_csr = dlx.CsrMatrix(A_dense).solve(b)
    x_csc = dlx.CscMatrix(A_dense).solve(b)
    close_all(x_csc, x_csr)


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------
def test_bool_rejected_for_decompositions():
    with pytest.raises(TypeError):
        dlx.det(dlx.array([[True, False], [False, True]]))
    with pytest.raises(TypeError):
        dlx.lu(dlx.array([[True, False], [False, True]]))
    with pytest.raises(TypeError):
        dlx.dot(dlx.array([True, False]), dlx.array([False, True]))


def test_eig_svd_reject_ints():
    with pytest.raises(TypeError):
        dlx.eig(dlx.array([[1, 2], [3, 4]], dtype=dlx.i64))
    with pytest.raises(TypeError):
        dlx.svd(dlx.array([[1, 2], [3, 4]], dtype=dlx.i32))


def test_dtype_mismatch_raises():
    with pytest.raises(TypeError):
        dlx.solve(dlx.array(A), dlx.array([1, 1], dtype=dlx.i64))


def test_norm_requires_1d():
    with pytest.raises(ValueError):
        dlx.norm(dlx.array(A))