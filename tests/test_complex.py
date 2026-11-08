"""Complex number support: dtypes, arithmetic, linalg, and buffer interop.

The runtime has zero NumPy dependency; numpy is imported lazily inside the
interop tests only (``pytest.importorskip``).
"""

import cmath

import dracolix as dlx
import pytest

TOL = 1e-9


def close(x, y, tol=TOL):
    assert abs(complex(x) - complex(y)) < tol


def close_all(a, b, tol=TOL):
    a = a.tolist() if hasattr(a, "tolist") else a
    b = b.tolist() if hasattr(b, "tolist") else b
    assert isinstance(a, list) and isinstance(b, list)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if isinstance(x, list):
            close_all(x, y, tol)
        else:
            close(x, y, tol)


# ---------------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------------
def test_complex_dtype_singletons():
    assert dlx.c64 is dlx.c64
    assert dlx.c128 is dlx.c128
    assert dlx.c64 != dlx.c128
    assert dlx.c64.name == "c64"
    assert dlx.c128.name == "c128"
    assert repr(dlx.c128) == "dtype('c128')"
    assert isinstance(dlx.c64, dlx.DType)


def test_dtype_resolver_complex():
    assert dlx.dtype(dlx.c64) is dlx.c64
    assert dlx.dtype("c128") is dlx.c128
    assert dlx.dtype("complex") is dlx.c128
    assert dlx.dtype("complex64") is dlx.c64
    assert dlx.dtype("complex128") is dlx.c128
    assert dlx.dtype(complex) is dlx.c128
    assert dlx.dtype(dlx.array([1j])) is dlx.c128


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def test_array_from_complex_list():
    a = dlx.array([1 + 2j, 3 - 4j])
    assert a.dtype is dlx.c128
    assert a.tolist() == [1 + 2j, 3 - 4j]
    assert a[0] == 1 + 2j
    assert list(a) == [1 + 2j, 3 - 4j]


def test_array_mixed_int_and_complex_promotes_to_c128():
    a = dlx.array([1, 2j, 3])
    assert a.dtype is dlx.c128
    assert a.tolist() == [1 + 0j, 2j, 3 + 0j]


def test_dtype_keyword_accepts_complex_aliases():
    assert dlx.array([1, 2], dtype=complex).dtype is dlx.c128
    assert dlx.array([1, 2], dtype=dlx.c64).dtype is dlx.c64
    assert dlx.array([1, 2], dtype=dlx.c128).dtype is dlx.c128


def test_nested_complex_list():
    a = dlx.array([[1 + 1j, 2j], [3, 4 - 1j]])
    assert a.shape == (2, 2)
    assert a.dtype is dlx.c128
    assert a.tolist() == [[1 + 1j, 2j], [3 + 0j, 4 - 1j]]


def test_complex_factories():
    assert dlx.zeros(2, dtype=dlx.c128).tolist() == [0j, 0j]
    assert dlx.ones(2, dtype=dlx.c64).tolist() == [1 + 0j, 1 + 0j]
    assert dlx.empty(2, dtype=dlx.c128).tolist() == [0j, 0j]
    assert dlx.full(2, 1 + 2j).tolist() == [1 + 2j, 1 + 2j]
    assert dlx.full(2, 3, dtype=dlx.c64).tolist() == [3 + 0j, 3 + 0j]


def test_arange_rejects_complex():
    with pytest.raises(TypeError):
        dlx.arange(1 + 0j, 5)


# ---------------------------------------------------------------------------
# arithmetic and promotion
# ---------------------------------------------------------------------------
def test_complex_arithmetic():
    a = dlx.array([1 + 2j, 3 - 4j])
    b = dlx.array([2 - 1j, 1 + 1j])
    assert (a + b).tolist() == [3 + 1j, 4 - 3j]
    assert (a - b).tolist() == [-1 + 3j, 2 - 5j]
    assert (a * b).tolist() == [(1 + 2j) * (2 - 1j), (3 - 4j) * (1 + 1j)]
    close_all(a / b, [(1 + 2j) / (2 - 1j), (3 - 4j) / (1 + 1j)])
    assert (-a).tolist() == [-1 - 2j, -3 + 4j]


def test_complex_scalar_arithmetic():
    a = dlx.array([1 + 1j, 2 + 0j])
    assert (a + (1 - 1j)).tolist() == [2 + 0j, 3 - 1j]
    assert (a * 2).tolist() == [2 + 2j, 4 + 0j]
    assert (2 * a).tolist() == [2 + 2j, 4 + 0j]


def test_complex_promotion():
    c64 = dlx.array([1 + 1j], dtype=dlx.c64)
    c128 = dlx.array([1 + 1j], dtype=dlx.c128)
    assert (c64 + c64).dtype is dlx.c64
    assert (c64 + c128).dtype is dlx.c128
    assert (c64 + dlx.array([1.0])).dtype is dlx.c128
    assert (c64 + dlx.array([1], dtype=dlx.i64)).dtype is dlx.c128
    assert (c64 + dlx.array([1.0], dtype=dlx.f32)).dtype is dlx.c64
    # true division keeps the promoted complex dtype
    assert (c64 / c64).dtype is dlx.c64
    assert (c64 / c128).dtype is dlx.c128


def test_complex_power():
    a = dlx.array([1 + 2j, 3 - 4j])
    close_all(a**2, [(1 + 2j) ** 2, (3 - 4j) ** 2])
    assert (a**2).dtype is dlx.c128
    close_all(a**2.0, [(1 + 2j) ** 2, (3 - 4j) ** 2])
    close_all(dlx.array([1j], dtype=dlx.c64) ** 2, [-1 + 0j])
    close_all(
        (1 + 1j) ** dlx.array([2 + 0j, 3 + 0j]),
        [(1 + 1j) ** 2, (1 + 1j) ** 3],
    )
    close_all(dlx.array([2.0]) ** dlx.array([1 + 1j]), [2 ** (1 + 1j)])
    close_all(a ** dlx.array([0j, 1 + 0j]), [1 + 0j, 3 - 4j])


def test_complex_comparisons():
    a = dlx.array([1 + 2j, 3 - 4j])
    b = dlx.array([1 + 2j, 0j])
    assert (a == b).tolist() == [True, False]
    assert (a != b).tolist() == [False, True]
    assert (a == (1 + 2j)).tolist() == [True, False]


def test_complex_bool_single_element():
    assert bool(dlx.array([1 + 2j]))
    assert not bool(dlx.array([0j]))


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------
def test_astype_complex():
    a = dlx.array([1 + 2j, 3 - 4j])
    close_all(a.astype(dlx.f64), [1.0, 3.0])  # imaginary part dropped
    close_all(a.astype(dlx.c64), [1 + 2j, 3 - 4j])
    assert a.astype(dlx.c64).dtype is dlx.c64
    close_all(dlx.array([1.5, 2.5]).astype(dlx.c128), [1.5 + 0j, 2.5 + 0j])


def test_complex_to_real_list_and_setitem_are_rejected():
    # Like NumPy, a complex value cannot be silently stored in a real array;
    # astype() remains the explicit way to drop the imaginary part.
    with pytest.raises(TypeError):
        dlx.array([1 + 2j], dtype=dlx.f64)
    with pytest.raises(TypeError):
        dlx.array([1 + 2j], dtype=dlx.f32)
    a = dlx.zeros(2)
    with pytest.raises(TypeError):
        a[0] = 1 + 2j


def test_complex_repr():
    assert repr(dlx.array([1 + 2j])) == "array([(1+2j)], dtype=c128)"
    assert "c64" in repr(dlx.array([1j], dtype=dlx.c64))


def test_complex_setitem_and_slice():
    a = dlx.zeros(3, dtype=dlx.c128)
    a[0] = 1 + 2j
    a[1] = 3
    assert a.tolist() == [1 + 2j, 3 + 0j, 0j]
    assert a[0:2].tolist() == [1 + 2j, 3 + 0j]


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------
def test_complex_sum_mean_norm():
    a = dlx.array([1 + 2j, 3 - 4j])
    assert a.sum() == 4 - 2j
    assert a.mean() == 2 - 1j
    close(dlx.norm(a), cmath.sqrt(abs(1 + 2j) ** 2 + abs(3 - 4j) ** 2))


def test_complex_reductions_along_axis():
    a = dlx.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    assert a.sum(0).tolist() == [6 + 8j, 10 + 12j]
    assert a.mean(0).tolist() == [3 + 4j, 5 + 6j]


def test_complex_min_max_raise():
    a = dlx.array([1 + 2j, 3 - 4j])
    for fn in (a.min, a.max, lambda: a.min(0), lambda: a.max(0)):
        with pytest.raises(TypeError):
            fn()


# ---------------------------------------------------------------------------
# element-wise complex helpers: real / imag / conj / abs / angle
# ---------------------------------------------------------------------------
def test_complex_real_imag():
    a = dlx.array([1 + 2j, -3 + 4j, 0 - 1j])
    assert a.real.tolist() == [1.0, -3.0, 0.0]
    assert a.imag.tolist() == [2.0, 4.0, -1.0]
    assert a.real.dtype is dlx.f64
    assert a.imag.dtype is dlx.f64
    assert dlx.real(a).tolist() == a.real.tolist()
    assert dlx.imag(a).tolist() == a.imag.tolist()

    b = dlx.array([1 + 2j], dtype=dlx.c64)
    assert b.real.dtype is dlx.f32
    assert b.imag.dtype is dlx.f32
    close_all(b.real, [1.0])
    close_all(b.imag, [2.0])


def test_complex_conj():
    a = dlx.array([1 + 2j, -3 - 4j])
    assert a.conj().tolist() == [1 - 2j, -3 + 4j]
    assert a.conjugate().tolist() == a.conj().tolist()
    assert dlx.conj(a).tolist() == a.conj().tolist()
    assert dlx.conjugate(a).tolist() == a.conj().tolist()
    assert a.conj().conj().tolist() == a.tolist()
    assert a.conj().dtype is dlx.c128


def test_complex_abs_and_builtin_abs():
    a = dlx.array([3 + 4j, -1 + 0j, 0 - 2j])
    assert a.abs().tolist() == [5.0, 1.0, 2.0]
    assert abs(a).tolist() == [5.0, 1.0, 2.0]
    assert dlx.abs(a).tolist() == [5.0, 1.0, 2.0]
    assert dlx.absolute(a).tolist() == [5.0, 1.0, 2.0]
    assert a.abs().dtype is dlx.f64
    assert abs(dlx.array([3 + 4j], dtype=dlx.c64)).dtype is dlx.f32


def test_complex_angle():
    a = dlx.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j])
    got = a.angle().tolist()
    for x, want in zip(got, [0.0, cmath.pi / 2, cmath.pi, -cmath.pi / 2]):
        close(x, want)
    close_all(dlx.angle(a), got)
    assert a.angle().dtype is dlx.f64
    assert dlx.array([1j], dtype=dlx.c64).angle().dtype is dlx.f32


def test_complex_helpers_preserve_shape():
    a = dlx.array([[1 + 1j, 2 + 2j], [3 - 3j, 4 - 4j]])
    assert a.real.shape == (2, 2)
    assert a.imag.shape == (2, 2)
    assert a.conj().shape == (2, 2)
    assert a.abs().shape == (2, 2)
    assert a.angle().shape == (2, 2)
    # Euler identity: z == |z| * (cos arg + i sin arg)
    r = a.abs().tolist()
    th = a.angle().tolist()
    for row_r, row_th, row_z in zip(r, th, a.tolist()):
        for rr, tt, zz in zip(row_r, row_th, row_z):
            close(rr * (cmath.cos(tt) + 1j * cmath.sin(tt)), zz, 1e-12)


def test_complex_helpers_on_real_arrays():
    a = dlx.array([3.0, -2.5, 0.0])
    assert a.real.tolist() == [3.0, -2.5, 0.0]
    assert a.imag.tolist() == [0.0, 0.0, 0.0]
    assert a.conj().tolist() == [3.0, -2.5, 0.0]
    assert a.abs().tolist() == [3.0, 2.5, 0.0]
    close_all(a.angle(), [0.0, cmath.pi, 0.0])
    assert a.real.dtype is dlx.f64
    assert a.imag.dtype is dlx.f64
    assert a.conj().dtype is dlx.f64


def test_complex_helpers_on_integer_arrays():
    a = dlx.array([-7, 0, 4], dtype=dlx.i32)
    assert a.real.tolist() == [-7, 0, 4]
    assert a.imag.tolist() == [0, 0, 0]
    assert a.conj().tolist() == [-7, 0, 4]
    assert a.abs().tolist() == [7, 0, 4]
    assert a.real.dtype is dlx.i32
    assert a.imag.dtype is dlx.i32
    # Phase is a floating-point quantity even for integer input.
    assert a.angle().dtype is dlx.f64
    close_all(a.angle(), [cmath.pi, 0.0, 0.0])


def test_complex_helpers_on_bool_arrays():
    a = dlx.array([False, True, True])
    assert a.real.tolist() == [False, True, True]
    assert a.imag.tolist() == [False, False, False]
    assert a.conj().tolist() == [False, True, True]
    assert a.abs().tolist() == [False, True, True]


# ---------------------------------------------------------------------------
# linear algebra
# ---------------------------------------------------------------------------
def test_complex_dot_is_not_conjugated():
    a = dlx.array([1 + 1j])
    # np.dot semantics: sum(a * b), not sum(conj(a) * b)
    assert dlx.dot(a, a) == (1 + 1j) ** 2
    assert dlx.dot(a, dlx.array([1 - 1j])) == 2 + 0j


def test_complex_matmul_matvec():
    A = dlx.array([[1 + 1j, 0j], [0j, 2 - 1j]])
    B = dlx.array([[1 + 0j, 0j], [0j, 1 + 0j]])
    close_all(A @ B, [[1 + 1j, 0j], [0j, 2 - 1j]])
    x = dlx.array([1 + 0j, 1 + 0j])
    close_all(dlx.matvec(A, x), [1 + 1j, 2 - 1j])
    assert (x @ x) == 2 + 0j


def test_real_complex_operand_promotion():
    A = dlx.array([[1 + 1j, 0j], [0j, 2 + 0j]])
    R = dlx.array([[2.0, 0.0], [0.0, 3.0]], dtype=dlx.f64)
    I = dlx.array([[1, 0], [0, 1]], dtype=dlx.i64)
    close_all(A @ R, [[2 + 2j, 0j], [0j, 6 + 0j]])
    close_all(R @ A, [[2 + 2j, 0j], [0j, 6 + 0j]])
    close_all(A @ I, A)
    close_all(I @ A, A)
    assert (A @ R).dtype is dlx.c128
    assert (I @ A).dtype is dlx.c128

    x = dlx.array([1.0, 2.0])
    got = dlx.matvec(A, x)
    assert got.dtype is dlx.c128
    close_all(got, [1 + 1j, 4 + 0j])
    assert dlx.dot(dlx.array([1 + 1j, 3 - 4j]), x) == (1 + 1j) + 2 * (3 - 4j)
    close_all(dlx.solve(A, x), [0.5 - 0.5j, 1 + 0j])


def test_complex_diagonal():
    A = dlx.array([[1 + 1j, 2j], [3, 4 - 1j]])
    assert dlx.diagonal(A).tolist() == [1 + 1j, 4 - 1j]
    close_all(dlx.diag(dlx.array([1 + 1j, 2 - 2j])), [[1 + 1j, 0j], [0j, 2 - 2j]])


def test_complex_det_solve_inv():
    A = dlx.array([[1 + 1j, 0j], [0j, 2 - 1j]])
    assert dlx.det(A) == 3 + 1j
    close_all(dlx.solve(A, dlx.array([1 + 0j, 1 + 0j])), [0.5 - 0.5j, 0.4 + 0.2j])
    close_all(dlx.inv(A), [[0.5 - 0.5j, 0j], [0j, 0.4 + 0.2j]])
    assert dlx.rank(A) == 2


def test_complex_qr_is_unitary():
    A = dlx.array([[1 + 1j, 0j], [0j, 2 - 1j]])
    Q, R = dlx.qr(A)
    close_all(Q @ R, A)
    # Q^H Q == I for a unitary Q
    q = Q.tolist()
    for i in range(2):
        for j in range(2):
            s = sum(complex(q[k][i]).conjugate() * q[k][j] for k in range(2))
            close(s, 1.0 if i == j else 0.0)


def test_complex_cholesky_hermitian():
    # Hermitian positive-definite: diag is real, off-diagonals conjugate
    A = dlx.array([[4 + 0j, 2 - 1j], [2 + 1j, 5 + 0j]])
    L = dlx.cholesky(A)
    l = L.tolist()
    n = 2
    for i in range(n):
        for j in range(n):
            s = sum(l[i][k] * complex(l[j][k]).conjugate() for k in range(n))
            close(s, A.tolist()[i][j])


def test_complex_decompositions_match_for_diagonal():
    A = dlx.array([[2 + 0j, 0j], [0j, 3 + 0j]])
    L, U, _ = dlx.lu(A)
    close_all(L @ U, A)


def test_eig_svd_reject_complex():
    A = dlx.array([[1 + 0j, 0j], [0j, 1 + 0j]])
    with pytest.raises(TypeError):
        dlx.eig(A)
    with pytest.raises(TypeError):
        dlx.svd(A)


def test_sparse_rejects_complex():
    with pytest.raises(TypeError):
        dlx.CsrMatrix(dlx.array([[1 + 1j, 0j], [0j, 1 + 0j]]))
    with pytest.raises(TypeError):
        dlx.CscMatrix([[1 + 1j, 0j], [0j, 1 + 0j]])


# ---------------------------------------------------------------------------
# PEP 3118 buffer interop
# ---------------------------------------------------------------------------
def test_complex_buffer_format():
    assert memoryview(dlx.array([1j], dtype=dlx.c64)).format == "Zf"
    assert memoryview(dlx.array([1j], dtype=dlx.c64)).itemsize == 8
    assert memoryview(dlx.array([1j], dtype=dlx.c128)).format == "Zd"
    assert memoryview(dlx.array([1j], dtype=dlx.c128)).itemsize == 16


def test_numpy_complex_interop():
    np = pytest.importorskip("numpy")

    a = dlx.array([1 + 2j, 3 - 4j])
    arr = np.asarray(a)
    assert arr.dtype == np.complex128
    assert arr.tolist() == [1 + 2j, 3 - 4j]

    b = dlx.array([1 + 2j, 3 - 4j], dtype=dlx.c64)
    assert np.asarray(b).dtype == np.complex64

    for np_dtype, dlx_dtype in (
        (np.complex64, dlx.c64),
        (np.complex128, dlx.c128),
    ):
        src = np.array([1 + 2j, 3 - 4j], dtype=np_dtype)
        got = dlx.array(src)
        assert got.dtype is dlx_dtype
        close_all(got, src)

    # Round trip through numpy.
    close_all(dlx.array(np.asarray(a)), a.tolist())

    # Forcing a real dtype from a complex buffer keeps the real part.
    close_all(dlx.array(np.asarray(a), dtype=dlx.f64), [1.0, 3.0])


# ---------------------------------------------------------------------------
# 0.3.4: complex std/var and elementwise math
# ---------------------------------------------------------------------------
def test_complex_var_std():
    z = dlx.array([1 + 1j, 3 - 1j])
    assert z.var() == pytest.approx(2.0)
    assert z.std() == pytest.approx(2.0**0.5)
    assert z.var(0).tolist() == pytest.approx([2.0])
    assert z.std(ddof=1) == pytest.approx(2.0)


def test_complex_ufuncs():
    close_all(dlx.sqrt(dlx.array([-1 + 0j])), [1j])
    close_all(dlx.sqrt(dlx.array([3 + 4j])), [cmath.sqrt(3 + 4j)])
    close_all(dlx.exp(dlx.array([1j])), [cmath.exp(1j)])
    close_all(dlx.log(dlx.array([-1 + 0j])), [cmath.log(-1 + 0j)])
    close_all(dlx.log1p(dlx.array([1j])), [cmath.log(1 + 1j)])
    close_all(dlx.log2(dlx.array([8 + 0j])), [3 + 0j])
    close_all(dlx.log10(dlx.array([1000 + 0j])), [3 + 0j])
    close_all(dlx.sin(dlx.array([1 + 1j])), [cmath.sin(1 + 1j)])
    close_all(dlx.cos(dlx.array([1 + 1j])), [cmath.cos(1 + 1j)])
    close_all(dlx.tanh(dlx.array([1 + 1j])), [cmath.tanh(1 + 1j)])

    assert dlx.sqrt(dlx.array([1j], dtype=dlx.c64)).dtype is dlx.c64
    assert dlx.sqrt(dlx.array([1j], dtype=dlx.c128)).dtype is dlx.c128

    with pytest.raises(TypeError):
        dlx.floor(dlx.array([1 + 0j]))
    with pytest.raises(TypeError):
        dlx.ceil(dlx.array([1 + 0j]))
    with pytest.raises(TypeError):
        dlx.trunc(dlx.array([1 + 0j]))


def test_complex_square_and_predicates():
    sq = dlx.square(dlx.array([1 + 2j], dtype=dlx.c64))
    assert sq.dtype is dlx.c64
    close_all(sq, [(1 + 2j) ** 2])

    z = dlx.array([complex(float("nan"), 0.0), 1 + 2j])
    assert dlx.isnan(z).tolist() == [True, False]
    z2 = dlx.array([complex(0.0, float("inf")), 1 + 2j])
    assert dlx.isinf(z2).tolist() == [True, False]
    assert dlx.isfinite(z2).tolist() == [False, True]
