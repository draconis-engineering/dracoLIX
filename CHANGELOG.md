# Changelog

All notable changes to DracoLIX are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.3] - 2026-11-08

### Added

- Python: `Array.prod()` and `Array.prod(axis)` reductions (empty product is
  `1`; bool input returns `i64` 0/1, matching NumPy).
- Python: `Array.argmin()` / `Array.argmax()` and axis forms. Scalar forms
  return a Python `int` flat C-order index; axis forms return `i64` index
  arrays. Empty inputs raise `ValueError`, complex inputs raise `TypeError`.
- Python: `Array.any()` / `Array.all()` and axis forms (bool scalar / bool
  array; empty identities follow NumPy: `any() -> False`, `all() -> True`).
- Python: binary operators now accept list/tuple operands, so
  `arr < [1, 2, 3]` and `arr + [1, 2, 3]` behave like NumPy.

### Fixed

- Core: `Array::prod()` and `Array::prod(axis)` always returned `0` because the
  accumulator was seeded with `T{}`; products now seed with the
  multiplicative identity.
- Core: `Array::argmin(axis)` / `Array::argmax(axis)` returned the min/max
  **values** with the element dtype instead of indices; they now return `i64`
  index arrays.
- Python: `min()` on bool arrays returned logical OR (`any`); it now returns
  logical AND, while `max()` remains logical OR.

### Changed

- Python: `bindings/python/dracolix/__init__.pyi` is now the authoritative type
  stub for `import dracolix`.
- Python: `std`/`var` are intentionally not exposed in 0.3.3; the axis
  implementations were incorrect and are deferred to 0.3.4.

## [0.3.2] - 2026-11-07

### Changed

- Version bump only: `CMakeLists.txt`, `core/include/dracolix/dracolix.hpp`,
  and `pyproject.toml`.

## [0.3.1] - 2026-11-06

### Added

- Python: additional integer dtypes `i8`, `i16`, `u8`, `u16`, `u32`, and `u64`,
  with NumPy-compatible promotion, overflow-checked construction and
  assignment, PEP 3118 buffer interop, and NumPy-style dtype aliases.
- Python: `linspace(start, stop, num=50, endpoint=True, dtype=None)` with
  real and complex support.
- Python: `Array.item(index=None)` and `Array.__index__` (size-1 integer
  arrays), following NumPy semantics.
- C++: `Array` now accepts shapes containing zero-length dimensions, so empty
  arrays can be represented (for example `linspace(..., num=0)`).

### Changed

- Python: integer/float promotion now matches NumPy (for example `i32 + f32`
  promotes to `f64`, `i8 + u8` promotes to `i16`).
- Python: importing a byte buffer (`b"..."`) now infers `u8` instead of `i32`;
  integer buffer formats map to their exact-width dtypes.

## [0.3.0] - 2026-11-05

### Added

- Complex number support across the C++ core and Python bindings: `c64`/`c128`
  dtypes, complex list/buffer construction, arithmetic and NumPy-compatible
  promotion, complex-aware `norm`/`dot`/`matmul`/`matvec`, and complex
  LU/QR/Cholesky/solve/inv/det/rank.
- Python: element-wise exponentiation (`array ** scalar`, `scalar ** array`, and
  `array ** array`) with broadcasting, dtype promotion, complex support, and
  NumPy-like rejection of negative integer exponents.
- Complex element-wise API: `real`, `imag`, `conj`/`conjugate`, `abs`/`__abs__`,
  and `angle`, available on `Array` and as module functions. Real, integer, and
  bool arrays get identity/zero/phase semantics.
- PEP 3118 complex buffer interop (`Zf`/`Zd`), matching NumPy `complex64` and
  `complex128` layouts.
- C++: unified `solve_ivp` with adaptive RK45, plus stiff backward Euler and
  variable-step BDF2 solvers with numerical or analytic Jacobians.
- C++: header-only PDE module with 1-D/2-D grids, Dirichlet/periodic
  boundaries, heat (explicit and Crank-Nicolson), wave, upwind advection,
  tridiagonal 1-D Poisson, and SOR-based 2-D Poisson solvers.
- C++: golden-section, Nelder-Mead, and BFGS optimizers; QR-based linear least
  squares with polynomial fitting.
- C++: seedable random engine (uniform, normal, integer, Bernoulli,
  exponential, Poisson) and five probability distributions with pdf/pmf, cdf,
  quantile, and sampling.
- C++: compile-time dimensional analysis (`Quantity`/`Dimension`), SI units,
  CODATA 2018 constants, DFT/FFT and signal processing, and time-series
  primitives.

### Changed

- Python: mixed real/complex operands are promoted for `matmul`, `matvec`,
  `dot`, and `solve`; mismatches between two real dtypes remain errors.
- Python: complex values are rejected when constructing real arrays from lists
  or assigning into real arrays. Use `astype()` to explicitly discard the
  imaginary part.
- Complex `eig`/`svd` and sparse operations remain unsupported and raise
  `TypeError`.

## [0.2.8] - 2026-11-02

### Added

- Python: full dense linear algebra and decomposition API, bound from the C++
  core — `matmul`, `matvec`, `dot`, `norm`, `diagonal`, `diag`, `lu`,
  `solve`, `inv`/`inverse`, `det`/`determinant`, `rank`, `qr`, `cholesky`,
  `eig`, `svd`, and sparse `cg_solve`.
- Python: `CsrMatrix.solve(b)` and `CscMatrix.solve(b)`.
- Core: `Array::avg()` reduction (serial, and parallel for large arrays),
  alongside `sum`/`min`/`max`/`mean`.

### Fixed

- Core: `avg()` was inserted without its closing brace, which nested the
  remaining reductions and initializers inside its body and broke compilation.
- Packaging: `dracolix.__version__` now derives from the distribution version
  (`pyproject.toml`) instead of a stale hardcoded string.

### Changed

- Packaging: wheels are built for CPython 3.10–3.15 via cibuildwheel; musl,
  32-bit, and free-threaded (`cp31?t`) ABIs are explicitly skipped for now.
- CI: Python jobs build/test on 3.12, 3.14, and 3.15, and the wheel test
  command now runs the full test suite.

[Unreleased]: https://github.com/draconis-engineering/dracolix/compare/v0.3.3...HEAD
[0.3.3]: https://github.com/draconis-engineering/dracolix/releases/tag/v0.3.3
[0.3.2]: https://github.com/draconis-engineering/dracolix/releases/tag/v0.3.2
[0.3.1]: https://github.com/draconis-engineering/dracolix/releases/tag/v0.3.1
[0.3.0]: https://github.com/draconis-engineering/dracolix/releases/tag/v0.3.0
[0.2.8]: https://github.com/draconis-engineering/dracolix/releases/tag/v0.2.8
