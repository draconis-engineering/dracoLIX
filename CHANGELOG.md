# Changelog

All notable changes to DracoLIX are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/draconis-engineering/dracolix/compare/v0.2.8...HEAD
[0.2.8]: https://github.com/draconis-engineering/dracolix/releases/tag/v0.2.8
