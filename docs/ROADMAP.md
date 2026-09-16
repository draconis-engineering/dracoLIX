# DracoLIX Roadmap

## Phase 0 — Foundation

Mål: Gjøre prosjektet til en ordentlig numerisk runtime.

- [x] Bestem endelig prosjektstruktur
- [x] Definer core API
- [x] Definer DType-systemet
- [x] Implementer native scalar types
  - [x] bool
  - [x] signed integers (i32, i64)
  - [ ] unsigned integers (reserved in enum, not yet dispatched)
  - [x] f32
  - [x] f64
- [x] Definer Array<T> / dynamisk array-representasjon
  - [x] Shape + strides
  - [x] Memory ownership
  - [x] Contiguous memory
  - [x] Views / slices
- [x] Grunnleggende testing
- [x] Benchmarking-infrastruktur (`benchmarks/bench_core.cpp` + `bench.hpp`, `compare_baseline.py`)

Exit condition: Du kan lage og manipulere native DracoLIX-arrays uten NumPy.

## Phase 1 — Core Array Engine

Mål: Bygge selve datamotoren.

- [x] Array
  - [x] dtype
  - [x] shape
  - [x] strides
  - [x] layout (RowMajor / ColMajor)
  - [x] memory (std::vector-backed)
- [x] Vector / Matrix / Tensor-basics
- [x] Indexing
- [x] Slicing
- [x] Reshape
- [x] Transpose
- [x] Copy / clone
- [x] Views (ArrayView, non-owning strided)
- [x] Type conversion (astype)
- [x] Broadcasting — basic
- [x] Element-wise operations
  - [x] +
  - [x] -
  - [x] *
  - [x] /
- [x] Reductions
  - [x] sum (global + axis)
  - [x] min (global + axis)
  - [x] max (global + axis)
  - [x] mean (global + axis)

Exit condition: DracoLIX kan fungere som en liten, selvstendig numerical array library.

## Phase 2 — Linear Algebra

Mål: Gjøre DracoLIX genuint nyttig for matematikk.

- [x] Matrix multiplication — prototype
  - [x] Optimalisert matrix multiplication (cache-friendly ikj ordering, scalar baseline)
  - [x] Dot product
  - [x] Vector norms
  - [x] Transpose
  - [x] Matrix-vector multiplication
  - [x] Diagonal operations
  - [x] Batched matmul/matvec/dot (atleast-3D)
  - [x] LU decomposition (`core/include/dracolix/decomp.hpp:12`)
  - [x] QR decomposition (`decomp::qr`, Modified Gram-Schmidt, `tests/cpp/test_decomp.cpp:22`)
  - [x] Cholesky decomposition (`decomp::cholesky`, `solve_cholesky`)
  - [x] Matrix inverse (`decomp::inverse`)
  - [x] Determinant (`decomp::determinant`)
  - [x] Rank (`decomp::rank`)
  - [x] Linear system solving (`decomp::solve`, `solve_lu`, multi-RHS)
  - [ ] Eigenvalues / eigenvectors
  - [ ] SVD
  - [ ] Sparse
    - [ ] Sparse matrix representation
    - [ ] CSR
    - [ ] CSC
    - [ ] Sparse matrix multiplication
    - [ ] Sparse linear solvers

Exit condition: DracoLIX kan håndtere reelle engineering/scientific-computing workloads.

## Phase 3 — Performance Engine

Her ville jeg begynt å bli litt gal på performance.

- [x] Benchmark suite (`benchmarks/bench_core.cpp` — elementwise, reductions, matmul, batched matmul, sparse, transpose)
- [x] Baseline implementations (scalar `ikj` GEMM in `core/src/kernels/gemm.cpp`, driven via `bench_core`)
- [ ] Memory profiling
- [x] Cache-aware algorithms (`ikj` ordering, contiguous checks in `linalg.hpp:35`)
- [x] SIMD (`gemm_f64/f32_avx2` FMA in `core/src/kernels/gemm.cpp:33`, `-mavx2 -mfma`)
- [ ] Multithreading
- [ ] Parallel reductions
- [ ] Thread pool
- [x] Kernel dispatch (`core/include/dracolix/kernels/dispatch.hpp:18`, `select_gemm_kernel`, `dispatch_gemm_*`)
- [x] CPU feature detection (`core/include/dracolix/cpu.hpp:12`, AVX/AVX2/AVX512F/FMA)
- [x] Alignment (`core/include/dracolix/alloc.hpp:14`, 64B `AlignedAllocator`, `is_aligned`)
- [x] Optimized memory allocation (`AlignedAllocator` via `Array` storage `core/include/dracolix/array.hpp:355`)

Og viktig:

Benchmark mot etablerte biblioteker

Ikke for å "slå NumPy" for enhver pris, men for å vite hvor du faktisk står.

- [x] DracoLIX (bench_core, ~20 GFLOP/s on 512²)
- [x] NumPy (`benchmarks/compare_baseline.py`, `bench_gemm.py`)
- [x] BLAS (plumbed via `DRACOLIX_USE_FORTRAN` / OpenBLAS)
- [x] OpenBLAS (linked optionally, `compare_baseline.py` probes)
- [ ] MKL

for relevante workloads.

Exit condition: Du vet hvorfor DracoLIX er rask eller treg, i stedet for bare å anta at språket = raskt.

## Phase 4 — Python API

Nå kan du lage den Python-opplevelsen ordentlig.

```python
import dracolix as dlx

A = dlx.matrix(..., dtype=dlx.f64)
B = dlx.matrix(..., dtype=dlx.f64)

C = A @ B
```

- [ ] nanobind bindings
- [ ] Native Python objects
- [ ] Python-side DTypes
- [ ] Array API
- [ ] Error handling
- [ ] Documentation
- [ ] Type hints
- [ ] Packaging
- [ ] Wheels
- [ ] Windows/Linux support

Viktig mål:

```
Python
   ↓
nanobind
   ↓
DracoLIX
```

## Phase 5 — Julia

Når core-en er stabil, blir Julia utrolig interessant.

- [ ] Julia bindings
- [ ] Native DracoLIX arrays
- [ ] Julia ↔ C++ memory handling
- [ ] DType mapping
- [ ] Julia broadcasting
- [ ] Julia linear algebra interface
- [ ] Documentation/examples

Da begynner DracoLIX å bli ordentlig multi-language:

```

             DracoLIX Core
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
     Python     Julia      C++
```     

## Phase 6 — Numerical Computing

Nå beveger vi oss fra "linear algebra library" til SciComp.

- [ ] Numerical methods
- [ ] Root finding
- [ ] Interpolation
- [ ] Numerical differentiation
- [ ] Numerical integration
- [ ] Optimization
- [ ] Least squares
- [ ] Random number generation
- [ ] Probability distributions
- [ ] Differential equations
- [ ] ODE framework
- [ ] Euler
- [ ] RK4
- [ ] Adaptive Runge-Kutta
- [ ] Stiff solvers
- [ ] PDE abstractions

Exit condition: DracoLIX kan brukes til å implementere faktiske matematiske modeller uten at du må finne frem NumPy/SciPy.

## Phase 7 — Scientific Computing

Her begynner den opprinnelige visjonen å bli real.

- [ ] Physical units / dimensions
- [ ] Scientific constants
- [ ] Signal processing
- [ ] Time-series primitives
- [ ] Optimization framework
- [ ] Numerical simulation framework
- [ ] Mesh/data structures
- [ ] PDE infrastructure
- [ ] Engineering utilities

Og dette er hvor Olympus begynner å bli en seriøs testbruker.

```
Olympus
   │
   ├── training data
   ├── time series
   ├── statistics
   ├── signal processing
   └── analytics
             │
             ▼
        DracoLIX
```

## Phase 8 — HPC

Ikke start her før alt over fungerer.

- [ ] Advanced SIMD
- [ ] NUMA awareness
- [ ] CPU topology
- [ ] Work stealing
- [ ] Distributed arrays
- [ ] MPI
- [ ] Distributed linear algebra
- [ ] Domain decomposition
- [ ] Checkpointing
- [ ] Large-scale simulation

## Phase 9 — Accelerators

Først når CPU-backenden er solid:

- [ ] GPU abstraction
- [ ] CUDA
- [ ] ROCm
- [ ] GPU memory management
- [ ] GPU kernels
- [ ] CPU/GPU dispatch
- [ ] Unified execution model

## Phase 10 — CFD / Simulation

Dette er moonshot-fasen.

- [ ] Mesh engine
- [ ] Finite difference
- [ ] Finite volume
- [ ] Finite element research
- [ ] Boundary conditions
- [ ] Navier–Stokes
- [ ] Fluid simulation
- [ ] Parallel CFD
- [ ] Visualization/export

Da er DracoLIX plutselig ikke bare et bibliotek lenger, men en ordentlig SciComp-platform.
