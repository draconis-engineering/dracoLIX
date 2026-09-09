# DracoLIX Roadmap

## Phase 0 — Foundation

Mål: Gjøre prosjektet til en ordentlig numerisk runtime.

- [ ] Bestem endelig prosjektstruktur
- [ ] Definer core API
- [ ] Definer DType-systemet
- [ ] Implementer native scalar types
- [ ] bool
- [ ] signed integers
- [ ] unsigned integers
- [ ] f32
- [ ] f64
- [ ] Definer Array<T> / dynamisk array-representasjon
- [ ] Shape + strides
- [ ] Memory ownership
- [ ] Contiguous memory
- [ ] Views / slices
- [ ] Grunnleggende testing
- [ ] Benchmarking-infrastruktur

Exit condition: Du kan lage og manipulere native DracoLIX-arrays uten NumPy.

## Phase 1 — Core Array Engine

Mål: Bygge selve datamotoren.

- [ ] Array
  - [ ] dtype
  - [ ] shape
  - [ ] strides
  - [ ] layout
  - [ ] memory
- [ ] Vector
- [ ] Matrix
- [ ] Tensor-basics
- [ ] Indexing
- [ ] Slicing
- [ ] Reshape
- [ ] Transpose
- [ ] Copy / clone
- [ ] Views
- [ ] Type conversion
- [ ] Broadcasting — basic
- [ ] Element-wise operations
  - [ ] +
  - [ ] -
  - [ ] *
  - [ ] /
- [ ] Reductions
  - [ ] sum
  - [ ] min
  - [ ] max
  - [ ] mean

Exit condition: DracoLIX kan fungere som en liten, selvstendig numerical array library.

## Phase 2 — Linear Algebra

Mål: Gjøre DracoLIX genuint nyttig for matematikk.

- [ ] Matrix multiplication — prototype
  - [ ] Optimalisert matrix multiplication
  - [ ] Dot product
  - [ ] Vector norms
  - [ ] Transpose
  - [ ] Matrix-vector multiplication
  - [ ] Diagonal operations
  - [ ] LU decomposition
  - [ ] QR decomposition
  - [ ] Cholesky decomposition
  - [ ] Matrix inverse
  - [ ] Determinant
  - [ ] Rank
  - [ ] Linear system solving
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

- [ ] Benchmark suite
- [ ] Baseline implementations
- [ ] Memory profiling
- [ ] Cache-aware algorithms
- [ ] SIMD
- [ ] Multithreading
- [ ] Parallel reductions
- [ ] Thread pool
- [ ] Kernel dispatch
- [ ] CPU feature detection
- [ ] Alignment
- [ ] Optimized memory allocation

Og viktig:

Benchmark mot etablerte biblioteker

Ikke for å "slå NumPy" for enhver pris, men for å vite hvor du faktisk står.

- [ ] DracoLIX
- [ ] NumPy
- [ ] BLAS
- [ ] OpenBLAS
- [ ] MKL

for relevante workloads.

Exit condition: Du vet hvorfor DracoLIX er rask eller treg, i stedet for bare å anta at Rust = raskt.

## Phase 4 — Python API

Nå kan du lage den Python-opplevelsen ordentlig.

```python
import dracolix as dlx

A = dlx.matrix(..., dtype=dlx.f64)
B = dlx.matrix(..., dtype=dlx.f64)

C = A @ B
```

- [ ] PyO3 bindings
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
PyO3
   ↓
DracoLIX
```

## Phase 5 — Julia

Når core-en er stabil, blir Julia utrolig interessant.

- [ ] Julia bindings
- [ ] Native DracoLIX arrays
- [ ] Julia ↔ Rust memory handling
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
     Python     Julia      Rust
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
