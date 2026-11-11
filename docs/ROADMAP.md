# DracoLIX Roadmap

## [x] Phase 0 — Foundation

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

## [x] Phase 1 — Core Array Engine

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

## [x] Phase 2 — Linear Algebra

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
    - [x] Eigenvalues / eigenvectors (`core/include/dracolix/eigen.hpp:11` Jacobi, symmetric `eig_sym`)
    - [x] SVD (`core/include/dracolix/svd.hpp:11` via `A^T A` eigen, tall/wide)
    - [x] Sparse
        - [x] Sparse matrix representation (`core/include/dracolix/sparse.hpp:30` `CsrMatrix`/`CscMatrix`)
        - [x] CSR
        - [x] CSC
        - [x] Sparse matrix multiplication (`CsrMatrix::matmul`)
        - [x] Sparse linear solvers (`cg_solve` CG for SPD CSR)

Exit condition: DracoLIX kan håndtere reelle engineering/scientific-computing workloads.

## [ ] Phase 3 — Performance Engine

Her ville jeg begynt å bli litt gal på performance.

- [x] Benchmark suite (`benchmarks/bench_core.cpp` — elementwise, reductions, matmul, batched matmul, sparse, transpose)
- [x] Baseline implementations (scalar `ikj` GEMM in `core/src/kernels/gemm.cpp`, driven via `bench_core`)
- [x] Memory profiling (`core/include/dracolix/mem.hpp:11` `current_rss_bytes`/`peak_rss_bytes` via `/proc/self/status`, `format_bytes`, wired into `bench_core --mem`)
- [x] Cache-aware algorithms (`ikj` ordering, contiguous checks in `linalg.hpp:35`)
- [x] SIMD (register-blocked microkernels in `core/src/kernels/gemm_micro.hpp`: f64 AVX-512 8×24 / AVX2 6×8, f32 AVX-512 8×32 / AVX2 6×16, selected at runtime via `target` attributes + CPUID — no global `-march=native`)
- [x] Multithreading (`core/src/kernels/gemm_blocked.cpp`, row-panel parallel over disjoint C tiles)
- [x] Parallel reductions (`core/include/dracolix/array.hpp:266` `sum`/`min`/`max` via `ThreadPool::global`)
- [x] Thread pool (`core/include/dracolix/thread_pool.hpp:14`, global pool `hardware_concurrency`)
- [x] Kernel dispatch (`core/include/dracolix/kernels/dispatch.hpp`, `gemm_backend()` honours `DLX_GEMM_BACKEND`, `dispatch_gemm_*`)
- [x] CPU feature detection (`core/include/dracolix/cpu.hpp:12`, AVX/AVX2/AVX512F/DQ/VL/BW/FMA)
- [x] Alignment (`core/include/dracolix/alloc.hpp:14`, 64B `AlignedAllocator`/`AlignedBuffer`, `is_aligned`)
- [x] Optimized memory allocation (`AlignedAllocator` via `Array` storage `core/include/dracolix/array.hpp:355`)
- [x] Cache blocking + packing (`core/src/kernels/gemm_blocked.cpp`, MC/NC/KC panels, packed `Ap`/`Bp`)
- [x] GEMM backend matrix (`benchmarks/bench_gemm_backends.cpp`: scalar / avx2 / avx512 / openblas, validated vs the scalar oracle)
- [x] GEMM correctness matrix (`tests/cpp/test_gemm.cpp`, 1200 shapes — f64 and f32 — across 5 backends incl. remainder tiles and degenerate m/n/p)

Og viktig:

Benchmark mot etablerte biblioteker

Ikke for å "slå NumPy" for enhver pris, men for å vite hvor du faktisk står.

- [x] DracoLIX (~340 GFLOP/s on 4000², see `docs/perf-report-02102026.md`)
- [x] NumPy (`benchmarks/compare_baseline.py`, `bench_gemm.py`)
- [x] BLAS (plumbed via `DRACOLIX_USE_BLAS` / `core/src/kernels/gemm_blas.cpp`)
- [x] OpenBLAS (linked optionally, `compare_baseline.py` probes)
- [ ] MKL

for relevante workloads.

Exit condition: Du vet hvorfor DracoLIX er rask eller treg, i stedet for bare å anta at språket = raskt.

## [x] Phase 4 — Python API

Nå kan du lage den Python-opplevelsen ordentlig.

```python
import dracolix as dlx

A = dlx.array([[1,2],[3,4]], dtype=dlx.f64)
B = dlx.array([[5,6],[7,8]], dtype=dlx.f64)

C = A @ B
```

- [x] nanobind bindings (`bindings/python/src/dracolix_nb.cpp:1` PEP 3118, `DlxArray` variant `f32/f64/i32/i64/bool`)
- [x] Native Python objects (`Array`, `CsrMatrix`, `CscMatrix` via `nb::class_`)
- [x] Python-side DTypes (`f32`/`f64`/`i32`/`i64`/`bool_` singleton `DlxDType`)
- [x] Array API (`array`/`asarray`/`zeros`/`ones`/`arange`/`astype`/`__array__`/`__buffer__`)
- [x] Error handling (`type_error`/`invalid_argument` → `TypeError`/`ValueError`, bounds checks)
- [x] Documentation (`docs/README.md` + docstrings in `dracolix_nb.cpp`)
- [x] Type hints (`bindings/python/dracolix/py.typed` + `__init__.pyi` typed `Array`/`matmul`/`svd`)
- [x] Packaging (`pyproject.toml:1` `scikit-build-core` `0.2.0` PyPI)
- [x] Wheels (`.github/workflows/wheels.yml:1` cibuildwheel manylinux_2_28)
- [x] Windows/Linux support (`ci.yml` `linux`/`windows msys2/UCRT64`/`python` green)

## [o] Phase 5 — Julia

Når core-en er stabil, blir Julia utrolig interessant.

- [x] Julia bindings (`bindings/julia/src/DracoLIX.jl:1` `ccall` to `core/include/dracolix/c_api.h`)
- [x] Native DracoLIX arrays (`DracoArray` opaque `Array<double>` via `c_api.cpp`)
- [x] Julia ↔ C++ memory handling (`finalizer` → `dracolix_array_destroy`, row↔col transpose in `from_matrix`/`to_matrix`)
- [x] DType mapping (`f64` prototype, pattern for `f32/i32/i64`)
- [ ] Julia broadcasting
- [x] Julia linear algebra interface (`matmul`/`matvec` via `dracolix_matmul_f64`)
- [x] Documentation/examples (`bindings/julia/README.md:1`, `test/runtests.jl:1`)

Da begynner DracoLIX å bli ordentlig multi-language:

```

             DracoLIX Core
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
     Python     Julia      C++
```

## [x] Phase 6 — Numerical Computing

Nå beveger vi oss fra "linear algebra library" til SciComp.

- [x] Numerical methods (`core/include/dracolix/numerics.hpp:11`)
- [x] Root finding (`bisect`/`newton` bisection + Newton, tol 1e-10)
- [x] Interpolation (`interp_linear` 1-D)
- [x] Numerical differentiation (`derivative` central 1e-6)
- [x] Numerical integration (`integrate_simpson`/`trapezoidal` 1e-6 on x²)
- [x] Optimization (`golden_section`, `nelder_mead`, `bfgs` with numerical gradients)
- [x] Least squares (`lstsq` via QR, `polyfit`/`polyval`)
- [x] Random number generation (seedable `Rng` + array helpers)
- [x] Probability distributions (normal, uniform, exponential, Bernoulli, Poisson)
- [x] Differential equations (`core/include/dracolix/ode.hpp:11`)
- [x] ODE framework (`State` vector, `RHS` functor, traj vector)
- [x] Euler (fixed-step)
- [x] RK4 (classical 4th order)
- [x] Adaptive Runge-Kutta (`adaptive_rk45` step-doubling, tol 1e-6)
- [x] Stiff solvers (backward Euler + variable-step BDF2 with damped Newton, optional analytic Jacobian)
- [x] Unified `solve_ivp` entry point (method dispatch + recorded time points)
- [x] PDE abstractions (uniform grids; 1-D heat explicit/Crank-Nicolson, wave, upwind advection; 1-D and 2-D Poisson)

## [ ] Phase 7 — Scientific Computing

Her begynner den opprinnelige visjonen å bli real.

- [x] Physical units / dimensions (`units::Quantity` with compile-time dimension checking and SI units)
- [x] Scientific constants (`core/include/dracolix/constants.hpp`, CODATA 2018 values with units)
- [x] Signal processing (DFT/radix-2 FFT, inverse transforms, convolution/correlation, windows, power spectra)
- [x] Time-series primitives (differencing, cumsum, rolling statistics, EWMA, autocorrelation, z-score)
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

## [ ] Phase 8 — HPC

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

## [ ] Phase 9 — Accelerators

1. [ ] GPU Abstraction & Unified Execution Model

- [ ] Device-grensesnitt: Definer en abstrakt Device- eller Backend-klasse/trait som både CPU, CUDA og ROCm implementerer. Dette gjør at høyere lag i koden ikke trenger å bry seg om hvor beregningene skjer.
- [ ] Tensor/Buffer-representasjon: Sørg for at datastrukturene dine holder styr på hvilken enhet (DeviceType::CPU, DeviceType::CUDA, osv.) dataene ligger på, og støtter eksplisitt kopiering mellom dem (to(device)).
- [ ] Asynkron eksekvering: Både CUDA og ROCm opererer med strømmer (streams) og køer (queues). Bygg inn støtte for asynkrone kall og synkroniseringspunkter tidlig i abstraksjonslaget.

2. [ ] GPU Memory Management

- [ ] Allokering og frigjøring: Implementer wrappers rundt cudaMalloc/cudaFree og hipMalloc/hipFree.
- [ ] Memory Pooling (Caching Allocator): GPU-allokering er tregt. En enkel caching allocator som gjenbruker frigjorte minneblokker i stedet for å gi dem tilbake til driveren umiddelbart, vil gi massiv ytelsesgevinst.
- [ ] Pinned Memory: Vurder støtte for page-locked (pinned) host-minne for raskere dataoverføring mellom CPU og GPU.

3. [ ] CUDA & ROCm (HIP)

- [WIP] Felles kodebase med HIP: Siden AMD sitt ROCm/HIP-API er designet for å ligne veldig på CUDA, kan du ofte skrive kjernene (kernels) én gang og kompilere dem til begge plattformer ved hjelp av makroer eller HIP-verktøy, i stedet for å vedlikeholde to helt separate kodebaser.

4. [ ] GPU Kernels & CPU/GPU Dispatch

- [ ] Dispatch-tabell: Utvid dispatch-mekanismen din slik at operasjoner rutes til riktig kjerne basert på hvilken enhet input-dataene befinner seg på.
- [ ] Feilhåndtering: Husk å sjekke returverdier fra både kjerneoppsett (launch) og asynkron eksekvering (cudaGetLastError / hipGetLastError), da feil på GPU ofte dukker opp først ved neste synkronisering.
- [ ] Verifisering: Bruk den solide CPU-backenden din til å kjøre automatiske tester som sammenligner output fra GPU-kernels mot CPU-resultater (med en liten toleranse for flyttallsavvik).

## [ ] Phase 10 — CFD / Simulation

1. [ ] Core Discretization & Solvers

Before solving fluid equations, we need a spatial domain and numerical methods to approximate derivatives.

- [ ] Mesh engine: Build or integrate a grid generator supporting structured Cartesian grids first, then expand to unstructured triangular/tetrahedral meshes (e.g., Delaunay triangulation).
- [ ] Finite difference (FDM): Implement basic spatial and temporal discretization schemes (central difference, upwind schemes, and Runge-Kutta time stepping) on uniform grids to validate your math.
- [ ] Finite volume (FVM): Transition to FVM for conservation laws, focusing on flux calculations across cell faces, gradient reconstruction, and slope limiters for stability.
- [ ] Finite element research (FEM): Prototype weak-form formulations and basis functions (Galerkin method) to evaluate whether FEM offers advantages over FVM for complex geometries or multiphysics in your engine.

2. [ ] Physics & Fluid Dynamics

Once our numerical schemes can solve simple scalar transport (like heat diffusion or advection), introduce fluid mechanics.

- [ ] Boundary conditions: Implement ghost cells or face-flux overrides for Dirichlet (fixed velocity/pressure), Neumann (zero-gradient outflow), no-slip walls, and periodic boundaries.
- [ ] Navier–Stokes: Couple mass and momentum conservation using pressure-velocity coupling algorithms like SIMPLE, PISO, or fractional-step projection methods.
- [ ] Fluid simulation: Combine the solver components into a complete transient simulation loop, starting with 2D incompressible laminar flows (such as lid-driven cavity or flow past a cylinder) before scaling to 3D or turbulence models.

3. [ ] Performance & Output

CFD is computationally expensive, making optimization and data inspection essential for usable simulations.

- [ ] Parallel CFD: Partition the mesh into subdomains and parallelize solver loops using shared-memory threading (OpenMP/Rust Rayon), GPU compute shaders (CUDA/Vulkan/WGPU), or distributed MPI.
- [ ] Visualization/export: Export field data (velocity vectors, pressure scalars, vorticity) to standard formats like VTK/VTU for ParaView, or build a real-time renderer using streamlines, vector glyphs, and color maps.
