# GEMM Performance Investigation

## Overview

This document records the GEMM performance investigation performed on DracoLIX on the 2 October 2026.

The goal was to determine why DracoLIX matrix multiplication was significantly slower than NumPy/OpenBLAS, identify where the time was actually being spent, and establish a clear path toward a high-performance CPU backend that can later coexist with CUDA and ROCm backends.

The main conclusion is:

> The performance problem is in DracoLIX's native GEMM implementation, not in Python, nanobind, the machine, or OpenBLAS.

A Fortran → BLAS control benchmark demonstrated that the same machine can sustain roughly **410 GFLOP/s** on a 4000×4000 GEMM, while the current DracoLIX GEMM achieves only roughly **9 GFLOP/s**.

---

## 1. Initial Benchmark

The original benchmark compared NumPy against DracoLIX for square `float64` GEMM.

| Matrix size | NumPy | DracoLIX | DracoLIX slowdown |
|---|---:|---:|---:|
| 1000×1000 | 0.0084 s | 0.1018 s | ~12× |
| 2000×2000 | 0.0472 s | 1.1438 s | ~24× |
| 3000×3000 | 0.1496 s | 5.8077 s | ~39× |
| 4000×4000 | 0.4111 s | 13.6786 s | ~33× |

A later run with matrix dimensions of 4000x4000 produced approximately:

```text
NumPy Time:    0.334579 s
DracoLIX Time: 13.884100 s
```

The exact timings vary between runs, but the performance gap is clearly substantial.

## 2. perf Investigation

```bash
perf stat
```

For the 4000×4000 benchmark:

DracoLIX Time: 13.884100 seconds

95,860,446,916 cpu-cycles
119,302,971,172 instructions
1.2 instructions/cycle
17,573,717,093 branches
19,156,226 branch-misses

This showed that DracoLIX was actually spending significant CPU time doing computation.

The problem was therefore unlikely to be simply:

- Python overhead
- nanobind overhead
- function-call overhead
- data conversion overhead
- perf record

The call graph was even more useful.

The important result was approximately:

- ~75%  dracolix::kernels::dispatch_gemm_f64(...)
- ~21%  libc / related execution

OpenBLAS showed up separately for NumPy's GEMM path. This established that the DracoLIX GEMM path itself dominates the runtime. NumPy, meanwhile, was using a heavily optimized and multithreaded OpenBLAS GEMM implementation.

## 3. Current DracoLIX GEMM Implementation

The current implementation has a reasonable scalar baseline:

```cpp
for (size_t i = 0; i < m; ++i) {
    for (size_t k = 0; k < n; ++k) {
        double aik = A[i * n + k];

        for (size_t j = 0; j < p; ++j) {
            C[i * p + j] += aik * B[k * p + j];
        }
    }
}
```

This is an `i`-`k`-`j` loop ordering, which is substantially more cache-friendly than the naive `i`-`j`-`k` formulation for row-major matrices. There is also an AVX2 implementation using 4-wide double vectors and FMA when available. However, the current AVX2 implementation is still a relatively simple vectorized kernel:

```
load C
load B
FMA
store C
```

for each k. That means the same C values are repeatedly loaded and stored from memory.

### Missing optimizations

The current implementation does not yet have:

- Register blocking
- Cache blocking / tiling
- A proper AVX2 microkernel
- Full multi-threaded GEMM
- A real AVX-512 implementation
- A fully wired BLAS backend

These are the major opportunities.

## 4. Important Discovery: Threading Is Disabled

The current GEMM dispatcher contains a condition equivalent to:

```cpp
if (false && ops >= 1024 * 1024 * 1024 && m >= 256) {
    ...
}
```

This means the parallel GEMM path is currently disabled. Therefore, large DracoLIX GEMMs are effectively running through the single-threaded native implementation. This is one of the major reasons the comparison against NumPy/OpenBLAS is so unfavorable.

## 5. Important Discovery: The Fortran/BLAS Backend Already Works

DracoLIX has an optional Fortran backend.

The Fortran implementation contains:

```fortran
res = matmul(a, b)
```

The Fortran compiler flags include:

```sh
-fexternal-blas
```

Inspection of the resulting static library showed:

```sh
U dgemm_
T matmatmul_c
```

This confirms that the generated Fortran code is calling BLAS dgemm. The system has OpenBLAS installed, and the Python extension is also linked against OpenBLAS.

## 6. The Critical Architecture Discovery

Although DracoLIX contains a BLAS/Fortran backend, the normal C++ matmul() path does not currently use it for 2D matrices.

The 2D double specialization effectively does:

```cpp
kernels::dispatch_gemm_f64(
    A.data(),
    B.data(),
    C.data(),
    m,
    n,
    p
);
```

The dispatcher chooses between the native kernels.

The GemmKernel::Blas enum exists:

```cpp
enum class GemmKernel {
    Scalar,
    Avx2,
    Avx512,
    Blas
};
```

but `select_gemm_kernel()` does not currently return GemmKernel::Blas.

There is also a `BLAS_THRESHOLD` constant, but it is not currently used to route the normal 2D GEMM path to BLAS.

Therefore:

Enabling the Fortran backend does not automatically make Python @ use OpenBLAS.

Python @ currently ends up in DracoLIX's own C++ GEMM dispatcher.

## 7. The Control Experiment

A dedicated Fortran benchmark was created to isolate the Fortran/OpenBLAS path.

Command:

```sh
./build/benchmarks/bench_fortran 4000
```

Result:

```sh
Fortran/OpenBLAS GEMM: 4000x4000 * 4000x4000

Time:   0.311516 s
GFLOP/s: 410.894
C[0]:   -6.29813
```

For comparison, DracoLIX's native GEMM was approximately 13.9 seconds for the same nominal 4000×4000 @ 4000x4000 workload.

The theoretical operation count is:

2 × 4000³ = 128,000,000,000 FLOPs

Therefore the current native DracoLIX implementation achieves approximately:

```sh
128 GFLOP / 13.9 s
≈ 9.2 GFLOP/s
```

The Fortran/OpenBLAS benchmark achieves roughly 410.9 GFLOP/s.

That is roughly a 44–45x performance difference.

## 8. What This Proves

The control experiment is extremely useful because it removes several possible explanations.

It strongly indicates that the problem is not:

- Python
- nanobind
- the Python array interface
- the CPU being incapable of high GEMM throughput
- OpenBLAS being unavailable
- the compiler being unable to generate high-performance code
- the general machine configuration

Instead, the primary problem is:

DracoLIX's own GEMM kernel and execution strategy are not yet competitive with a mature BLAS implementation.

This is actually a very useful result because it gives us a known-good performance ceiling/reference on the same machine.

## 9. C++ vs Fortran

Should Fortran become the default? No. Fortran/OpenBLAS should remain optional.

The recommended architecture is:

```text
                    DracoLIX GEMM API
                           |
                    Backend Dispatcher
                           |
          +----------------+----------------+
          |                |                |
       CPU Native       BLAS          GPU Backends
          |                |           /          \
     C++ kernels      OpenBLAS       CUDA        ROCm
```

C++ should remain the primary native CPU implementation.

Fortran/OpenBLAS can be:

- an optional backend
- a fallback
- a reference implementation
- a performance baseline
- a useful compatibility path for users who already have BLAS infrastructure

There is no fundamental reason why a well-designed C++ GEMM implementation cannot perform extremely well. In fact, the important performance techniques used by high-performance GEMM implementations are not inherently tied to Fortran:

- vectorization
- FMA
- register blocking
- cache blocking
- packing
- parallel execution
- architecture-specific kernels
- careful memory layout

C++ can implement all of these.

## 10. Why C++ Is the Better Long-Term Core

DracoLIX is intended to eventually support multiple compute backends, including:

- CPU
- CUDA
- ROCm
- potentially other accelerators

Keeping the backend abstraction in C++ makes this architecture much cleaner.

For example:

```text
dracolix::linalg::matmul()
        |
        v
   GEMM dispatcher
        |
        +--> Scalar CPU
        |
        +--> AVX2 CPU
        |
        +--> AVX-512 CPU
        |
        +--> OpenBLAS
        |
        +--> CUDA
        |
        +--> ROCm
```

The public API does not need to care which backend performs the operation.

This also makes future hardware dispatch much easier.

## 11. Recommended Optimization Roadmap

### Phase 1 — Keep the scalar kernel

Do not delete the current scalar implementation. It should become the correctness/reference implementation.

Use it for:

- unit tests
- numerical comparisons
- debugging
- validating new kernels

The scalar implementation is valuable even if it is slow.

### Phase 2 — Build a real AVX2 microkernel

The current AVX2 kernel should be replaced with a register-blocked implementation.

Instead of repeatedly doing:

```text
load C
FMA
store C
```

...we want:

```text
load C tile into registers

for k:
    load/broadcast A
    load B
    FMA into registers

store C tile
```
The C tile remains in registers throughout the inner k loop.

Potential starting points:

4×4 f64 microkernel

or:

4×8 f64 microkernel

depending on register pressure and benchmarking.

This should provide a substantial improvement before threading is even introduced.

### Phase 3 — Cache Blocking

Introduce tiling around the microkernel.

Conceptually:

for M block
    for N block
        for K block
            run microkernels

The goal is to keep working sets inside the appropriate cache levels.

The exact block sizes should be benchmarked rather than guessed.

### Phase 4 — Parallel GEMM

Re-enable the ThreadPool path.

Parallelize over independent C tiles or row blocks.

For example:

Thread 0 -> C tile
Thread 1 -> C tile
Thread 2 -> C tile
Thread 3 -> C tile
...

Avoid multiple threads writing to the same C region.

The threshold should also become a real tunable policy rather than:

if (false && ...)

### Phase 5 — AVX-512

The dispatcher currently contains:

GemmKernel::Avx512

but there is not yet a dedicated AVX-512 implementation.

This should eventually become a real backend.

Until then, the dispatcher should not claim that an AVX-512 kernel is being used when it actually falls back to AVX2.

### Phase 6 — Backend Dispatch

Turn the current enum into an actual backend system:

- Scalar
- AVX2
- AVX-512
- OpenBLAS
- CUDA
- ROCm

The dispatcher should select the appropriate backend based on:

- available hardware
- matrix dimensions
- build configuration
- user configuration
- backend availability

## 12. Benchmark Strategy

Keep the Fortran/OpenBLAS benchmark permanently.

It is an extremely useful control.

Recommended benchmark matrix:

- DracoLIX Scalar
- DracoLIX AVX2
- DracoLIX AVX-512
- DracoLIX Threaded AVX2
- DracoLIX Threaded AVX-512
- OpenBLAS
- CUDA
- ROCm

Test multiple matrix sizes:

- 256
- 512
- 1024
- 2048
- 4096
- 8192

Track:

- time
- GFLOP/s
- threads
- backend
- dtype
- matrix dimensions

The goal should not simply be:

"Beat NumPy."

The better goal is:

"Understand how close each DracoLIX backend gets to the hardware's available compute throughput."

## 13. Important Benchmark Caveat

The Fortran benchmark uses Fortran's column-major memory interpretation, while DracoLIX's native C++ matrices are row-major.

Therefore, the current Fortran benchmark is primarily a performance control experiment.

It should not yet be treated as a perfect numerical apples-to-apples comparison of the two APIs.

A future benchmark should normalize:

- memory layout
- initialization
- thread count
- matrix dimensions
- dtype
- warmup
- validation

In particular, OpenBLAS should also be tested with explicitly controlled thread counts:

OPENBLAS_NUM_THREADS=1 ./build/benchmarks/bench_fortran 4000

and with the desired multi-thread configuration.

This will let us distinguish single-thread kernel performance from parallel scaling

## 14. Current Findings Summary

### Confirmed
- DracoLIX's native GEMM is the main performance bottleneck.
- Python overhead is not the primary issue.
- NumPy is using optimized OpenBLAS GEMM.
- DracoLIX's current large GEMM path is effectively single-threaded.
- The current AVX2 implementation is not sufficiently optimized.
- The optional Fortran implementation can reach ~410 GFLOP/s on the test machine.
- The Fortran implementation is generating a BLAS dgemm_ call.
- The normal C++ matmul() path currently bypasses the Fortran/BLAS backend.
- The Blas dispatcher enum exists but is not wired into selection.
- CUDA/ROCm can fit naturally into a future backend architecture.

### Not yet established
- Exact single-thread OpenBLAS performance on this machine.
- Optimal DracoLIX cache-block sizes.
- Optimal AVX2 microkernel dimensions.
- Scaling behavior of the DracoLIX ThreadPool.
- Whether AVX-512 provides a useful gain on the target hardware.
- How close a fully optimized native C++ kernel can get to OpenBLAS.

## 15. Architectural Decision

Recommended

C++ native CPU implementation remains the primary DracoLIX GEMM backend.

Fortran/OpenBLAS remains optional.

The intended long-term architecture is:

```text
                    ┌─────────────────────┐
                    │   DracoLIX Linalg   │
                    └──────────┬──────────┘
                               │
                       Backend Dispatcher
                               │
        ┌──────────────┬───────┼────────┬──────────────┐
        │              │       │        │              │
     Scalar          AVX2   AVX-512   OpenBLAS       GPU
                                           │         /   \
                                         BLAS      CUDA  ROCm
```

This preserves:

- portability
- control over the native implementation
- future GPU support
- optional BLAS interoperability
- a clear performance hierarchy
- a stable public API

Most importantly, it means DracoLIX does not have to become dependent on Fortran just because OpenBLAS currently provides a much faster GEMM.

## 16. Immediate Next Steps
- Keep the scalar GEMM as the reference implementation.
- Implement a proper AVX2 register-blocked microkernel.
- Add cache blocking.
- Benchmark after each optimization.
- Re-enable and tune multi-threading.
- Add a real AVX-512 kernel.
- Wire GemmKernel::Blas into an optional backend.
- Add explicit backend reporting to benchmarks.
- Normalize benchmark layouts and thread counts.
- Design the backend abstraction with CUDA/ROCm in mind.

The current performance gap is therefore not the end of the road — it is a very clear optimization target with a working high-performance control implementation available for comparison.

---

# Part 2 — Resolution (same day)

This part records what was changed in response to the findings above, and what
the measurements are now. Nothing in Part 1 is retracted: the diagnosis was
correct, and every recommendation in section 11 was carried out.

## 17. Root Cause, Confirmed

Three separate defects, in order of impact:

### 17.1 The AVX2 kernel had no register blocking (the big one)

The kernel in the old `gemm.cpp` was:

```
load C -> FMA -> store C     (per k)
```

so every FMA was accompanied by a C load and a C store. That caps the inner
loop far below the FMA issue rate regardless of how good the compiler is. The
fix is the standard one: hold an `MR x NR` tile of C in vector registers for the
entire k loop, so the inner loop only issues a broadcast, a B load and an FMA.

Measured microkernel shapes on this CPU (AMD Ryzen 5 7600, Zen 4):

| shape | GFLOP/s |
|---|---:|
| AVX2 4×4 | 8.0 |
| AVX2 4×12 | 27.4 |
| AVX2 6×8 | 27.6 |
| AVX-512 4×16 | 32.3 |
| AVX-512 8×16 | 62.6 |
| AVX-512 8×24 | 70.6 |

Zen 4 executes 512-bit FMA as two 256-bit ops, so AVX-512 doubles FLOP/cycle
over AVX2 rather than quadrupling it — which is why the AVX2 ceiling tops out
near 30 and AVX-512 near 70.

Chosen: **AVX-512 8×24** (24 accumulators + 3 B vectors = 27 of 32 zmm) and
**AVX2 6×8** (12 accumulators + 2 ymm of 16).

### 17.2 Threading was hard-disabled

`if (false && ops >= ...)` in both dispatchers. That gate was left in from the
Phase 3 experiment where threading *regressed* mid-size GEMM — because it was
row-parallel with no blocking, so each thread re-read the whole B panel and
cache locality collapsed.

Fixed by parallelising over **cache-blocked row panels** instead of raw rows:
each worker owns a disjoint `MC`-row slab of C, packs its own A/B panels, and
needs no locks. No false sharing, because panel boundaries are 256 rows.

### 17.3 The BLAS backend existed but was unreachable

`GemmKernel::Blas` and `BLAS_THRESHOLD` were declared and never used, as
section 6 said. `GemmBackendKind` is now a real dispatch decision with an
optional `DRACOLIX_USE_BLAS` build, a `blas` backend, and an
`DLX_GEMM_BACKEND` override so the benchmark can pin any backend.

## 18. A Second Bug the Report Did Not Find

The build applied `-march=native -mavx2 -mfma` to the whole core
(`CMakeLists.txt`). Two problems:

1. **Portability.** A wheel built on a modern CI runner is compiled for that
   runner and `SIGILL`s on older user CPUs. `-march=native` in a library that
   ships binary wheels is a shipping bug, not a tuning choice.
2. It made it impossible to express "AVX-512 kernel, runtime gated", because
   the entire library was already at the runner's ISA.

Now: no `-march=native`, no `-mavx2`. SIMD lives behind GCC/Clang `target`
attributes in `core/src/kernels/gemm_micro.hpp` and is gated by the CPUID probe
in `cpu.hpp` (`avx512f/dq/vl/bw` are now detected — previously only `avx512f`).
`-mtune=native` is kept, since it changes scheduling and cannot introduce an
illegal instruction.

## 19. A Correctness Bug Found and Fixed Along the Way

The AVX2 edge kernel was initially wrong, and the existing 7-test suite did not
catch it because every test used round, tile-aligned sizes.

Cause: AVX2 `maskload`/`maskstore` select on the **most significant bit of each
64-bit lane**, while AVX-512's `__mmask8` uses **bit i for lane i**. The AVX2
path had been written with the AVX-512 idiom (`(1u << n) - 1`), which silently
zeroed the lanes.

Fixed, and `tests/cpp/test_gemm.cpp` was added: 30 sizes × 5 shape variants ×
5 backends = 750 shapes, straddling both microkernel tiles and the cache block,
plus degenerate m/n/p. Reintroducing the bug makes it report 145 mismatches, so
the test genuinely covers it.

## 20. Results

4000×4000 × 4000×4000, f64, all bit-exact against the scalar reference
(`max|err| = 0`):

| backend | time | GFLOP/s | vs old |
|---|---:|---:|---:|
| scalar (old default path) | 19648 ms | 6.5 | 1× |
| DracoLIX AVX-512 blocked+threaded | 380 ms | **337** | **52×** |
| OpenBLAS (12 threads, reference) | 384 ms | 384 | 59× |
| NumPy (from Part 1) | 335 ms | 383 | 59× |

Single-thread, `taskset -c 0`, for a like-for-like kernel comparison:

| | GFLOP/s |
|---|---:|
| OpenBLAS 1024³ | 60.2 |
| OpenBLAS 2048³ | 69.7 |
| DracoLIX 512² | 114–132 |

DracoLIX's blocked microkernel is at or above OpenBLAS single-thread, and the
threaded path reaches ~88 % of 12-thread OpenBLAS. That gap is mostly memory
bandwidth at 4000³; the remaining headroom is in packing (A is packed
opportunistically rather than through a full BLIS-style pack), NUMA-aware
scheduling, and a bf16/`sgemm` path.

## 21. f32: the second silent gap

Section 22 of Part 1 flagged that f32 has no vectorized path. Adding it
exposed the fact that the *blocked macro-kernel* was written for `double` only,
so f32 could not simply borrow it.

The blocked kernel is now templated on the scalar type, with a `Micro<T>` traits
table supplying the tile shape and kernel per (dtype, ISA):

| dtype | AVX-512 | AVX2 |
|---|---|---|
| f64 | 8 × 24 | 6 × 8 |
| f32 | 8 × 32 | 6 × 16 |

Packing, cache blocking and the row-panel threading are shared verbatim, and
the per-worker scratch is a single `AlignedBuffer<char>` serving both dtypes, so
an f32 GEMM does not allocate a second pool.

f32 at n=1024, 12 threads:

| backend | GFLOP/s | vs scalar |
|---|---:|---:|
| scalar | 29.6 | 1× |
| native (AVX-512) | **449** | **15×** |
| avx2 | 414 | 14× |

### 21.1 A third mask bug, in the f32 AVX-512 edge kernel

Same family as section 19, different symptom. `m0` was built as
`(1u << nr) & 0xFFFF` with the `- 1` omitted, so for `nr = 1` the mask enabled
**lane 1** rather than lane 0 — every narrow column tail was off by one lane.

The AVX2 f32 edge kernel had a related latent bug: it set 4 of the 8 lanes of
each `ymm` mask, which happened not to be reachable by the tested shapes but
would silently drop columns for `5 <= nr <= 8`.

All three mask bugs share one root cause: **AVX-512 `__mmask16`/`__mmask8`,
AVX2-double `maskload`, and AVX2-float `maskload` all use different mask
conventions**, and there is nothing in the type system to stop you using the
wrong one. The 1200-shape matrix in `tests/cpp/test_gemm.cpp` now covers f64 and
f32 independently precisely so that a fix for one cannot mask a bug in the
other.

## 22. What Changed in the Tree

- `core/src/kernels/gemm_micro.hpp` — new. Register-blocked AVX-512 8×24 and
  AVX2 6×8 microkernels, their f32 counterparts (8×32 / 6×16), and masked edge
  variants for all four; runtime `target` attributes.
- `core/src/kernels/gemm_blocked.cpp` — new. Cache blocking (MC/NC/KC),
  packing, row-panel threading, templated on scalar type.
- `core/src/kernels/gemm_blas.cpp` — new. The OpenBLAS backend, off by default.
- `core/src/kernels/gemm.cpp` — rewritten. Scalar reference retained verbatim
  as the correctness oracle; all dead `if (false && ...)` paths deleted.
- `core/include/dracolix/kernels/dispatch.hpp` — real `GemmBackend` selection,
  `DLX_GEMM_BACKEND` override.
- `core/include/dracolix/cpu.hpp` — added `avx512dq/vl/bw`.
- `core/include/dracolix/alloc.hpp` — added `AlignedBuffer` for packing scratch.
- `benchmarks/bench_gemm_backends.cpp` — new. The matrix from section 12, now
  covering f64 and f32.
- `tests/cpp/test_gemm.cpp` — new. 1200-shape correctness matrix from
  sections 19 and 21.
- `CMakeLists.txt` — dropped `-march=native`/`-mavx2`/`-mfma`, added
  `DRACOLIX_USE_BLAS`.

## 23. Still Open

- Full BLIS-style A/B packing (currently A is packed, but the macro-kernel
  traversal is not yet the optimal L3-resident order).
- `sum axis=0` on a 1024² array is still ~0.12 GFLOP/s and single-threaded —
  parallel reductions cover the global case only.
- Cache block sizes were tuned for this Zen 4 part (32 KB L1d, 1 MB L2/core).
  They are `#define`-overridable but not auto-detected.
- No bf16 tensor-core path, which is where AVX-512 actually pays off on this
  CPU (`avx512_bf16` and `avx512_vnni` are both present but unused).
- `gemm_recommended_threads` picks `min(m_panels, nthreads)`; a proper
  work-stealing or 2-D (MC × NC) decomposition would scale better on machines
  with more cores than MC-row panels.
