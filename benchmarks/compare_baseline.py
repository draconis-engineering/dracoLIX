#!/usr/bin/env python3
"""compare_baseline.py — Run C++ bench_core and NumPy baseline side-by-side.
Phase 3 baseline comparison (ROADMAP: DracoLIX vs NumPy vs OpenBLAS).

Usage:
  cmake --build build && ./build/benchmarks/bench_core 50 10
  python benchmarks/compare_baseline.py  # also runs NumPy GEMM
"""
import subprocess, sys, time, pathlib

def run_cpp():
    exe = pathlib.Path("build/benchmarks/bench_core")
    if not exe.exists():
        print(f"[skip] {exe} not built (cmake --build build)")
        return
    print("="*60)
    print("C++ bench_core")
    print("="*60)
    subprocess.run([str(exe), "50", "10"], check=False)

def run_numpy():
    try:
        import numpy as np
    except ImportError:
        print("[skip] numpy not installed")
        return
    print("\n" + "="*60)
    print(f"NumPy {np.__version__} — GEMM baseline")
    print(" BLAS:", np.__config__.blas_opt_info.get('libraries', '?') if hasattr(np.__config__, 'blas_opt_info') else '?')
    print("="*60)
    for n in [64, 256, 512, 1000]:
        a = np.random.rand(n,n)
        b = np.random.rand(n,n)
        # warmup
        _ = a @ b
        t0 = time.perf_counter()
        iters = 10 if n >= 512 else 20
        for _ in range(iters):
            c = a @ b
        elapsed = time.perf_counter() - t0
        flops = 2 * n**3 * iters
        gflops = flops / elapsed / 1e9
        print(f" matmul {n}x{n:4d}  {iters:3d} iters  {elapsed:7.4f}s  {elapsed/iters*1000:7.2f}ms  {gflops:6.2f} GFLOP/s")
    # batched (numpy matmul broadcasts)
    try:
        a = np.random.rand(8,128,128); b = np.random.rand(8,128,128)
        t0 = time.perf_counter()
        for _ in range(20): c = a @ b
        elapsed = time.perf_counter() - t0
        flops = 2*8*128**3 * 20
        print(f" batched 8x128x128  20 iters  {elapsed:7.4f}s  {gflops:6.2f} GFLOP/s (approx)")
    except Exception as e:
        print(" batched skip:", e)

if __name__ == "__main__":
    run_cpp()
    run_numpy()
    print("\nDone. For BLAS/OpenBLAS/MKL, compare `ldd build/benchmarks/bench_core` and `numpy.show_config()`.")
