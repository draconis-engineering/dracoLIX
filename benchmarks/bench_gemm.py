"""GEMM benchmark: dracolix core vs NumPy.

NumPy is used for matrix generation and as the reference baseline
(dev-only dependency, not a runtime dependency of dracolix).
"""

import time

try:
    import numpy as np
except ImportError:
    np = None

try:
    import dracolix as dlx
except ImportError:
    dlx = None

# Scale up to a size that requires heavy computing power
N = 1000
print(f"scale: {N}x{N}")

a = np.random.rand(N, N).astype(np.float64)
b = np.random.rand(N, N).astype(np.float64)
a_dlx = dlx.array(a)
b_dlx = dlx.array(b)


def bench_numpy():
    start = time.perf_counter()
    res = np.dot(a, b)
    return res, time.perf_counter() - start


def bench_dracolix():
    if dlx is None:
        return None, None
    start = time.perf_counter()
    res = a_dlx @ b_dlx
    return res, time.perf_counter() - start


def main():
    res_np, numpy_time = bench_numpy()
    print(f"NumPy Time:    {numpy_time:.6f} seconds")

    res_dl, dracolix_time = bench_dracolix()
    if res_dl is None:
        print("DracoLIX Time: skipped (Python binding not built yet)")
    else:
        print(f"DracoLIX Time: {dracolix_time:.6f} seconds")
        assert np.allclose(res_np, np.asarray(res_dl)), "Mathematical mismatch!"
        print("Output validation passed successfully.")


if __name__ == "__main__":
    main()
