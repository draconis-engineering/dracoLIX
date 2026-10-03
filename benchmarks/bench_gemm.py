"""GEMM benchmark: dracolix core vs NumPy.

NumPy is used for matrix generation and as the reference baseline
(dev-only dependency, not a runtime dependency of dracolix).
"""

import time

import dracolix as dlx
import numpy as np


def bench_numpy(array1, array2):
    start = time.perf_counter()
    res = np.dot(array1, array2)
    return res, time.perf_counter() - start


def bench_dracolix(array1, array2):
    dxa = dlx.array(array1)
    dxb = dlx.array(array2)
    if dlx is None:
        print("DracoLIX Time: skipped (Python binding not built yet)")
        return None, None
    start = time.perf_counter()
    res = dxa @ dxb
    return res, time.perf_counter() - start


def main():
    for n in [100, 500, 1000, 2000, 3000, 4000]:
        array1 = np.random.rand(n, n).astype(np.float64)
        array2 = np.random.rand(n, n).astype(np.float64)
        print(f"size: {n}")

        res_np, numpy_time = bench_numpy(array1, array2)
        print(f"NumPy Time:    {numpy_time:.6f} seconds")

        res_dl, dracolix_time = bench_dracolix(array1, array2)
        print(f"DracoLIX Time: {dracolix_time:.6f} seconds")

        assert np.allclose(res_np, np.asarray(res_dl)), "Mathematical mismatch!"

        print("Output validation passed successfully.")


if __name__ == "__main__":
    main()
