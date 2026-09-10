import time

import dracolix
import numpy as np

# Scale up to a size that requires heavy computing power
N = 4000
print(f"scale: {N}x{N}")
a = np.random.rand(N, N).astype(np.float64)
b = np.random.rand(N, N).astype(np.float64)

# Benchmark NumPy
start = time.perf_counter()
res_np = np.dot(a, b)
numpy_time = time.perf_counter() - start

# Benchmark DracoLIX (Fortran/Rust)
start = time.perf_counter()
res_dl = dracolix.matmatmul(a, b)
dracolix_time = time.perf_counter() - start

print(f"NumPy Time:    {numpy_time:.6f} seconds")
print(f"DracoLIX Time: {dracolix_time:.6f} seconds")

# Assert correctness
assert np.allclose(res_np, res_dl), "Mathematical mismatch detected!"
print("✅ Output validation passed successfully.")
