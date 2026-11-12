// GPU CUDA/ROCm primitives for accelerated computing - Phase 9
// Licensed under GPLv3+ (see LICENSE)

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#elif defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/hip_runtime.h>
#endif

#if defined(__CUDACC__) || defined(__HIP_PLATFORM_AMD__) ||                    \
	defined(__HIP_PLATFORM_NVIDIA__)
#define GPU_CU

#include <functional>
#include <iostream>

namespace dracolix {
namespace detail {}
namespace gpu {

// Elementwise lambdas with the __device__ specifier
// Arithmetic
auto ewise_add = [] __device__(float x, float y) -> float { return x + y; };
auto ewise_sub = [] __device__(float x, float y) -> float { return x - y; };
auto ewise_mul = [] __device__(float x, float y) -> float { return x * y; };
auto ewise_div = [] __device__(float x, float y) -> float { return x / y; };

template <typename T, typename F>
__global__ void gpu_scalar_elementwise(const T *a, const T b, T *c, int n,
									   F fn) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n) {
		c[idx] = fn(a[idx], b);
	}
}

template <typename T, typename F>
__global__ void gpu_array_elementwise(const T *a, const T *b, T *c, int n,
									  F fn) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n) {
		c[idx] = fn(a[idx], b[idx]);
	}
}

template <typename T>
void launch_kernel(const T *d_a, const T *d_b, T *d_c, int n) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	gpu_array_elementwise<T>
		<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n, ewise_add);
}

} // namespace gpu
} // namespace dracolix

#endif
