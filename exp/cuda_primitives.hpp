#pragma once

// CUDA primitives for GPU-accelerated HPC - Phase 9
//
// Licenced under GPLv3+, see LICENSE

#include <cuda_runtime.h>
#include <functional>
#include <iostream>

namespace dracolix::cuda {

// Rewrite the kernel using a template parameter for the callable type
template <typename F>
__global__ void cuda_elementwise(const float *a, const float *b, float *c,
								 int n, F fn) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n) {
		// The callable is completely inlined here by the compiler
		c[idx] = fn(a[idx], b[idx]);
	}
}

// Launch it from Host (CPU) code using a __device__ lambda
void launch_kernel(const float *d_a, const float *d_b, float *d_c, int n) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	// --- Elementwise lambdas with the __device__ specifier --- //

	// Arithmetic
	auto ew_add = [] __device__(float x, float y) { return x + y; };
	auto ew_sub = [] __device__(float x, float y) { return x - y; };
	auto ew_mul = [] __device__(float x, float y) { return x * y; };
	auto ew_div = [] __device__(float x, float y) { return x / y; };

	cuda_elementwise<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n,
														 ew_add);
}

} // namespace dracolix::cuda
