#pragma once

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

#include "kernels/gpu.cu"
#include <functional>
#include <iostream>

template <typename T, typename F>
__global__ void gpu_scalar_elementwise(const T *a, const T b, T *c, int n,
									   F fn);

template <typename T, typename F>
__global__ void gpu_array_elementwise(const T *a, const T *b, T *c, int n,
									  F fn);

template <typename T>
void launch_kernel(const T *d_a, const T *d_b, T *d_c, int n);
