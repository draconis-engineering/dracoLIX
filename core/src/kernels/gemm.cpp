// DracoLIX GEMM dispatch — chooses the backend, then calls it.
// Licensed under GPL-3.0-or-later
//
// The scalar ikj kernel in this file is the *reference* implementation: it
// stays exactly as it was so it can be used to validate the vectorized and
// blocked kernels. Everything else routes through gemm_blocked.cpp.
//
// Backend selection is a runtime decision (see dispatch.hpp): the ISA is probed
// on the running CPU and the vector kernels are compiled in behind `target`
// attributes, so a single portable binary picks the best available path.

#include "gemm.hpp"
#include "dracolix/kernels/dispatch.hpp"
#include "dracolix/thread_pool.hpp"

namespace dracolix::kernels {

// Cache-blocked / packed / threaded double kernel (gemm_blocked.cpp).
// `kernel` pins the ISA so the backend benchmark can measure avx2 vs avx512
// separately; pass GemmKernel::Scalar to get the reference path.
void gemm_blocked_f64(const double *A, const double *B, double *C, size_t m,
					  size_t n, size_t p, size_t nthreads, GemmKernel kernel);
void gemm_blocked_f32(const float *A, const float *B, float *C, size_t m,
					  size_t n, size_t p, size_t nthreads, GemmKernel kernel);
size_t gemm_recommended_threads(size_t m, size_t n, size_t p);

// ---------------------------------------------------------------------------
// Reference scalar kernels — cache-friendly (i,k,j). Do not "optimize" these:
// they are the correctness oracle for every other backend.
// ---------------------------------------------------------------------------
#if defined(__GNUC__) && !defined(__clang__)
// Tell GCC globally to disable auto-vectorization for these specific functions
__attribute__((optimize("no-tree-vectorize")))
#endif
void gemm_f64(const double *A, const double *B, double *C, size_t m, size_t n,
			  size_t p) {
	// Caller must zero C. Core owns this contract explicitly.
	for (size_t i = 0; i < m; ++i) {
		for (size_t k = 0; k < n; ++k) {
			const double aik = A[i * n + k];

#if defined(_MSC_VER) && !defined(__clang__)
#pragma loop(no_vector)
#elif defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
			for (size_t j = 0; j < p; ++j)
				C[i * p + j] += aik * B[k * p + j];
		}
	}
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
void gemm_f32(const float *A, const float *B, float *C, size_t m, size_t n,
			  size_t p) {
	for (size_t i = 0; i < m; ++i) {
		for (size_t k = 0; k < n; ++k) {
			const float aik = A[i * n + k];

#if defined(_MSC_VER) && !defined(__clang__)
#pragma loop(no_vector)
#elif defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
			for (size_t j = 0; j < p; ++j)
				C[i * p + j] += aik * B[k * p + j];
		}
	}
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
void dispatch_gemm_f64(const double *A, const double *B, double *C, size_t m,
					   size_t n, size_t p) {
	if (m == 0 || n == 0 || p == 0)
		return;

	const GemmBackend be = gemm_backend();
	switch (be.backend) {
	case GemmBackendKind::Blas:
		if (gemm_blas_f64(A, B, C, m, n, p))
			return;
		[[fallthrough]];
	case GemmBackendKind::Native:
	default: {
		const size_t nt = gemm_recommended_threads(m, n, p);
		gemm_blocked_f64(A, B, C, m, n, p, nt, be.kernel);
		return;
	}
	case GemmBackendKind::Scalar:
		gemm_f64(A, B, C, m, n, p);
		return;
	}
}

void dispatch_gemm_f32(const float *A, const float *B, float *C, size_t m,
					   size_t n, size_t p) {
	if (m == 0 || n == 0 || p == 0)
		return;

	const GemmBackend be = gemm_backend();
	switch (be.backend) {
	case GemmBackendKind::Blas:
		if (gemm_blas_f32(A, B, C, m, n, p))
			return;
		[[fallthrough]];
	case GemmBackendKind::Native:
	default:
		// f32 now shares the same packed/blocked/threaded machinery as f64,
		// with its own microkernels (AVX-512 8x32, AVX2 6x16).
		gemm_blocked_f32(A, B, C, m, n, p, gemm_recommended_threads(m, n, p),
						 be.kernel);
		return;
	case GemmBackendKind::Scalar:
		gemm_f32(A, B, C, m, n, p);
		return;
	}
}

} // namespace dracolix::kernels
