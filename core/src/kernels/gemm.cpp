#include "gemm.hpp"
#include "dracolix/kernels/dispatch.hpp"
#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

namespace dracolix::kernels {

// Cache-friendly (i,k,j) ordering. Scalar baseline for Phase 3 to replace with
// SIMD/BLAS.
void gemm_f64(const double *A, const double *B, double *C, size_t m, size_t n,
			  size_t p) {
	// Caller must zero C. Core owns this contract explicitly.
	for (size_t i = 0; i < m; ++i) {
		for (size_t k = 0; k < n; ++k) {
			double aik = A[i * n + k];
			for (size_t j = 0; j < p; ++j) {
				C[i * p + j] += aik * B[k * p + j];
			}
		}
	}
}

void gemm_f32(const float *A, const float *B, float *C, size_t m, size_t n,
			  size_t p) {
	for (size_t i = 0; i < m; ++i) {
		for (size_t k = 0; k < n; ++k) {
			float aik = A[i * n + k];
			for (size_t j = 0; j < p; ++j) {
				C[i * p + j] += aik * B[k * p + j];
			}
		}
	}
}

// ---- SIMD variants (compile-time specialization, runtime dispatch checks
// feature) ----
#if defined(__AVX2__)
static void gemm_f64_avx2(const double *A, const double *B, double *C, size_t m,
						  size_t n, size_t p) {
	// 4-wide FMA: C row tiled, B row broadcast
	for (size_t i = 0; i < m; ++i) {
		for (size_t k = 0; k < n; ++k) {
			__m256d aik = _mm256_broadcast_sd(&A[i * n + k]);
			size_t j = 0;
			for (; j + 4 <= p; j += 4) {
				__m256d c = _mm256_loadu_pd(&C[i * p + j]);
				__m256d b = _mm256_loadu_pd(&B[k * p + j]);
#if defined(__FMA__)
				c = _mm256_fmadd_pd(aik, b, c);
#else
				c = _mm256_add_pd(c, _mm256_mul_pd(aik, b));
#endif
				_mm256_storeu_pd(&C[i * p + j], c);
			}
			for (; j < p; ++j)
				C[i * p + j] += A[i * n + k] * B[k * p + j];
		}
	}
}
static void gemm_f32_avx2(const float *A, const float *B, float *C, size_t m,
						  size_t n, size_t p) {
	for (size_t i = 0; i < m; ++i) {
		for (size_t k = 0; k < n; ++k) {
			__m256 aik = _mm256_broadcast_ss(&A[i * n + k]);
			size_t j = 0;
			for (; j + 8 <= p; j += 8) {
				__m256 c = _mm256_loadu_ps(&C[i * p + j]);
				__m256 b = _mm256_loadu_ps(&B[k * p + j]);
#if defined(__FMA__)
				c = _mm256_fmadd_ps(aik, b, c);
#else
				c = _mm256_add_ps(c, _mm256_mul_ps(aik, b));
#endif
				_mm256_storeu_ps(&C[i * p + j], c);
			}
			for (; j < p; ++j)
				C[i * p + j] += A[i * n + k] * B[k * p + j];
		}
	}
}
#endif

void dispatch_gemm_f64(const double *A, const double *B, double *C, size_t m,
					   size_t n, size_t p) {
	auto k = select_gemm_kernel(m, n, p);
#if defined(__AVX2__)
	if (k == GemmKernel::Avx2 || k == GemmKernel::Avx512) {
// runtime guard: if CPU actually lacks AVX2, fallback
#if defined(__GNUC__)
		if (__builtin_cpu_supports("avx2")) {
			gemm_f64_avx2(A, B, C, m, n, p);
			return;
		}
#endif
	}
#endif
	(void)k;
	gemm_f64(A, B, C, m, n, p);
}

void dispatch_gemm_f32(const float *A, const float *B, float *C, size_t m,
					   size_t n, size_t p) {
	auto k = select_gemm_kernel(m, n, p);
#if defined(__AVX2__)
	if (k == GemmKernel::Avx2 || k == GemmKernel::Avx512) {
#if defined(__GNUC__)
		if (__builtin_cpu_supports("avx2")) {
			gemm_f32_avx2(A, B, C, m, n, p);
			return;
		}
#endif
	}
#endif
	(void)k;
	gemm_f32(A, B, C, m, n, p);
}

} // namespace dracolix::kernels
