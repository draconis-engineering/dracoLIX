// DracoLIX OpenBLAS backend — optional, linked only when DRACOLIX_USE_FORTRAN=ON
// Licensed under GPL-3.0-or-later
//
// This is the `GemmKernel::Blas` path that dispatch.hpp advertises. When the
// core is built without BLAS, every entry point here returns false / unavailable
// and the dispatcher transparently falls back to the native kernels — so the
// default build stays pure C++ with no Fortran or BLAS runtime dependency.
//
// Row-major throughout: we pass CblasRowMajor and let the BLAS handle layout,
// which is the same memory contract DracoLIX's own kernels use.

#include "dracolix/kernels/dispatch.hpp"

#if defined(DRACOLIX_HAVE_BLAS)

#if defined(DRACOLIX_BLAS_FORTRAN_INTERFACE)
#include <cblas.h>
#else
#include <cblas.h>
#endif

namespace dracolix::kernels {

bool gemm_blas_available() noexcept { return true; }

// CBLAS row-major dgemm with beta = 1: the core's contract is "C is pre-zeroed,
// accumulate into it", which is exactly beta=1 on a zeroed C.
bool gemm_blas_f64(const double *A, const double *B, double *C, size_t m,
				   size_t n, size_t p) {
	if (!A || !B || !C)
		return false;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)p,
				(int)n, 1.0, A, (int)n, B, (int)p, 1.0, C, (int)p);
	return true;
}

bool gemm_blas_f32(const float *A, const float *B, float *C, size_t m,
				   size_t n, size_t p) {
	if (!A || !B || !C)
		return false;
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)p,
				(int)n, 1.0f, A, (int)n, B, (int)p, 1.0f, C, (int)p);
	return true;
}

} // namespace dracolix::kernels

#else // !DRACOLIX_HAVE_BLAS

namespace dracolix::kernels {

bool gemm_blas_available() noexcept { return false; }
bool gemm_blas_f64(const double *, const double *, double *, size_t, size_t,
				   size_t) {
	return false;
}
bool gemm_blas_f32(const float *, const float *, float *, size_t, size_t,
				   size_t) {
	return false;
}

} // namespace dracolix::kernels

#endif // DRACOLIX_HAVE_BLAS