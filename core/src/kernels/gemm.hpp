#pragma once
#include <cstddef>

// Core kernels - pure C++ with no Python/Julia dependency.
// Fortran fallback is linked separately via iso_c_binding.
namespace dracolix::kernels {

// Row-major GEMM: C(m x p) = A(m x n) * B(n x p)
// All pointers must be contiguous and non-aliased.
void gemm_f64(const double* A, const double* B, double* C,
              size_t m, size_t n, size_t p);
void gemm_f32(const float* A, const float* B, float* C,
              size_t m, size_t n, size_t p);

// Fortran BLAS fallback (implemented in linalg.f90)
extern "C" {
void matmatmul_c(const double* a, const double* b, double* res,
                 int n, int m, int p);
void matvecmul_c(const double* a, const double* b, double* res,
                 int n, int m);
}

} // namespace dracolix::kernels
