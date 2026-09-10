#include "gemm.hpp"

namespace dracolix::kernels {

// Cache-friendly (i,k,j) ordering. Scalar baseline for Phase 3 to replace with SIMD/BLAS.
void gemm_f64(const double* A, const double* B, double* C,
              size_t m, size_t n, size_t p) {
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

void gemm_f32(const float* A, const float* B, float* C,
              size_t m, size_t n, size_t p) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t k = 0; k < n; ++k) {
            float aik = A[i * n + k];
            for (size_t j = 0; j < p; ++j) {
                C[i * p + j] += aik * B[k * p + j];
            }
        }
    }
}

} // namespace dracolix::kernels
