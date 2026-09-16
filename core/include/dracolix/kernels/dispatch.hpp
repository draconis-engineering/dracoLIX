#pragma once
// Kernel dispatch — Phase 3: Alignment + CPU detection + selection
// Licensed under GPL-3.0-only
#include "../cpu.hpp"
#include "../alloc.hpp"
#include <cstddef>
#include <functional>

namespace dracolix::kernels {

enum class GemmKernel {
    Scalar = 0,
    Avx2,
    Avx512,
    Blas // via DRACOLIX_USE_FORTRAN / OpenBLAS
};

// Runtime selection — pure C++, no BLAS dependency by default
inline GemmKernel select_gemm_kernel(size_t m, size_t n, size_t p) {
    // Threshold from linalg.hpp BLAS_THRESHOLD (64^3)
    size_t ops = m * n * p;
    if (ops < 4096) return GemmKernel::Scalar; // small: scalar wins (no dispatch overhead)

    static const auto feats = cpu::detect();
    // Prefer widest available; payload kernels probe at runtime via ifunc/builtin
    if (feats.avx512f) return GemmKernel::Avx512;
    if (feats.avx2) return GemmKernel::Avx2;
    if (feats.avx) return GemmKernel::Avx2; // fallback
    return GemmKernel::Scalar;
}

inline const char* kernel_name(GemmKernel k) {
    switch (k) {
        case GemmKernel::Scalar: return "scalar-ikj";
        case GemmKernel::Avx2: return "avx2";
        case GemmKernel::Avx512: return "avx512";
        case GemmKernel::Blas: return "openblas";
    }
    return "unknown";
}

// Dispatch entry — selects and calls. Scalar is always available.
void dispatch_gemm_f64(const double* A, const double* B, double* C, size_t m, size_t n, size_t p);
void dispatch_gemm_f32(const float* A, const float* B, float* C, size_t m, size_t n, size_t p);

} // namespace dracolix::kernels
