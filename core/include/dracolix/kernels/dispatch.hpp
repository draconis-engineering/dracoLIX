#pragma once

// GEMM backend dispatch — selects CPU scalar / AVX2 / AVX-512 / OpenBLAS at
// runtime. Licensed under GPL-3.0-or-later
//
// Selection is deliberately a *runtime* decision driven by three inputs:
//   - what the running CPU actually supports (cpu::detect, cpuid)
//   - what the build was compiled with (DRACOLIX_USE_FORTRAN / BLAS symbols)
//   - an optional user override (env var, for benchmarking and debugging)
//
// Keeping it runtime rather than compile-time is what lets the AVX-512 kernel
// live in the same portable binary as the scalar reference: the vector code is
// compiled behind GCC/Clang `target` attributes and only called when the CPU
// gate passes.

#include "../cpu.hpp"
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace dracolix::kernels {

enum class GemmKernel {
	Scalar = 0, // reference ikj
	Avx2,	   // register-blocked + packed + blocked
	Avx512,	   // ditto, wider
	Blas	   // OpenBLAS via DRACOLIX_USE_FORTRAN / direct cblas
};

enum class GemmBackendKind {
	Scalar, // force the reference kernel
	Native, // best in-tree vectorized kernel
	Blas,	 // external BLAS
};

// Optional direct BLAS bindings. Compiled only when a BLAS was found at build
// time; declared here so the dispatcher can call them without leaking BLAS
// headers into the core API.
bool gemm_blas_available() noexcept;
bool gemm_blas_f64(const double *A, const double *B, double *C, size_t m,
				   size_t n, size_t p);
bool gemm_blas_f32(const float *A, const float *B, float *C, size_t m,
				   size_t n, size_t p);

struct GemmBackend {
	GemmBackendKind backend = GemmBackendKind::Native;
	GemmKernel kernel = GemmKernel::Scalar;
};

// Probe the running CPU once. Cheap enough to call per GEMM, and keeps the
// result honest if a process migrates between heterogeneous cores.
inline const cpu::Features &gemm_cpu_features() {
	static const cpu::Features f = cpu::detect();
	return f;
}

// User override, read exactly once (getenv is a linear scan of environ, and this
// runs on every GEMM):
//   DLX_GEMM_BACKEND = scalar | native | avx2 | avx512 | blas
// Lets the backend benchmark pin a specific path and gives a bisection handle
// when a result looks wrong.
inline const char *gemm_backend_override() {
	static const char *e = [] {
		const char *v = std::getenv("DLX_GEMM_BACKEND");
		return (v && *v) ? v : "";
	}();
	return e;
}

inline GemmBackend gemm_backend() {
	GemmBackend out;

	// Widest vector kernel this CPU can actually run. The requirements mirror
	// the `target` attributes on the kernels themselves.
	const auto &f = gemm_cpu_features();
	const bool cpu_avx512 = f.avx512f && f.avx512dq && f.avx512vl && f.avx2;
	if (cpu_avx512)
		out.kernel = GemmKernel::Avx512;
	else if (f.avx2)
		out.kernel = GemmKernel::Avx2;
	else
		out.kernel = GemmKernel::Scalar;

	out.backend = (out.kernel == GemmKernel::Scalar) ? GemmBackendKind::Scalar
													  : GemmBackendKind::Native;

	const char *e = gemm_backend_override();
	if (std::strcmp(e, "scalar") == 0) {
		out.backend = GemmBackendKind::Scalar;
		out.kernel = GemmKernel::Scalar;
	} else if (std::strcmp(e, "blas") == 0) {
		if (gemm_blas_available())
			out.backend = GemmBackendKind::Blas;
	} else if (std::strcmp(e, "avx2") == 0) {
		// Clamped to what this CPU can run, so kernel_name() never lies.
		out.backend = GemmBackendKind::Native;
		out.kernel = f.avx2 ? GemmKernel::Avx2 : GemmKernel::Scalar;
	} else if (std::strcmp(e, "avx512") == 0) {
		out.backend = GemmBackendKind::Native;
		out.kernel = cpu_avx512
						 ? GemmKernel::Avx512
						 : (f.avx2 ? GemmKernel::Avx2 : GemmKernel::Scalar);
	}
	// "native" and unset both mean: keep the probed choice.

	// If no vector path is available but BLAS is, BLAS still beats scalar.
	// An explicit "scalar" request still wins.
	if (out.kernel == GemmKernel::Scalar && std::strcmp(e, "scalar") != 0 &&
		gemm_blas_available())
		out.backend = GemmBackendKind::Blas;

	return out;
}

inline const char *kernel_name(GemmKernel k) {
	switch (k) {
	case GemmKernel::Scalar:
		return "scalar-ikj";
	case GemmKernel::Avx2:
		return "avx2-blocked";
	case GemmKernel::Avx512:
		return "avx512-blocked";
	case GemmKernel::Blas:
		return "openblas";
	}
	return "unknown";
}

inline const char *backend_name(GemmBackendKind b) {
	switch (b) {
	case GemmBackendKind::Scalar:
		return "scalar";
	case GemmBackendKind::Native:
		return "native";
	case GemmBackendKind::Blas:
		return "blas";
	}
	return "unknown";
}

// Thread count the native path would use for this shape (0 = serial).
size_t gemm_recommended_threads(size_t m, size_t n, size_t p);

// Dispatch entry — selects and calls. Scalar is always available.
void dispatch_gemm_f64(const double *A, const double *B, double *C, size_t m,
					   size_t n, size_t p);
void dispatch_gemm_f32(const float *A, const float *B, float *C, size_t m,
					   size_t n, size_t p);

} // namespace dracolix::kernels