// GEMM correctness across shapes, backends and thread counts.
//
// Licensed under GPL-3.0-or-later
//
// This exists because the optimized GEMM uses register blocking, masked edge
// tiles and cache blocking — all of which introduce remainder handling that a
// round-size test never reaches. The sizes below deliberately straddle the
// microkernel tile (8x24 for AVX-512, 6x8 for AVX2) and the cache block
// (256x256x256) in every direction, and every backend is checked against the
// scalar reference kernel.
//
// A real bug this caught: AVX2 maskload/maskstore select on the *most
// significant bit* of each 64-bit lane, whereas AVX-512's __mmask8 uses bit i
// for lane i. Using the AVX-512 idiom in the AVX2 path silently disabled the
// lanes and produced wrong results for any n not a multiple of 4.

#include "dracolix/cpu.hpp"
#include "dracolix/kernels/dispatch.hpp"
#include "dracolix/thread_pool.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace dracolix;

namespace {

int failures = 0;

// Independent oracle. Deliberately the naive i,j,k ordering: it shares no code
// path with the optimized kernel, so it can catch blocking/masking mistakes
// that a shared helper would hide.
template <typename T>
void reference_gemm(const T *A, const T *B, T *C, size_t m, size_t n,
					size_t p) {
	for (size_t i = 0; i < m; ++i)
		for (size_t j = 0; j < p; ++j) {
			T acc = T(0);
			for (size_t k = 0; k < n; ++k)
				acc += A[i * n + k] * B[k * p + j];
			C[i * p + j] = acc;
		}
}

template <typename T> void fill(std::vector<T> &v, uint64_t seed) {
	for (size_t i = 0; i < v.size(); ++i) {
		seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
		v[i] = static_cast<T>((seed >> 33) % 1000) / T(500) - T(1);
	}
}

void check(const std::string &backend, size_t m, size_t n, size_t p) {
	std::vector<double> A(m * n), B(n * p);
	fill(A, 0x243F6A8885A308D3ULL ^ m);
	fill(B, 0x13198A2E03707344ULL ^ p);

	std::vector<double> want(m * p, 0.0);
	reference_gemm(A.data(), B.data(), want.data(), m, n, p);

	std::vector<double> got(m * p, 0.0);
	kernels::dispatch_gemm_f64(A.data(), B.data(), got.data(), m, n, p);

	double worst = 0.0;
	for (size_t i = 0; i < want.size(); ++i)
		worst = std::max(worst, std::fabs(want[i] - got[i]));

	// Scaled tolerance: accumulation order differs from the reference, so allow
	// a few ulps of the magnitudes involved rather than demanding bit equality.
	const double tol = 1e-11 * double(n);
	if (!(worst <= tol)) {
		std::printf(
			"  FAIL f64 %-8s %zux%zu * %zux%zu   max|diff| = %.3g (tol %.3g)\n",
			backend.c_str(), m, n, n, p, worst, tol);
		++failures;
	}
}

// f32 exercises a separate set of microkernels (AVX-512 8x32, AVX2 6x16) with
// different tile shapes, and — importantly — a different mask convention:
// AVX2 float masks select on the MSB of each 32-bit lane, not 64-bit.
void check_f32(const std::string &backend, size_t m, size_t n, size_t p) {
	std::vector<float> A(m * n), B(n * p);
	fill(A, 0xA4093822299F31D0ULL ^ m);
	fill(B, 0x082EFA98EC4E6C89ULL ^ p);

	std::vector<float> want(m * p, 0.0f);
	reference_gemm(A.data(), B.data(), want.data(), m, n, p);

	std::vector<float> got(m * p, 0.0f);
	kernels::dispatch_gemm_f32(A.data(), B.data(), got.data(), m, n, p);

	double worst = 0.0;
	for (size_t i = 0; i < want.size(); ++i)
		worst = std::max<double>(worst,
								 std::fabs(double(want[i]) - double(got[i])));

	// f32 accumulates in single precision, so the tolerance is eps * n.
	const double tol = 1e-6 * double(n);
	if (!(worst <= tol)) {
		std::printf(
			"  FAIL f32 %-8s %zux%zu * %zux%zu   max|diff| = %.3g (tol %.3g)\n",
			backend.c_str(), m, n, n, p, worst, tol);
		++failures;
	}
}

} // namespace

int main() {
	const auto feats = cpu::detect();
	std::printf("GEMM correctness\n");
	std::printf("  cpu: %s\n", cpu::to_string(feats).c_str());
	std::printf("  selected backend: %s / %s, pool %zu workers\n",
				kernels::backend_name(kernels::gemm_backend().backend),
				kernels::kernel_name(kernels::gemm_backend().kernel),
				ThreadPool::global().size());

	// Sizes chosen to hit every remainder combination against the 8x24
	// (AVX-512) and 6x8 (AVX2) microkernels, and the 256^3 cache block.
	const size_t sizes[] = {1,	 2,	  3,   5,	7,	 8,	  9,   15,	16,	 17,
							23,	 24,  25,  31,	32,	 47,  48,  63,	64,	 65,
							127, 128, 129, 191, 255, 256, 257, 300, 511, 512};

	std::vector<std::string> backends = {"native"};
	if (feats.avx2)
		backends.push_back("avx2");
	if (feats.avx512f && feats.avx512dq && feats.avx512vl)
		backends.push_back("avx512");
	backends.push_back("scalar");
	if (kernels::gemm_blas_available())
		backends.push_back("blas");

	size_t cases = 0;
	for (const auto &be : backends) {
		kernels::set_gemm_backend_override(be.c_str());

		// Prove the override took. If this ever falls back to reading the env
		// var once at startup, every row below would silently run the same
		// kernel under four different names — the harness would keep passing
		// while covering a single path, and a failure would be logged against
		// the wrong backend. Fail loudly rather than report a false matrix.
		const auto sel = kernels::gemm_backend();
		const bool cpu_avx512 = feats.avx512f && feats.avx512dq &&
								feats.avx512vl && feats.avx2;
		// The test only ever asks for backends this CPU can run.
		const kernels::GEMMKernel best = cpu_avx512
											 ? kernels::GEMMKernel::Avx512
											 : (feats.avx2 ? kernels::GEMMKernel::Avx2
														   : kernels::GEMMKernel::Scalar);

		kernels::GEMMBackendKind want_backend;
		kernels::GEMMKernel want_kernel;
		if (be == "scalar") {
			want_backend = kernels::GEMMBackendKind::Scalar;
			want_kernel = kernels::GEMMKernel::Scalar;
		} else if (be == "blas") {
			want_backend = kernels::GEMMBackendKind::Blas;
			want_kernel = sel.kernel; // BLAS keeps the probed vector kernel
		} else if (be == "avx2") {
			want_backend = kernels::GEMMBackendKind::Native;
			want_kernel = feats.avx2 ? kernels::GEMMKernel::Avx2
									 : kernels::GEMMKernel::Scalar;
		} else if (be == "avx512") {
			want_backend = kernels::GEMMBackendKind::Native;
			want_kernel = best;
		} else { // "native": untouched probe
			want_kernel = best;
			want_backend = (best == kernels::GEMMKernel::Scalar)
							   ? kernels::GEMMBackendKind::Scalar
							   : kernels::GEMMBackendKind::Native;
		}
		if (sel.backend != want_backend || sel.kernel != want_kernel) {
			std::printf("  OVERRIDE FAILED: asked for '%s', got %s / %s\n",
						be.c_str(), kernels::backend_name(sel.backend),
						kernels::kernel_name(sel.kernel));
			++failures;
		}

		std::printf("  backend %-8s -> %s / %s\n", be.c_str(),
					kernels::backend_name(sel.backend),
					kernels::kernel_name(sel.kernel));
		for (size_t n : sizes) {
			check(be, n, n, n); // square
			++cases;
			check(be, n, n + 1, n - 1); // ragged m,n,p
			++cases;
			check(be, 1, n, n); // degenerate m (row vector)
			++cases;
			check(be, n, 1, n); // degenerate n (outer product)
			++cases;
			check(be, n, n, 1); // degenerate p (column vector)
			++cases;
		}
		// f32 microkernels are separate code with different tile shapes and a
		// different AVX2 mask convention; they need their own coverage.
		for (size_t n : sizes) {
			check_f32(be, n, n, n);
			++cases;
			check_f32(be, n, n + 1, n - 1);
			++cases;
			check_f32(be, 1, n, n);
			++cases;
			check_f32(be, n, 1, n);
			++cases;
			check_f32(be, n, n, 1);
			++cases;
		}
		kernels::set_gemm_backend_override(nullptr);
	}
	std::printf("  %zu shapes checked across %zu backends\n", cases,
				backends.size());

	if (failures) {
		std::printf("FAILED: %d mismatches\n", failures);
		return 1;
	}
	std::printf("gemm correctness ok\n");
	return 0;
}
