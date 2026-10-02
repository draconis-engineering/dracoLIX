// GEMM backend comparison harness
// Licensed under GPL-3.0-or-later
//
// Sweeps every GEMM backend DracoLIX can dispatch to, for f64 and f32, at
// several sizes, and validates each against the scalar reference kernel. This
// is the measurement tool the GEMM optimisation work was driven with.
//
//   ./bench_gemm_backends                 # default sweep
//   ./bench_gemm_backends 2048            # single size
//
// Backends under test:
//   scalar      reference ikj (the correctness oracle)
//   native      whatever the runtime CPUID probe selects
//   avx2        pinned to the AVX2 microkernels
//   avx512      pinned to the AVX-512 microkernels
//   openblas    external BLAS, if the build has DRACOLIX_USE_BLAS=ON
//
// Exit code is non-zero if any backend deviates from the reference, so this
// doubles as a regression test.

#include "dracolix/kernels/dispatch.hpp"
#include "dracolix/thread_pool.hpp"
#include "dracolix/cpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace dracolix;
using Clock = std::chrono::steady_clock;

namespace {

double now_s() {
	return std::chrono::duration<double>(Clock::now().time_since_epoch())
		.count();
}

// Deterministic pseudo-random fill so runs are comparable.
template <typename T> void fill_rand(std::vector<T> &v, uint64_t seed) {
	for (size_t i = 0; i < v.size(); ++i) {
		seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
		v[i] = static_cast<T>((seed >> 33) % 100000) / T(100000) - T(0.5);
	}
}

// The reference implementation, inlined so we always have an oracle even when
// the build forces another backend.
template <typename T>
void reference_gemm(const T *A, const T *B, T *C, size_t m, size_t n,
					size_t p) {
	for (size_t i = 0; i < m; ++i)
		for (size_t k = 0; k < n; ++k) {
			const T aik = A[i * n + k];
			for (size_t j = 0; j < p; ++j)
				C[i * p + j] += aik * B[k * p + j];
		}
}

template <typename T>
double max_abs_diff(const std::vector<T> &a, const std::vector<T> &b) {
	double worst = 0;
	for (size_t i = 0; i < a.size(); ++i)
		worst = std::max<double>(worst, std::fabs(double(a[i]) - double(b[i])));
	return worst;
}

// Route a call through either the dispatcher or the local oracle.
template <typename T>
void run(const std::string &backend, const T *A, const T *B, T *C, size_t n) {
	if (backend == "scalar")
		reference_gemm(A, B, C, n, n, n);
	else if constexpr (std::is_same<T, double>::value)
		kernels::dispatch_gemm_f64(A, B, C, n, n, n);
	else
		kernels::dispatch_gemm_f32(A, B, C, n, n, n);
}

struct Row {
	std::string backend;
	const char *dtype;
	size_t n;
	double gflops;
	double secs;
	double maxerr;
};

// One (backend, dtype, size) measurement, validated against the reference.
template <typename T>
Row measure(const std::string &backend, const char *dtype, size_t n,
			const std::vector<T> &A, const std::vector<T> &B) {
	std::vector<T> C(n * n, T(0));

	// Warm-up (also pays first-touch page faults).
	run(backend, A.data(), B.data(), C.data(), n);

	// Correctness vs the reference.
	std::vector<T> R(n * n, T(0));
	reference_gemm(A.data(), B.data(), R.data(), n, n, n);
	const double err = max_abs_diff(C, R);

	// Aim for >= ~0.4 s of work so the clock is not dominated by noise.
	int iters = static_cast<int>(0.4 * 1e9 / (2.0 * n * n * n));
	iters = std::max(iters, 1);

	std::fill(C.begin(), C.end(), T(0));
	const double t0 = now_s();
	for (int it = 0; it < iters; ++it)
		run(backend, A.data(), B.data(), C.data(), n);
	const double t1 = now_s();

	Row r;
	r.backend = backend;
	r.dtype = dtype;
	r.n = n;
	r.secs = (t1 - t0) / iters;
	r.gflops = 2.0 * n * n * n / r.secs / 1e9;
	r.maxerr = err;
	return r;
}

} // namespace

int main(int argc, char **argv) {
	const auto feats = cpu::detect();
	std::printf("CPU: %s\n", cpu::to_string(feats).c_str());
	std::printf("pool: %zu workers   blas: %s\n\n", ThreadPool::global().size(),
				kernels::gemm_blas_available() ? "yes" : "no");

	std::vector<size_t> sizes;
	if (argc > 1) {
		sizes.push_back(static_cast<size_t>(std::atoi(argv[1])));
	} else {
		sizes = {256, 512, 1024, 2048, 4000};
	}

	std::vector<std::string> backends = {"scalar", "native"};
	if (feats.avx2)
		backends.push_back("avx2");
	if (feats.avx512f)
		backends.push_back("avx512");
	if (kernels::gemm_blas_available())
		backends.push_back("openblas");

	std::printf("%-6s %-10s %6s %12s %12s %11s  %s\n", "dtype", "backend", "n",
				"time(ms)", "GFLOP/s", "max|err|", "vs scalar");
	std::printf("%s\n", std::string(80, '-').c_str());

	std::vector<Row> rows;
	int rc = 0;
	for (size_t n : sizes) {
		std::vector<double> A64(n * n), B64(n * n);
		std::vector<float> A32(n * n), B32(n * n);
		fill_rand(A64, 0x9e3779b97f4a7c15ULL ^ n);
		fill_rand(B64, 0xc2b2ae3d27d4eb4fULL ^ n);
		fill_rand(A32, 0x9e3779b97f4a7c15ULL ^ n);
		fill_rand(B32, 0xc2b2ae3d27d4eb4fULL ^ n);

		for (const char *dt : {"f64", "f32"}) {
			double scalar_gf = 0;
			for (const auto &be : backends) {
				// Pick the backend through the documented env override.
				setenv("DLX_GEMM_BACKEND",
					   be == "openblas" ? "blas" : be.c_str(), 1);
				const Row r =
					(std::strcmp(dt, "f64") == 0)
						? measure<double>(be, dt, n, A64, B64)
						: measure<float>(be, dt, n, A32, B32);
				if (be == "scalar")
					scalar_gf = r.gflops;
				char rel[32] = "  -";
				if (scalar_gf > 0 && be != "scalar")
					std::snprintf(rel, sizeof(rel), "%5.1fx",
								  r.gflops / scalar_gf);
				std::printf("%-6s %-10s %6zu %12.3f %12.2f %11.3g  %s\n", dt,
							be.c_str(), n, r.secs * 1e3, r.gflops, r.maxerr,
							rel);
				rows.push_back(r);
			}
			unsetenv("DLX_GEMM_BACKEND");
			std::printf("\n");
		}
	}

	// Tolerance is relative to the accumulation length, since the optimized
	// kernels accumulate in a different order than the reference.
	for (const auto &r : rows) {
		const double tol = (std::strcmp(r.dtype, "f32") == 0 ? 1e-4 : 1e-11) *
						   double(r.n);
		if (r.backend != "scalar" && r.maxerr > tol) {
			std::fprintf(stderr,
						 "FAIL: %s %s at n=%zu deviates from reference by %g "
						 "(tol %g)\n",
						 r.dtype, r.backend.c_str(), r.n, r.maxerr, tol);
			rc = 1;
		}
	}
	if (rc == 0)
		std::printf("all backends match the scalar reference within tolerance\n");
	return rc;
}