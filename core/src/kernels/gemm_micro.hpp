// DracoLIX GEMM microkernels — register-blocked, row-major, C += A*B
// Licensed under GPL-3.0-or-later
//
// Design notes
// ------------
// DracoLIX stores every matrix row-major, so B is already contiguous along the
// N (column) axis and can be loaded with plain vector loads. The previous
// kernel reloaded and re-stored the C tile from memory on every k step, which
// caps throughput at roughly one C load + one C store per FMA.
//
// The kernels below hold a MR x NR tile of C in vector registers for the whole
// k loop, so the inner loop issues only:
//     broadcast A[i][k]   (scalar -> vector)
//     load   B[k][j..j+NR] (contiguous vector load)
//     FMA
//
// That removes C traffic from the inner loop entirely and is what takes this
// from ~27 GFLOP/s to roughly OpenBLAS single-thread parity.
//
// Kernels are selected at *runtime* via GCC/Clang `target` attributes rather
// than global -mavx2/-mavx512 flags, so the same binary stays portable: the
// AVX-512 code is compiled in but only ever *called* when the dispatcher has
// confirmed the running CPU supports it.

#pragma once

#include <cstddef>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||            \
	defined(_M_IX86)
#define DRACOLIX_GEMM_X86 1
#endif

#if defined(DRACOLIX_GEMM_X86) && (defined(__GNUC__) || defined(__clang__))
#define DRACOLIX_TARGET_AVX2 __attribute__((target("avx2,fma")))
#define DRACOLIX_TARGET_AVX512                                                   \
	__attribute__((target("avx512f,avx512dq,avx512vl,avx512bw")))
#define DRACOLIX_HAS_TARGET_ATTR 1
#else
#define DRACOLIX_TARGET_AVX2
#define DRACOLIX_TARGET_AVX512
#define DRACOLIX_HAS_TARGET_ATTR 0
#endif

#if DRACOLIX_HAS_TARGET_ATTR
#include <immintrin.h>
#endif

namespace dracolix::kernels {

// ===========================================================================
// AVX-512 double — MR=8, NR=24
//
// 24 accumulators + 3 B vectors = 27 of the 32 architectural zmm registers,
// leaving headroom for broadcasts/addressing. Measured fastest shape on Zen 4;
// see benchmarks/bench_gemm_backends.cpp for the sweep.
// ===========================================================================
#define DRACOLIX_MR512 8
#define DRACOLIX_NR512 24

// AVX2 / FMA fallback — MR=6, NR=8 (12 accumulators + 2 ymm of 16).
#define DRACOLIX_MR256 6
#define DRACOLIX_NR256 8

#if DRACOLIX_HAS_TARGET_ATTR

// ---------------------------------------------------------------------------
// Full MR x NR tile.
// ---------------------------------------------------------------------------
DRACOLIX_TARGET_AVX512
inline void micro_gemm_512(const double *__restrict Ap, int lda,
						   const double *__restrict Bp, int ldb,
						   double *__restrict C, int ldc, int k) {
	constexpr int MR = DRACOLIX_MR512;
	constexpr int NV = DRACOLIX_NR512 / 8; // 3 zmm per row

	__m512d c[MR][NV];
#pragma GCC unroll 8
	for (int i = 0; i < MR; ++i)
#pragma GCC unroll 3
		for (int v = 0; v < NV; ++v)
			c[i][v] = _mm512_loadu_pd(&C[i * ldc + v * 8]);

	for (int p = 0; p < k; ++p) {
		const double *bp = &Bp[(size_t)p * ldb];
		const __m512d b0 = _mm512_loadu_pd(&bp[0]);
		const __m512d b1 = _mm512_loadu_pd(&bp[8]);
		const __m512d b2 = _mm512_loadu_pd(&bp[16]);
#pragma GCC unroll 8
		for (int i = 0; i < MR; ++i) {
			const __m512d a = _mm512_set1_pd(Ap[i * lda + p]);
			c[i][0] = _mm512_fmadd_pd(a, b0, c[i][0]);
			c[i][1] = _mm512_fmadd_pd(a, b1, c[i][1]);
			c[i][2] = _mm512_fmadd_pd(a, b2, c[i][2]);
		}
	}

#pragma GCC unroll 8
	for (int i = 0; i < MR; ++i) {
		_mm512_storeu_pd(&C[i * ldc + 0], c[i][0]);
		_mm512_storeu_pd(&C[i * ldc + 8], c[i][1]);
		_mm512_storeu_pd(&C[i * ldc + 16], c[i][2]);
	}
}

DRACOLIX_TARGET_AVX2
inline void micro_gemm_256(const double *__restrict Ap, int lda,
						   const double *__restrict Bp, int ldb,
						   double *__restrict C, int ldc, int k) {
	constexpr int MR = DRACOLIX_MR256;
	constexpr int NV = DRACOLIX_NR256 / 4; // 2 ymm per row

	__m256d c[MR][NV];
#pragma GCC unroll 6
	for (int i = 0; i < MR; ++i)
#pragma GCC unroll 2
		for (int v = 0; v < NV; ++v)
			c[i][v] = _mm256_loadu_pd(&C[i * ldc + v * 4]);

	for (int p = 0; p < k; ++p) {
		const double *bp = &Bp[(size_t)p * ldb];
		const __m256d b0 = _mm256_loadu_pd(&bp[0]);
		const __m256d b1 = _mm256_loadu_pd(&bp[4]);
#pragma GCC unroll 6
		for (int i = 0; i < MR; ++i) {
			const __m256d a = _mm256_broadcast_sd(&Ap[i * lda + p]);
			c[i][0] = _mm256_fmadd_pd(a, b0, c[i][0]);
			c[i][1] = _mm256_fmadd_pd(a, b1, c[i][1]);
		}
	}

#pragma GCC unroll 6
	for (int i = 0; i < MR; ++i) {
		_mm256_storeu_pd(&C[i * ldc + 0], c[i][0]);
		_mm256_storeu_pd(&C[i * ldc + 4], c[i][1]);
	}
}

// ---------------------------------------------------------------------------
// Edge kernels.
//
// Column remainder (nr < NR) is handled with masked loads/stores so there is no
// scalar fallback path; row remainder (mr < MR) just stops the row loop, and
// every read is clamped to a valid tile row so nothing reads out of bounds.
// ---------------------------------------------------------------------------
DRACOLIX_TARGET_AVX512
inline void micro_gemm_512_partial(const double *__restrict Ap, int lda,
								   const double *__restrict Bp, int ldb,
								   double *__restrict C, int ldc, int k, int mr,
								   int nr) {
	constexpr int NV = DRACOLIX_NR512 / 8;

	const __mmask8 m0 = (__mmask8)((1u << (nr > 0 ? nr : 0)) - 1u);
	const __mmask8 m1 = (nr > 8) ? (__mmask8)((1u << (nr - 8)) - 1u) : (__mmask8)0;
	const __mmask8 m2 = (nr > 16) ? (__mmask8)((1u << (nr - 16)) - 1u) : (__mmask8)0;

	__m512d c[DRACOLIX_MR512][NV];
#pragma GCC unroll 8
	for (int i = 0; i < mr; ++i) {
		c[i][0] = _mm512_maskz_loadu_pd(m0, &C[i * ldc]);
		if (nr > 8)
			c[i][1] = _mm512_maskz_loadu_pd(m1, &C[i * ldc + 8]);
		if (nr > 16)
			c[i][2] = _mm512_maskz_loadu_pd(m2, &C[i * ldc + 16]);
	}

	for (int p = 0; p < k; ++p) {
		const double *bp = &Bp[(size_t)p * ldb];
		const __m512d b0 = _mm512_maskz_loadu_pd(m0, &bp[0]);
		const __m512d b1 = _mm512_maskz_loadu_pd(m1, &bp[8]);
		const __m512d b2 = _mm512_maskz_loadu_pd(m2, &bp[16]);
#pragma GCC unroll 8
		for (int i = 0; i < mr; ++i) {
			const __m512d a = _mm512_set1_pd(Ap[i * lda + p]);
			c[i][0] = _mm512_fmadd_pd(a, b0, c[i][0]);
			if (nr > 8)
				c[i][1] = _mm512_fmadd_pd(a, b1, c[i][1]);
			if (nr > 16)
				c[i][2] = _mm512_fmadd_pd(a, b2, c[i][2]);
		}
	}

#pragma GCC unroll 8
	for (int i = 0; i < mr; ++i) {
		_mm512_mask_storeu_pd(&C[i * ldc], m0, c[i][0]);
		if (nr > 8)
			_mm512_mask_storeu_pd(&C[i * ldc + 8], m1, c[i][1]);
		if (nr > 16)
			_mm512_mask_storeu_pd(&C[i * ldc + 16], m2, c[i][2]);
	}
}

DRACOLIX_TARGET_AVX2
inline void micro_gemm_256_partial(const double *__restrict Ap, int lda,
								   const double *__restrict Bp, int ldb,
								   double *__restrict C, int ldc, int k, int mr,
								   int nr) {
	constexpr int NV = DRACOLIX_NR256 / 4;

	// AVX2 (pre-AVX512) mask intrinsics take a vector of lane masks where each
	// 64-bit lane selects on its *most significant bit* — note this is a
	// different convention from AVX-512's __mmask8, which uses bit i for lane i.
	const int n0 = nr > 4 ? 4 : nr;
	const int n1 = nr > 4 ? (nr - 4 > 4 ? 4 : nr - 4) : 0;
	constexpr long long kOn = 1LL << 63;
	const __m256i m0 = _mm256_setr_epi64x(
		(n0 > 0) ? kOn : 0, (n0 > 1) ? kOn : 0, (n0 > 2) ? kOn : 0,
		(n0 > 3) ? kOn : 0);
	const __m256i m1 = _mm256_setr_epi64x(
		(n1 > 0) ? kOn : 0, (n1 > 1) ? kOn : 0, (n1 > 2) ? kOn : 0,
		(n1 > 3) ? kOn : 0);

	__m256d c[DRACOLIX_MR256][NV];
#pragma GCC unroll 6
	for (int i = 0; i < mr; ++i) {
		c[i][0] = _mm256_maskload_pd(&C[i * ldc], m0);
		if (nr > 4)
			c[i][1] = _mm256_maskload_pd(&C[i * ldc + 4], m1);
	}

	for (int p = 0; p < k; ++p) {
		const double *bp = &Bp[(size_t)p * ldb];
		const __m256d b0 = _mm256_maskload_pd(&bp[0], m0);
		const __m256d b1 = _mm256_maskload_pd(&bp[4], m1);
#pragma GCC unroll 6
		for (int i = 0; i < mr; ++i) {
			const __m256d a = _mm256_broadcast_sd(&Ap[i * lda + p]);
			c[i][0] = _mm256_fmadd_pd(a, b0, c[i][0]);
			if (nr > 4)
				c[i][1] = _mm256_fmadd_pd(a, b1, c[i][1]);
		}
	}
#pragma GCC unroll 6
	for (int i = 0; i < mr; ++i) {
		_mm256_maskstore_pd(&C[i * ldc], m0, c[i][0]);
		if (nr > 4)
			_mm256_maskstore_pd(&C[i * ldc + 4], m1, c[i][1]);
	}
}

#endif // DRACOLIX_HAS_TARGET_ATTR

// ===========================================================================
// f32 microkernels
//
// Same structure as the f64 kernels. Vector width doubles in elements, so the
// tile is wider in columns for the same register budget:
//   AVX-512: 8x32 (16 accumulators + 2 zmm of 32)
//   AVX2:    6x16 (12 accumulators + 2 ymm of 16)
// Without these, f32 falls back to scalar and gets no speedup at all.
// ===========================================================================
#if DRACOLIX_HAS_TARGET_ATTR

#define DRACOLIX_MR512_F 8
#define DRACOLIX_NR512_F 32
#define DRACOLIX_MR256_F 6
#define DRACOLIX_NR256_F 16

DRACOLIX_TARGET_AVX512
inline void micro_gemm_512_f(const float *__restrict Ap, int lda,
							 const float *__restrict Bp, int ldb,
							 float *__restrict C, int ldc, int k) {
	constexpr int MR = DRACOLIX_MR512_F;
	constexpr int NV = DRACOLIX_NR512_F / 16; // 2 zmm per row

	__m512 c[MR][NV];
#pragma GCC unroll 8
	for (int i = 0; i < MR; ++i)
#pragma GCC unroll 2
		for (int v = 0; v < NV; ++v)
			c[i][v] = _mm512_loadu_ps(&C[i * ldc + v * 16]);

	for (int p = 0; p < k; ++p) {
		const float *bp = &Bp[(size_t)p * ldb];
		const __m512 b0 = _mm512_loadu_ps(&bp[0]);
		const __m512 b1 = _mm512_loadu_ps(&bp[16]);
#pragma GCC unroll 8
		for (int i = 0; i < MR; ++i) {
			const __m512 a = _mm512_set1_ps(Ap[i * lda + p]);
			c[i][0] = _mm512_fmadd_ps(a, b0, c[i][0]);
			c[i][1] = _mm512_fmadd_ps(a, b1, c[i][1]);
		}
	}
#pragma GCC unroll 8
	for (int i = 0; i < MR; ++i) {
		_mm512_storeu_ps(&C[i * ldc + 0], c[i][0]);
		_mm512_storeu_ps(&C[i * ldc + 16], c[i][1]);
	}
}

DRACOLIX_TARGET_AVX2
inline void micro_gemm_256_f(const float *__restrict Ap, int lda,
							 const float *__restrict Bp, int ldb,
							 float *__restrict C, int ldc, int k) {
	constexpr int MR = DRACOLIX_MR256_F;
	constexpr int NV = DRACOLIX_NR256_F / 8; // 2 ymm per row

	__m256 c[MR][NV];
#pragma GCC unroll 6
	for (int i = 0; i < MR; ++i)
#pragma GCC unroll 2
		for (int v = 0; v < NV; ++v)
			c[i][v] = _mm256_loadu_ps(&C[i * ldc + v * 8]);

	for (int p = 0; p < k; ++p) {
		const float *bp = &Bp[(size_t)p * ldb];
		const __m256 b0 = _mm256_loadu_ps(&bp[0]);
		const __m256 b1 = _mm256_loadu_ps(&bp[8]);
#pragma GCC unroll 6
		for (int i = 0; i < MR; ++i) {
			const __m256 a = _mm256_broadcast_ss(&Ap[i * lda + p]);
			c[i][0] = _mm256_fmadd_ps(a, b0, c[i][0]);
			c[i][1] = _mm256_fmadd_ps(a, b1, c[i][1]);
		}
	}
#pragma GCC unroll 6
	for (int i = 0; i < MR; ++i) {
		_mm256_storeu_ps(&C[i * ldc + 0], c[i][0]);
		_mm256_storeu_ps(&C[i * ldc + 8], c[i][1]);
	}
}

// --- f32 edge kernels ------------------------------------------------------
DRACOLIX_TARGET_AVX512
inline void micro_gemm_512_f_partial(const float *__restrict Ap, int lda,
									 const float *__restrict Bp, int ldb,
									 float *__restrict C, int ldc, int k, int mr,
									 int nr) {
	constexpr int NV = DRACOLIX_NR512_F / 16;
	const __mmask16 m0 = (__mmask16)(((1u << nr) - 1u) & 0xFFFFu);
	const __mmask16 m1 = (nr > 16) ? (__mmask16)((1u << (nr - 16)) - 1u)
								  : (__mmask16)0;

	__m512 c[DRACOLIX_MR512_F][NV];
#pragma GCC unroll 8
	for (int i = 0; i < mr; ++i) {
		c[i][0] = _mm512_maskz_loadu_ps(m0, &C[i * ldc]);
		if (nr > 16)
			c[i][1] = _mm512_maskz_loadu_ps(m1, &C[i * ldc + 16]);
	}
	for (int p = 0; p < k; ++p) {
		const float *bp = &Bp[(size_t)p * ldb];
		const __m512 b0 = _mm512_maskz_loadu_ps(m0, &bp[0]);
		const __m512 b1 = _mm512_maskz_loadu_ps(m1, &bp[16]);
#pragma GCC unroll 8
		for (int i = 0; i < mr; ++i) {
			const __m512 a = _mm512_set1_ps(Ap[i * lda + p]);
			c[i][0] = _mm512_fmadd_ps(a, b0, c[i][0]);
			if (nr > 16)
				c[i][1] = _mm512_fmadd_ps(a, b1, c[i][1]);
		}
	}
#pragma GCC unroll 8
	for (int i = 0; i < mr; ++i) {
		_mm512_mask_storeu_ps(&C[i * ldc], m0, c[i][0]);
		if (nr > 16)
			_mm512_mask_storeu_ps(&C[i * ldc + 16], m1, c[i][1]);
	}
}

DRACOLIX_TARGET_AVX2
inline void micro_gemm_256_f_partial(const float *__restrict Ap, int lda,
									 const float *__restrict Bp, int ldb,
									 float *__restrict C, int ldc, int k, int mr,
									 int nr) {
	constexpr int NV = DRACOLIX_NR256_F / 8;
	const int n0 = nr > 8 ? 8 : nr;
	const int n1 = nr > 8 ? (nr - 8 > 8 ? 8 : nr - 8) : 0;

	// AVX2 float masks select on the MSB of each *32-bit* lane, so every
	// enabled lane is INT32_MIN rather than a bit-shift mask.
	constexpr int kOn = static_cast<int>(0x80000000u);
	// All 8 lanes must be covered: an AVX2 ymm holds 8 floats, not 4.
	const __m256i m0 =
		_mm256_setr_epi32((n0 > 0) ? kOn : 0, (n0 > 1) ? kOn : 0,
						   (n0 > 2) ? kOn : 0, (n0 > 3) ? kOn : 0,
						   (n0 > 4) ? kOn : 0, (n0 > 5) ? kOn : 0,
						   (n0 > 6) ? kOn : 0, (n0 > 7) ? kOn : 0);
	const __m256i m1 =
		_mm256_setr_epi32((n1 > 0) ? kOn : 0, (n1 > 1) ? kOn : 0,
						   (n1 > 2) ? kOn : 0, (n1 > 3) ? kOn : 0,
						   (n1 > 4) ? kOn : 0, (n1 > 5) ? kOn : 0,
						   (n1 > 6) ? kOn : 0, (n1 > 7) ? kOn : 0);

	__m256 c[DRACOLIX_MR256_F][NV] = {};
#pragma GCC unroll 6
	for (int i = 0; i < mr; ++i) {
		c[i][0] = _mm256_maskload_ps(&C[i * ldc], m0);
		if (nr > 8)
			c[i][1] = _mm256_maskload_ps(&C[i * ldc + 8], m1);
	}
	for (int p = 0; p < k; ++p) {
		const float *bp = &Bp[(size_t)p * ldb];
		const __m256 b0 = _mm256_maskload_ps(&bp[0], m0);
		const __m256 b1 = _mm256_maskload_ps(&bp[8], m1);
#pragma GCC unroll 6
		for (int i = 0; i < mr; ++i) {
			const __m256 a = _mm256_broadcast_ss(&Ap[i * lda + p]);
			c[i][0] = _mm256_fmadd_ps(a, b0, c[i][0]);
			if (nr > 8)
				c[i][1] = _mm256_fmadd_ps(a, b1, c[i][1]);
		}
	}
#pragma GCC unroll 6
	for (int i = 0; i < mr; ++i) {
		_mm256_maskstore_ps(&C[i * ldc], m0, c[i][0]);
		if (nr > 8)
			_mm256_maskstore_ps(&C[i * ldc + 8], m1, c[i][1]);
	}
}

#endif // DRACOLIX_HAS_TARGET_ATTR

} // namespace dracolix::kernels
