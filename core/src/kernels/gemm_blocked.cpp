// DracoLIX blocked GEMM — cache blocking + packing + threading
// Licensed under GPL-3.0-or-later
//
// Three levels of blocking, the standard Goto-style hierarchy:
//
//   for jc (NC over N)          <- B panel stays resident in L2
//     for pc (KC over K)
//       pack B[pc:pc+kc, jc:jc+nc] -> Bp     (dense, 64B aligned)
//       for ic (MC over M)                <- A panel stays resident in L2
//         pack A[ic:ic+mc, pc:pc+kc] -> Ap  (dense, 64B aligned)
//         for jr (NR over nc)
//           for ir (MR over mc)
//             microkernel(Ap+ir, Bp+jr, C)   <- C tile lives in registers
//
// Row panels of C are independent, so the outermost ic loop is the parallel
// unit: every thread writes a disjoint set of C rows and no synchronisation or
// false sharing is needed.

#include "dracolix/alloc.hpp"
#include "dracolix/cpu.hpp"
#include "dracolix/kernels/dispatch.hpp"
#include "dracolix/thread_pool.hpp"
#include "gemm.hpp"
#include "gemm_micro.hpp"


#include <algorithm>
#include <atomic>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

namespace dracolix::kernels {

// ---------------------------------------------------------------------------
// Cache blocking parameters (f64). Tuned on Zen 4 (32 KB L1d, 1 MB L2/core,
// 32 MB shared L3); override at compile time if you retune for another CPU.
//
//   Bp panel  KC*NC*8 = 256*256*8 = 512 KB  -> L2
//   Ap panel  MC*KC*8 = 256*256*8 = 512 KB  -> L2
//   micro-panel KC*NR*8 = 256*24*8 = 48 KB   -> borderline L1, see tune below
// ---------------------------------------------------------------------------
#ifndef DRACOLIX_GEMM_MC
#define DRACOLIX_GEMM_MC 256
#endif
#ifndef DRACOLIX_GEMM_NC
#define DRACOLIX_GEMM_NC 256
#endif
#ifndef DRACOLIX_GEMM_KC
#define DRACOLIX_GEMM_KC 256
#endif

// Below this many FLOPs a thread dispatch costs more than it saves.
#ifndef DRACOLIX_GEMM_PAR_MIN
#define DRACOLIX_GEMM_PAR_MIN (32 * 1024 * 1024)
#endif

namespace {

constexpr size_t kMC = DRACOLIX_GEMM_MC;
constexpr size_t kNC = DRACOLIX_GEMM_NC;
constexpr size_t kKC = DRACOLIX_GEMM_KC;

#if DRACOLIX_HAS_TARGET_ATTR
// Per-thread packing scratch, grown on demand and reused across calls. One raw
// byte buffer serves both dtypes, so an f32 GEMM never has to allocate a second
// pool. Guarded by a mutex because ThreadPool hands blocks to workers
// dynamically, so the worker index is not stable across calls.
struct Scratch {
	// Two distinct regions: Bp and Ap are live at the same time inside the
	// kc loop, so they must not share storage.
	AlignedBuffer<char> Bp, Ap;

	char *bbuf(size_t need) {
		Bp.reset(need);
		return Bp.get();
	}
	char *abuf(size_t need) {
		Ap.reset(need);
		return Ap.get();
	}
};

Scratch &scratch_for(size_t idx) {
	// One scratch per pool worker, allocated lazily on first use.
	static std::vector<std::unique_ptr<Scratch>> pool;
	static std::mutex pool_mtx;
	std::lock_guard<std::mutex> lk(pool_mtx);
	if (pool.size() <= idx)
		pool.resize(idx + 1);
	if (!pool[idx])
		pool[idx] = std::make_unique<Scratch>();
	return *pool[idx];
}

// --- packing ---------------------------------------------------------------
// A[ic:ic+mc, pc:pc+kc] -> Ap, row-major, dense. Row-major source is already
// contiguous along k, so this is a strided copy that also guarantees each
// packed row starts on a 64-byte boundary.
template <typename T>
void pack_a(const T *A, size_t lda, size_t mc, size_t kc, T *Ap, size_t lda_p) {
	for (size_t i = 0; i < mc; ++i)
		std::memcpy(Ap + i * lda_p, A + i * lda, kc * sizeof(T));
}

// B[pc:pc+kc, jc:jc+nc] -> Bp, row-major, dense, each row aligned to a whole
// number of vectors. Columns beyond nc are zeroed so a full-width microkernel
// can never read past the end of a short final panel.
template <typename T>
void pack_b(const T *B, size_t ldb, size_t kc, size_t nc, T *Bp, size_t ldb_p) {
	for (size_t p = 0; p < kc; ++p) {
		T *dst = Bp + p * ldb_p;
		std::memcpy(dst, B + p * ldb, nc * sizeof(T));
		if (nc < ldb_p)
			std::memset(dst + nc, 0, (ldb_p - nc) * sizeof(T));
	}
}

#endif // DRACOLIX_HAS_TARGET_ATTR

} // namespace

// ---------------------------------------------------------------------------
// Microkernel dispatch table per scalar type. Blocking / packing / threading
// are identical for f64 and f32; only the tile shape and the kernel differ.
// ---------------------------------------------------------------------------
#if DRACOLIX_HAS_TARGET_ATTR
template <typename T> struct Micro;

template <> struct Micro<double> {
	static constexpr size_t MR512 = DRACOLIX_MR512, NR512 = DRACOLIX_NR512;
	static constexpr size_t MR256 = DRACOLIX_MR256, NR256 = DRACOLIX_NR256;
	static void full512(const double *a, int la, const double *b, int lb,
						double *c, int lc, int k) {
		micro_gemm_512(a, la, b, lb, c, lc, k);
	}
	static void full256(const double *a, int la, const double *b, int lb,
						double *c, int lc, int k) {
		micro_gemm_256(a, la, b, lb, c, lc, k);
	}
	static void part512(const double *a, int la, const double *b, int lb,
						double *c, int lc, int k, int mr, int nr) {
		micro_gemm_512_partial(a, la, b, lb, c, lc, k, mr, nr);
	}
	static void part256(const double *a, int la, const double *b, int lb,
						double *c, int lc, int k, int mr, int nr) {
		micro_gemm_256_partial(a, la, b, lb, c, lc, k, mr, nr);
	}
};

template <> struct Micro<float> {
	static constexpr size_t MR512 = DRACOLIX_MR512_F, NR512 = DRACOLIX_NR512_F;
	static constexpr size_t MR256 = DRACOLIX_MR256_F, NR256 = DRACOLIX_NR256_F;
	static void full512(const float *a, int la, const float *b, int lb,
						float *c, int lc, int k) {
		micro_gemm_512_f(a, la, b, lb, c, lc, k);
	}
	static void full256(const float *a, int la, const float *b, int lb,
						float *c, int lc, int k) {
		micro_gemm_256_f(a, la, b, lb, c, lc, k);
	}
	static void part512(const float *a, int la, const float *b, int lb,
						float *c, int lc, int k, int mr, int nr) {
		micro_gemm_512_f_partial(a, la, b, lb, c, lc, k, mr, nr);
	}
	static void part256(const float *a, int la, const float *b, int lb,
						float *c, int lc, int k, int mr, int nr) {
		micro_gemm_256_f_partial(a, la, b, lb, c, lc, k, mr, nr);
	}
};

// One row-panel of C: rows [0,m) x [0,n), all k. Templated on scalar type so
// f64 and f32 share the blocking structure.
template <typename T>
static void gemm_rowpanel_serial(const T *A, size_t lda, const T *B, size_t ldb,
								 T *C, size_t ldc, size_t m, size_t n, size_t k,
								 bool use512, Scratch &sc) {
	using Mi = Micro<T>;
	const size_t MR = use512 ? Mi::MR512 : Mi::MR256;
	const size_t NR = use512 ? Mi::NR512 : Mi::NR256;

	for (size_t jc = 0; jc < n; jc += kNC) {
		const size_t nc = std::min(kNC, n - jc);
		for (size_t pc = 0; pc < k; pc += kKC) {
			const size_t kc = std::min(kKC, k - pc);
			// B panel column width rounded up to a whole number of microkernel
			// tiles, so packed rows stay vector-aligned and the microkernel can
			// never read past the end of a short final panel.
			const size_t ncp = ((nc + NR - 1) / NR) * NR;
			T *Bp = reinterpret_cast<T *>(sc.bbuf(kc * ncp * sizeof(T)));
			pack_b(B + pc * ldb + jc, ldb, kc, nc, Bp, ncp);

			for (size_t ic = 0; ic < m; ic += kMC) {
				const size_t mc = std::min(kMC, m - ic);
				T *Ap = reinterpret_cast<T *>(sc.abuf(mc * kc * sizeof(T)));
				pack_a(A + ic * lda + pc, lda, mc, kc, Ap, kc);

				T *Cp = C + ic * ldc + jc;
				for (size_t jr = 0; jr < nc; jr += NR) {
					const size_t nr = std::min<size_t>(NR, nc - jr);
					for (size_t ir = 0; ir < mc; ir += MR) {
						const size_t mr = std::min<size_t>(MR, mc - ir);
						const T *ap = Ap + ir * kc;
						const T *bp = Bp + jr;
						T *cp = Cp + ir * ldc + jr;
						if (mr == MR && nr == NR) {
							if (use512)
								Mi::full512(ap, (int)kc, bp, (int)ncp, cp,
											(int)ldc, (int)kc);
							else
								Mi::full256(ap, (int)kc, bp, (int)ncp, cp,
											(int)ldc, (int)kc);
						} else {
							if (use512)
								Mi::part512(ap, (int)kc, bp, (int)ncp, cp,
											(int)ldc, (int)kc, (int)mr,
											(int)nr);
							else
								Mi::part256(ap, (int)kc, bp, (int)ncp, cp,
											(int)ldc, (int)kc, (int)mr,
											(int)nr);
						}
					}
				}
			}
		}
	}
}
#endif

// ---------------------------------------------------------------------------
// Public entries: C += A*B, C assumed zeroed by the caller (beta = 1).
// ---------------------------------------------------------------------------
template <typename T>
static void gemm_blocked_impl(const T *A, const T *B, T *C, size_t m, size_t n,
							  size_t p, size_t nthreads, GEMMKernel kernel,
							  void (*scalar_ref)(const T *, const T *, T *,
												 size_t, size_t, size_t)) {
#if !DRACOLIX_HAS_TARGET_ATTR
	(void)nthreads;
	(void)kernel;
	scalar_ref(A, B, C, m, n, p);
#else
	// Honour the requested kernel, but never run an ISA this CPU lacks.
	const auto feats = cpu::detect();
#if defined(__GNUC__) || defined(__clang__)
	const bool cpu_avx512 = feats.avx512f && feats.avx512dq && feats.avx512vl;
#else
	const bool cpu_avx512 = false;
#endif

	bool use512 = false;
	if (kernel == GEMMKernel::Avx512)
		use512 = cpu_avx512;
	else if (kernel == GEMMKernel::Avx2)
		use512 = false;

	if (!use512 && !feats.avx2) {
		scalar_ref(A, B, C, m, n, p);
		return;
	}

	// Split M into cache-blocked row panels, one per worker. Panels are
	// disjoint in C, so no locking and no false sharing. One panel when it is
	// small.
	const size_t rows_per_panel = kMC;
	const size_t m_panels = (m + rows_per_panel - 1) / rows_per_panel;
	size_t panels = 1;
	if (m_panels > 1 && nthreads > 1)
		panels = std::min(m_panels, nthreads);

	ThreadPool &pool = ThreadPool::global();
	if (panels == 1) {
		gemm_rowpanel_serial(A, n, B, p, C, p, m, p, n, use512, scratch_for(0));
		return;
	}

	std::atomic<size_t> next{0};
	std::vector<std::future<void>> futs;
	futs.reserve(panels);
	for (size_t t = 0; t < panels; ++t) {
		futs.emplace_back(pool.enqueue([&, t] {
			for (;;) {
				const size_t pi = next.fetch_add(1);
				if (pi >= m_panels)
					break;
				const size_t i0 = pi * rows_per_panel;
				const size_t mc = std::min(rows_per_panel, m - i0);
				gemm_rowpanel_serial(A + i0 * n, n, B, p, C + i0 * p, p, mc, p,
									 n, use512, scratch_for(t));
			}
		}));
	}
	for (auto &f : futs)
		f.get();
#endif
}

void gemm_blocked_f64(const double *A, const double *B, double *C, size_t m,
					  size_t n, size_t p, size_t nthreads, GEMMKernel kernel) {
	gemm_blocked_impl(A, B, C, m, n, p, nthreads, kernel, gemm_f64);
}

void gemm_blocked_f32(const float *A, const float *B, float *C, size_t m,
					  size_t n, size_t p, size_t nthreads, GEMMKernel kernel) {
	gemm_blocked_impl(A, B, C, m, n, p, nthreads, kernel, gemm_f32);
}

size_t gemm_recommended_threads(size_t m, size_t n, size_t p) {
	const size_t ops = m * n * p;
	if (ops < DRACOLIX_GEMM_PAR_MIN)
		return 1;
	size_t hw = ThreadPool::global().size();
	// Row panels are the only parallel axis; if we cannot fill the machine with
	// at least one cache-blocked panel each, threading costs more than it
	// saves.
	const size_t m_panels = (m + kMC - 1) / kMC;
	return std::max<size_t>(1, std::min(hw, m_panels));
}

} // namespace dracolix::kernels
