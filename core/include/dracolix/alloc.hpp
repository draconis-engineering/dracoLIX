#pragma once

// Aligned allocation — Phase 3: Alignment + Optimized allocation
// Licensed under GPL-3.0+

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>

// Check for aligned allocation support
#if defined(_WIN32) || defined(_WIN64)
#include <malloc.h>
#endif

namespace dracolix {

constexpr size_t kCacheLine = 64;
constexpr size_t kSIMDAlign = 64; // covers AVX-512

// Aligned allocator for std::vector
// Provides aligned memory allocation for std::vector,
// using platform-specific aligned allocation functions
template <typename T, size_t Align = kSIMDAlign> struct AlignedAllocator {
	using value_type = T;
	using size_type = size_t;
	using difference_type = ptrdiff_t;
	using propagate_on_container_move_assignment = std::true_type;

	AlignedAllocator() noexcept {}
	template <typename U>
	AlignedAllocator(const AlignedAllocator<U, Align> &) noexcept {}

	T *allocate(size_t n) {
		if (n == 0)
			return nullptr;
		size_t bytes = n * sizeof(T);
		size_t aligned_bytes = (bytes + Align - 1) & ~(Align - 1);
		void *p = nullptr;
#if defined(_WIN32) || defined(_WIN64)
		p = _aligned_malloc(aligned_bytes, Align);
		if (!p)
			throw std::bad_alloc();
#elif defined(_ISOC11_SOURCE)
		p = std::aligned_alloc(Align, aligned_bytes);
		if (!p)
			throw std::bad_alloc();
#else
		if (posix_memalign(&p, Align, aligned_bytes) != 0)
			throw std::bad_alloc();
#endif
		return static_cast<T *>(p);
	}
	void deallocate(T *p, size_t) noexcept {
#if defined(_WIN32) || defined(_WIN64)
		_aligned_free(p);
#else
		std::free(p);
#endif
	}
	template <typename U> struct rebind {
		using other = AlignedAllocator<U, Align>;
	};
	bool operator==(const AlignedAllocator &) const noexcept { return true; }
	bool operator!=(const AlignedAllocator &) const noexcept { return false; }
};

// Check if a pointer is aligned
// returns true if the pointer is aligned to the given alignment
inline bool is_aligned(const void *p, size_t align = kSIMDAlign) {
	return (reinterpret_cast<uintptr_t>(p) % align) == 0;
}

// Minimal owning buffer of 64-byte aligned elements.
//
// Used for GEMM packing scratch, where the alignment is load-bearing (each
// packed row must start on a vector boundary) and the size grows on demand.
// Trivially copyable elements only — it never constructs or destructs.
template <typename T, size_t Align = kSIMDAlign> class AlignedBuffer {
  public:
	AlignedBuffer() = default;
	explicit AlignedBuffer(size_t n) { reset(n); }
	~AlignedBuffer() { release(ptr_); }

	AlignedBuffer(const AlignedBuffer &) = delete;
	AlignedBuffer &operator=(const AlignedBuffer &) = delete;
	AlignedBuffer(AlignedBuffer &&o) noexcept : ptr_(o.ptr_), n_(o.n_) {
		o.ptr_ = nullptr;
		o.n_ = 0;
	}
	AlignedBuffer &operator=(AlignedBuffer &&o) noexcept {
		if (this != &o) {
			release(ptr_);
			ptr_ = o.ptr_;
			n_ = o.n_;
			o.ptr_ = nullptr;
			o.n_ = 0;
		}
		return *this;
	}

	void reset(size_t n) {
		if (n <= n_)
			return;
		void *p = nullptr;
		const size_t bytes = ((n * sizeof(T) + Align - 1) / Align) * Align;
#if defined(_WIN32) || defined(_WIN64)
		p = _aligned_malloc(bytes, Align);
#else
		if (posix_memalign(&p, Align, bytes) != 0)
			p = nullptr;
#endif
		if (!p)
			throw std::bad_alloc();
		release(ptr_);
		ptr_ = static_cast<T *>(p);
		n_ = n;
	}

	T *get() const noexcept { return ptr_; }
	T *data() const noexcept { return ptr_; }
	size_t capacity() const noexcept { return n_; }

  private:
	// Free whatever reset() allocated. On Windows the buffer comes from
	// _aligned_malloc(), which the CRT *requires* to be released with
	// _aligned_free(): passing it to std::free() walks a misaligned header and
	// corrupts the heap (STATUS_HEAP_CORRUPTION / 0xc0000374). POSIX buffers
	// come from posix_memalign(), which std::free() is correct for.
	static void release(void *p) noexcept {
#if defined(_WIN32) || defined(_WIN64)
		_aligned_free(p);
#else
		std::free(p);
#endif
	}

	T *ptr_ = nullptr;
	size_t n_ = 0;
};

} // namespace dracolix
