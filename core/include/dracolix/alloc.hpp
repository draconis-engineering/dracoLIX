#pragma once
// Aligned allocation — Phase 3: Alignment + Optimized allocation
// Licensed under GPL-3.0-only
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>

#if defined(_WIN32) || defined(_WIN64)
#include <malloc.h>
#endif

namespace dracolix {

constexpr size_t kCacheLine = 64;
constexpr size_t kSimdAlign = 64; // covers AVX-512

// Aligned allocator for std::vector
template <typename T, size_t Align = kSimdAlign> struct AlignedAllocator {
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

inline bool is_aligned(const void *p, size_t align = kSimdAlign) {
	return (reinterpret_cast<uintptr_t>(p) % align) == 0;
}

} // namespace dracolix
