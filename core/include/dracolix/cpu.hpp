#pragma once
// CPU feature detection — Phase 3
// Licensed under GPL-3.0-only
#include <cstring>
#include <string>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace dracolix::cpu {

struct Features {
	bool sse2 = false;
	bool sse4_1 = false;
	bool avx = false;
	bool avx2 = false;
	bool avx512f = false;
	bool fma = false;
	std::string brand;
};

// Fill regs[4] with eax, ebx, ecx, edx for the given CPUID leaf/subleaf.
// GCC/Clang: cpuid.h; MSVC: intrin.h (__cpuidex).
inline void cpuid(unsigned int regs[4], unsigned int leaf, unsigned int subleaf) {
#if defined(_MSC_VER)
	__cpuidex(reinterpret_cast<int *>(regs), leaf, subleaf);
#elif defined(__x86_64__) || defined(__i386__)
	__get_cpuid_count(leaf, subleaf, &regs[0], &regs[1], &regs[2], &regs[3]);
#else
	(void)regs;
	(void)leaf;
	(void)subleaf;
	std::memset(regs, 0, 4 * sizeof(unsigned int));
#endif
}

inline Features detect() {
	Features f;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
	unsigned int regs[4];
	cpuid(regs, 1, 0);
	f.sse2 = regs[3] & (1u << 26);
	f.sse4_1 = regs[2] & (1u << 19);
	f.avx = regs[2] & (1u << 28);
	f.fma = regs[2] & (1u << 12);
	cpuid(regs, 7, 0);
	f.avx2 = regs[1] & (1u << 5);
	f.avx512f = regs[1] & (1u << 16);
	// brand string (optional)
	char brand[49] = {0};
	for (unsigned int leaf = 0x80000002; leaf <= 0x80000004; ++leaf) {
		cpuid(regs, leaf, 0);
		std::memcpy(brand + (leaf - 0x80000002) * 16, regs, 16);
	}
	f.brand = std::string(brand);
	// trim
	f.brand.erase(f.brand.find_last_not_of(" \0") + 1);
	f.brand.erase(0, f.brand.find_first_not_of(" "));
#else
	f.brand = "unknown";
#endif
	return f;
}

inline std::string to_string(const Features &fe) {
	std::string s = fe.brand.empty() ? "unknown" : fe.brand;
	s += " [";
	if (fe.sse2)
		s += "SSE2 ";
	if (fe.sse4_1)
		s += "SSE4.1 ";
	if (fe.avx)
		s += "AVX ";
	if (fe.avx2)
		s += "AVX2 ";
	if (fe.avx512f)
		s += "AVX512F ";
	if (fe.fma)
		s += "FMA ";
	s += "]";
	return s;
}

} // namespace dracolix::cpu