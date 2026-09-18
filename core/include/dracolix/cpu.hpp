#pragma once
// CPU feature detection — Phase 3
// Licensed under GPL-3.0-only
#include <string>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  #if defined(_MSC_VER)
    #include <intrin.h>
  #else
    #include <cpuid.h>
  #endif
#endif

namespace dracolix {


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

inline Features detect() {
	Features f;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
	unsigned int eax, ebx, ecx, edx;
	if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
		f.sse2 = edx & (1u << 26);
		f.sse4_1 = ecx & (1u << 19);
		f.avx = ecx & (1u << 28);
		f.fma = ecx & (1u << 12);
	}
	if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
		f.avx2 = ebx & (1u << 5);
		f.avx512f = ebx & (1u << 16);
	}
	// brand string (optional)
	char brand[49] = {0};
	unsigned int regs[4];
	__get_cpuid(0x80000002, &eax, &ebx, &ecx, &edx);
	*reinterpret_cast<unsigned int *>(&brand[0]) = eax;
	*reinterpret_cast<unsigned int *>(&brand[4]) = ebx;
	*reinterpret_cast<unsigned int *>(&brand[8]) = ecx;
	*reinterpret_cast<unsigned int *>(&brand[12]) = edx;
	__get_cpuid(0x80000003, &eax, &ebx, &ecx, &edx);
	*reinterpret_cast<unsigned int *>(&brand[16]) = eax;
	*reinterpret_cast<unsigned int *>(&brand[20]) = ebx;
	*reinterpret_cast<unsigned int *>(&brand[24]) = ecx;
	*reinterpret_cast<unsigned int *>(&brand[28]) = edx;
	__get_cpuid(0x80000004, &eax, &ebx, &ecx, &edx);
	*reinterpret_cast<unsigned int *>(&brand[32]) = eax;
	*reinterpret_cast<unsigned int *>(&brand[36]) = ebx;
	*reinterpret_cast<unsigned int *>(&brand[40]) = ecx;
	*reinterpret_cast<unsigned int *>(&brand[44]) = edx;
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
