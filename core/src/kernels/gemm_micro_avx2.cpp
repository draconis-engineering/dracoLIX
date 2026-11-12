// AVX2/FMA GEMM microkernels, out-of-line for MSVC.
// Licensed under GPL-3.0-or-later
//
// MSVC has no GCC/Clang-style per-function target attributes, so the ISA is
// selected per translation unit instead. This file is compiled with
// /arch:AVX2 (see CMakeLists.txt) and only the AVX2 kernels are emitted; the
// AVX-512 kernels live in gemm_micro_avx512.cpp. Runtime CPUID dispatch makes
// sure these instructions only execute on CPUs that support them.

#if defined(_MSC_VER) && !defined(__clang__)

#pragma warning(disable : 4068) // #pragma GCC unroll is a no-op here

#define DRACOLIX_GEMM_MICRO_DEFINE 1
#define DRACOLIX_GEMM_MICRO_NO_AVX512 1
#include "gemm_micro.hpp"

#endif
