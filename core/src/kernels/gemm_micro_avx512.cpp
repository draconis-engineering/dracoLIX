// AVX-512 GEMM microkernels, out-of-line for MSVC.
// Licensed under GPL-3.0-or-later
//
// Compiled with /arch:AVX512 (see CMakeLists.txt). The functions are present in
// the DLL but are only called after the CPUID gate in dispatch.hpp confirms
// the CPU (and OS) support AVX-512. No other code in this TU runs on a CPU
// without AVX-512.

#if defined(_MSC_VER) && !defined(__clang__)

#pragma warning(disable : 4068) // #pragma GCC unroll is a no-op here

#define DRACOLIX_GEMM_MICRO_DEFINE 1
#define DRACOLIX_GEMM_MICRO_NO_AVX2 1
#include "gemm_micro.hpp"

#endif
