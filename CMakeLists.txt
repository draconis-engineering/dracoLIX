cmake_minimum_required(VERSION 3.20)
project(dracolix VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The classic HPC Fortran kernels
# (fortran/linalg.f90 -> core/src/kernels/linalg.f90)
# are optional. Off by default: the core builds as pure C++ kernels. Enable with
#   -DDRACOLIX_USE_FORTRAN=ON
# Requires a Fortran compiler (gfortran) and an OpenBLAS install.
option(DRACOLIX_USE_FORTRAN
	"Build Fortran kernels and link gfortran + OpenBLAS" OFF)

set(DRACOLIX_CORE_SOURCES
	core/src/kernels/gemm.cpp
)

if(DRACOLIX_USE_FORTRAN)
	enable_language(Fortran)
	set(CMAKE_Fortran_FLAGS
	"-O3 -march=native -ffast-math -funroll-loops -fexternal-blas -fPIC")
	list(APPEND DRACOLIX_CORE_SOURCES core/src/kernels/linalg.f90)
endif()

# ---- Core (outgoing-only, no Python/Julia deps) ----
add_library(dracolix_core STATIC ${DRACOLIX_CORE_SOURCES})
# Must be PIC so it can be linked into the nanobind shared module on Linux
set_target_properties(dracolix_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(dracolix_core PUBLIC Threads::Threads)

# FORCE C++20 directly onto this library and anything that uses it
target_compile_features(dracolix_core PUBLIC cxx_std_20)

target_include_directories(dracolix_core PUBLIC
	core/include
)

# x86 SIMD + warnings only for GCC and Clang (NOT AppleClang/MSVC, where
# -mavx2/-mfma/-march=native are undefined or unsupported on arm64). gated to
# C++ so a mixed Fortran build does not receive C++-only flags.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	target_compile_options(dracolix_core PRIVATE
		$<$<COMPILE_LANGUAGE:CXX>:-O3>
		$<$<COMPILE_LANGUAGE:CXX>:-march=native>
		$<$<COMPILE_LANGUAGE:CXX>:-Wall>
		$<$<COMPILE_LANGUAGE:CXX>:-Wextra>
		$<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
		$<$<COMPILE_LANGUAGE:CXX>:-mavx2>
		$<$<COMPILE_LANGUAGE:CXX>:-mfma>
	)
endif()

if(DRACOLIX_USE_FORTRAN)
	# Fortran runtime + BLAS
	find_package(BLAS QUIET)
	if(BLAS_FOUND)
		target_link_libraries(dracolix_core PUBLIC ${BLAS_LIBRARIES} gfortran)
	else()
		target_link_libraries(dracolix_core PUBLIC gfortran openblas)
	endif()
endif()

# ---- Tests (C++ only, no bindings) ----
# Skipped during wheel builds (SKBUILD) so wheels don't compile needless
# test/benchmark executables (bench_core prints GCC's __VERSION__ macro).
if(NOT SKBUILD)
	enable_testing()
	add_subdirectory(tests/cpp)
endif()

# ---- Language bindings are separate, depend on dracolix_core ----
# The Python bindings build either:
#   - as part of a wheel build (scikit-build-core sets SKBUILD), or
#   - explicitly with -DDRACOLIX_BUILD_PYTHON=ON.
option(DRACOLIX_BUILD_PYTHON "Build the nanobind Python bindings" OFF)
if(SKBUILD OR DRACOLIX_BUILD_PYTHON)
	add_subdirectory(bindings/python)
endif()
# add_subdirectory(bindings/julia)   # CxxWrap - outgoing only (later)

# ---- Benchmarks ----
if(NOT SKBUILD)
	add_subdirectory(benchmarks)
endif()
