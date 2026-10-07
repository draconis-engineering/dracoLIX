set(DLIXVERSION "0.2.5")
set(DLIXVERSION_BASE "0.2.5")

cmake_minimum_required(VERSION 3.20)
project(dracolix VERSION ${DLIXVERSION_BASE} LANGUAGES CXX)
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

# Direct OpenBLAS/CBLAS backend (GemmKernel::Blas). Independent of the Fortran
# kernels above: needs no Fortran compiler, just cblas.h + a BLAS library.
# Off by default so wheels stay pure C++ with no BLAS runtime dependency.
option(DRACOLIX_USE_BLAS "Enable the OpenBLAS GEMM backend" OFF)

set(DRACOLIX_CORE_SOURCES
	core/src/kernels/gemm.cpp
	core/src/kernels/gemm_blocked.cpp
	core/src/kernels/gemm_blas.cpp
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

# SIMD + warnings only for GCC and Clang (NOT AppleClang/MSVC).
#
# Deliberately NO -march=native and no -mavx2/-mfma here.
#
# `-march=native` compiles the whole library for the *build* machine, which
# makes any binary (or wheel) produced on a modern CI runner fault with SIGILL
# on older user CPUs. SIMD is instead selected per-function at runtime through
# GCC/Clang `target` attributes in core/src/kernels/gemm_micro.hpp, gated by the
# CPUID probe in cpu.hpp — one portable binary, best available kernel.
#
# -mtune is safe to keep: it changes scheduling, never the accepted instruction
# set, so it cannot introduce an illegal instruction.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	target_compile_options(dracolix_core PRIVATE
		$<$<COMPILE_LANGUAGE:CXX>:-O3>
		$<$<COMPILE_LANGUAGE:CXX>:-mtune=native>
		$<$<COMPILE_LANGUAGE:CXX>:-Wall>
		$<$<COMPILE_LANGUAGE:CXX>:-Wextra>
		$<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
	)
endif()

if(DRACOLIX_USE_BLAS)
	find_path(CBLAS_INCLUDE_DIR cblas.h)
	find_library(CBLAS_LIBRARY NAMES openblas blas)
	if(CBLAS_INCLUDE_DIR AND CBLAS_LIBRARY)
		target_compile_definitions(dracolix_core PUBLIC DRACOLIX_HAVE_BLAS)
		target_include_directories(dracolix_core PRIVATE ${CBLAS_INCLUDE_DIR})
		target_link_libraries(dracolix_core PUBLIC ${CBLAS_LIBRARY})
		message(STATUS "DracoLIX: OpenBLAS GEMM backend enabled (${CBLAS_LIBRARY})")
	else()
		message(WARNING
			"DracoLIX: DRACOLIX_USE_BLAS=ON but cblas.h/BLAS library not found; "
			"falling back to the native GEMM kernels.")
	endif()
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
option(DRACOLIX_BUILD_JULIA "Build Julia C API shared lib" OFF)
if(DRACOLIX_BUILD_JULIA)
	add_library(dracolix_c SHARED core/src/c_api.cpp)
	target_link_libraries(dracolix_c PUBLIC dracolix_core)
	set_target_properties(dracolix_c PROPERTIES OUTPUT_NAME dracolix VERSION ${DLIXVERSION})
endif()
# add_subdirectory(bindings/julia)   # ccall-based, no build needed

# ---- Benchmarks ----
if(NOT SKBUILD)
	add_subdirectory(benchmarks)
endif()

add_compile_options(/fsanitize=address)
