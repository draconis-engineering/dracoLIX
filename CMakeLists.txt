cmake_minimum_required(VERSION 3.20)
project(dracolix VERSION 0.1.0 LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_Fortran_FLAGS "-O3 -march=native -ffast-math -funroll-loops -fexternal-blas -fPIC")

# ---- Core (outgoing-only, no Python/Julia deps) ----
add_library(dracolix_core STATIC
    core/src/kernels/gemm.cpp
    core/src/kernels/linalg.f90
)

target_include_directories(dracolix_core PUBLIC
    core/include
)

target_compile_options(dracolix_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
)

# Fortran runtime + BLAS (same as old build.rs)
find_package(BLAS QUIET)
if(BLAS_FOUND)
    target_link_libraries(dracolix_core PUBLIC ${BLAS_LIBRARIES} gfortran)
else()
    target_link_libraries(dracolix_core PUBLIC gfortran openblas)
endif()

# ---- Tests (C++ only, no bindings) ----
enable_testing()
add_subdirectory(tests/cpp OPTIONAL)

# ---- Language bindings are separate, depend on dracolix_core ----
# add_subdirectory(bindings/python)  # nanobind/pybind11 - outgoing only
# add_subdirectory(bindings/julia)   # CxxWrap - outgoing only

# ---- Benchmarks ----
# add_subdirectory(benchmarks) # later
