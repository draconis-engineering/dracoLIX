add_executable(bench_core bench_core.cpp)
target_link_libraries(bench_core PRIVATE dracolix_core)
target_include_directories(bench_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optional: add to ctest as benchmark (not a pass/fail test)
add_test(NAME bench_core COMMAND bench_core 20 5)
set_tests_properties(bench_core PROPERTIES LABELS benchmark)

# Backend comparison harness: scalar vs avx2 vs avx512 vs openblas, single- and
# multi-threaded, plus correctness validation against the scalar reference.
# This is the benchmark the GEMM optimisation work is measured with.
add_executable(bench_gemm_backends bench_gemm_backends.cpp)
target_link_libraries(bench_gemm_backends PRIVATE dracolix_core)
target_include_directories(bench_gemm_backends PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Fortran/BLAS control benchmark. Only meaningful when the Fortran kernels are
# built (it calls matmatmul_c, the symbol gfortran emits for MATMUL).
if(DRACOLIX_USE_FORTRAN)
	add_executable(bench_fortran bench_fortran.cpp)
	target_link_libraries(bench_fortran PRIVATE dracolix_core)
endif()
