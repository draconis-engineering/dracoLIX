add_executable(test_array test_array.cpp)
target_link_libraries(test_array PRIVATE dracolix_core)
add_test(NAME test_array COMMAND test_array)

add_executable(test_linalg test_linalg.cpp)
target_link_libraries(test_linalg PRIVATE dracolix_core)
add_test(NAME test_linalg COMMAND test_linalg)

add_executable(test_sparse test_sparse.cpp)
target_link_libraries(test_sparse PRIVATE dracolix_core)
add_test(NAME test_sparse COMMAND test_sparse)

add_executable(test_decomp test_decomp.cpp)
target_link_libraries(test_decomp PRIVATE dracolix_core)
add_test(NAME test_decomp COMMAND test_decomp)

add_executable(test_eigen_svd test_eigen_svd.cpp)
target_link_libraries(test_eigen_svd PRIVATE dracolix_core)
add_test(NAME test_eigen_svd COMMAND test_eigen_svd)

add_executable(test_complex test_complex.cpp)
target_link_libraries(test_complex PRIVATE dracolix_core)
add_test(NAME test_complex COMMAND test_complex)

add_executable(test_numerics_ode test_numerics_ode.cpp)
target_link_libraries(test_numerics_ode PRIVATE dracolix_core)
add_test(NAME test_numerics_ode COMMAND test_numerics_ode)

add_executable(test_pde test_pde.cpp)
target_link_libraries(test_pde PRIVATE dracolix_core)
add_test(NAME test_pde COMMAND test_pde)

add_executable(test_random test_random.cpp)
target_link_libraries(test_random PRIVATE dracolix_core)
add_test(NAME test_random COMMAND test_random)

add_executable(test_distributions test_distributions.cpp)
target_link_libraries(test_distributions PRIVATE dracolix_core)
add_test(NAME test_distributions COMMAND test_distributions)

add_executable(test_optimize test_optimize.cpp)
target_link_libraries(test_optimize PRIVATE dracolix_core)
add_test(NAME test_optimize COMMAND test_optimize)

add_executable(test_least_squares test_least_squares.cpp)
target_link_libraries(test_least_squares PRIVATE dracolix_core)
add_test(NAME test_least_squares COMMAND test_least_squares)

add_executable(test_units test_units.cpp)
target_link_libraries(test_units PRIVATE dracolix_core)
add_test(NAME test_units COMMAND test_units)

add_executable(test_constants test_constants.cpp)
target_link_libraries(test_constants PRIVATE dracolix_core)
add_test(NAME test_constants COMMAND test_constants)

add_executable(test_signal test_signal.cpp)
target_link_libraries(test_signal PRIVATE dracolix_core)
add_test(NAME test_signal COMMAND test_signal)

add_executable(test_timeseries test_timeseries.cpp)
target_link_libraries(test_timeseries PRIVATE dracolix_core)
add_test(NAME test_timeseries COMMAND test_timeseries)

# Exercises every GEMM backend across awkward shapes (remainder tiles,
# degenerate m/n/p) against a naive independent oracle.
add_executable(test_gemm test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE dracolix_core)
add_test(NAME test_gemm COMMAND test_gemm)
