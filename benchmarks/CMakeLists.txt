add_executable(bench_core bench_core.cpp)
target_link_libraries(bench_core PRIVATE dracolix_core)
target_include_directories(bench_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optional: add to ctest as benchmark (not a pass/fail test)
add_test(NAME bench_core COMMAND bench_core 20 5)
set_tests_properties(bench_core PROPERTIES LABELS benchmark)
