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
