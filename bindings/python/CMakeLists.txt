# Outgoing-only Python binding. Depends on dracolix_core, not vice versa.
# Example with nanobind (preferred) or pybind11.
# find_package(nanobind CONFIG REQUIRED)
# nanobind_add_module(_dracolix_nb bindings/python/src/dracolix_nb.cpp)
# target_link_libraries(_dracolix_nb PRIVATE dracolix_core)
cmake_minimum_required(VERSION 3.20)
message(STATUS "Python bindings not yet implemented - core builds standalone")
