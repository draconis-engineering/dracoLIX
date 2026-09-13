# DracoLIX Python bindings (nanobind) - outgoing only.
# Depends on the core target dracolix_core being defined by the top-level CMake.

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# nanobind ships its CMake config via the Python package:
#   python -m nanobind --cmake_dir   ->  <prefix>/nanobind/cmake
set(nanobind_ROOT "")
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_dracolix_nb src/dracolix_nb.cpp)
target_link_libraries(_dracolix_nb PRIVATE dracolix_core)

if(MSVC)
    # The core headers are shared with the pure-C++ build where these warnings
    # are treated as errors on GCC/Clang; silence size/ptr-noise for the bindings.
    target_compile_options(_dracolix_nb PRIVATE /wd4244 /wd4267 /wd4804 /bigobj)
endif()

install(TARGETS _dracolix_nb
        LIBRARY DESTINATION dracolix
        RUNTIME DESTINATION dracolix)