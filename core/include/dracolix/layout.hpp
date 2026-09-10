#pragma once
#include <cstdint>
namespace dracolix {
enum class Layout : uint8_t {
    RowMajor = 0, // C contiguous
    ColMajor = 1  // Fortran contiguous
};
} // namespace dracolix
