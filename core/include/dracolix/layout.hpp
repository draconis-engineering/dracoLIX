#pragma once

// Layout enumeration
// Licensed under GPL-3.0+

#include <cstdint>
namespace dracolix {
enum class Layout : uint8_t {
	RowMajor = 0, // C contiguous
	ColMajor = 1  // Fortran contiguous
};
} // namespace dracolix
