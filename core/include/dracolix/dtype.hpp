#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace dracolix {

// Full planned type universe (spec). Only a subset is implemented initially.
// Keep enum values stable - they are part of the ABI.
enum class DType : uint32_t {
    // signed integers
    I4 = 0, I8, I16, I32, I64, I128, I256,
    // unsigned integers (reserve, not yet exposed)
    U4, U8, U16, U32, U64, U128, U256,
    // floats
    F8, F16, F32, F64, F128, F256, F512,
    // complex (C16 = 2*F8 etc.)
    C16, C32, C64, C128, C256, C512, C1024,
    Bool,
    Char,
    Str,
    Void
};

// Implemented subset for Phase 0 - gate all kernel dispatch through this.
inline bool is_implemented(DType dt) {
    switch (dt) {
        case DType::F32:
        case DType::F64:
        case DType::I32:
        case DType::I64:
        case DType::Bool:
            return true;
        default:
            return false;
    }
}

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::I4: return 1; // packed, placeholder
        case DType::I8:  return 1;
        case DType::I16: return 2;
        case DType::I32: return 4;
        case DType::I64: return 8;
        case DType::I128: return 16;
        case DType::I256: return 32;
        case DType::U4: return 1;
        case DType::U8: return 1;
        case DType::U16: return 2;
        case DType::U32: return 4;
        case DType::U64: return 8;
        case DType::U128: return 16;
        case DType::U256: return 32;
        case DType::F8:  return 1;
        case DType::F16: return 2;
        case DType::F32: return 4;
        case DType::F64: return 8;
        case DType::F128: return 16;
        case DType::F256: return 32;
        case DType::F512: return 64;
        case DType::C16: return 2;
        case DType::C32: return 4;
        case DType::C64: return 8;
        case DType::C128: return 16;
        case DType::C256: return 32;
        case DType::C512: return 64;
        case DType::C1024: return 128;
        case DType::Bool: return 1;
        case DType::Char: return 1;
        default: throw std::invalid_argument("Unknown DType");
    }
}

// Returns the alignment of the given DType.
// For now alignment == size for implemented types; SIMD/cache tuning later.
inline size_t dtype_alignment(DType dt) {
    return dtype_size(dt);
}

// Returns the name of the given DType.
inline std::string dtype_name(DType dt) {
    switch (dt) {
        case DType::F32: return "f32";
        case DType::F64: return "f64";
        case DType::I32: return "i32";
        case DType::I64: return "i64";
        case DType::Bool: return "bool";
        default: return "unimplemented";
    }
}

// C++ type -> DType mapping (only implemented types)
template <typename T> struct DTypeOf { static constexpr DType value = DType::Void; };
template <> struct DTypeOf<float>  { static constexpr DType value = DType::F32; };
template <> struct DTypeOf<double> { static constexpr DType value = DType::F64; };
template <> struct DTypeOf<int32_t>{ static constexpr DType value = DType::I32; };
template <> struct DTypeOf<int64_t>{ static constexpr DType value = DType::I64; };
template <> struct DTypeOf<bool>   { static constexpr DType value = DType::Bool; };

} // namespace dracolix
