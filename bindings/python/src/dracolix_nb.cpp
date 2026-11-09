// DracoLIX Python bindings (nanobind).
//
// Interop is done through the PEP 3118 buffer protocol.
// We export buffers from dracolix Array (bf_getbuffer) and consume buffers
// from numpy/memoryview (PyObject_GetBuffer).
//
// Licensed under GPL-3.0-only - see LICENSE

#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "dracolix/dracolix.hpp"

// Injected by bindings/python/CMakeLists.txt from the distribution version
// (pyproject.toml under scikit-build-core, or the CMake project version).
#ifndef DRACOLIX_PY_VERSION
#define DRACOLIX_PY_VERSION "0.0.0"
#endif

#include <Python.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace nb = nanobind;

namespace {

// ---------------------------------------------------------------------------
// Detect the "kind" of a Python scalar (bool / int / float / complex)
// ---------------------------------------------------------------------------
enum class NumKind { None, Bool, Int, Float, Complex };

NumKind num_kind(nb::handle h) noexcept {
	if (PyBool_Check(h.ptr()))
		return NumKind::Bool;
	if (PyLong_Check(h.ptr()))
		return NumKind::Int;
	if (PyFloat_Check(h.ptr()))
		return NumKind::Float;
	if (PyComplex_Check(h.ptr()))
		return NumKind::Complex;
	return NumKind::None;
}

bool is_number(nb::handle h) noexcept { return num_kind(h) != NumKind::None; }

// ---------------------------------------------------------------------------
// The Python-facing array: a type-erased container over the core Array<T>
// ---------------------------------------------------------------------------

struct DlxArray {
	using Variant =
		std::variant<dracolix::Array<float>, dracolix::Array<double>,
					 dracolix::Array<int8_t>, dracolix::Array<int16_t>,
					 dracolix::Array<int32_t>, dracolix::Array<int64_t>,
					 dracolix::Array<uint8_t>, dracolix::Array<uint16_t>,
					 dracolix::Array<uint32_t>, dracolix::Array<uint64_t>,
					 dracolix::Array<std::complex<float>>,
					 dracolix::Array<std::complex<double>>,
					 dracolix::Array<bool>>;
	Variant v;

	DlxArray() = delete;
	explicit DlxArray(Variant value) : v(std::move(value)) {}

	template <typename T>
	explicit DlxArray(dracolix::Array<T> a) : v(std::move(a)) {}

	template <typename T> dracolix::Array<T> *as() {
		return std::get_if<dracolix::Array<T>>(&v);
	}
	template <typename T> const dracolix::Array<T> *as() const {
		return std::get_if<dracolix::Array<T>>(&v);
	}

	dracolix::DType dtype() const {
		return std::visit([](const auto &a) { return a.dtype(); }, v);
	}
	size_t size() const {
		return std::visit([](const auto &a) { return a.size(); }, v);
	}
	size_t ndim() const {
		return std::visit([](const auto &a) { return a.ndim(); }, v);
	}
	const std::vector<size_t> &shape() const {
		return std::visit(
			[](const auto &a) -> const std::vector<size_t> & {
				return a.shape();
			},
			v);
	}
	const std::vector<size_t> &strides() const {
		return std::visit(
			[](const auto &a) -> const std::vector<size_t> & {
				return a.strides();
			},
			v);
	}
	dracolix::Layout layout() const {
		return std::visit([](const auto &a) { return a.layout(); }, v);
	}
	size_t itemsize() const { return dracolix::dtype_size(dtype()); }
	void *data_ptr() {
		return std::visit([](auto &a) -> void * { return a.data(); }, v);
	}
	const void *data_ptr() const {
		return std::visit(
			[](const auto &a) -> const void * { return a.data(); }, v);
	}
};

// Extract the element type from dracolix::Array<T> (the core has no value_type;
// std::visit gives us the Array type, not the scalar).
template <typename ArrayT> struct array_scalar;
template <typename T> struct array_scalar<dracolix::Array<T>> {
	using type = T;
};

// Registered Python type objects (filled during NB_MODULE).
PyTypeObject *g_dlx_type = nullptr;
PyTypeObject *g_dtype_type = nullptr;

bool is_array(nb::handle h) noexcept {
	return g_dlx_type && PyObject_TypeCheck(h.ptr(), g_dlx_type);
}
bool is_dtype(nb::handle h) noexcept {
	return g_dtype_type && PyObject_TypeCheck(h.ptr(), g_dtype_type);
}

DlxArray &as_array(nb::handle h) {
	if (!is_array(h))
		throw nb::type_error("expected a dracolix Array");
	return *nb::cast<DlxArray *>(h);
}

// ---------------------------------------------------------------------------
// DType wrapper + singleton registry
// ---------------------------------------------------------------------------

struct DlxDType {
	dracolix::DType dt = dracolix::DType::Void;
	constexpr explicit DlxDType(dracolix::DType d) : dt(d) {}
};

nb::object dtype_singleton(dracolix::DType dt) {
	// The singletons are Python-owned (stored as attributes of this module).
	// Look them up on demand instead of caching them in C++ static storage:
	// a static nb::object would outlive the interpreter and never be released,
	// producing nanobind "leaked instances" warnings at interpreter shutdown.
	auto mod = nb::module_::import_("dracolix._dracolix_nb");
	switch (dt) {
	case dracolix::DType::F32:
		return mod.attr("f32");
	case dracolix::DType::F64:
		return mod.attr("f64");
	case dracolix::DType::I8:
		return mod.attr("i8");
	case dracolix::DType::I16:
		return mod.attr("i16");
	case dracolix::DType::I32:
		return mod.attr("i32");
	case dracolix::DType::I64:
		return mod.attr("i64");
	case dracolix::DType::U8:
		return mod.attr("u8");
	case dracolix::DType::U16:
		return mod.attr("u16");
	case dracolix::DType::U32:
		return mod.attr("u32");
	case dracolix::DType::U64:
		return mod.attr("u64");
	case dracolix::DType::C64:
		return mod.attr("c64");
	case dracolix::DType::C128:
		return mod.attr("c128");
	case dracolix::DType::Bool:
		return mod.attr("bool");
	default:
		throw nb::type_error("unsupported dtype");
	}
}

std::optional<dracolix::DType> optional_dtype(nb::handle h) {
	if (h.is_none())
		return std::nullopt;
	// Allow the builtin `complex` type as an alias for c128 (NumPy-style
	// `dtype=complex`). `float`/`int`/`bool` are intentionally not accepted
	// here; use dlx.f64 / dlx.i64 / dlx.bool_ for an explicit dtype.
	if (h.ptr() == (PyObject *)&PyComplex_Type)
		return dracolix::DType::C128;
	if (!is_dtype(h))
		throw nb::type_error(
			"dtype must be one of dlx.f32, dlx.f64, dlx.i8, dlx.i16, "
			"dlx.i32, dlx.i64, dlx.u8, dlx.u16, dlx.u32, dlx.u64, "
			"dlx.c64, dlx.c128, dlx.bool_, or complex");
	return nb::cast<DlxDType *>(h)->dt;
}

// ---------------------------------------------------------------------------
// Scalar conversion
//
// nanobind 3.x has no std::complex<T> caster, so complex scalars cross the
// Python boundary through the CPython complex API.
// ---------------------------------------------------------------------------

std::complex<double> to_complex(nb::handle h) {
	if (PyComplex_Check(h.ptr())) {
		Py_complex z = PyComplex_AsCComplex(h.ptr());
		return {z.real, z.imag};
	}
	return {nb::cast<double>(h), 0.0};
}

double to_double(nb::handle h) {
	if (PyComplex_Check(h.ptr()))
		return PyComplex_AsCComplex(h.ptr()).real;
	return nb::cast<double>(h);
}

template <typename T> nb::object scalar_to_py(T v) {
	if constexpr (dracolix::is_complex_v<T>) {
		return nb::steal<nb::object>(PyComplex_FromDoubles(
			static_cast<double>(v.real()), static_cast<double>(v.imag())));
	} else {
		return nb::cast(v);
	}
}

// Convert a Python scalar to an integer element type with NumPy-like
// restrictions: floats must be integral, complex is rejected, and values
// must fit the target type (no silent wrap-around).
template <typename T> T integer_from_py(nb::handle h) {
	static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>);
	std::string name = dracolix::dtype_name(dracolix::DTypeOf<T>::value);
	if (num_kind(h) == NumKind::Complex)
		throw nb::type_error(("cannot convert complex to " + name).c_str());
	if (num_kind(h) == NumKind::Float) {
		double d = nb::cast<double>(h);
		if (d != std::trunc(d))
			throw nb::type_error(
				("cannot convert non-integral float to " + name).c_str());
		long double ld = (long double)d;
		if (ld < (long double)std::numeric_limits<T>::min() ||
			ld > (long double)std::numeric_limits<T>::max())
			throw nb::value_error(("float out of range for " + name).c_str());
		return (T)d;
	}
	if constexpr (std::is_signed_v<T>) {
		long long v = nb::cast<long long>(h);
		if ((long double)v < (long double)std::numeric_limits<T>::min() ||
			(long double)v > (long double)std::numeric_limits<T>::max())
			throw nb::value_error(("integer out of range for " + name).c_str());
		return (T)v;
	} else {
		// PyLong_AsUnsignedLongLong rejects negative values (OverflowError)
		// while nb::cast would surface an opaque "bad cast" RuntimeError.
		unsigned long long v;
		if (PyLong_Check(h.ptr())) {
			v = PyLong_AsUnsignedLongLong(h.ptr());
			if (PyErr_Occurred()) {
				PyErr_Clear();
				throw nb::value_error(
					("integer out of range for " + name).c_str());
			}
		} else {
			v = nb::cast<unsigned long long>(h);
		}
		if ((long double)v > (long double)std::numeric_limits<T>::max())
			throw nb::value_error(("integer out of range for " + name).c_str());
		return (T)v;
	}
}

nb::object py_scalar_at(const DlxArray &a, int64_t offset) {
	return std::visit(
		[offset](const auto &arr) {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			return scalar_to_py<T>(arr[(size_t)offset]);
		},
		a.v);
}

DlxArray scalar_slot(nb::handle h) {
	switch (num_kind(h)) {
	case NumKind::Bool: {
		dracolix::Array<int32_t> a(std::vector<size_t>{1});
		a[0] = nb::cast<bool>(h) ? 1 : 0;
		return DlxArray(std::move(a));
	}
	case NumKind::Int: {
		// Python integers that do not fit int64 are represented as u64.
		long long v = PyLong_AsLongLong(h.ptr());
		if (v == -1 && PyErr_Occurred()) {
			PyErr_Clear();
			unsigned long long u = PyLong_AsUnsignedLongLong(h.ptr());
			if (u == (unsigned long long)-1 && PyErr_Occurred())
				throw nb::python_error();
			dracolix::Array<uint64_t> a(std::vector<size_t>{1});
			a[0] = (uint64_t)u;
			return DlxArray(std::move(a));
		}
		dracolix::Array<int64_t> a(std::vector<size_t>{1});
		a[0] = (int64_t)v;
		return DlxArray(std::move(a));
	}
	case NumKind::Float: {
		double v = nb::cast<double>(h);
		dracolix::Array<double> a(std::vector<size_t>{1});
		a[0] = v;
		return DlxArray(std::move(a));
	}
	case NumKind::Complex: {
		std::complex<double> v = to_complex(h);
		dracolix::Array<std::complex<double>> a(std::vector<size_t>{1});
		a[0] = v;
		return DlxArray(std::move(a));
	}
	default:
		throw nb::type_error("expected a scalar");
	}
}

// ---------------------------------------------------------------------------
// dtype promotion (rules documented in __init__.pyi)
// ---------------------------------------------------------------------------

bool is_complex_dtype(dracolix::DType d) {
	return d == dracolix::DType::C64 || d == dracolix::DType::C128;
}

bool is_unsigned_dtype(dracolix::DType d) {
	switch (d) {
	case dracolix::DType::U8:
	case dracolix::DType::U16:
	case dracolix::DType::U32:
	case dracolix::DType::U64:
		return true;
	default:
		return false;
	}
}

bool is_signed_dtype(dracolix::DType d) {
	switch (d) {
	case dracolix::DType::I8:
	case dracolix::DType::I16:
	case dracolix::DType::I32:
	case dracolix::DType::I64:
		return true;
	default:
		return false;
	}
}

// Width in bits for integer dtypes (0 for non-integers; bool counts as 8).
int int_bits(dracolix::DType d) {
	switch (d) {
	case dracolix::DType::Bool:
		return 8;
	case dracolix::DType::I8:
	case dracolix::DType::U8:
		return 8;
	case dracolix::DType::I16:
	case dracolix::DType::U16:
		return 16;
	case dracolix::DType::I32:
	case dracolix::DType::U32:
		return 32;
	case dracolix::DType::I64:
	case dracolix::DType::U64:
		return 64;
	default:
		return 0;
	}
}

dracolix::DType signed_int_of_bits(int bits) {
	switch (bits) {
	case 8:
		return dracolix::DType::I8;
	case 16:
		return dracolix::DType::I16;
	case 32:
		return dracolix::DType::I32;
	case 64:
		return dracolix::DType::I64;
	default:
		return dracolix::DType::Void;
	}
}

// NumPy-compatible promotion for two non-complex dtypes.
dracolix::DType promote_real_pair(dracolix::DType a, dracolix::DType b) {
	if (a == b)
		return a;
	if (a == dracolix::DType::Bool)
		return b;
	if (b == dracolix::DType::Bool)
		return a;
	bool a_f = a == dracolix::DType::F32 || a == dracolix::DType::F64;
	bool b_f = b == dracolix::DType::F32 || b == dracolix::DType::F64;
	if (a_f || b_f) {
		if (a == dracolix::DType::F64 || b == dracolix::DType::F64)
			return dracolix::DType::F64;
		// One side is f32 and the other is an integer: f32 holds integers up
		// to 16 bits exactly; wider integers need f64.
		dracolix::DType id = a_f ? b : a;
		return int_bits(id) <= 16 ? dracolix::DType::F32 : dracolix::DType::F64;
	}
	bool au = is_unsigned_dtype(a), bu = is_unsigned_dtype(b);
	int aw = int_bits(a), bw = int_bits(b);
	if (au == bu)
		return aw >= bw ? a : b;
	// Mixed signed/unsigned. If the unsigned type is narrower, the signed
	// type of that width covers both ranges; otherwise widen to the next
	// signed type, falling back to f64 for uint64 vs any signed type.
	int uw = au ? aw : bw;
	int sw = au ? bw : aw;
	if (uw < sw)
		return signed_int_of_bits(sw);
	if (sw < 64) {
		dracolix::DType widened = signed_int_of_bits(std::max(uw, sw) * 2);
		if (widened != dracolix::DType::Void)
			return widened;
	}
	return dracolix::DType::F64;
}

dracolix::DType promote_pair(dracolix::DType a, dracolix::DType b) {
	if (a == b)
		return a;
	bool ac = is_complex_dtype(a), bc = is_complex_dtype(b);
	if (ac || bc) {
		if (ac && bc)
			return (a == dracolix::DType::C128 || b == dracolix::DType::C128)
					   ? dracolix::DType::C128
					   : dracolix::DType::C64;
		dracolix::DType c = ac ? a : b;
		dracolix::DType other = ac ? b : a;
		dracolix::DType component = c == dracolix::DType::C128
										? dracolix::DType::F64
										: dracolix::DType::F32;
		dracolix::DType r = promote_real_pair(component, other);
		return r == dracolix::DType::F32 ? dracolix::DType::C64
										 : dracolix::DType::C128;
	}
	return promote_real_pair(a, b);
}

// ---------------------------------------------------------------------------
// dtype casting (core Array<T>::astype)
// ---------------------------------------------------------------------------

template <typename T> DlxArray cast_cpp(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) { return DlxArray(arr.template astype<T>()); },
		a.v);
}

DlxArray cast_dtype(const DlxArray &a, dracolix::DType dt) {
	switch (dt) {
	case dracolix::DType::F32:
		return cast_cpp<float>(a);
	case dracolix::DType::F64:
		return cast_cpp<double>(a);
	case dracolix::DType::I8:
		return cast_cpp<int8_t>(a);
	case dracolix::DType::I16:
		return cast_cpp<int16_t>(a);
	case dracolix::DType::I32:
		return cast_cpp<int32_t>(a);
	case dracolix::DType::I64:
		return cast_cpp<int64_t>(a);
	case dracolix::DType::U8:
		return cast_cpp<uint8_t>(a);
	case dracolix::DType::U16:
		return cast_cpp<uint16_t>(a);
	case dracolix::DType::U32:
		return cast_cpp<uint32_t>(a);
	case dracolix::DType::U64:
		return cast_cpp<uint64_t>(a);
	case dracolix::DType::C64:
		return cast_cpp<std::complex<float>>(a);
	case dracolix::DType::C128:
		return cast_cpp<std::complex<double>>(a);
	case dracolix::DType::Bool:
		return cast_cpp<bool>(a);
	default:
		throw nb::type_error("unsupported dtype");
	}
}

// Real/complex operand mixing is promoted to the complex dtype, matching
// NumPy (``complex_mat @ real_mat`` works). Mismatches between two real
// dtypes stay an error, preserving the existing strict semantics.
void promote_complex_mismatch(std::optional<DlxArray> &a,
							  std::optional<DlxArray> &b, const char *op) {
	if (a->dtype() == b->dtype())
		return;
	if (!is_complex_dtype(a->dtype()) && !is_complex_dtype(b->dtype()))
		throw nb::type_error(
			(std::string(op) + " requires matching dtypes").c_str());
	dracolix::DType target = promote_pair(a->dtype(), b->dtype());
	if (a->dtype() != target)
		a.emplace(cast_dtype(*a, target));
	if (b->dtype() != target)
		b.emplace(cast_dtype(*b, target));
}

// ---------------------------------------------------------------------------
// Broadcast helpers (same semantics as core Array<T>::elementwise)
// ---------------------------------------------------------------------------

enum class BinOp {
	Add,
	Sub,
	Mul,
	Div,
	FloorDiv,
	Mod,
	Pow,
	And,
	Or,
	Xor,
	LShift,
	RShift,
	Eq,
	Ne,
	Lt,
	Le,
	Gt,
	Ge
};

std::vector<size_t> bcast_shape(const std::vector<size_t> &a,
								const std::vector<size_t> &b) {
	size_t na = a.size(), nb = b.size();
	size_t n = std::max(na, nb);
	std::vector<size_t> res(n);
	for (int i = (int)n - 1, ia = (int)na - 1, ib = (int)nb - 1; i >= 0;
		 --i, --ia, --ib) {
		size_t da = ia >= 0 ? a[ia] : 1;
		size_t db = ib >= 0 ? b[ib] : 1;
		if (da != db && da != 1 && db != 1)
			throw std::invalid_argument("broadcast: incompatible shapes");
		res[i] = std::max(da, db);
	}
	return res;
}

size_t bcast_offset(const std::vector<size_t> &shape,
					const std::vector<size_t> &strides,
					const std::vector<size_t> &ridx, size_t rndim) {
	size_t orig = shape.size();
	size_t off = 0;
	for (size_t i = 0; i < rndim; ++i) {
		int orig_d = (int)i - (int)(rndim - orig);
		if (orig_d < 0)
			continue;
		if (shape[orig_d] == 1)
			continue;
		off += ridx[i] * strides[orig_d];
	}
	return off;
}

// Ordering comparisons are implemented below (after the BinOp enum below is
// declared); forward-declared here so cmp_bcast can reference it.
enum class BinOp : int;

template <typename R, BinOp op>
dracolix::Array<bool> cmp_bcast(const dracolix::Array<R> &a,
								const dracolix::Array<R> &b) {
	if constexpr (dracolix::is_complex_v<R> && op != BinOp::Eq &&
				  op != BinOp::Ne) {
		throw nb::type_error(
			"ordering comparisons are not supported for complex dtypes");
	} else {
		auto comp = [](const R &x, const R &y) -> bool {
			if constexpr (dracolix::is_complex_v<R>) {
				if constexpr (op == BinOp::Eq)
					return x == y;
				else
					return x != y;
			} else {
				if constexpr (op == BinOp::Eq)
					return x == y;
				else if constexpr (op == BinOp::Ne)
					return x != y;
				else if constexpr (op == BinOp::Lt)
					return x < y;
				else if constexpr (op == BinOp::Le)
					return x <= y;
				else if constexpr (op == BinOp::Gt)
					return x > y;
				else
					return x >= y;
			}
		};

		if (a.shape() == b.shape()) {
			dracolix::Array<bool> out(a.shape());
			for (size_t i = 0; i < a.size(); ++i)
				out[i] = comp(a[i], b[i]);
			return out;
		}

		auto bs = bcast_shape(a.shape(), b.shape());
		dracolix::Array<bool> out(bs);
		size_t bnd = bs.size();
		std::vector<size_t> idx(bnd);

		for (size_t flat = 0; flat < out.size(); ++flat) {
			size_t rem = flat;
			for (int d = (int)bnd - 1; d >= 0; --d) {
				idx[d] = rem % bs[d];
				rem /= bs[d];
			}

			size_t off_a = bcast_offset(a.shape(), a.strides(), idx, bnd);
			size_t off_b = bcast_offset(b.shape(), b.strides(), idx, bnd);
			out[flat] = comp(a[off_a], b[off_b]);
		}
		return out;
	}
}

// ---------------------------------------------------------------------------
// Binary operations (promotion + broadcasting; no bool arithmetic)
// ---------------------------------------------------------------------------

// Element-wise power. Integer arrays keep their integer dtype (NumPy
// rejects negative integer exponents), while floating/complex arrays use
// std::pow.
template <typename R> R pow_scalar(R base, R exponent) {
	if constexpr (std::is_integral_v<R>) {
		if constexpr (std::is_signed_v<R>) {
			if (static_cast<int64_t>(exponent) < 0)
				throw nb::value_error(
					"integers to negative integer powers are not allowed");
		}
		R result = R(1);
		R factor = base;
		uint64_t n = static_cast<uint64_t>(exponent);
		while (n != 0) {
			if ((n & 1u) != 0)
				result *= factor;
			n >>= 1;
			if (n != 0)
				factor *= factor;
		}
		return result;
	} else {
		return static_cast<R>(std::pow(base, exponent));
	}
}

[[noreturn]] void throw_zero_division(const char *msg) {
	PyErr_SetString(PyExc_ZeroDivisionError, msg);
	throw nb::python_error();
}

// Python/NumPy floor division: quotient floored towards -inf.
template <typename R> R floor_div_scalar(R a, R b) {
	if constexpr (std::is_integral_v<R>) {
		if (b == R(0))
			throw_zero_division("integer division or modulo by zero");
		R q = static_cast<R>(a / b);
		R r = static_cast<R>(a % b);
		if constexpr (std::is_signed_v<R>) {
			if (r != R(0) && ((r < R(0)) != (b < R(0))))
				--q;
		}
		return q;
	} else {
		return static_cast<R>(std::floor(a / b));
	}
}

// Python/NumPy remainder: the result has the divisor's sign.
template <typename R> R mod_scalar(R a, R b) {
	if constexpr (std::is_integral_v<R>) {
		if (b == R(0))
			throw_zero_division("integer division or modulo by zero");
		R r = static_cast<R>(a % b);
		if constexpr (std::is_signed_v<R>) {
			if (r != R(0) && ((r < R(0)) != (b < R(0))))
				r = static_cast<R>(r + b);
		}
		return r;
	} else {
		R r = static_cast<R>(std::fmod(a, b));
		if (r != R(0) && ((r < R(0)) != (b < R(0))))
			r = static_cast<R>(r + b);
		else if (r == R(0))
			r = std::copysign(R(0), b);
		return r;
	}
}

// Shift with explicit range checks (avoids C++ UB for bad counts).
template <typename R> R shift_scalar(R a, R b, bool left) {
	if constexpr (std::is_signed_v<R>) {
		if (b < R(0))
			throw nb::value_error("negative shift count");
	}
	uint64_t n = static_cast<uint64_t>(b);
	if (n >= sizeof(R) * 8)
		throw nb::value_error("shift count out of range");
	using U = std::make_unsigned_t<R>;
	if (left)
		return static_cast<R>(static_cast<U>(a) << n);
	return static_cast<R>(a >> n);
}

// Elementwise bind with broadcasting.
template <typename R, typename F>
dracolix::Array<R> zip_bcast(const dracolix::Array<R> &a,
							 const dracolix::Array<R> &b, F fn) {
	if (a.shape() == b.shape()) {
		dracolix::Array<R> out(a.shape());
		for (size_t i = 0; i < a.size(); ++i)
			out[i] = fn(a[i], b[i]);
		return out;
	}
	auto bs = bcast_shape(a.shape(), b.shape());
	dracolix::Array<R> out(bs);
	size_t bnd = bs.size();
	std::vector<size_t> idx(bnd);
	for (size_t flat = 0; flat < out.size(); ++flat) {
		size_t rem = flat;
		for (int d = (int)bnd - 1; d >= 0; --d) {
			idx[d] = rem % bs[d];
			rem /= bs[d];
		}
		size_t off_a = bcast_offset(a.shape(), a.strides(), idx, bnd);
		size_t off_b = bcast_offset(b.shape(), b.strides(), idx, bnd);
		out[flat] = fn(a[off_a], b[off_b]);
	}
	return out;
}

template <typename R>
DlxArray bin_typed(const DlxArray &A, const DlxArray &B, BinOp op) {
	const dracolix::Array<R> &a = *A.as<R>();
	const dracolix::Array<R> &b = *B.as<R>();
	switch (op) {
	case BinOp::Add:
		return DlxArray(a + b);
	case BinOp::Sub:
		return DlxArray(a - b);
	case BinOp::Mul:
		return DlxArray(a * b);
	case BinOp::Div:
		return DlxArray(a / b);
	case BinOp::Pow:
		return DlxArray(
			zip_bcast(a, b, [](R x, R y) { return pow_scalar(x, y); }));
	case BinOp::FloorDiv:
		if constexpr (dracolix::is_complex_v<R>)
			throw nb::type_error(
				"floor division is not supported for complex dtypes");
		else
			return DlxArray(zip_bcast(
				a, b, [](R x, R y) { return floor_div_scalar(x, y); }));
	case BinOp::Mod:
		if constexpr (dracolix::is_complex_v<R>)
			throw nb::type_error("modulo is not supported for complex dtypes");
		else
			return DlxArray(
				zip_bcast(a, b, [](R x, R y) { return mod_scalar(x, y); }));
	case BinOp::And:
		if constexpr (std::is_integral_v<R>)
			return DlxArray(zip_bcast(
				a, b, [](R x, R y) -> R { return static_cast<R>(x & y); }));
		else
			throw nb::type_error(
				"bitwise operators require integer or bool arrays");
	case BinOp::Or:
		if constexpr (std::is_integral_v<R>)
			return DlxArray(zip_bcast(
				a, b, [](R x, R y) -> R { return static_cast<R>(x | y); }));
		else
			throw nb::type_error(
				"bitwise operators require integer or bool arrays");
	case BinOp::Xor:
		if constexpr (std::is_integral_v<R>)
			return DlxArray(zip_bcast(
				a, b, [](R x, R y) -> R { return static_cast<R>(x ^ y); }));
		else
			throw nb::type_error(
				"bitwise operators require integer or bool arrays");
	case BinOp::LShift:
	case BinOp::RShift: {
		if constexpr (std::is_integral_v<R> && !std::is_same_v<R, bool>) {
			return DlxArray(zip_bcast(a, b, [op](R x, R y) -> R {
				return shift_scalar(x, y, op == BinOp::LShift);
			}));
		} else {
			throw nb::type_error("shift operators require integer arrays");
		}
	}
	case BinOp::Eq:
		return DlxArray(cmp_bcast<R, BinOp::Eq>(a, b));
	case BinOp::Ne:
		return DlxArray(cmp_bcast<R, BinOp::Ne>(a, b));
	case BinOp::Lt:
		if constexpr (dracolix::is_complex_v<R>)
			throw nb::type_error("ordering comparisons are not supported "
								 "for complex dtypes");
		else
			return DlxArray(cmp_bcast<R, BinOp::Lt>(a, b));
	case BinOp::Le:
		if constexpr (dracolix::is_complex_v<R>)
			throw nb::type_error("ordering comparisons are not supported "
								 "for complex dtypes");
		else
			return DlxArray(cmp_bcast<R, BinOp::Le>(a, b));
	case BinOp::Gt:
		if constexpr (dracolix::is_complex_v<R>)
			throw nb::type_error("ordering comparisons are not supported "
								 "for complex dtypes");
		else
			return DlxArray(cmp_bcast<R, BinOp::Gt>(a, b));
	case BinOp::Ge:
		if constexpr (dracolix::is_complex_v<R>)
			throw nb::type_error("ordering comparisons are not supported "
								 "for complex dtypes");
		else
			return DlxArray(cmp_bcast<R, BinOp::Ge>(a, b));
	}
	throw std::logic_error("unreachable");
}

DlxArray bin_switch(dracolix::DType r, const DlxArray &A, const DlxArray &B,
					BinOp op) {
	switch (r) {
	case dracolix::DType::F32:
		return bin_typed<float>(A, B, op);
	case dracolix::DType::F64:
		return bin_typed<double>(A, B, op);
	case dracolix::DType::I8:
		return bin_typed<int8_t>(A, B, op);
	case dracolix::DType::I16:
		return bin_typed<int16_t>(A, B, op);
	case dracolix::DType::I32:
		return bin_typed<int32_t>(A, B, op);
	case dracolix::DType::I64:
		return bin_typed<int64_t>(A, B, op);
	case dracolix::DType::U8:
		return bin_typed<uint8_t>(A, B, op);
	case dracolix::DType::U16:
		return bin_typed<uint16_t>(A, B, op);
	case dracolix::DType::U32:
		return bin_typed<uint32_t>(A, B, op);
	case dracolix::DType::U64:
		return bin_typed<uint64_t>(A, B, op);
	case dracolix::DType::Bool:
		return bin_typed<bool>(A, B, op);
	case dracolix::DType::C64:
		return bin_typed<std::complex<float>>(A, B, op);
	case dracolix::DType::C128:
		return bin_typed<std::complex<double>>(A, B, op);
	default:
		throw nb::type_error("unsupported dtype");
	}
}

DlxArray dlx_array(nb::object data, nb::object dtype);

bool coerce_operand(nb::handle h, std::optional<DlxArray> &out) {
	if (is_array(h)) {
		out = *nb::cast<DlxArray *>(h);
		return true;
	}
	if (is_number(h)) {
		out = scalar_slot(h);
		return true;
	}
	// Sequences (list/tuple) and buffers are coerced through array(), so
	// ``arr < [1, 2, 3]`` and ``arr + [1, 2, 3]`` behave like NumPy.
	if (PyList_Check(h.ptr()) || PyTuple_Check(h.ptr()) ||
		PyObject_CheckBuffer(h.ptr())) {
		out = dlx_array(nb::borrow<nb::object>(h), nb::none());
		return true;
	}
	return false;
}

nb::object dlx_bin_op(nb::object x, nb::object y, BinOp op) {
	std::optional<DlxArray> A, B;
	if (!coerce_operand(x, A) || !coerce_operand(y, B))
		return nb::not_implemented();
	assert(A && B);

	dracolix::DType ra = A->dtype(), rb = B->dtype();
	dracolix::DType r = promote_pair(ra, rb);
	bool bitwise = op == BinOp::And || op == BinOp::Or || op == BinOp::Xor;
	bool shift = op == BinOp::LShift || op == BinOp::RShift;
	if (bitwise || shift) {
		auto bad = [](dracolix::DType d) {
			return d == dracolix::DType::F32 || d == dracolix::DType::F64 ||
				   is_complex_dtype(d);
		};
		if (bad(ra) || bad(rb))
			throw nb::type_error(
				"bitwise operators require integer or bool arrays");
	}
	if (shift && (ra == dracolix::DType::Bool || rb == dracolix::DType::Bool))
		return nb::not_implemented();
	if (r == dracolix::DType::Bool) {
		// Bool never participates in arithmetic (NumPy's bool promotions are
		// surprising); comparisons and bitwise &/|/^ remain well defined.
		if (op == BinOp::Add || op == BinOp::Sub || op == BinOp::Mul ||
			op == BinOp::Div || op == BinOp::FloorDiv || op == BinOp::Mod ||
			op == BinOp::Pow || shift)
			return nb::not_implemented();
	}
	if (op == BinOp::Div && r != dracolix::DType::C64 &&
		r != dracolix::DType::C128) {
		// __truediv__: NumPy-like semantics (int/int -> f64, f32/f32 ->
		// f32). Complex division keeps its promoted complex dtype.
		r = (ra == dracolix::DType::F32 && rb == dracolix::DType::F32)
				? dracolix::DType::F32
				: dracolix::DType::F64;
	}
	DlxArray Ca = cast_dtype(*A, r);
	DlxArray Cb = cast_dtype(*B, r);
	return nb::cast<DlxArray>(bin_switch(r, Ca, Cb, op));
}

nb::object dlx_add(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::Add);
}
nb::object dlx_sub(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::Sub);
}
nb::object dlx_mul(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::Mul);
}
nb::object dlx_div(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::Div);
}
nb::object dlx_pow(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::Pow);
}
nb::object dlx_rpow(nb::object x, nb::object y) {
	// __rpow__(self, other) means ``other ** self``.
	return dlx_bin_op(y, x, BinOp::Pow);
}
nb::object dlx_floordiv(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::FloorDiv);
}
nb::object dlx_rfloordiv(nb::object x, nb::object y) {
	return dlx_bin_op(y, x, BinOp::FloorDiv);
}
nb::object dlx_mod(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::Mod);
}
nb::object dlx_rmod(nb::object x, nb::object y) {
	return dlx_bin_op(y, x, BinOp::Mod);
}
nb::object dlx_and(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::And);
}
nb::object dlx_or(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::Or);
}
nb::object dlx_xor(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::Xor);
}
nb::object dlx_lshift(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::LShift);
}
nb::object dlx_rlshift(nb::object x, nb::object y) {
	return dlx_bin_op(y, x, BinOp::LShift);
}
nb::object dlx_rshift(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::RShift);
}
nb::object dlx_rrshift(nb::object x, nb::object y) {
	return dlx_bin_op(y, x, BinOp::RShift);
}
nb::object dlx_eq(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::Eq);
}
nb::object dlx_ne(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::Ne);
}
nb::object dlx_lt(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::Lt);
}
nb::object dlx_le(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::Le);
}
nb::object dlx_gt(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::Gt);
}
nb::object dlx_ge(nb::object x, nb::object y) {
	return dlx_bin_op(x, y, BinOp::Ge);
}

nb::object dlx_neg(nb::object x) {
	if (!is_array(x))
		return nb::not_implemented();
	DlxArray &a = as_array(x);
	return std::visit(
		[](auto &arr) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				throw nb::type_error(
					"unary negation is not supported for bool arrays");
			} else {
				return nb::cast<DlxArray>(DlxArray(arr * T(-1)));
			}
		},
		a.v);
}

DlxArray dlx_pos(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> DlxArray {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				throw nb::type_error(
					"unary plus is not supported for bool arrays");
			} else {
				return DlxArray(arr.clone());
			}
		},
		a.v);
}

DlxArray dlx_invert(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> DlxArray {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				dracolix::Array<bool> out(arr.shape());
				for (size_t i = 0; i < arr.size(); ++i)
					out[i] = !static_cast<bool>(arr[i]);
				return DlxArray(std::move(out));
			} else if constexpr (std::is_integral_v<T>) {
				dracolix::Array<T> out(arr.shape());
				for (size_t i = 0; i < arr.size(); ++i)
					out[i] = static_cast<T>(~arr[i]);
				return DlxArray(std::move(out));
			} else {
				throw nb::type_error(
					"bitwise NOT requires an integer or bool array");
			}
		},
		a.v);
}

// ---------------------------------------------------------------------------
// matmul (core dracolix::linalg::matmul; 1-D @ 1-D -> scalar dot)
// ---------------------------------------------------------------------------

nb::object dlx_matmul(nb::object x, nb::object y) {
	std::optional<DlxArray> A, B;
	if (!coerce_operand(x, A) || !coerce_operand(y, B))
		return nb::not_implemented();
	assert(A && B);
	dracolix::DType dt = A->dtype();
	promote_complex_mismatch(A, B, "matmul");
	if (dt == dracolix::DType::Bool)
		throw nb::type_error("matmul does not support bool arrays");

	if (A->ndim() == 1 && B->ndim() == 1)
		return std::visit(
			[&](const auto &a) -> nb::object {
				using T =
					typename array_scalar<std::decay_t<decltype(a)>>::type;
				const auto &b = *B->as<T>();
				return scalar_to_py<T>(dracolix::linalg::dot(a, b));
			},
			A->v);

	return std::visit(
		[&](const auto &a) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(a)>>::type;
			const auto &b = *B->as<T>();
			return nb::cast<DlxArray>(DlxArray(dracolix::linalg::matmul(a, b)));
		},
		A->v);
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

struct KeySpec {
	bool is_int = false;
	int64_t idx = 0;
	dracolix::Slice slice;
};

// Parse a Python index (int, slice, or tuple of int/slice) into KeySpecs.
// PySlice_Unpack marks an omitted component with PY_SSIZE_T_MAX (start)
// or PY_SSIZE_T_MIN (stop); step is left as given.
void parse_slice_h(nb::handle obj, KeySpec &ks) {
	Py_ssize_t start, stop, step;
	if (PySlice_Unpack(obj.ptr(), &start, &stop, &step) != 0)
		throw nb::python_error();
	if (step == 0)
		throw nb::value_error("slice step cannot be 0");
	ks.is_int = false;
	ks.slice.step = step;
	if (start != PY_SSIZE_T_MAX)
		ks.slice.start = (int64_t)start;
	if (stop != PY_SSIZE_T_MIN)
		ks.slice.stop = (int64_t)stop;
}

std::vector<KeySpec> parse_key(nb::object key, size_t ndim) {
	std::vector<KeySpec> out;
	auto parse_one = [&](nb::handle h, KeySpec &ks) {
		if (PySlice_Check(h.ptr())) {
			parse_slice_h(h, ks);
		} else if (PyLong_Check(h.ptr())) {
			ks.is_int = true;
			ks.idx = nb::cast<int64_t>(h);
		} else {
			throw nb::type_error("indices must be integers or slices");
		}
	};

	if (PyTuple_Check(key.ptr())) {
		Py_ssize_t n = PyTuple_GET_SIZE(key.ptr());
		for (Py_ssize_t i = 0; i < n; ++i) {
			KeySpec ks;
			parse_one(PyTuple_GET_ITEM(key.ptr(), i), ks);
			out.push_back(ks);
		}
	} else {
		KeySpec ks;
		parse_one(key, ks);
		out.push_back(ks);
	}

	if (out.size() > ndim)
		throw nb::index_error("too many indices");
	while (out.size() < ndim)
		out.push_back(KeySpec{}); // slice(None)
	return out;
}

nb::object dlx_scalar_index(const DlxArray &a,
							const std::vector<int64_t> &idx) {
	const std::vector<size_t> &shape = a.shape();
	const std::vector<size_t> &strides = a.strides();
	size_t off = 0;
	for (size_t d = 0; d < shape.size(); ++d) {
		int64_t i = idx[d];
		if (i < 0)
			i += (int64_t)shape[d];
		if (i < 0 || (size_t)i >= shape[d])
			throw nb::index_error("index out of bounds");
		off += (size_t)i * strides[d];
	}
	return py_scalar_at(a, (int64_t)off);
}

// Materialize a (possibly mixed int/slice) index into a new contiguous
// array.
template <typename T>
dracolix::Array<T> slice_materialize(const dracolix::Array<T> &arr,
									 const std::vector<KeySpec> &keys) {
	size_t nd = arr.ndim();
	std::vector<int64_t> fixed_off(nd, 0);
	std::vector<int64_t> starts(nd, 0), steps(nd, 1);
	std::vector<int64_t> lens(nd, 0);
	std::vector<size_t> out_shape;

	for (size_t d = 0; d < nd; ++d) {
		const KeySpec &ks = keys[d];
		if (ks.is_int) {
			int64_t i = ks.idx;
			if (i < 0)
				i += (int64_t)arr.shape()[d];
			if (i < 0 || (size_t)i >= arr.shape()[d])
				throw nb::index_error("index out of bounds");
			fixed_off[d] = i * (int64_t)arr.strides()[d];
			lens[d] = 0;
			continue;
		}
		size_t off0, len;
		int64_t step;
		dracolix::normalize_slice(ks.slice, arr.shape()[d], off0, len, step);
		if (len == 0)
			throw nb::value_error("zero-length slices are not supported yet");
		starts[d] = (int64_t)off0;
		steps[d] = step;
		lens[d] = (int64_t)len;
		out_shape.push_back(len);
	}

	dracolix::Array<T> out(out_shape);
	std::vector<size_t> idx(out_shape.size());
	for (size_t flat = 0; flat < out.size(); ++flat) {
		size_t rem = flat;
		for (int dd = (int)out_shape.size() - 1; dd >= 0; --dd) {
			idx[dd] = rem % out_shape[dd];
			rem /= out_shape[dd];
		}
		int64_t src = 0;
		size_t od = 0;
		for (size_t d = 0; d < nd; ++d) {
			if (lens[d] == 0) {
				src += fixed_off[d];
			} else {
				int64_t src_d = starts[d] + (int64_t)idx[od] * steps[d];
				src += src_d * (int64_t)arr.strides()[d];
				++od;
			}
		}
		out[flat] = arr[(size_t)src];
	}
	return out;
}

nb::object dlx_getitem(const DlxArray &self, nb::object key) {
	std::vector<KeySpec> keys = parse_key(key, self.ndim());

	bool all_int = true;
	for (auto &k : keys)
		if (!k.is_int) {
			all_int = false;
			break;
		}
	if (all_int) {
		std::vector<int64_t> idx;
		idx.reserve(keys.size());
		for (auto &k : keys)
			idx.push_back(k.idx);
		return dlx_scalar_index(self, idx);
	}

	return std::visit(
		[&](const auto &arr) -> nb::object {
			DlxArray out(slice_materialize(arr, keys));
			return nb::cast<DlxArray>(std::move(out));
		},
		self.v);
}

nb::object dlx_setitem(DlxArray &self, nb::object key, nb::object val) {
	std::vector<KeySpec> keys = parse_key(key, self.ndim());
	bool all_int = true;
	for (auto &ks : keys)
		if (!ks.is_int) {
			all_int = false;
			break;
		}
	if (!all_int)
		throw nb::type_error("slice assignment is not supported yet");
	if (keys.size() != self.ndim())
		throw nb::index_error("index rank mismatch");
	if (!is_number(val))
		throw nb::type_error("assignment value must be a scalar");

	size_t off = 0;
	for (size_t d = 0; d < self.ndim(); ++d) {
		int64_t i = keys[d].idx;
		if (i < 0)
			i += (int64_t)self.shape()[d];
		if (i < 0 || (size_t)i >= self.shape()[d])
			throw nb::index_error("index out of bounds");
		off += (size_t)i * self.strides()[d];
	}
	std::visit(
		[&](auto &arr) {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			T v;
			if constexpr (std::is_same_v<T, bool>) {
				v = nb::cast<bool>(val);
			} else if constexpr (dracolix::is_complex_v<T>) {
				v = static_cast<T>(to_complex(val));
			} else if constexpr (std::is_integral_v<T>) {
				v = integer_from_py<T>(val);
			} else {
				if (num_kind(val) == NumKind::Complex)
					throw nb::type_error(
						"cannot assign complex to a real array; "
						"use a complex dtype");
				v = (T)to_double(val);
			}
			arr[(size_t)off] = v;
		},
		self.v);
	return nb::none();
}

// ---------------------------------------------------------------------------
// Buffer protocol export (PEP 3118).
// nanobind does not ship def_buffer(); we wire bf_getbuffer /
// bf_releasebuffer on the registered type ourselves.
// ---------------------------------------------------------------------------

const char *buffer_format(dracolix::DType dt) {
	switch (dt) {
	case dracolix::DType::F32:
		return "f";
	case dracolix::DType::F64:
		return "d";
	case dracolix::DType::I8:
		return "b";
	case dracolix::DType::I16:
		return "h";
	case dracolix::DType::I32:
		return "i";
	case dracolix::DType::I64:
		return "q";
	case dracolix::DType::U8:
		return "B";
	case dracolix::DType::U16:
		return "H";
	case dracolix::DType::U32:
		return "I";
	case dracolix::DType::U64:
		return "Q";
	case dracolix::DType::C64:
		return "Zf"; // PEP 3118 complex64 (two floats)
	case dracolix::DType::C128:
		return "Zd"; // PEP 3118 complex128 (two doubles)
	case dracolix::DType::Bool:
		return "?";
	default:
		return "";
	}
}

int dlx_getbuffer(PyObject *obj, Py_buffer *view, int flags) {
	if (view == nullptr) {
		PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
		return -1;
	}
	if (!g_dlx_type || !PyObject_TypeCheck(obj, g_dlx_type)) {
		PyErr_SetString(PyExc_TypeError, "not a dracolix Array");
		return -1;
	}
	DlxArray *self = nb::cast<DlxArray *>(nb::handle(obj));
	const std::vector<size_t> &shape = self->shape();
	const std::vector<size_t> &strides = self->strides();
	size_t ndim = self->ndim();

	bool is_rowmajor = true;
	bool is_colmajor = true;
	if (ndim > 0) {
		{
			size_t expect = 1;
			for (int i = (int)ndim - 1; i >= 0; --i) {
				if (strides[i] != expect) {
					is_rowmajor = false;
					break;
				}
				expect *= shape[i];
			}
		}
		{
			size_t expect = 1;
			for (size_t i = 0; i < ndim; ++i) {
				if (strides[i] != expect) {
					is_colmajor = false;
					break;
				}
				expect *= shape[i];
			}
		}
	}

	// Contiguity checks must compare the full flag mask:
	// PyBUF_C/F_CONTIGUOUS each include the ND+STRIDES bits, which plain
	// requests always carry.
	if ((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS && !is_rowmajor) {
		PyErr_Format(PyExc_BufferError, "dracolix array is not C-contiguous");
		return -1;
	}
	if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS && !is_colmajor) {
		PyErr_Format(PyExc_BufferError,
					 "dracolix array is not Fortran-contiguous");
		return -1;
	}
	// PyBUF_INDIRECT relaxes the format requirements but never suboffsets;
	// we are suboffset-capable exporter (we simply never produce
	// suboffsets).

	bool need_shape = (flags & PyBUF_ND) != 0;
	bool need_strides = (flags & PyBUF_STRIDES) != 0;

	Py_ssize_t itemsize = (Py_ssize_t)self->itemsize();

	Py_ssize_t *c_shape = nullptr;
	Py_ssize_t *c_strides = nullptr;
	if (need_shape) {
		c_shape = (Py_ssize_t *)PyMem_Malloc(sizeof(Py_ssize_t) *
											 std::max<size_t>(ndim, 1));
		if (!c_shape) {
			PyErr_NoMemory();
			return -1;
		}
		for (size_t d = 0; d < ndim; ++d)
			c_shape[d] = (Py_ssize_t)shape[d];
	}
	if (need_strides) {
		c_strides = (Py_ssize_t *)PyMem_Malloc(sizeof(Py_ssize_t) *
											   std::max<size_t>(ndim, 1));
		if (!c_strides) {
			PyMem_Free(c_shape);
			PyErr_NoMemory();
			return -1;
		}
		// PEP 3118 specifies byte strides, not element strides.
		for (size_t d = 0; d < ndim; ++d)
			c_strides[d] = (Py_ssize_t)strides[d] * itemsize;
	}

	// Simple requests with no ND/STRIDES and a 1-D array: hand out a flat
	// contiguous view (shape NULL is fine for simple consumers).
	const char *fmt = buffer_format(self->dtype());
	char *fmt_copy = (char *)PyMem_Malloc(strlen(fmt) + 1);
	if (!fmt_copy) {
		PyMem_Free(c_shape);
		PyMem_Free(c_strides);
		PyErr_NoMemory();
		return -1;
	}
	std::strcpy(fmt_copy, fmt);

	if (!need_shape && !need_strides && ndim == 1) {
		c_shape = (Py_ssize_t *)PyMem_Malloc(sizeof(Py_ssize_t));
		if (!c_shape) {
			PyErr_NoMemory();
			return -1;
		}
		c_shape[0] = (Py_ssize_t)shape[0];
		view->ndim = 1;
		view->shape = c_shape;
		view->strides = nullptr;
		view->suboffsets = nullptr;
		view->internal = nullptr;
		view->obj = obj;
		Py_INCREF(obj);
		view->buf = const_cast<void *>(self->data_ptr());
		view->len = (Py_ssize_t)self->size() * itemsize;
		view->readonly = 0;
		view->itemsize = itemsize;
		view->format = fmt_copy;
		return 0;
	}

	view->obj = obj;
	Py_INCREF(obj);
	view->buf = const_cast<void *>(self->data_ptr());
	view->len = (Py_ssize_t)self->size() * itemsize;
	view->readonly = 0;
	view->format = fmt_copy;
	view->itemsize = itemsize;
	view->ndim = need_shape ? (Py_ssize_t)ndim : 0;
	view->shape = c_shape;
	view->strides = c_strides;
	view->suboffsets = nullptr;
	view->internal = nullptr;
	return 0;
}

void dlx_releasebuffer(PyObject *, Py_buffer *view) {
	if (view->shape)
		PyMem_Free(view->shape);
	if (view->strides)
		PyMem_Free(view->strides);
	if (view->format)
		PyMem_Free(view->format);
	view->shape = nullptr;
	view->strides = nullptr;
	view->format = nullptr;
}

void install_buffer_slots() {
	if (!g_dlx_type)
		return;
	PyBufferProcs *bp = g_dlx_type->tp_as_buffer;
	if (!bp)
		return;
	bp->bf_getbuffer = &dlx_getbuffer;
	bp->bf_releasebuffer = &dlx_releasebuffer;
}

// ---------------------------------------------------------------------------
// Buffer input (numpy / memoryview -> DlxArray)
// ---------------------------------------------------------------------------

dracolix::DType infer_format_dtype(const char *fmt, Py_ssize_t itemsize) {
	if (fmt == nullptr || *fmt == 0) {
		if (itemsize == 1)
			return dracolix::DType::U8;
		return dracolix::DType::Void;
	}
	switch (*fmt) {
	case 'f':
		return dracolix::DType::F32;
	case 'd':
		return dracolix::DType::F64;
	case 'q':
		return dracolix::DType::I64;
	case 'Q':
		return dracolix::DType::U64;
	case 'L':
	case 'N':
		// unsigned long is 32-bit on Windows, unlike LP64 platforms.
		return itemsize >= 8 ? dracolix::DType::U64 : dracolix::DType::U32;
	case 'i':
	case 'l':
		return itemsize >= 8 ? dracolix::DType::I64 : dracolix::DType::I32;
	case 'n':
		return itemsize >= 8 ? dracolix::DType::I64 : dracolix::DType::I32;
	case 'b':
		return dracolix::DType::I8;
	case 'h':
		return dracolix::DType::I16;
	case 'B':
		return dracolix::DType::U8;
	case 'H':
		return dracolix::DType::U16;
	case 'I':
		return dracolix::DType::U32;
	case 'e':
		return dracolix::DType::F32;
	case 'Z':
		// PEP 3118 complex: 'Zf' (complex64) or 'Zd' (complex128).
		// Fall back to itemsize when the precision character is absent.
		if (fmt[1] == 'f' || (fmt[1] == 0 && itemsize == 8))
			return dracolix::DType::C64;
		return dracolix::DType::C128;
	case '?':
		return dracolix::DType::Bool;
	default:
		return dracolix::DType::Void;
	}
}

double read_double_at(const char *p, char tag, Py_ssize_t itemsize) {
	if (tag == 'Z') {
		// Complex source; a real destination takes the real part.
		if (itemsize == 8)
			return (double)(*(const float *)p);
		return *(const double *)p;
	}
	switch (tag) {
	case 'f':
		return (double)(*(const float *)p);
	case 'd':
		return *(const double *)p;
	case 'e': {
		uint16_t h;
		std::memcpy(&h, p, 2);
		uint32_t sign = (uint32_t)(h >> 15) & 1u;
		uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
		uint32_t mant = (uint32_t)h & 0x3FFu;
		double d;
		if (exp == 0)
			d = mant ? std::ldexp((double)mant, -24) : 0.0;
		else if (exp == 31)
			d = mant ? std::numeric_limits<double>::quiet_NaN()
					 : std::numeric_limits<double>::infinity();
		else
			d = std::ldexp((double)(mant + 1024), (int)exp - 25);
		return sign ? -d : d;
	}
	case '?':
		return *(const bool *)p ? 1.0 : 0.0;
	default: {
		bool is_unsigned = (tag == 'B' || tag == 'H' || tag == 'I' ||
							tag == 'L' || tag == 'N' || tag == 'Q');
		if (itemsize == 1)
			return is_unsigned ? (double)(*(const uint8_t *)p)
							   : (double)(*(const int8_t *)p);
		if (itemsize == 2)
			return is_unsigned ? (double)(*(const uint16_t *)p)
							   : (double)(*(const int16_t *)p);
		if (itemsize == 4)
			return is_unsigned ? (double)(*(const uint32_t *)p)
							   : (double)(*(const int32_t *)p);
		return is_unsigned ? (double)(*(const uint64_t *)p)
						   : (double)(*(const int64_t *)p);
	}
	}
}

int64_t read_int_at(const char *p, char tag, Py_ssize_t itemsize) {
	if (tag == 'Z')
		return (int64_t)read_double_at(p, tag, itemsize);
	bool is_unsigned = (tag == 'B' || tag == 'H' || tag == 'I' || tag == 'L' ||
						tag == 'N' || tag == 'Q');
	bool is_bool = tag == '?';
	if (is_bool)
		return *(const bool *)p ? 1 : 0;
	if (itemsize == 1)
		return is_unsigned ? (int64_t)(*(const uint8_t *)p)
						   : (int64_t)(*(const int8_t *)p);
	if (itemsize == 2)
		return is_unsigned ? (int64_t)(*(const uint16_t *)p)
						   : (int64_t)(*(const int16_t *)p);
	if (itemsize == 4)
		return is_unsigned ? (int64_t)(*(const uint32_t *)p)
						   : (int64_t)(*(const int32_t *)p);
	return is_unsigned ? (int64_t)(*(const uint64_t *)p)
					   : (*(const int64_t *)p);
}

uint64_t read_uint_at(const char *p, char tag, Py_ssize_t itemsize) {
	if (tag == 'Z') {
		double d = read_double_at(p, tag, itemsize);
		if (d <= 0.0)
			return 0;
		if (d >= 18446744073709551616.0)
			return std::numeric_limits<uint64_t>::max();
		return (uint64_t)d;
	}
	bool is_unsigned = (tag == 'B' || tag == 'H' || tag == 'I' || tag == 'L' ||
						tag == 'N' || tag == 'Q');
	if (tag == '?')
		return *(const bool *)p ? 1 : 0;
	if (tag == 'f' || tag == 'd' || tag == 'e') {
		double d = read_double_at(p, tag, itemsize);
		if (d <= 0.0)
			return 0;
		if (d >= 18446744073709551616.0)
			return std::numeric_limits<uint64_t>::max();
		return (uint64_t)d;
	}
	if (is_unsigned) {
		if (itemsize == 1)
			return (uint64_t)(*(const uint8_t *)p);
		if (itemsize == 2)
			return (uint64_t)(*(const uint16_t *)p);
		if (itemsize == 4)
			return (uint64_t)(*(const uint32_t *)p);
		return *(const uint64_t *)p;
	}
	return (uint64_t)read_int_at(p, tag, itemsize);
}

template <typename T> T from_raw(const char *p, char tag, Py_ssize_t itemsize) {
	if constexpr (dracolix::is_complex_v<T>) {
		using R = typename T::value_type;
		if (tag == 'Z') {
			Py_ssize_t comp = itemsize / 2;
			if (comp == 8) {
				const double *q = (const double *)p;
				return {static_cast<R>(q[0]), static_cast<R>(q[1])};
			}
			const float *q = (const float *)p;
			return {static_cast<R>(q[0]), static_cast<R>(q[1])};
		}
		return {static_cast<R>(read_double_at(p, tag, itemsize)), R(0)};
	} else if constexpr (std::is_same_v<T, bool>) {
		if (tag == '?')
			return *(const bool *)p;
		return read_double_at(p, tag, itemsize) != 0.0;
	} else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
		return (T)read_uint_at(p, tag, itemsize);
	} else if constexpr (std::is_integral_v<T>) {
		return (T)read_int_at(p, tag, itemsize);
	} else {
		return (T)read_double_at(p, tag, itemsize);
	}
}

template <typename T>
DlxArray buffer_to_array(const Py_buffer &view, char tag) {
	if (view.ndim == 0)
		throw nb::value_error("0-dimensional buffers are not supported yet");
	std::vector<size_t> shape((size_t)view.ndim);
	Py_ssize_t n = 1;
	for (Py_ssize_t d = 0; d < view.ndim; ++d) {
		shape[(size_t)d] = (size_t)view.shape[d];
		n *= view.shape[d];
	}
	for (auto s : shape)
		if (s == 0)
			throw nb::value_error("empty buffers are not supported yet");

	dracolix::Array<T> out(shape);
	if (n != (Py_ssize_t)out.size())
		throw std::logic_error("buffer size mismatch");

	Py_ssize_t itemsize = view.itemsize;
	const Py_ssize_t *st = view.strides;
	std::vector<Py_ssize_t> fallback_strides;
	if (st == nullptr) {
		fallback_strides.resize((size_t)view.ndim);
		for (Py_ssize_t d = view.ndim - 1, acc = 1; d >= 0; --d) {
			fallback_strides[(size_t)d] = acc;
			acc *= view.shape[d];
		}
		st = fallback_strides.data();
	}

	for (Py_ssize_t flat = 0; flat < n; ++flat) {
		size_t rem = (size_t)flat;
		size_t off = 0;
		for (Py_ssize_t d = view.ndim - 1; d >= 0; --d) {
			size_t idx = rem % (size_t)view.shape[d];
			rem /= (size_t)view.shape[d];
			off += idx * (size_t)st[d];
		}
		const char *p = (const char *)view.buf + off * itemsize;
		out[(size_t)flat] = from_raw<T>(p, tag, itemsize);
	}
	return DlxArray(std::move(out));
}

DlxArray from_buffer(nb::object o, std::optional<dracolix::DType> dt) {
	Py_buffer view;
	if (PyObject_GetBuffer(o.ptr(), &view, PyBUF_CONTIG_RO | PyBUF_FORMAT) != 0)
		throw nb::python_error();
	struct Guard {
		Py_buffer &v;
		~Guard() { PyBuffer_Release(&v); }
	} guard{view};

	dracolix::DType target =
		dt.value_or(infer_format_dtype(view.format, view.itemsize));
	if (target == dracolix::DType::Void)
		throw nb::type_error("unsupported buffer format");

	char tag = (view.format && *view.format) ? *view.format : 'B';
	switch (target) {
	case dracolix::DType::F32:
		return buffer_to_array<float>(view, tag);
	case dracolix::DType::F64:
		return buffer_to_array<double>(view, tag);
	case dracolix::DType::I8:
		return buffer_to_array<int8_t>(view, tag);
	case dracolix::DType::I16:
		return buffer_to_array<int16_t>(view, tag);
	case dracolix::DType::I32:
		return buffer_to_array<int32_t>(view, tag);
	case dracolix::DType::I64:
		return buffer_to_array<int64_t>(view, tag);
	case dracolix::DType::U8:
		return buffer_to_array<uint8_t>(view, tag);
	case dracolix::DType::U16:
		return buffer_to_array<uint16_t>(view, tag);
	case dracolix::DType::U32:
		return buffer_to_array<uint32_t>(view, tag);
	case dracolix::DType::U64:
		return buffer_to_array<uint64_t>(view, tag);
	case dracolix::DType::C64:
		return buffer_to_array<std::complex<float>>(view, tag);
	case dracolix::DType::C128:
		return buffer_to_array<std::complex<double>>(view, tag);
	case dracolix::DType::Bool:
		return buffer_to_array<bool>(view, tag);
	default:
		throw nb::type_error("unsupported dtype");
	}
}

// ---------------------------------------------------------------------------
// list / tuple input
// ---------------------------------------------------------------------------

void walk_sequence(nb::object o, size_t depth, std::vector<size_t> &shape,
				   std::vector<nb::object> &leaves, bool &any_float,
				   bool &any_int, bool &any_bool, bool &any_complex) {
	if (!PyList_Check(o.ptr()) && !PyTuple_Check(o.ptr()))
		throw nb::type_error("expected list or tuple of numbers");

	Py_ssize_t len = PySequence_Size(o.ptr());
	if (depth >= shape.size()) {
		shape.push_back((size_t)len);
	} else if ((size_t)len != shape[depth]) {
		throw nb::value_error("ragged nested sequence");
	}
	for (Py_ssize_t i = 0; i < len; ++i) {
		nb::object item = nb::steal(PySequence_GetItem(o.ptr(), i));
		if (PyList_Check(item.ptr()) || PyTuple_Check(item.ptr())) {
			walk_sequence(item, depth + 1, shape, leaves, any_float, any_int,
						  any_bool, any_complex);
		} else if (is_number(item)) {
			leaves.push_back(item);
			switch (num_kind(item)) {
			case NumKind::Float:
				any_float = true;
				break;
			case NumKind::Int:
				any_int = true;
				break;
			case NumKind::Complex:
				any_complex = true;
				break;
			default:
				any_bool = true;
				break;
			}
		} else {
			throw nb::type_error(
				"dracolix arrays only support bool, int, float and "
				"complex elements");
		}
	}
}

dracolix::DType infer_list_dtype(bool any_float, bool any_int, bool any_bool,
								 bool any_complex) {
	if (any_complex)
		return dracolix::DType::C128; // Python complex is double precision
	if (any_float)
		return dracolix::DType::F64;
	if (any_int)
		return dracolix::DType::I64;
	(void)any_bool;
	return dracolix::DType::Bool;
}

template <typename T>
void fill_leaf(dracolix::Array<T> &out, size_t i, nb::object leaf) {
	if constexpr (std::is_same_v<T, bool>) {
		if (num_kind(leaf) == NumKind::Complex)
			throw nb::type_error("cannot convert complex to bool dtype");
		out[i] = num_kind(leaf) == NumKind::Bool ? nb::cast<bool>(leaf)
												 : to_double(leaf) != 0.0;
	} else if constexpr (dracolix::is_complex_v<T>) {
		out[i] = static_cast<T>(to_complex(leaf));
	} else if constexpr (std::is_integral_v<T>) {
		out[i] = integer_from_py<T>(leaf);
	} else {
		// Like NumPy's list conversion, a complex leaf cannot be stored
		// silently in a real dtype (use astype() to drop the imaginary
		// part).
		if (num_kind(leaf) == NumKind::Complex)
			throw nb::type_error("cannot convert complex to float dtype");
		out[i] = (T)to_double(leaf);
	}
}

DlxArray from_list(nb::object o, std::optional<dracolix::DType> dt) {
	std::vector<size_t> shape;
	std::vector<nb::object> leaves;
	bool any_float = false, any_int = false, any_bool = false;
	bool any_complex = false;
	walk_sequence(o, 0, shape, leaves, any_float, any_int, any_bool,
				  any_complex);
	dracolix::DType target = dt.value_or(
		infer_list_dtype(any_float, any_int, any_bool, any_complex));
	if (shape.empty() || leaves.empty())
		throw nb::value_error("empty arrays are not supported yet");

	switch (target) {
	case dracolix::DType::F32: {
		dracolix::Array<float> out(shape);
		for (size_t i = 0; i < leaves.size(); ++i)
			fill_leaf<float>(out, i, leaves[i]);
		return DlxArray(std::move(out));
	}
	case dracolix::DType::F64: {
		dracolix::Array<double> out(shape);
		for (size_t i = 0; i < leaves.size(); ++i)
			fill_leaf<double>(out, i, leaves[i]);
		return DlxArray(std::move(out));
	}
	case dracolix::DType::I8: {
		dracolix::Array<int8_t> out(shape);
		for (size_t i = 0; i < leaves.size(); ++i)
			fill_leaf<int8_t>(out, i, leaves[i]);
		return DlxArray(std::move(out));
	}
	case dracolix::DType::I16: {
		dracolix::Array<int16_t> out(shape);
		for (size_t i = 0; i < leaves.size(); ++i)
			fill_leaf<int16_t>(out, i, leaves[i]);
		return DlxArray(std::move(out));
	}
	case dracolix::DType::I32: {
		dracolix::Array<int32_t> out(shape);
		for (size_t i = 0; i < leaves.size(); ++i)
			fill_leaf<int32_t>(out, i, leaves[i]);
		return DlxArray(std::move(out));
	}
	case dracolix::DType::I64: {
		dracolix::Array<int64_t> out(shape);
		for (size_t i = 0; i < leaves.size(); ++i)
			fill_leaf<int64_t>(out, i, leaves[i]);
		return DlxArray(std::move(out));
	}
	case dracolix::DType::U8: {
		dracolix::Array<uint8_t> out(shape);
		for (size_t i = 0; i < leaves.size(); ++i)
			fill_leaf<uint8_t>(out, i, leaves[i]);
		return DlxArray(std::move(out));
	}
	case dracolix::DType::U16: {
		dracolix::Array<uint16_t> out(shape);
		for (size_t i = 0; i < leaves.size(); ++i)
			fill_leaf<uint16_t>(out, i, leaves[i]);
		return DlxArray(std::move(out));
	}
	case dracolix::DType::U32: {
		dracolix::Array<uint32_t> out(shape);
		for (size_t i = 0; i < leaves.size(); ++i)
			fill_leaf<uint32_t>(out, i, leaves[i]);
		return DlxArray(std::move(out));
	}
	case dracolix::DType::U64: {
		dracolix::Array<uint64_t> out(shape);
		for (size_t i = 0; i < leaves.size(); ++i)
			fill_leaf<uint64_t>(out, i, leaves[i]);
		return DlxArray(std::move(out));
	}
	case dracolix::DType::C64: {
		dracolix::Array<std::complex<float>> out(shape);
		for (size_t i = 0; i < leaves.size(); ++i)
			fill_leaf<std::complex<float>>(out, i, leaves[i]);
		return DlxArray(std::move(out));
	}
	case dracolix::DType::C128: {
		dracolix::Array<std::complex<double>> out(shape);
		for (size_t i = 0; i < leaves.size(); ++i)
			fill_leaf<std::complex<double>>(out, i, leaves[i]);
		return DlxArray(std::move(out));
	}
	case dracolix::DType::Bool: {
		dracolix::Array<bool> out(shape);
		for (size_t i = 0; i < leaves.size(); ++i)
			fill_leaf<bool>(out, i, leaves[i]);
		return DlxArray(std::move(out));
	}
	default:
		throw nb::type_error("unsupported dtype");
	}
}

// ---------------------------------------------------------------------------
// dlx.array / dlx.asarray
// ---------------------------------------------------------------------------

DlxArray dlx_array(nb::object data, nb::object dtype) {
	std::optional<dracolix::DType> dt = optional_dtype(dtype);
	if (is_array(data))
		return dt ? cast_dtype(*nb::cast<DlxArray *>(data), *dt)
				  : DlxArray(*nb::cast<DlxArray *>(data));
	if (PyObject_CheckBuffer(data.ptr()))
		return from_buffer(data, dt);
	if (PyList_Check(data.ptr()) || PyTuple_Check(data.ptr()))
		return from_list(data, dt);
	throw nb::type_error("expected a sequence, buffer, or dracolix Array");
}

nb::object dlx_asarray(nb::object data, nb::object dtype) {
	std::optional<dracolix::DType> dt = optional_dtype(dtype);
	if (is_array(data)) {
		if (dt && *dt != nb::cast<DlxArray *>(data)->dtype())
			return nb::cast<DlxArray>(
				cast_dtype(*nb::cast<DlxArray *>(data), *dt));
		return data; // no copy
	}
	return nb::cast<DlxArray>(dlx_array(data, dtype));
}

// ---------------------------------------------------------------------------
// shape parsing for zeros / ones / full / empty
// ---------------------------------------------------------------------------

std::vector<size_t> parse_shape(nb::object o) {
	std::vector<size_t> shape;
	if (PyLong_Check(o.ptr())) {
		int64_t n = nb::cast<int64_t>(o);
		if (n <= 0)
			throw nb::value_error("dimensions must be positive");
		shape.push_back((size_t)n);
		return shape;
	}
	if (PyList_Check(o.ptr()) || PyTuple_Check(o.ptr())) {
		Py_ssize_t len = PySequence_Size(o.ptr());
		for (Py_ssize_t i = 0; i < len; ++i) {
			nb::object item = nb::steal(PySequence_GetItem(o.ptr(), i));
			int64_t n = nb::cast<int64_t>(item);
			if (n <= 0)
				throw nb::value_error("dimensions must be positive");
			shape.push_back((size_t)n);
		}
		return shape;
	}
	throw nb::type_error("shape must be an int or a sequence of ints");
}

template <typename T> T scalar_to_dtype(nb::handle h) {
	if constexpr (std::is_same_v<T, bool>) {
		return nb::cast<bool>(h);
	} else if constexpr (dracolix::is_complex_v<T>) {
		return static_cast<T>(to_complex(h));
	} else if constexpr (std::is_integral_v<T>) {
		return integer_from_py<T>(h);
	} else {
		return (T)to_double(h);
	}
}

template <typename T>
DlxArray make_filled(const std::vector<size_t> &shape, T v) {
	dracolix::Array<T> a(shape);
	for (size_t i = 0; i < a.size(); ++i)
		a[i] = v;
	return DlxArray(std::move(a));
}

DlxArray dlx_zeros(nb::object shape, nb::object dtype) {
	std::vector<size_t> s = parse_shape(shape);
	std::optional<dracolix::DType> dt = optional_dtype(dtype);
	dracolix::DType t = dt.value_or(dracolix::DType::F64);
	switch (t) {
	case dracolix::DType::F32:
		return DlxArray(dracolix::Array<float>::zeros(s));
	case dracolix::DType::F64:
		return DlxArray(dracolix::Array<double>::zeros(s));
	case dracolix::DType::I8:
		return DlxArray(dracolix::Array<int8_t>::zeros(s));
	case dracolix::DType::I16:
		return DlxArray(dracolix::Array<int16_t>::zeros(s));
	case dracolix::DType::I32:
		return DlxArray(dracolix::Array<int32_t>::zeros(s));
	case dracolix::DType::I64:
		return DlxArray(dracolix::Array<int64_t>::zeros(s));
	case dracolix::DType::U8:
		return DlxArray(dracolix::Array<uint8_t>::zeros(s));
	case dracolix::DType::U16:
		return DlxArray(dracolix::Array<uint16_t>::zeros(s));
	case dracolix::DType::U32:
		return DlxArray(dracolix::Array<uint32_t>::zeros(s));
	case dracolix::DType::U64:
		return DlxArray(dracolix::Array<uint64_t>::zeros(s));
	case dracolix::DType::C64:
		return DlxArray(dracolix::Array<std::complex<float>>::zeros(s));
	case dracolix::DType::C128:
		return DlxArray(dracolix::Array<std::complex<double>>::zeros(s));
	case dracolix::DType::Bool:
		return DlxArray(dracolix::Array<bool>::zeros(s));
	default:
		throw nb::type_error("unsupported dtype");
	}
}

DlxArray dlx_ones(nb::object shape, nb::object dtype) {
	std::vector<size_t> s = parse_shape(shape);
	std::optional<dracolix::DType> dt = optional_dtype(dtype);
	dracolix::DType t = dt.value_or(dracolix::DType::F64);
	switch (t) {
	case dracolix::DType::F32:
		return DlxArray(dracolix::Array<float>::ones(s));
	case dracolix::DType::F64:
		return DlxArray(dracolix::Array<double>::ones(s));
	case dracolix::DType::I8:
		return DlxArray(dracolix::Array<int8_t>::ones(s));
	case dracolix::DType::I16:
		return DlxArray(dracolix::Array<int16_t>::ones(s));
	case dracolix::DType::I32:
		return DlxArray(dracolix::Array<int32_t>::ones(s));
	case dracolix::DType::I64:
		return DlxArray(dracolix::Array<int64_t>::ones(s));
	case dracolix::DType::U8:
		return DlxArray(dracolix::Array<uint8_t>::ones(s));
	case dracolix::DType::U16:
		return DlxArray(dracolix::Array<uint16_t>::ones(s));
	case dracolix::DType::U32:
		return DlxArray(dracolix::Array<uint32_t>::ones(s));
	case dracolix::DType::U64:
		return DlxArray(dracolix::Array<uint64_t>::ones(s));
	case dracolix::DType::C64:
		return DlxArray(dracolix::Array<std::complex<float>>::ones(s));
	case dracolix::DType::C128:
		return DlxArray(dracolix::Array<std::complex<double>>::ones(s));
	case dracolix::DType::Bool:
		return DlxArray(dracolix::Array<bool>::ones(s));
	default:
		throw nb::type_error("unsupported dtype");
	}
}

DlxArray dlx_empty(nb::object shape, nb::object dtype) {
	// The core always zero-initializes storage; empty() == zeros() for now.
	return dlx_zeros(shape, dtype);
}

DlxArray dlx_full(nb::object shape, nb::object value, nb::object dtype) {
	std::vector<size_t> s = parse_shape(shape);
	if (!is_number(value))
		throw nb::type_error("full(): fill value must be a scalar");
	std::optional<dracolix::DType> dt = optional_dtype(dtype);
	dracolix::DType t =
		dt.value_or(num_kind(value) == NumKind::Complex ? dracolix::DType::C128
					: num_kind(value) == NumKind::Float ? dracolix::DType::F64
					: num_kind(value) == NumKind::Bool	? dracolix::DType::Bool
														: dracolix::DType::I64);
	switch (t) {
	case dracolix::DType::F32:
		return make_filled<float>(s, scalar_to_dtype<float>(value));
	case dracolix::DType::F64:
		return make_filled<double>(s, scalar_to_dtype<double>(value));
	case dracolix::DType::I8:
		return make_filled<int8_t>(s, scalar_to_dtype<int8_t>(value));
	case dracolix::DType::I16:
		return make_filled<int16_t>(s, scalar_to_dtype<int16_t>(value));
	case dracolix::DType::I32:
		return make_filled<int32_t>(s, scalar_to_dtype<int32_t>(value));
	case dracolix::DType::I64:
		return make_filled<int64_t>(s, scalar_to_dtype<int64_t>(value));
	case dracolix::DType::U8:
		return make_filled<uint8_t>(s, scalar_to_dtype<uint8_t>(value));
	case dracolix::DType::U16:
		return make_filled<uint16_t>(s, scalar_to_dtype<uint16_t>(value));
	case dracolix::DType::U32:
		return make_filled<uint32_t>(s, scalar_to_dtype<uint32_t>(value));
	case dracolix::DType::U64:
		return make_filled<uint64_t>(s, scalar_to_dtype<uint64_t>(value));
	case dracolix::DType::C64:
		return make_filled<std::complex<float>>(
			s, scalar_to_dtype<std::complex<float>>(value));
	case dracolix::DType::C128:
		return make_filled<std::complex<double>>(
			s, scalar_to_dtype<std::complex<double>>(value));
	case dracolix::DType::Bool:
		return make_filled<bool>(s, scalar_to_dtype<bool>(value));
	default:
		throw nb::type_error("unsupported dtype");
	}
}

DlxArray dlx_arange(nb::object start, nb::object stop, nb::object step) {
	bool has_stop = !stop.is_none();
	nb::object a0 = has_stop ? start : nb::object(nb::int_(0));
	nb::object a1 = has_stop ? stop : start;
	nb::object a2 = step.is_none() ? nb::object(nb::int_(1)) : step;

	if (!is_number(a0) || !is_number(a1) || !is_number(a2))
		throw nb::type_error("arange arguments must be numbers");
	if (num_kind(a0) == NumKind::Complex || num_kind(a1) == NumKind::Complex ||
		num_kind(a2) == NumKind::Complex)
		throw nb::type_error("arange does not support complex arguments");
	if (num_kind(a2) == NumKind::Int && nb::cast<int64_t>(a2) == 0)
		throw nb::value_error("arange step cannot be 0");

	bool any_float = false;
	for (nb::handle h : {nb::handle(a0), nb::handle(a1), nb::handle(a2)})
		if (num_kind(h) == NumKind::Float)
			any_float = true;

	if (!any_float) {
		int64_t s0 = nb::cast<int64_t>(a0), s1 = nb::cast<int64_t>(a1),
				s2 = nb::cast<int64_t>(a2);
		int64_t span = s1 - s0;
		int64_t n64 = 0;
		if (s2 > 0 && span > 0)
			n64 = (span + s2 - 1) / s2;
		if (s2 < 0 && span < 0)
			n64 = (span + s2 + 1) / s2;
		if (n64 < 0)
			n64 = 0;
		dracolix::Array<int64_t> a(std::vector<size_t>{(size_t)n64});
		for (int64_t i = 0; i < n64; ++i)
			a[(size_t)i] = s0 + i * s2;
		return DlxArray(std::move(a));
	}
	double d0 = nb::cast<double>(a0), d1 = nb::cast<double>(a1),
		   d2 = nb::cast<double>(a2);
	double span = d1 - d0;
	int64_t n64 = 0;
	if (d2 != 0.0) {
		if (d2 > 0 && span > 0)
			n64 = (int64_t)std::ceil(span / d2);
		if (d2 < 0 && span < 0)
			n64 = (int64_t)std::ceil(span / d2);
	}
	if (n64 < 0)
		n64 = 0;
	dracolix::Array<double> a(std::vector<size_t>{(size_t)n64});
	for (int64_t i = 0; i < n64; ++i)
		a[(size_t)i] = d0 + (double)i * d2;
	return DlxArray(std::move(a));
}

DlxArray dlx_linspace(nb::object start, nb::object stop, nb::object num,
					  nb::object endpoint, nb::object dtype) {
	if (!is_number(start) || !is_number(stop))
		throw nb::type_error("linspace arguments must be numbers");
	int64_t n = num.is_none() ? 50 : nb::cast<int64_t>(num);
	if (n < 0)
		throw nb::value_error("number of samples must be non-negative");
	bool ep = endpoint.is_none() ? true : nb::cast<bool>(endpoint);
	std::optional<dracolix::DType> dt = optional_dtype(dtype);
	bool complex_result = num_kind(start) == NumKind::Complex ||
						  num_kind(stop) == NumKind::Complex ||
						  (dt && is_complex_dtype(*dt));

	if (n == 0) {
		DlxArray base =
			complex_result
				? DlxArray(dracolix::Array<std::complex<double>>(
					  std::vector<size_t>{0}))
				: DlxArray(dracolix::Array<double>(std::vector<size_t>{0}));
		return dt ? cast_dtype(base, *dt) : base;
	}

	if (complex_result) {
		std::complex<double> a = to_complex(start);
		std::complex<double> b = to_complex(stop);
		dracolix::Array<std::complex<double>> out(
			std::vector<size_t>{(size_t)n});
		if (n == 1) {
			out[0] = a;
		} else {
			double denom = ep ? (double)(n - 1) : (double)n;
			std::complex<double> step = (b - a) / denom;
			for (int64_t i = 0; i < n; ++i)
				out[(size_t)i] = a + (double)i * step;
			if (ep)
				out[(size_t)(n - 1)] = b;
		}
		DlxArray base(std::move(out));
		return dt ? cast_dtype(base, *dt) : base;
	}

	double a = to_double(start);
	double b = to_double(stop);
	dracolix::Array<double> out(std::vector<size_t>{(size_t)n});
	if (n == 1) {
		out[0] = a;
	} else {
		double denom = ep ? (double)(n - 1) : (double)n;
		double step = (b - a) / denom;
		for (int64_t i = 0; i < n; ++i)
			out[(size_t)i] = a + (double)i * step;
		if (ep)
			out[(size_t)(n - 1)] = b;
	}
	DlxArray base(std::move(out));
	return dt ? cast_dtype(base, *dt) : base;
}

// ---------------------------------------------------------------------------
// Iteration
// ---------------------------------------------------------------------------

struct FlatIter {
	const DlxArray *arr = nullptr;
	int64_t i = 0;
	FlatIter &operator++() {
		++i;
		return *this;
	}
	FlatIter operator++(int) {
		FlatIter c = *this;
		++i;
		return c;
	}
	bool operator==(const FlatIter &o) const {
		return i == o.i && arr == o.arr;
	}
	bool operator!=(const FlatIter &o) const { return !(*this == o); }
	nb::object operator*() const { return py_scalar_at(*arr, i); }

	using iterator_category = std::forward_iterator_tag;
	using value_type = nb::object;
	using difference_type = std::ptrdiff_t;
	using pointer = nb::object *;
	using reference = nb::object;
};

nb::object make_iter(const DlxArray &self, nb::handle scope) {
	return nb::make_iterator<nb::rv_policy::copy>(
		scope, "dracolix_array_iterator", FlatIter{&self, 0},
		FlatIter{&self, (int64_t)self.size()});
}

// ---------------------------------------------------------------------------
// reductions
// ---------------------------------------------------------------------------

nb::object dlx_sum(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				size_t cnt = 0;
				for (size_t i = 0; i < arr.size(); ++i)
					if (arr[i])
						++cnt;
				return nb::cast((int64_t)cnt);
			} else {
				return scalar_to_py<T>(arr.sum());
			}
		},
		a.v);
}

nb::object dlx_min(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				// Logical AND: min([True, False]) is False.
				bool all = true;
				for (size_t i = 0; i < arr.size(); ++i)
					if (!arr[i]) {
						all = false;
						break;
					}
				return nb::cast(all);
			} else if constexpr (dracolix::is_complex_v<T>) {
				throw nb::type_error(
					"min() is undefined for complex arrays (no ordering)");
			} else {
				return nb::cast(arr.min());
			}
		},
		a.v);
}

nb::object dlx_max(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				bool any = false;
				for (size_t i = 0; i < arr.size(); ++i)
					if (arr[i]) {
						any = true;
						break;
					}
				return nb::cast(any);
			} else if constexpr (dracolix::is_complex_v<T>) {
				throw nb::type_error(
					"max() is undefined for complex arrays (no ordering)");
			} else {
				return nb::cast(arr.max());
			}
		},
		a.v);
}

nb::object dlx_mean(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				size_t cnt = 0;
				for (size_t i = 0; i < arr.size(); ++i)
					if (arr[i])
						++cnt;
				return nb::cast((double)cnt / (double)arr.size());
			} else if constexpr (dracolix::is_complex_v<T>) {
				return scalar_to_py<T>(arr.avg());
			} else {
				return nb::cast(arr.mean());
			}
		},
		a.v);
}

int64_t normalize_axis(const DlxArray &a, int64_t axis) {
	if (axis < 0)
		axis += (int64_t)a.ndim();
	if (axis < 0 || (size_t)axis >= a.ndim())
		throw nb::index_error("axis out of bounds");
	return axis;
}

DlxArray dlx_sum_axis(const DlxArray &a, int64_t axis) {
	axis = normalize_axis(a, axis);
	return std::visit(
		[&](const auto &arr) -> DlxArray {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				// logical OR along axis (documented in __init__.pyi)
				dracolix::Array<bool> r = arr.sum((size_t)axis);
				return DlxArray(std::move(r));
			} else {
				return DlxArray(arr.sum((size_t)axis));
			}
		},
		a.v);
}

DlxArray dlx_min_axis(const DlxArray &a, int64_t axis) {
	axis = normalize_axis(a, axis);
	return std::visit(
		[&](const auto &arr) -> DlxArray {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (dracolix::is_complex_v<T>)
				throw nb::type_error("min(axis) is undefined for complex "
									 "arrays (no ordering)");
			else
				return DlxArray(arr.min_axis((size_t)axis));
		},
		a.v);
}

DlxArray dlx_max_axis(const DlxArray &a, int64_t axis) {
	axis = normalize_axis(a, axis);
	return std::visit(
		[&](const auto &arr) -> DlxArray {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (dracolix::is_complex_v<T>)
				throw nb::type_error("max(axis) is undefined for complex "
									 "arrays (no ordering)");
			else
				return DlxArray(arr.max_axis((size_t)axis));
		},
		a.v);
}

DlxArray dlx_mean_axis(const DlxArray &a, int64_t axis) {
	axis = normalize_axis(a, axis);
	return std::visit(
		[&](const auto &arr) -> DlxArray {
			return DlxArray(arr.mean_axis((size_t)axis));
		},
		a.v);
}

nb::object dlx_var(const DlxArray &a, int64_t ddof) {
	if (a.size() == 0)
		throw nb::value_error("variance of empty array");
	return std::visit(
		[&](const auto &arr) -> nb::object {
			return nb::cast(arr.var((int)ddof));
		},
		a.v);
}

nb::object dlx_std(const DlxArray &a, int64_t ddof) {
	if (a.size() == 0)
		throw nb::value_error("standard deviation of empty array");
	return std::visit(
		[&](const auto &arr) -> nb::object {
			return nb::cast(arr.std((int)ddof));
		},
		a.v);
}

DlxArray dlx_var_axis(const DlxArray &a, int64_t axis, int64_t ddof) {
	axis = normalize_axis(a, axis);
	return std::visit(
		[&](const auto &arr) -> DlxArray {
			return DlxArray(arr.var_axis((size_t)axis, (int)ddof));
		},
		a.v);
}

DlxArray dlx_std_axis(const DlxArray &a, int64_t axis, int64_t ddof) {
	axis = normalize_axis(a, axis);
	return std::visit(
		[&](const auto &arr) -> DlxArray {
			return DlxArray(arr.std_axis((size_t)axis, (int)ddof));
		},
		a.v);
}

nb::object dlx_var_dispatch(const DlxArray &a, nb::object axis, int64_t ddof) {
	if (axis.is_none())
		return dlx_var(a, ddof);
	return nb::cast<DlxArray>(dlx_var_axis(a, nb::cast<int64_t>(axis), ddof));
}

nb::object dlx_std_dispatch(const DlxArray &a, nb::object axis, int64_t ddof) {
	if (axis.is_none())
		return dlx_std(a, ddof);
	return nb::cast<DlxArray>(dlx_std_axis(a, nb::cast<int64_t>(axis), ddof));
}

nb::object dlx_prod(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				// Logical AND, returned as i64 0/1 (NumPy parity).
				bool p = true;
				for (size_t i = 0; i < arr.size(); ++i)
					if (!arr[i]) {
						p = false;
						break;
					}
				return nb::cast((int64_t)(p ? 1 : 0));
			} else {
				return scalar_to_py<T>(arr.prod());
			}
		},
		a.v);
}

DlxArray dlx_prod_axis(const DlxArray &a, int64_t axis) {
	axis = normalize_axis(a, axis);
	return std::visit(
		[&](const auto &arr) -> DlxArray {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				auto r = arr.prod((size_t)axis);
				return DlxArray(r.template astype<int64_t>());
			} else {
				return DlxArray(arr.prod((size_t)axis));
			}
		},
		a.v);
}

nb::object dlx_argmin(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (dracolix::is_complex_v<T>) {
				throw nb::type_error(
					"argmin is undefined for complex arrays (no ordering)");
			} else {
				if (arr.size() == 0)
					throw nb::value_error("argmin of empty array");
				return nb::cast((int64_t)arr.argmin());
			}
		},
		a.v);
}

nb::object dlx_argmax(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (dracolix::is_complex_v<T>) {
				throw nb::type_error(
					"argmax is undefined for complex arrays (no ordering)");
			} else {
				if (arr.size() == 0)
					throw nb::value_error("argmax of empty array");
				return nb::cast((int64_t)arr.argmax());
			}
		},
		a.v);
}

DlxArray dlx_argmin_axis(const DlxArray &a, int64_t axis) {
	axis = normalize_axis(a, axis);
	return std::visit(
		[&](const auto &arr) -> DlxArray {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (dracolix::is_complex_v<T>) {
				throw nb::type_error("argmin(axis) is undefined for complex "
									 "arrays (no ordering)");
			} else {
				if (arr.size() == 0 || arr.shape()[(size_t)axis] == 0)
					throw nb::value_error("argmin(axis) of an empty sequence");
				return DlxArray(arr.argmin((size_t)axis));
			}
		},
		a.v);
}

DlxArray dlx_argmax_axis(const DlxArray &a, int64_t axis) {
	axis = normalize_axis(a, axis);
	return std::visit(
		[&](const auto &arr) -> DlxArray {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (dracolix::is_complex_v<T>) {
				throw nb::type_error("argmax(axis) is undefined for complex "
									 "arrays (no ordering)");
			} else {
				if (arr.size() == 0 || arr.shape()[(size_t)axis] == 0)
					throw nb::value_error("argmax(axis) of an empty sequence");
				return DlxArray(arr.argmax((size_t)axis));
			}
		},
		a.v);
}

nb::object dlx_any(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> nb::object { return nb::cast(arr.any()); }, a.v);
}

nb::object dlx_all(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> nb::object { return nb::cast(arr.all()); }, a.v);
}

DlxArray dlx_any_axis(const DlxArray &a, int64_t axis) {
	axis = normalize_axis(a, axis);
	return std::visit(
		[&](const auto &arr) -> DlxArray {
			return DlxArray(arr.any_axis((size_t)axis));
		},
		a.v);
}

DlxArray dlx_all_axis(const DlxArray &a, int64_t axis) {
	axis = normalize_axis(a, axis);
	return std::visit(
		[&](const auto &arr) -> DlxArray {
			return DlxArray(arr.all_axis((size_t)axis));
		},
		a.v);
}

// ---------------------------------------------------------------------------
// comparisons
// -----------
// All comparators implement two functions: one array-array elementwise and
// one single-array that compares against a scalar
//
// NOTE: not sure how to do this at all
// ---------------------------------------------------------------------------

static DlxArray dlx_compare_arrays(const DlxArray &a, const DlxArray &b,
								   BinOp op) {
	dracolix::DType r = promote_pair(a.dtype(), b.dtype());
	DlxArray Ca = cast_dtype(a, r);
	DlxArray Cb = cast_dtype(b, r);
	return bin_switch(r, Ca, Cb, op);
}

DlxArray dlx_equal(const DlxArray &a, const DlxArray &b) {
	return dlx_compare_arrays(a, b, BinOp::Eq);
}

DlxArray dlx_equal(const DlxArray &a, const int64_t x) {
	return dlx_compare_arrays(a, dlx_array(nb::int_(x), nb::none()), BinOp::Eq);
}

DlxArray dlx_not_equal(const DlxArray &a, const DlxArray &b) {
	return dlx_compare_arrays(a, b, BinOp::Ne);
}

DlxArray dlx_not_equal(const DlxArray &a, const int64_t x) {
	return dlx_compare_arrays(a, dlx_array(nb::int_(x), nb::none()), BinOp::Ne);
}

DlxArray dlx_less_than(const DlxArray &a, const DlxArray &b) {
	return dlx_compare_arrays(a, b, BinOp::Lt);
}

DlxArray dlx_less_than(const DlxArray &a, const int64_t x) {
	return dlx_compare_arrays(a, dlx_array(nb::int_(x), nb::none()), BinOp::Lt);
}

DlxArray dlx_less_or_equal_than(const DlxArray &a, const DlxArray &b) {
	return dlx_compare_arrays(a, b, BinOp::Le);
}

DlxArray dlx_less_or_equal_than(const DlxArray &a, const int64_t x) {
	return dlx_compare_arrays(a, dlx_array(nb::int_(x), nb::none()), BinOp::Le);
}

DlxArray dlx_greater_than(const DlxArray &a, const DlxArray &b) {
	return dlx_compare_arrays(a, b, BinOp::Gt);
}

DlxArray dlx_greater_than(const DlxArray &a, const int64_t x) {
	return dlx_compare_arrays(a, dlx_array(nb::int_(x), nb::none()), BinOp::Gt);
}

DlxArray dlx_greater_or_equal_than(const DlxArray &a, const DlxArray &b) {
	return dlx_compare_arrays(a, b, BinOp::Ge);
}

DlxArray dlx_greater_or_equal_than(const DlxArray &a, const int64_t x) {
	return dlx_compare_arrays(a, dlx_array(nb::int_(x), nb::none()), BinOp::Ge);
}

// ---------------------------------------------------------------------------
// complex-valued helpers: real / imag / conj / abs / angle
//
// These work for every dtype. For real arrays the imaginary part is zero,
// conjugation and the real part are identity, and the phase is 0 or pi.
// ---------------------------------------------------------------------------
DlxArray dlx_real(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> DlxArray {
			return DlxArray(dracolix::cplx::real(arr));
		},
		a.v);
}

DlxArray dlx_imag(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> DlxArray {
			return DlxArray(dracolix::cplx::imag(arr));
		},
		a.v);
}

DlxArray dlx_conj(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> DlxArray {
			return DlxArray(dracolix::cplx::conj(arr));
		},
		a.v);
}

DlxArray dlx_abs(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> DlxArray {
			return DlxArray(dracolix::cplx::abs(arr));
		},
		a.v);
}

DlxArray dlx_angle(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> DlxArray {
			return DlxArray(dracolix::cplx::angle(arr));
		},
		a.v);
}

// -------------------------------------------------------------------------
// Elementwise math (ufuncs)
// -------------------------------------------------------------------------

enum class UFunc {
	Sqrt,
	Exp,
	Expm1,
	Log,
	Log1p,
	Log2,
	Log10,
	Sin,
	Cos,
	Tan,
	Asin,
	Acos,
	Atan,
	Sinh,
	Cosh,
	Tanh,
	Floor,
	Ceil,
	Trunc
};

bool ufunc_complex_ok(UFunc f) {
	return f != UFunc::Floor && f != UFunc::Ceil && f != UFunc::Trunc;
}

template <typename T> T apply_real_ufunc(UFunc f, T v) {
	switch (f) {
	case UFunc::Sqrt:
		return static_cast<T>(std::sqrt(v));
	case UFunc::Exp:
		return static_cast<T>(std::exp(v));
	case UFunc::Expm1:
		return static_cast<T>(std::expm1(v));
	case UFunc::Log:
		return static_cast<T>(std::log(v));
	case UFunc::Log1p:
		return static_cast<T>(std::log1p(v));
	case UFunc::Log2:
		return static_cast<T>(std::log2(v));
	case UFunc::Log10:
		return static_cast<T>(std::log10(v));
	case UFunc::Sin:
		return static_cast<T>(std::sin(v));
	case UFunc::Cos:
		return static_cast<T>(std::cos(v));
	case UFunc::Tan:
		return static_cast<T>(std::tan(v));
	case UFunc::Asin:
		return static_cast<T>(std::asin(v));
	case UFunc::Acos:
		return static_cast<T>(std::acos(v));
	case UFunc::Atan:
		return static_cast<T>(std::atan(v));
	case UFunc::Sinh:
		return static_cast<T>(std::sinh(v));
	case UFunc::Cosh:
		return static_cast<T>(std::cosh(v));
	case UFunc::Tanh:
		return static_cast<T>(std::tanh(v));
	case UFunc::Floor:
		return static_cast<T>(std::floor(v));
	case UFunc::Ceil:
		return static_cast<T>(std::ceil(v));
	case UFunc::Trunc:
		return static_cast<T>(std::trunc(v));
	}
	return v;
}

template <typename T>
std::complex<T> apply_complex_ufunc(UFunc f, std::complex<T> z) {
	using C = std::complex<T>;
	switch (f) {
	case UFunc::Sqrt:
		return std::sqrt(z);
	case UFunc::Exp:
		return std::exp(z);
	case UFunc::Expm1:
		return std::exp(z) - C(1);
	case UFunc::Log:
		return std::log(z);
	case UFunc::Log1p:
		return std::log(z + C(1));
	case UFunc::Log2:
		return std::log(z) / std::log(C(2));
	case UFunc::Log10:
		return std::log(z) / std::log(C(10));
	case UFunc::Sin:
		return std::sin(z);
	case UFunc::Cos:
		return std::cos(z);
	case UFunc::Tan:
		return std::tan(z);
	case UFunc::Asin:
		return std::asin(z);
	case UFunc::Acos:
		return std::acos(z);
	case UFunc::Atan:
		return std::atan(z);
	case UFunc::Sinh:
		return std::sinh(z);
	case UFunc::Cosh:
		return std::cosh(z);
	case UFunc::Tanh:
		return std::tanh(z);
	case UFunc::Floor:
	case UFunc::Ceil:
	case UFunc::Trunc:
		throw nb::type_error(
			"floor/ceil/trunc are not supported for complex dtypes");
	}
	return z;
}

DlxArray dlx_ufunc(const DlxArray &a, UFunc f) {
	dracolix::DType dt = a.dtype();
	dracolix::DType out_dt;
	if (is_complex_dtype(dt)) {
		if (!ufunc_complex_ok(f))
			throw nb::type_error(
				"floor/ceil/trunc are not supported for complex dtypes");
		out_dt = dt;
	} else if (dt == dracolix::DType::F32) {
		out_dt = dracolix::DType::F32;
	} else {
		out_dt = dracolix::DType::F64;
	}
	DlxArray c = cast_dtype(a, out_dt);
	return std::visit(
		[f](const auto &arr) -> DlxArray {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			dracolix::Array<T> out(arr.shape());
			for (size_t i = 0; i < arr.size(); ++i) {
				if constexpr (dracolix::is_complex_v<T>)
					out[i] = apply_complex_ufunc(f, arr[i]);
				else
					out[i] = apply_real_ufunc<T>(f, static_cast<T>(arr[i]));
			}
			return DlxArray(std::move(out));
		},
		c.v);
}

DlxArray dlx_square(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> DlxArray {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				return DlxArray(arr.clone());
			} else {
				dracolix::Array<T> out(arr.shape());
				for (size_t i = 0; i < arr.size(); ++i)
					out[i] = static_cast<T>(arr[i] * arr[i]);
				return DlxArray(std::move(out));
			}
		},
		a.v);
}

enum class Pred { Finite, Nan, Inf };

template <typename T> bool apply_pred(Pred p, T v) {
	if constexpr (dracolix::is_complex_v<T>) {
		double re = static_cast<double>(v.real());
		double im = static_cast<double>(v.imag());
		switch (p) {
		case Pred::Finite:
			return std::isfinite(re) && std::isfinite(im);
		case Pred::Nan:
			return std::isnan(re) || std::isnan(im);
		case Pred::Inf:
			return std::isinf(re) || std::isinf(im);
		}
		return false;
	} else {
		double d = static_cast<double>(v);
		switch (p) {
		case Pred::Finite:
			return std::isfinite(d);
		case Pred::Nan:
			return std::isnan(d);
		case Pred::Inf:
			return std::isinf(d);
		}
		return false;
	}
}

DlxArray dlx_predicate(const DlxArray &a, Pred p) {
	return std::visit(
		[p](const auto &arr) -> DlxArray {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			dracolix::Array<bool> out(arr.shape());
			for (size_t i = 0; i < arr.size(); ++i)
				out[i] = apply_pred<T>(p, static_cast<T>(arr[i]));
			return DlxArray(std::move(out));
		},
		a.v);
}

// ---------------------------------------------------------------------------
// elementwise maximum / minimum
// ---------------------------------------------------------------------------

enum class MinMax { Maximum, Minimum };

template <typename T> T minmax_scalar(T x, T y, MinMax op) {
	if constexpr (dracolix::is_complex_v<T>) {
		// NumPy orders complex values lexicographically: real part first,
		// then imaginary part. NaN in either component propagates.
		if (std::isnan(x.real()) || std::isnan(x.imag()))
			return x;
		if (std::isnan(y.real()) || std::isnan(y.imag()))
			return y;
		bool x_gt = x.real() > y.real() ||
					(x.real() == y.real() && x.imag() > y.imag());
		bool x_lt = x.real() < y.real() ||
					(x.real() == y.real() && x.imag() < y.imag());
		return op == MinMax::Maximum ? (x_gt ? x : y) : (x_lt ? x : y);
	} else if constexpr (std::is_floating_point_v<T>) {
		// IEEE maximum/minimum propagate NaN (unlike fmax/fmin).
		if (std::isnan(x))
			return x;
		if (std::isnan(y))
			return y;
		return op == MinMax::Maximum ? (x >= y ? x : y) : (x <= y ? x : y);
	} else {
		return op == MinMax::Maximum ? (x >= y ? x : y) : (x <= y ? x : y);
	}
}

template <typename T>
DlxArray minmax_typed(const DlxArray &A, const DlxArray &B, MinMax op) {
	const auto &a = *A.as<T>();
	const auto &b = *B.as<T>();
	return DlxArray(zip_bcast(a, b, [op](T x, T y) -> T {
		return minmax_scalar(x, y, op);
	}));
}

DlxArray minmax_switch(dracolix::DType dt, const DlxArray &A, const DlxArray &B,
					   MinMax op) {
	switch (dt) {
	case dracolix::DType::F32:
		return minmax_typed<float>(A, B, op);
	case dracolix::DType::F64:
		return minmax_typed<double>(A, B, op);
	case dracolix::DType::I8:
		return minmax_typed<int8_t>(A, B, op);
	case dracolix::DType::I16:
		return minmax_typed<int16_t>(A, B, op);
	case dracolix::DType::I32:
		return minmax_typed<int32_t>(A, B, op);
	case dracolix::DType::I64:
		return minmax_typed<int64_t>(A, B, op);
	case dracolix::DType::U8:
		return minmax_typed<uint8_t>(A, B, op);
	case dracolix::DType::U16:
		return minmax_typed<uint16_t>(A, B, op);
	case dracolix::DType::U32:
		return minmax_typed<uint32_t>(A, B, op);
	case dracolix::DType::U64:
		return minmax_typed<uint64_t>(A, B, op);
	case dracolix::DType::Bool:
		return minmax_typed<bool>(A, B, op);
	case dracolix::DType::C64:
		return minmax_typed<std::complex<float>>(A, B, op);
	case dracolix::DType::C128:
		return minmax_typed<std::complex<double>>(A, B, op);
	default:
		throw nb::type_error("unsupported dtype");
	}
}

// Scalars participate like in the binary operators, except that a Python bool
// stays bool so that ``maximum(bool_array, True)`` keeps the bool dtype.
bool coerce_minmax_operand(nb::handle h, std::optional<DlxArray> &out) {
	if (is_number(h) && num_kind(h) == NumKind::Bool) {
		dracolix::Array<bool> a(std::vector<size_t>{1});
		a[0] = nb::cast<bool>(h);
		out = DlxArray(std::move(a));
		return true;
	}
	return coerce_operand(h, out);
}

nb::object dlx_minmax(nb::object x, nb::object y, MinMax op) {
	bool both_scalar = is_number(x) && is_number(y);
	std::optional<DlxArray> A, B;
	if (!coerce_minmax_operand(x, A) || !coerce_minmax_operand(y, B))
		throw nb::type_error(op == MinMax::Maximum
								 ? "maximum expected arrays or numbers"
								 : "minimum expected arrays or numbers");
	dracolix::DType dt = promote_pair(A->dtype(), B->dtype());
	DlxArray Ca = cast_dtype(*A, dt);
	DlxArray Cb = cast_dtype(*B, dt);
	DlxArray out = minmax_switch(dt, Ca, Cb, op);
	if (!both_scalar)
		return nb::cast<DlxArray>(std::move(out));
	return std::visit(
		[](const auto &arr) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			return scalar_to_py<T>(static_cast<T>(arr[0]));
		},
		out.v);
}

// ---------------------------------------------------------------------------
// item / __index__
// ---------------------------------------------------------------------------

nb::object dlx_item(const DlxArray &a, nb::object index) {
	size_t off = 0;
	if (index.is_none()) {
		if (a.size() != 1)
			throw nb::value_error(
				"can only convert an array of size 1 to a Python scalar");
	} else {
		int64_t i = nb::cast<int64_t>(index);
		int64_t n = (int64_t)a.size();
		if (i < 0)
			i += n;
		if (i < 0 || i >= n)
			throw nb::index_error("index out of bounds");
		size_t rem = (size_t)i;
		for (int d = (int)a.ndim() - 1; d >= 0; --d) {
			off += (rem % a.shape()[(size_t)d]) * a.strides()[(size_t)d];
			rem /= a.shape()[(size_t)d];
		}
	}
	return py_scalar_at(a, (int64_t)off);
}

nb::object dlx_index(const DlxArray &a) {
	if (a.size() != 1)
		throw nb::type_error(
			"only size-1 arrays can be converted to Python scalars");
	dracolix::DType dt = a.dtype();
	if (dt == dracolix::DType::F32 || dt == dracolix::DType::F64 ||
		is_complex_dtype(dt))
		throw nb::type_error(
			"only integer arrays can be converted to a Python int");
	return py_scalar_at(a, 0);
}

// ---------------------------------------------------------------------------
// tolist / repr
// ---------------------------------------------------------------------------

template <typename T>
nb::object build_list(const dracolix::Array<T> &a, size_t d, size_t offset,
					  const std::vector<size_t> &rst) {
	nb::list l;
	size_t n = a.shape()[d];
	if (d + 1 == a.ndim()) {
		for (size_t i = 0; i < n; ++i)
			l.append(scalar_to_py<T>(a[offset + i * rst[d]]));
		return l;
	}
	for (size_t i = 0; i < n; ++i)
		l.append(build_list(a, d + 1, offset + i * rst[d], rst));
	return l;
}

nb::object dlx_tolist(const DlxArray &a) {
	return std::visit(
		[&](const auto &arr) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			std::vector<size_t> rst(a.ndim());
			size_t acc = 1;
			for (int d = (int)a.ndim() - 1; d >= 0; --d) {
				rst[(size_t)d] = acc;
				acc *= a.shape()[(size_t)d];
			}
			return build_list<T>(arr, 0, 0, rst);
		},
		a.v);
}

std::string repr_py(nb::object o) {
	nb::object r = nb::steal(PyObject_Repr(o.ptr()));
	return nb::cast<std::string>(r);
}

std::string dlx_repr(const DlxArray &a) {
	std::string out = "array(";
	out += repr_py(dlx_tolist(a));
	out += ", dtype=";
	out += dracolix::dtype_name(a.dtype());
	out += ")";
	return out;
}

// ---------------------------------------------------------------------------
// reshape / transpose / astype / copy
// ---------------------------------------------------------------------------

DlxArray dlx_reshape(const DlxArray &a, nb::object shape) {
	std::vector<size_t> s = parse_shape(shape);
	return std::visit(
		[&](const auto &arr) -> DlxArray {
			dracolix::Array<
				typename array_scalar<std::decay_t<decltype(arr)>>::type>
				c = arr;
			c.reshape(s);
			return DlxArray(std::move(c));
		},
		a.v);
}

DlxArray dlx_transpose(const DlxArray &a) {
	if (a.ndim() != 2)
		throw nb::value_error("transpose() requires 2-D");
	return std::visit([](const auto &arr) { return DlxArray(arr.transpose()); },
					  a.v);
}

DlxArray dlx_copy(const DlxArray &a) {
	return std::visit([](const auto &arr) { return DlxArray(arr.clone()); },
					  a.v);
}

DlxArray dlx_astype(const DlxArray &a, nb::object dtype) {
	std::optional<dracolix::DType> dt = optional_dtype(dtype);
	if (!dt)
		throw nb::type_error("astype() requires a dtype");
	return cast_dtype(a, *dt);
}

// ---------------------------------------------------------------------------
// Sparse matrices (CSR / CSC wrappers over the core header)
// ---------------------------------------------------------------------------
// Scalar types supported by sparse storage (no bool: bool x bool matmul is
// meaningless and would need dedicated semantics).
using CsrVariant =
	std::variant<dracolix::CsrMatrix<float>, dracolix::CsrMatrix<double>,
				 dracolix::CsrMatrix<int32_t>, dracolix::CsrMatrix<int64_t>>;
using CscVariant =
	std::variant<dracolix::CscMatrix<float>, dracolix::CscMatrix<double>,
				 dracolix::CscMatrix<int32_t>, dracolix::CscMatrix<int64_t>>;

// Scalar types that have sparse storage; used to keep std::get_if calls
// instantiated only for alternatives that actually exist in CsrVariant.
template <typename T>
inline constexpr bool is_sparse_scalar_v =
	std::is_same_v<T, float> || std::is_same_v<T, double> ||
	std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;

template <typename M> struct matrix_scalar;
template <typename T> struct matrix_scalar<dracolix::CsrMatrix<T>> {
	using type = T;
};
template <typename T> struct matrix_scalar<dracolix::CscMatrix<T>> {
	using type = T;
};

// MSVC refuses the std::variant converting constructor when the alternative
// type is a dependent name inside a template. These helpers narrow T to a
// concrete scalar via if constexpr so every std::variant construction sees
// a concrete, non-dependent matrix type.
template <typename T> CsrVariant make_csr(dracolix::CsrMatrix<T> m) {
	if constexpr (std::is_same_v<T, float>)
		return CsrVariant(std::move(m));
	else if constexpr (std::is_same_v<T, double>)
		return CsrVariant(std::move(m));
	else if constexpr (std::is_same_v<T, int32_t>)
		return CsrVariant(std::move(m));
	else
		return CsrVariant(std::move(m));
}
template <typename T> CscVariant make_csc(dracolix::CscMatrix<T> m) {
	if constexpr (std::is_same_v<T, float>)
		return CscVariant(std::move(m));
	else if constexpr (std::is_same_v<T, double>)
		return CscVariant(std::move(m));
	else if constexpr (std::is_same_v<T, int32_t>)
		return CscVariant(std::move(m));
	else
		return CscVariant(std::move(m));
}

struct DlxCsr {
	CsrVariant v;
	explicit DlxCsr(CsrVariant value) : v(std::move(value)) {}
	template <typename T>
	explicit DlxCsr(dracolix::CsrMatrix<T> m) : v(make_csr(std::move(m))) {}
	template <typename T> dracolix::CsrMatrix<T> *as() {
		return std::get_if<dracolix::CsrMatrix<T>>(&v);
	}
	template <typename T> const dracolix::CsrMatrix<T> *as() const {
		return std::get_if<dracolix::CsrMatrix<T>>(&v);
	}
};

struct DlxCsc {
	CscVariant v;
	explicit DlxCsc(CscVariant value) : v(std::move(value)) {}
	template <typename T>
	explicit DlxCsc(dracolix::CscMatrix<T> m) : v(make_csc(std::move(m))) {}
	template <typename T> dracolix::CscMatrix<T> *as() {
		return std::get_if<dracolix::CscMatrix<T>>(&v);
	}
	template <typename T> const dracolix::CscMatrix<T> *as() const {
		return std::get_if<dracolix::CscMatrix<T>>(&v);
	}
};

template <typename V> dracolix::DType sparse_dtype(const V &m) {
	return std::visit(
		[](const auto &mm) {
			using M = std::decay_t<decltype(mm)>;
			using T = typename matrix_scalar<M>::type;
			if constexpr (std::is_same_v<T, float>)
				return dracolix::DType::F32;
			if constexpr (std::is_same_v<T, double>)
				return dracolix::DType::F64;
			if constexpr (std::is_same_v<T, int32_t>)
				return dracolix::DType::I32;
			return dracolix::DType::I64;
		},
		m.v);
}

PyTypeObject *g_csr_type = nullptr;
PyTypeObject *g_csc_type = nullptr;
bool is_csr(nb::handle h) noexcept {
	return g_csr_type && PyObject_TypeCheck(h.ptr(), g_csr_type);
}
bool is_csc(nb::handle h) noexcept {
	return g_csc_type && PyObject_TypeCheck(h.ptr(), g_csc_type);
}

template <typename T>
dracolix::CsrMatrix<T> dense_to_csr(const dracolix::Array<T> &a) {
	return dracolix::CsrMatrix<T>::from_dense(a);
}
template <typename T>
dracolix::CscMatrix<T> dense_to_csc(const dracolix::Array<T> &a) {
	return dracolix::CscMatrix<T>::from_dense(a);
}

DlxCsc csc_from_dense(const DlxArray &a) {
	if (a.ndim() != 2)
		throw nb::type_error("csc_matrix() requires a 2-D array");
	if (a.dtype() == dracolix::DType::Bool)
		throw nb::type_error("sparse matrices do not support bool entries");
	// Concrete per-dtype branches: MSVC cannot resolve std::variant ctors
	// when the scalar dracolix::Array<T> type comes from a nested alias
	// (see helpers below), so spell each alternative out.
	switch (a.dtype()) {
	case dracolix::DType::F32: {
		const dracolix::Array<float> &src = *a.as<float>();
		return DlxCsc(CscVariant(dense_to_csc<float>(src)));
	}
	case dracolix::DType::F64: {
		const dracolix::Array<double> &src = *a.as<double>();
		return DlxCsc(CscVariant(dense_to_csc<double>(src)));
	}
	case dracolix::DType::I32: {
		const dracolix::Array<int32_t> &src = *a.as<int32_t>();
		return DlxCsc(CscVariant(dense_to_csc<int32_t>(src)));
	}
	case dracolix::DType::I64: {
		const dracolix::Array<int64_t> &src = *a.as<int64_t>();
		return DlxCsc(CscVariant(dense_to_csc<int64_t>(src)));
	}
	default:
		throw nb::type_error("unsupported sparse dtype");
	}
}

// ------------------------------------------------------------
// ---- construction ------------------------------------------
// ------------------------------------------------------------

DlxCsr csr_from_dense(const DlxArray &a) {
	if (a.ndim() != 2)
		throw nb::type_error("csr_matrix() requires a 2-D array");
	if (a.dtype() == dracolix::DType::Bool)
		throw nb::type_error("sparse matrices do not support bool entries");
	switch (a.dtype()) {
	case dracolix::DType::F32: {
		const dracolix::Array<float> &src = *a.as<float>();
		return DlxCsr(CsrVariant(dense_to_csr<float>(src)));
	}
	case dracolix::DType::F64: {
		const dracolix::Array<double> &src = *a.as<double>();
		return DlxCsr(CsrVariant(dense_to_csr<double>(src)));
	}
	case dracolix::DType::I32: {
		const dracolix::Array<int32_t> &src = *a.as<int32_t>();
		return DlxCsr(CsrVariant(dense_to_csr<int32_t>(src)));
	}
	case dracolix::DType::I64: {
		const dracolix::Array<int64_t> &src = *a.as<int64_t>();
		return DlxCsr(CsrVariant(dense_to_csr<int64_t>(src)));
	}
	default:
		throw nb::type_error("unsupported sparse dtype");
	}
}

std::vector<size_t> idx_list(nb::object o) {
	if (is_array(o)) {
		DlxArray &a = *nb::cast<DlxArray *>(o);
		if (a.ndim() != 1)
			throw nb::value_error("index arrays must be 1-D");
		std::vector<size_t> out(a.size());
		std::visit(
			[&](const auto &arr) {
				using U =
					typename array_scalar<std::decay_t<decltype(arr)>>::type;
				if constexpr (dracolix::is_complex_v<U>) {
					throw nb::type_error(
						"complex arrays cannot be used as sparse indices");
				} else {
					for (size_t i = 0; i < a.size(); ++i) {
						int64_t v = (int64_t)arr[i];
						if (v < 0)
							throw nb::value_error(
								"indices must be non-negative");
						out[i] = (size_t)v;
					}
				}
			},
			a.v);
		return out;
	}
	if (PyList_Check(o.ptr()) || PyTuple_Check(o.ptr())) {
		Py_ssize_t len = PySequence_Size(o.ptr());
		std::vector<size_t> out((size_t)len);
		for (Py_ssize_t i = 0; i < len; ++i) {
			nb::object item = nb::steal(PySequence_GetItem(o.ptr(), i));
			int64_t v = nb::cast<int64_t>(item);
			if (v < 0)
				throw nb::value_error("indices must be non-negative");
			out[(size_t)i] = (size_t)v;
		}
		return out;
	}
	throw nb::type_error("indices must be a list/tuple/array of ints");
}

template <typename T> std::vector<T> dlx_values(const DlxArray &a) {
	std::vector<T> out(a.size());
	std::visit(
		[&](const auto &arr) {
			using U = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (dracolix::is_complex_v<U>) {
				throw nb::type_error(
					"sparse matrices do not support complex values");
			} else {
				for (size_t i = 0; i < a.size(); ++i)
					out[i] = (T)arr[i];
			}
		},
		a.v);
	return out;
}

template <typename T>
DlxCsr coo_to_csr_typed(size_t rows, size_t cols, std::vector<size_t> ri,
						std::vector<size_t> ci, std::vector<T> vv,
						bool drop_zeros) {
	return DlxCsr(dracolix::CsrMatrix<T>::from_coo(
		rows, cols, std::move(ri), std::move(ci), std::move(vv), drop_zeros));
}
template <typename T>
DlxCsc coo_to_csc_typed(size_t rows, size_t cols, std::vector<size_t> ri,
						std::vector<size_t> ci, std::vector<T> vv,
						bool drop_zeros) {
	return DlxCsc(dracolix::CscMatrix<T>::from_coo(
		rows, cols, std::move(ri), std::move(ci), std::move(vv), drop_zeros));
}

DlxCsr csr_from_coo(size_t rows, size_t cols, nb::object row_ind,
					nb::object col_ind, nb::object values, bool drop_zeros) {
	DlxArray va = dlx_array(values, nb::none());
	if (va.dtype() == dracolix::DType::Bool)
		throw nb::type_error("sparse matrices do not support bool entries");
	std::vector<size_t> ri = idx_list(row_ind);
	std::vector<size_t> ci = idx_list(col_ind);
	if (ri.size() != ci.size() || ci.size() != va.size())
		throw nb::value_error("COO triplet arrays must have equal length");
	switch (va.dtype()) {
	case dracolix::DType::F32:
		return coo_to_csr_typed<float>(rows, cols, std::move(ri), std::move(ci),
									   dlx_values<float>(va), drop_zeros);
	case dracolix::DType::F64:
		return coo_to_csr_typed<double>(rows, cols, std::move(ri),
										std::move(ci), dlx_values<double>(va),
										drop_zeros);
	case dracolix::DType::I32:
		return coo_to_csr_typed<int32_t>(rows, cols, std::move(ri),
										 std::move(ci), dlx_values<int32_t>(va),
										 drop_zeros);
	case dracolix::DType::I64:
		return coo_to_csr_typed<int64_t>(rows, cols, std::move(ri),
										 std::move(ci), dlx_values<int64_t>(va),
										 drop_zeros);
	default:
		throw nb::type_error("unsupported sparse dtype");
	}
}

DlxCsc csc_from_coo(size_t rows, size_t cols, nb::object row_ind,
					nb::object col_ind, nb::object values, bool drop_zeros) {
	DlxArray va = dlx_array(values, nb::none());
	if (va.dtype() == dracolix::DType::Bool)
		throw nb::type_error("sparse matrices do not support bool entries");
	std::vector<size_t> ri = idx_list(row_ind);
	std::vector<size_t> ci = idx_list(col_ind);
	if (ri.size() != ci.size() || ci.size() != va.size())
		throw nb::value_error("COO triplet arrays must have equal length");
	switch (va.dtype()) {
	case dracolix::DType::F32:
		return coo_to_csc_typed<float>(rows, cols, std::move(ri), std::move(ci),
									   dlx_values<float>(va), drop_zeros);
	case dracolix::DType::F64:
		return coo_to_csc_typed<double>(rows, cols, std::move(ri),
										std::move(ci), dlx_values<double>(va),
										drop_zeros);
	case dracolix::DType::I32:
		return coo_to_csc_typed<int32_t>(rows, cols, std::move(ri),
										 std::move(ci), dlx_values<int32_t>(va),
										 drop_zeros);
	case dracolix::DType::I64:
		return coo_to_csc_typed<int64_t>(rows, cols, std::move(ri),
										 std::move(ci), dlx_values<int64_t>(va),
										 drop_zeros);
	default:
		throw nb::type_error("unsupported sparse dtype");
	}
}

// ----------------------------------------------------------------
// ---- accessors -------------------------------------------------
// ----------------------------------------------------------------

nb::tuple int_tuple2(size_t a, size_t b) {
	nb::tuple t = nb::steal<nb::tuple>(PyTuple_New(2));
	PyTuple_SET_ITEM(t.ptr(), 0, nb::int_((int64_t)a).release().ptr());
	PyTuple_SET_ITEM(t.ptr(), 1, nb::int_((int64_t)b).release().ptr());
	return t;
}

DlxArray idx_to_array(const std::vector<size_t> &v) {
	dracolix::Array<int64_t> a(std::vector<size_t>{v.size()});
	for (size_t i = 0; i < v.size(); ++i)
		a[i] = (int64_t)v[i];
	return DlxArray(std::move(a));
}

template <typename V, typename GetIdx>
nb::object sparse_index_array(const V &m, GetIdx get) {
	return std::visit(
		[&](const auto &mm) -> nb::object {
			return nb::cast<DlxArray>(idx_to_array(get(mm)));
		},
		m.v);
}

nb::object csr_values_array(const DlxCsr &m) {
	return std::visit(
		[](const auto &mm) -> nb::object {
			using T = typename matrix_scalar<std::decay_t<decltype(mm)>>::type;
			const auto &vals = mm.values();
			dracolix::Array<T> a(std::vector<size_t>{vals.size()});
			for (size_t i = 0; i < vals.size(); ++i)
				a[i] = vals[i];
			return nb::cast<DlxArray>(DlxArray(std::move(a)));
		},
		m.v);
}
nb::object csc_values_array(const DlxCsc &m) {
	return std::visit(
		[](const auto &mm) -> nb::object {
			using T = typename matrix_scalar<std::decay_t<decltype(mm)>>::type;
			const auto &vals = mm.values();
			dracolix::Array<T> a(std::vector<size_t>{vals.size()});
			for (size_t i = 0; i < vals.size(); ++i)
				a[i] = vals[i];
			return nb::cast<DlxArray>(DlxArray(std::move(a)));
		},
		m.v);
}

nb::object sparse_at_csr(const DlxCsr &m, int64_t i, int64_t j) {
	return std::visit(
		[&](const auto &mm) { return nb::cast(mm.at((size_t)i, (size_t)j)); },
		m.v);
}
nb::object sparse_at_csc(const DlxCsc &m, int64_t i, int64_t j) {
	return std::visit(
		[&](const auto &mm) { return nb::cast(mm.at((size_t)i, (size_t)j)); },
		m.v);
}

nb::object sparse_to_dense_csr(const DlxCsr &m) {
	return std::visit(
		[](const auto &mm) {
			return nb::cast<DlxArray>(DlxArray(mm.to_dense()));
		},
		m.v);
}
nb::object sparse_to_dense_csc(const DlxCsc &m) {
	return std::visit(
		[](const auto &mm) {
			return nb::cast<DlxArray>(DlxArray(mm.to_dense()));
		},
		m.v);
}

nb::object csr_transpose(const DlxCsr &m) {
	return std::visit(
		[](const auto &mm) { return nb::cast<DlxCsc>(DlxCsc(mm.transpose())); },
		m.v);
}
nb::object csc_transpose(const DlxCsc &m) {
	return std::visit(
		[](const auto &mm) { return nb::cast<DlxCsr>(DlxCsr(mm.transpose())); },
		m.v);
}
nb::object csr_to_csc(const DlxCsr &m) {
	return std::visit(
		[](const auto &mm) { return nb::cast<DlxCsc>(DlxCsc(mm.to_csc())); },
		m.v);
}
nb::object csc_to_csr(const DlxCsc &m) {
	return std::visit(
		[](const auto &mm) { return nb::cast<DlxCsr>(DlxCsr(mm.to_csr())); },
		m.v);
}
nb::object csr_to_csr(const DlxCsr &m) {
	return std::visit(
		[](const auto &mm) { return nb::cast<DlxCsr>(DlxCsr(mm)); }, m.v);
}
nb::object csc_to_csc(const DlxCsc &m) {
	return std::visit(
		[](const auto &mm) { return nb::cast<DlxCsc>(DlxCsc(mm)); }, m.v);
}

// ---------------------------------------------------------------------
// ---- matmul ---------------------------------------------------------
// ---------------------------------------------------------------------

template <typename T>
nb::object csr_matmul_typed(const dracolix::CsrMatrix<T> &A, nb::object other) {
	if (is_csr(other)) {
		auto &B = *nb::cast<DlxCsr *>(other);
		const auto *Bm = B.as<T>();
		if (!Bm)
			throw nb::type_error("sparse @ sparse requires matching dtypes");
		return nb::cast<DlxCsr>(DlxCsr(A.matmul(*Bm)));
	}
	if (is_csc(other)) {
		auto &C = *nb::cast<DlxCsc *>(other);
		const auto *Cm = C.as<T>();
		if (!Cm)
			throw nb::type_error("sparse @ sparse requires matching dtypes");
		return nb::cast<DlxCsr>(DlxCsr(A.matmul(Cm->to_csr())));
	}
	bool dense_operand = is_array(other) || PyObject_CheckBuffer(other.ptr()) ||
						 PyList_Check(other.ptr()) ||
						 PyTuple_Check(other.ptr());
	if (dense_operand) {
		DlxArray b = dlx_array(other, nb::none());
		if (b.dtype() == dracolix::DType::Bool)
			throw nb::type_error("sparse @ dense does not support bool arrays");
		const auto *bd = b.as<T>();
		if (!bd)
			throw nb::type_error("sparse @ dense requires matching dtypes");
		if (bd->ndim() == 1)
			return nb::cast<DlxArray>(DlxArray(A.matvec(*bd)));
		if (bd->ndim() == 2) {
			auto B = dracolix::CsrMatrix<T>::from_dense(*bd);
			return nb::cast<DlxArray>(DlxArray(A.matmul(B).to_dense()));
		}
		throw nb::type_error(
			"sparse @ dense: dense operand must be 1-D or 2-D");
	}
	return nb::not_implemented();
}

nb::object csr_matmul(const DlxCsr &A, nb::object other) {
	return std::visit(
		[&](const auto &mm) -> nb::object {
			using T = typename matrix_scalar<std::decay_t<decltype(mm)>>::type;
			return csr_matmul_typed<T>(mm, other);
		},
		A.v);
}

nb::object csc_matmul(const DlxCsc &C, nb::object other) {
	// CSC has no matmul kernel of its own: materialize CSR (same matrix)
	// and delegate. Correctness is preserved; the CSR path is the reference
	// impl.
	return std::visit(
		[&](const auto &mm) -> nb::object {
			using T = typename matrix_scalar<std::decay_t<decltype(mm)>>::type;
			dracolix::CsrMatrix<T> csr = mm.to_csr();
			return csr_matmul_typed<T>(csr, other);
		},
		C.v);
}

nb::object sparse_rmatmul(nb::object x, nb::object y) {
	if (!(is_array(x) || PyObject_CheckBuffer(x.ptr()) ||
		  PyList_Check(x.ptr()) || PyTuple_Check(x.ptr())))
		return nb::not_implemented();
	DlxArray a = dlx_array(x, nb::none());
	if (a.dtype() == dracolix::DType::Bool)
		throw nb::type_error("dense @ sparse does not support bool arrays");
	if (a.dtype() == dracolix::DType::C64 || a.dtype() == dracolix::DType::C128)
		throw nb::type_error("sparse matrices do not support complex values");
	if (a.ndim() != 2)
		throw nb::type_error("dense @ sparse requires a 2-D dense array");
	if (is_csr(y)) {
		DlxCsr &s = *nb::cast<DlxCsr *>(y);
		return std::visit(
			[&](const auto &arr) -> nb::object {
				using T =
					typename array_scalar<std::decay_t<decltype(arr)>>::type;
				if constexpr (std::is_same_v<T, bool> ||
							  dracolix::is_complex_v<T>) {
					throw nb::type_error("dense @ sparse does not support bool "
										 "or complex arrays");
				} else if constexpr (!is_sparse_scalar_v<T>) {
					throw nb::type_error("unsupported sparse dtype");
				} else {
					const auto *Bm = s.as<T>();
					if (!Bm)
						throw nb::type_error(
							"dense @ sparse requires matching dtypes");
					auto A = dracolix::CsrMatrix<T>::from_dense(arr);
					return nb::cast<DlxArray>(
						DlxArray(A.matmul(*Bm).to_dense()));
				}
			},
			a.v);
	}
	if (is_csc(y)) {
		DlxCsc &s = *nb::cast<DlxCsc *>(y);
		return std::visit(
			[&](const auto &arr) -> nb::object {
				using T =
					typename array_scalar<std::decay_t<decltype(arr)>>::type;
				if constexpr (std::is_same_v<T, bool> ||
							  dracolix::is_complex_v<T>) {
					throw nb::type_error("dense @ sparse does not support bool "
										 "or complex arrays");
				} else if constexpr (!is_sparse_scalar_v<T>) {
					throw nb::type_error("unsupported sparse dtype");
				} else {
					const auto *Cm = s.as<T>();
					if (!Cm)
						throw nb::type_error(
							"dense @ sparse requires matching dtypes");
					dracolix::CsrMatrix<T> B = Cm->to_csr();
					auto A = dracolix::CsrMatrix<T>::from_dense(arr);
					return nb::cast<DlxArray>(DlxArray(A.matmul(B).to_dense()));
				}
			},
			a.v);
	}
	return nb::not_implemented();
}

// ---------------------------------------------------------------------------
// Linear algebra wrappers (module-level API mirroring the C++ core)
// ---------------------------------------------------------------------------

nb::object dlx_matvec(nb::object x, nb::object y) {
	std::optional<DlxArray> A, b;
	if (!coerce_operand(x, A) || !coerce_operand(y, b))
		throw nb::type_error("matvec expected arrays");
	promote_complex_mismatch(A, b, "matvec");
	return std::visit(
		[&](const auto &a) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(a)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				throw nb::type_error("matvec does not support bool arrays");
			} else {
				const auto &bv = *b->as<T>();
				return nb::cast<DlxArray>(
					DlxArray(dracolix::linalg::matvec(a, bv)));
			}
		},
		A->v);
}

nb::object dlx_dot(nb::object x, nb::object y) {
	std::optional<DlxArray> A, b;
	if (!coerce_operand(x, A) || !coerce_operand(y, b))
		throw nb::type_error("dot expected arrays");
	promote_complex_mismatch(A, b, "dot");
	return std::visit(
		[&](const auto &a) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(a)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				throw nb::type_error("dot does not support bool arrays");
			} else {
				const auto &bv = *b->as<T>();
				return scalar_to_py<T>(dracolix::linalg::dot(a, bv));
			}
		},
		A->v);
}

double dlx_norm(const DlxArray &a, int p) {
	return std::visit(
		[&](const auto &arr) { return dracolix::linalg::norm(arr, p); }, a.v);
}

nb::object dlx_diagonal(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) {
			return nb::cast<DlxArray>(
				DlxArray(dracolix::linalg::diagonal(arr)));
		},
		a.v);
}

nb::object dlx_diag(const DlxArray &d) {
	return std::visit(
		[](const auto &arr) {
			return nb::cast<DlxArray>(DlxArray(dracolix::linalg::diag(arr)));
		},
		d.v);
}

nb::object dlx_lu(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				throw nb::type_error("lu does not support bool arrays");
			} else {
				auto r = dracolix::decomp::lu(arr);
				return nb::make_tuple(
					nb::cast<DlxArray>(DlxArray(std::move(r.L))),
					nb::cast<DlxArray>(DlxArray(std::move(r.U))),
					nb::cast<DlxArray>(idx_to_array(r.perm)));
			}
		},
		a.v);
}

nb::object dlx_solve(nb::object x, nb::object y) {
	std::optional<DlxArray> A, b;
	if (!coerce_operand(x, A) || !coerce_operand(y, b))
		throw nb::type_error("solve expected arrays");
	promote_complex_mismatch(A, b, "solve");
	return std::visit(
		[&](const auto &a) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(a)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				throw nb::type_error("solve does not support bool arrays");
			} else {
				const auto &bv = *b->as<T>();
				return nb::cast<DlxArray>(
					DlxArray(dracolix::decomp::solve(a, bv)));
			}
		},
		A->v);
}

nb::object dlx_inverse(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				throw nb::type_error("inverse does not support bool arrays");
			} else {
				return nb::cast<DlxArray>(
					DlxArray(dracolix::decomp::inverse(arr)));
			}
		},
		a.v);
}

nb::object dlx_det(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				throw nb::type_error("det does not support bool arrays");
			} else {
				return scalar_to_py<T>(dracolix::decomp::determinant(arr));
			}
		},
		a.v);
}

int64_t dlx_rank(const DlxArray &a, double tol) {
	return std::visit(
		[&](const auto &arr) -> int64_t {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				throw nb::type_error("rank does not support bool arrays");
			} else {
				return (int64_t)dracolix::decomp::rank(arr, tol);
			}
		},
		a.v);
}

nb::object dlx_qr(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				throw nb::type_error("qr does not support bool arrays");
			} else {
				auto r = dracolix::decomp::qr(arr);
				return nb::make_tuple(
					nb::cast<DlxArray>(DlxArray(std::move(r.Q))),
					nb::cast<DlxArray>(DlxArray(std::move(r.R))));
			}
		},
		a.v);
}

nb::object dlx_cholesky(const DlxArray &a) {
	return std::visit(
		[](const auto &arr) -> nb::object {
			using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
			if constexpr (std::is_same_v<T, bool>) {
				throw nb::type_error("cholesky does not support bool arrays");
			} else {
				return nb::cast<DlxArray>(
					DlxArray(dracolix::decomp::cholesky(arr)));
			}
		},
		a.v);
}

nb::object dlx_eig(const DlxArray &a) {
	dracolix::DType dt = a.dtype();
	if (dt != dracolix::DType::F32 && dt != dracolix::DType::F64)
		throw nb::type_error(
			"eig requires a real float array (f32 or f64); complex is not "
			"supported yet");
	// The kernel is double-only; promote f32 to f64.
	DlxArray work(a.v);
	if (dt == dracolix::DType::F32)
		work = cast_dtype(a, dracolix::DType::F64);
	const auto *ad = work.as<double>();
	auto r = dracolix::eigen::eig_sym(*ad);
	return nb::make_tuple(nb::cast<DlxArray>(DlxArray(std::move(r.values))),
						  nb::cast<DlxArray>(DlxArray(std::move(r.vectors))));
}

nb::object dlx_svd(const DlxArray &a) {
	dracolix::DType dt = a.dtype();
	if (dt != dracolix::DType::F32 && dt != dracolix::DType::F64)
		throw nb::type_error(
			"svd requires a real float array (f32 or f64); complex is not "
			"supported yet");
	DlxArray work(a.v);
	if (dt == dracolix::DType::F32)
		work = cast_dtype(a, dracolix::DType::F64);
	const auto *ad = work.as<double>();
	auto r = dracolix::svd::svd(*ad);
	return nb::make_tuple(nb::cast<DlxArray>(DlxArray(std::move(r.U))),
						  nb::cast<DlxArray>(DlxArray(std::move(r.S))),
						  nb::cast<DlxArray>(DlxArray(std::move(r.Vt))));
}

nb::object csr_cg_solve(const DlxCsr &m, nb::object bobj) {
	DlxArray b = dlx_array(bobj, nb::none());
	if (b.dtype() == dracolix::DType::Bool)
		throw nb::type_error("cg_solve does not support bool arrays");
	return std::visit(
		[&](const auto &mm) -> nb::object {
			using T = typename matrix_scalar<std::decay_t<decltype(mm)>>::type;
			const auto *bd = b.as<T>();
			if (!bd)
				throw nb::type_error("cg_solve requires matching dtypes");
			return nb::cast<DlxArray>(DlxArray(dracolix::cg_solve(mm, *bd)));
		},
		m.v);
}

nb::object csc_cg_solve(const DlxCsc &m, nb::object bobj) {
	DlxArray b = dlx_array(bobj, nb::none());
	if (b.dtype() == dracolix::DType::Bool)
		throw nb::type_error("cg_solve does not support bool arrays");
	return std::visit(
		[&](const auto &mm) -> nb::object {
			using T = typename matrix_scalar<std::decay_t<decltype(mm)>>::type;
			const auto *bd = b.as<T>();
			if (!bd)
				throw nb::type_error("cg_solve requires matching dtypes");
			dracolix::CsrMatrix<T> csr = mm.to_csr();
			return nb::cast<DlxArray>(DlxArray(dracolix::cg_solve(csr, *bd)));
		},
		m.v);
}

} // namespace

// ---------------------------------------------------------------------------
// module registration
// ---------------------------------------------------------------------------

NB_MODULE(_dracolix_nb, m) {
	m.doc() =
		"DracoLIX - a scientific computing runtime from Draconis Engineering"
		"The Python API is written from scratch with Nanobind"
		"(interoperates via the PEP 3118 buffer protocol).";

	m.attr("__version__") = DRACOLIX_PY_VERSION;

	nb::class_<DlxDType>(m, "DType")
		.def("__eq__",
			 [](const DlxDType &a, const nb::object &other) -> nb::object {
				 if (!is_dtype(other))
					 return nb::not_implemented();
				 return nb::cast(a.dt == nb::cast<DlxDType *>(other)->dt);
			 })
		.def("__ne__",
			 [](const DlxDType &a, const nb::object &other) -> nb::object {
				 if (!is_dtype(other))
					 return nb::not_implemented();
				 return nb::cast(a.dt != nb::cast<DlxDType *>(other)->dt);
			 })
		.def("__hash__", [](const DlxDType &a) { return (size_t)a.dt; })
		.def("__repr__",
			 [](const DlxDType &a) {
				 return "dtype('" + dracolix::dtype_name(a.dt) + "')";
			 })
		.def_prop_ro("name", [](const DlxDType &a) {
			return dracolix::dtype_name(a.dt);
		});

	nb::class_<DlxArray> arr_cls(m, "Array");
	arr_cls
		.def_prop_ro(
			"dtype",
			[](const DlxArray &a) { return dtype_singleton(a.dtype()); })
		.def_prop_ro(
			"shape",
			[](const DlxArray &a) {
				nb::tuple t =
					nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t)a.ndim()));
				for (size_t d = 0; d < a.ndim(); ++d)
					PyTuple_SET_ITEM(
						t.ptr(), (Py_ssize_t)d,
						nb::int_((int64_t)a.shape()[d]).release().ptr());
				return t;
			})
		.def_prop_ro(
			"strides",
			[](const DlxArray &a) {
				nb::tuple t =
					nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t)a.ndim()));
				for (size_t d = 0; d < a.ndim(); ++d)
					PyTuple_SET_ITEM(
						t.ptr(), (Py_ssize_t)d,
						nb::int_((int64_t)a.strides()[d]).release().ptr());
				return t;
			})
		.def_prop_ro("ndim",
					 [](const DlxArray &a) { return (Py_ssize_t)a.ndim(); })
		.def_prop_ro("size",
					 [](const DlxArray &a) { return (Py_ssize_t)a.size(); })
		.def_prop_ro("itemsize",
					 [](const DlxArray &a) { return (Py_ssize_t)a.itemsize(); })
		.def_prop_ro("layout",
					 [](const DlxArray &a) {
						 return a.layout() == dracolix::Layout::RowMajor
									? "row-major"
									: "col-major";
					 })
		.def_prop_ro("T",
					 [](const DlxArray &a) -> DlxArray {
						 if (a.ndim() != 2)
							 throw nb::value_error(".T requires 2-D");
						 return dlx_transpose(a);
					 })

		.def("__len__",
			 [](const DlxArray &a) {
				 return (Py_ssize_t)(a.ndim() ? a.shape()[0] : 1);
			 })
		.def("__bool__",
			 [](const DlxArray &a) -> bool {
				 if (a.size() != 1)
					 throw nb::type_error(
						 "the truth value of an array with more "
						 "than one element is ambiguous");
				 return std::visit(
					 [](const auto &arr) -> bool {
						 using T = typename array_scalar<
							 std::decay_t<decltype(arr)>>::type;
						 if constexpr (dracolix::is_complex_v<T>)
							 return arr[0] != T{};
						 else
							 return (bool)arr[0];
					 },
					 a.v);
			 })
		.def(
			"__iter__",
			[scope = nb::handle(m)](const DlxArray &self) {
				return make_iter(self, scope);
			},
			nb::keep_alive<0, 1>())
		.def("__getitem__", &dlx_getitem)
		.def("__setitem__", &dlx_setitem)
		.def("__repr__", &dlx_repr)
		.def("__str__", &dlx_repr)

		.def("copy", &dlx_copy)
		.def("astype", &dlx_astype, nb::arg("dtype"))
		.def("reshape", &dlx_reshape, nb::arg("shape"))
		.def("transpose", &dlx_transpose)
		.def("tolist", &dlx_tolist)
		.def("item", &dlx_item, nb::arg("index") = nb::none())
		.def("__index__", &dlx_index)

		.def("sum", [](const DlxArray &a) { return dlx_sum(a); })
		.def(
			"sum",
			[](const DlxArray &a, int64_t axis) {
				return dlx_sum_axis(a, axis);
			},
			nb::arg("axis"))
		.def("min", [](const DlxArray &a) { return dlx_min(a); })
		.def(
			"min",
			[](const DlxArray &a, int64_t axis) {
				return dlx_min_axis(a, axis);
			},
			nb::arg("axis"))
		.def("max", [](const DlxArray &a) { return dlx_max(a); })
		.def(
			"max",
			[](const DlxArray &a, int64_t axis) {
				return dlx_max_axis(a, axis);
			},
			nb::arg("axis"))
		.def("mean", [](const DlxArray &a) { return dlx_mean(a); })
		.def(
			"mean",
			[](const DlxArray &a, int64_t axis) {
				return dlx_mean_axis(a, axis);
			},
			nb::arg("axis"))
		.def("prod", [](const DlxArray &a) { return dlx_prod(a); })
		.def(
			"prod",
			[](const DlxArray &a, int64_t axis) {
				return dlx_prod_axis(a, axis);
			},
			nb::arg("axis"))
		.def(
			"std",
			[](const DlxArray &a, nb::object axis, int64_t ddof) {
				return dlx_std_dispatch(a, axis, ddof);
			},
			nb::arg("axis") = nb::none(), nb::arg("ddof") = 0)
		.def(
			"var",
			[](const DlxArray &a, nb::object axis, int64_t ddof) {
				return dlx_var_dispatch(a, axis, ddof);
			},
			nb::arg("axis") = nb::none(), nb::arg("ddof") = 0)
		.def("argmin", [](const DlxArray &a) { return dlx_argmin(a); })
		.def(
			"argmin",
			[](const DlxArray &a, int64_t axis) {
				return dlx_argmin_axis(a, axis);
			},
			nb::arg("axis"))
		.def("argmax", [](const DlxArray &a) { return dlx_argmax(a); })
		.def(
			"argmax",
			[](const DlxArray &a, int64_t axis) {
				return dlx_argmax_axis(a, axis);
			},
			nb::arg("axis"))
		.def("any", [](const DlxArray &a) { return dlx_any(a); })
		.def(
			"any",
			[](const DlxArray &a, int64_t axis) {
				return dlx_any_axis(a, axis);
			},
			nb::arg("axis"))
		.def("all", [](const DlxArray &a) { return dlx_all(a); })
		.def(
			"all",
			[](const DlxArray &a, int64_t axis) {
				return dlx_all_axis(a, axis);
			},
			nb::arg("axis"))

		// ---- complex-valued helpers ----
		.def_prop_ro("real", &dlx_real)
		.def_prop_ro("imag", &dlx_imag)
		.def("conj", &dlx_conj)
		.def("conjugate", &dlx_conj)
		.def("abs", &dlx_abs)
		.def("angle", &dlx_angle)
		.def("__abs__", &dlx_abs)

		.def("__add__", &dlx_add)
		.def("__radd__", &dlx_add)
		.def("__sub__", &dlx_sub)
		.def("__rsub__",
			 [](nb::object x, nb::object y) { return dlx_sub(y, x); })
		.def("__mul__", &dlx_mul)
		.def("__rmul__", &dlx_mul)
		.def("__truediv__", &dlx_div)
		.def("__rtruediv__",
			 [](nb::object x, nb::object y) { return dlx_div(y, x); })
		.def("__pow__", &dlx_pow)
		.def("__rpow__", &dlx_rpow)
		.def("__floordiv__", &dlx_floordiv)
		.def("__rfloordiv__", &dlx_rfloordiv)
		.def("__mod__", &dlx_mod)
		.def("__rmod__", &dlx_rmod)
		.def("__and__", &dlx_and)
		.def("__rand__", &dlx_and)
		.def("__or__", &dlx_or)
		.def("__ror__", &dlx_or)
		.def("__xor__", &dlx_xor)
		.def("__rxor__", &dlx_xor)
		.def("__lshift__", &dlx_lshift)
		.def("__rlshift__", &dlx_rlshift)
		.def("__rshift__", &dlx_rshift)
		.def("__rrshift__", &dlx_rrshift)
		.def("__pos__", &dlx_pos)
		.def("__neg__", &dlx_neg)
		.def("__invert__", &dlx_invert)
		.def("__eq__", &dlx_eq)
		.def("__ne__", &dlx_ne)
		.def("__lt__", &dlx_lt)
		.def("__le__", &dlx_le)
		.def("__gt__", &dlx_gt)
		.def("__ge__", &dlx_ge)
		.def("__matmul__", &dlx_matmul)
		.def("__rmatmul__",
			 [](nb::object x, nb::object y) { return dlx_matmul(y, x); });

	// ---- sparse matrices ----------------------------------------------------
	nb::class_<DlxCsr> csr_cls(m, "CsrMatrix");
	csr_cls
		.def(
			"__init__",
			[](DlxCsr *self, nb::object data, nb::object dtype) {
				DlxArray a = dlx_array(data, dtype);
				new (self) DlxCsr(csr_from_dense(a));
			},
			nb::arg("data"), nb::arg("dtype") = nb::none())
		.def_static("from_coo", &csr_from_coo, nb::arg("rows"), nb::arg("cols"),
					nb::arg("row_ind"), nb::arg("col_ind"), nb::arg("values"),
					nb::arg("drop_zeros") = true)
		.def_prop_ro(
			"dtype",
			[](const DlxCsr &m) { return dtype_singleton(sparse_dtype(m)); })
		.def_prop_ro("shape",
					 [](const DlxCsr &m) {
						 return std::visit(
							 [](const auto &mm) {
								 return int_tuple2(mm.rows(), mm.cols());
							 },
							 m.v);
					 })
		.def_prop_ro(
			"rows",
			[](const DlxCsr &m) {
				return std::visit(
					[](const auto &mm) { return (Py_ssize_t)mm.rows(); }, m.v);
			})
		.def_prop_ro(
			"cols",
			[](const DlxCsr &m) {
				return std::visit(
					[](const auto &mm) { return (Py_ssize_t)mm.cols(); }, m.v);
			})
		.def_prop_ro(
			"nnz",
			[](const DlxCsr &m) {
				return std::visit(
					[](const auto &mm) { return (Py_ssize_t)mm.nnz(); }, m.v);
			})
		.def_prop_ro("row_ptr",
					 [](const DlxCsr &m) {
						 return std::visit(
							 [](const auto &mm) {
								 return nb::cast<DlxArray>(
									 idx_to_array(mm.row_ptr()));
							 },
							 m.v);
					 })
		.def_prop_ro("col_ind",
					 [](const DlxCsr &m) {
						 return std::visit(
							 [](const auto &mm) {
								 return nb::cast<DlxArray>(
									 idx_to_array(mm.col_ind()));
							 },
							 m.v);
					 })
		.def_prop_ro("values",
					 [](const DlxCsr &m) { return csr_values_array(m); })
		.def_prop_ro("T", [](const DlxCsr &m) { return csr_transpose(m); })
		.def("to_dense", [](const DlxCsr &m) { return sparse_to_dense_csr(m); })
		.def("to_csr", [](const DlxCsr &m) { return csr_to_csr(m); })
		.def("to_csc", [](const DlxCsr &m) { return csr_to_csc(m); })
		.def("at", [](const DlxCsr &m, int64_t i,
					  int64_t j) { return sparse_at_csr(m, i, j); })
		.def(
			"solve",
			[](const DlxCsr &m, nb::object b) { return csr_cg_solve(m, b); },
			nb::arg("b"))
		.def("__matmul__",
			 [](const DlxCsr &a, nb::object other) {
				 return csr_matmul(a, other);
			 })
		.def("__rmatmul__",
			 [](nb::object x, nb::object y) -> nb::object {
				 if (is_csr(x) || is_csc(x))
					 return sparse_rmatmul(y, x);
				 if (is_csr(y) || is_csc(y))
					 return sparse_rmatmul(x, y);
				 return nb::not_implemented();
			 })
		.def_prop_ro_static("__array_ufunc__",
							[](nb::handle) { return nb::none(); })
		.def("__repr__", [](const DlxCsr &m) {
			return std::visit(
				[](const auto &mm) {
					return "csr_matrix(" + std::to_string(mm.rows()) + " x " +
						   std::to_string(mm.cols()) +
						   ", nnz=" + std::to_string(mm.nnz()) + ")";
				},
				m.v);
		});

	nb::class_<DlxCsc> csc_cls(m, "CscMatrix");
	csc_cls
		.def(
			"__init__",
			[](DlxCsc *self, nb::object data, nb::object dtype) {
				DlxArray a = dlx_array(data, dtype);
				new (self) DlxCsc(csc_from_dense(a));
			},
			nb::arg("data"), nb::arg("dtype") = nb::none())
		.def_static("from_coo", &csc_from_coo, nb::arg("rows"), nb::arg("cols"),
					nb::arg("row_ind"), nb::arg("col_ind"), nb::arg("values"),
					nb::arg("drop_zeros") = true)
		.def_prop_ro(
			"dtype",
			[](const DlxCsc &m) { return dtype_singleton(sparse_dtype(m)); })
		.def_prop_ro("shape",
					 [](const DlxCsc &m) {
						 return std::visit(
							 [](const auto &mm) {
								 return int_tuple2(mm.rows(), mm.cols());
							 },
							 m.v);
					 })
		.def_prop_ro(
			"rows",
			[](const DlxCsc &m) {
				return std::visit(
					[](const auto &mm) { return (Py_ssize_t)mm.rows(); }, m.v);
			})
		.def_prop_ro(
			"cols",
			[](const DlxCsc &m) {
				return std::visit(
					[](const auto &mm) { return (Py_ssize_t)mm.cols(); }, m.v);
			})
		.def_prop_ro(
			"nnz",
			[](const DlxCsc &m) {
				return std::visit(
					[](const auto &mm) { return (Py_ssize_t)mm.nnz(); }, m.v);
			})
		.def_prop_ro("col_ptr",
					 [](const DlxCsc &m) {
						 return std::visit(
							 [](const auto &mm) {
								 return nb::cast<DlxArray>(
									 idx_to_array(mm.col_ptr()));
							 },
							 m.v);
					 })
		.def_prop_ro("row_ind",
					 [](const DlxCsc &m) {
						 return std::visit(
							 [](const auto &mm) {
								 return nb::cast<DlxArray>(
									 idx_to_array(mm.row_ind()));
							 },
							 m.v);
					 })
		.def_prop_ro("values",
					 [](const DlxCsc &m) { return csc_values_array(m); })
		.def_prop_ro("T", [](const DlxCsc &m) { return csc_transpose(m); })
		.def("to_dense", [](const DlxCsc &m) { return sparse_to_dense_csc(m); })
		.def("to_csr", [](const DlxCsc &m) { return csc_to_csr(m); })
		.def("to_csc", [](const DlxCsc &m) { return csc_to_csc(m); })
		.def("at", [](const DlxCsc &m, int64_t i,
					  int64_t j) { return sparse_at_csc(m, i, j); })
		.def(
			"solve",
			[](const DlxCsc &m, nb::object b) { return csc_cg_solve(m, b); },
			nb::arg("b"))
		.def("__matmul__",
			 [](const DlxCsc &a, nb::object other) {
				 return csc_matmul(a, other);
			 })
		.def("__rmatmul__",
			 [](nb::object x, nb::object y) -> nb::object {
				 if (is_csr(x) || is_csc(x))
					 return sparse_rmatmul(y, x);
				 if (is_csr(y) || is_csc(y))
					 return sparse_rmatmul(x, y);
				 return nb::not_implemented();
			 })
		.def_prop_ro_static("__array_ufunc__",
							[](nb::handle) { return nb::none(); })
		.def("__repr__", [](const DlxCsc &m) {
			return std::visit(
				[](const auto &mm) {
					return "csc_matrix(" + std::to_string(mm.rows()) + " x " +
						   std::to_string(mm.cols()) +
						   ", nnz=" + std::to_string(mm.nnz()) + ")";
				},
				m.v);
		});

	// singletons: Python-owned module attributes (no C++ static refs - see
	// dtype_singleton()). Keep the objects stable, so "dlx.f32 is dlx.f32".
	m.attr("f32") = nb::cast<DlxDType>(DlxDType(dracolix::DType::F32));
	m.attr("f64") = nb::cast<DlxDType>(DlxDType(dracolix::DType::F64));
	m.attr("i8") = nb::cast<DlxDType>(DlxDType(dracolix::DType::I8));
	m.attr("i16") = nb::cast<DlxDType>(DlxDType(dracolix::DType::I16));
	m.attr("i32") = nb::cast<DlxDType>(DlxDType(dracolix::DType::I32));
	m.attr("i64") = nb::cast<DlxDType>(DlxDType(dracolix::DType::I64));
	m.attr("u8") = nb::cast<DlxDType>(DlxDType(dracolix::DType::U8));
	m.attr("u16") = nb::cast<DlxDType>(DlxDType(dracolix::DType::U16));
	m.attr("u32") = nb::cast<DlxDType>(DlxDType(dracolix::DType::U32));
	m.attr("u64") = nb::cast<DlxDType>(DlxDType(dracolix::DType::U64));
	m.attr("c64") = nb::cast<DlxDType>(DlxDType(dracolix::DType::C64));
	m.attr("c128") = nb::cast<DlxDType>(DlxDType(dracolix::DType::C128));
	m.attr("bool") = nb::cast<DlxDType>(DlxDType(dracolix::DType::Bool));

	// factories
	m.def("array", &dlx_array, nb::arg("data"), nb::arg("dtype") = nb::none());
	m.def("asarray", &dlx_asarray, nb::arg("data"),
		  nb::arg("dtype") = nb::none());
	m.def("zeros", &dlx_zeros, nb::arg("shape"), nb::arg("dtype") = nb::none());
	m.def("ones", &dlx_ones, nb::arg("shape"), nb::arg("dtype") = nb::none());
	m.def("empty", &dlx_empty, nb::arg("shape"), nb::arg("dtype") = nb::none());
	m.def("full", &dlx_full, nb::arg("shape"), nb::arg("fill_value"),
		  nb::arg("dtype") = nb::none());
	m.def("arange", &dlx_arange, nb::arg("start"), nb::arg("stop") = nb::none(),
		  nb::arg("step") = nb::none());
	m.def("linspace", &dlx_linspace, nb::arg("start"), nb::arg("stop"),
		  nb::arg("num") = 50, nb::arg("endpoint") = true,
		  nb::arg("dtype") = nb::none());

	// ---- linear algebra ----------------------------------------------------
	m.def(
		"matmul",
		[](nb::object A, nb::object B) -> nb::object {
			nb::object r = dlx_matmul(A, B);
			if (r.ptr() == Py_NotImplemented)
				throw nb::type_error("matmul expected arrays");
			return r;
		},
		nb::arg("A"), nb::arg("B"));
	m.def("matvec", &dlx_matvec, nb::arg("A"), nb::arg("x"));
	m.def("dot", &dlx_dot, nb::arg("a"), nb::arg("b"));
	m.def("norm", &dlx_norm, nb::arg("a"), nb::arg("p") = 2);
	m.def("diagonal", &dlx_diagonal, nb::arg("A"));
	m.def("diag", &dlx_diag, nb::arg("d"));

	// ---- complex-valued helpers -------------------------------------------
	m.def("real", &dlx_real, nb::arg("a"));
	m.def("imag", &dlx_imag, nb::arg("a"));
	m.def("conj", &dlx_conj, nb::arg("a"));
	m.def("conjugate", &dlx_conj, nb::arg("a"));
	m.def("abs", &dlx_abs, nb::arg("a"));
	m.def("absolute", &dlx_abs, nb::arg("a"));
	m.def("angle", &dlx_angle, nb::arg("a"));

	// ---- elementwise math ---------------------------------------------------
	m.def(
		"sqrt", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Sqrt); },
		nb::arg("a"));
	m.def(
		"exp", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Exp); },
		nb::arg("a"));
	m.def(
		"expm1", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Expm1); },
		nb::arg("a"));
	m.def(
		"log", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Log); },
		nb::arg("a"));
	m.def(
		"log1p", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Log1p); },
		nb::arg("a"));
	m.def(
		"log2", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Log2); },
		nb::arg("a"));
	m.def(
		"log10", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Log10); },
		nb::arg("a"));
	m.def(
		"sin", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Sin); },
		nb::arg("a"));
	m.def(
		"cos", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Cos); },
		nb::arg("a"));
	m.def(
		"tan", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Tan); },
		nb::arg("a"));
	m.def(
		"asin", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Asin); },
		nb::arg("a"));
	m.def(
		"acos", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Acos); },
		nb::arg("a"));
	m.def(
		"atan", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Atan); },
		nb::arg("a"));
	m.def(
		"sinh", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Sinh); },
		nb::arg("a"));
	m.def(
		"cosh", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Cosh); },
		nb::arg("a"));
	m.def(
		"tanh", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Tanh); },
		nb::arg("a"));
	m.def(
		"floor", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Floor); },
		nb::arg("a"));
	m.def(
		"ceil", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Ceil); },
		nb::arg("a"));
	m.def(
		"trunc", [](const DlxArray &a) { return dlx_ufunc(a, UFunc::Trunc); },
		nb::arg("a"));
	m.def("square", &dlx_square, nb::arg("a"));
	m.def(
		"isfinite",
		[](const DlxArray &a) { return dlx_predicate(a, Pred::Finite); },
		nb::arg("a"));
	m.def(
		"isnan", [](const DlxArray &a) { return dlx_predicate(a, Pred::Nan); },
		nb::arg("a"));
	m.def(
		"isinf", [](const DlxArray &a) { return dlx_predicate(a, Pred::Inf); },
		nb::arg("a"));

	m.def(
		"maximum",
		[](nb::object x, nb::object y) {
			return dlx_minmax(x, y, MinMax::Maximum);
		},
		nb::arg("x"), nb::arg("y"));
	m.def(
		"minimum",
		[](nb::object x, nb::object y) {
			return dlx_minmax(x, y, MinMax::Minimum);
		},
		nb::arg("x"), nb::arg("y"));

	// ---- decompositions ----------------------------------------------------
	m.def("lu", &dlx_lu, nb::arg("A"));
	m.def("solve", &dlx_solve, nb::arg("A"), nb::arg("b"));
	m.def("inv", &dlx_inverse, nb::arg("A"));
	m.def("inverse", &dlx_inverse, nb::arg("A"));
	m.def("det", &dlx_det, nb::arg("A"));
	m.def("determinant", &dlx_det, nb::arg("A"));
	m.def("rank", &dlx_rank, nb::arg("A"), nb::arg("tol") = 1e-9);
	m.def("qr", &dlx_qr, nb::arg("A"));
	m.def("cholesky", &dlx_cholesky, nb::arg("A"));
	m.def("eig", &dlx_eig, nb::arg("A"));
	m.def("svd", &dlx_svd, nb::arg("A"));

	// ---- sparse solvers ----------------------------------------------------
	m.def("cg_solve", &csr_cg_solve, nb::arg("A"), nb::arg("b"));

	// wire PEP 3118 export slots after the class exists
	g_dlx_type = reinterpret_cast<PyTypeObject *>(nb::type<DlxArray>().ptr());
	g_dtype_type = reinterpret_cast<PyTypeObject *>(nb::type<DlxDType>().ptr());
	install_buffer_slots();

	g_csr_type = reinterpret_cast<PyTypeObject *>(nb::type<DlxCsr>().ptr());
	g_csc_type = reinterpret_cast<PyTypeObject *>(nb::type<DlxCsc>().ptr());
}
