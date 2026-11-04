#pragma once

// Element-wise complex operations: real, imag, conj, abs, angle.
//
// These are defined for every implemented element type. For real/integer
// arrays the imaginary part is zero, conjugation and the real part are
// identity, ``abs`` is the usual absolute value, and ``angle`` is 0 or pi.
//
// Licensed under GPL-3.0+

#include "array.hpp"
#include "dtype.hpp"

#include <cmath>
#include <complex>
#include <type_traits>

namespace dracolix {
namespace cplx {

// Real scalar type underlying T. For std::complex<R> this is R; every other
// (real/integer/bool) type maps to itself.
template <typename T> struct real_type {
	using type = T;
};
template <typename T> struct real_type<std::complex<T>> {
	using type = T;
};
template <typename T> using real_type_t = typename real_type<T>::type;

// Result type of angle(): floating point. Complex<F> yields F, float yields
// float, everything else (double and integers) yields double.
template <typename T> struct angle_type {
	using type = double;
};
template <> struct angle_type<float> {
	using type = float;
};
template <> struct angle_type<double> {
	using type = double;
};
template <typename T> struct angle_type<std::complex<T>> {
	using type = T;
};
template <typename T>
using angle_type_t = typename angle_type<std::remove_cv_t<T>>::type;

// pi in the underlying real type, used by angle() for real inputs.
template <typename R> constexpr R pi_v = R(3.141592653589793238462643383279502884L);

// Element-wise real part.
template <typename T> Array<real_type_t<T>> real(const Array<T> &a) {
	using R = real_type_t<T>;
	Array<R> out(a.shape());
	if constexpr (is_complex_v<T>) {
		for (size_t i = 0; i < a.size(); ++i)
			out[i] = static_cast<R>(a[i].real());
	} else {
		for (size_t i = 0; i < a.size(); ++i)
			out[i] = static_cast<R>(a[i]);
	}
	return out;
}

// Element-wise imaginary part (zeros for real inputs).
template <typename T> Array<real_type_t<T>> imag(const Array<T> &a) {
	using R = real_type_t<T>;
	Array<R> out(a.shape());
	if constexpr (is_complex_v<T>) {
		for (size_t i = 0; i < a.size(); ++i)
			out[i] = static_cast<R>(a[i].imag());
	} else {
		for (size_t i = 0; i < a.size(); ++i)
			out[i] = R{};
	}
	return out;
}

// Element-wise complex conjugate (identity for real inputs).
template <typename T> Array<T> conj(const Array<T> &a) {
	Array<T> out(a.shape());
	if constexpr (is_complex_v<T>) {
		for (size_t i = 0; i < a.size(); ++i)
			out[i] = std::conj(a[i]);
	} else {
		for (size_t i = 0; i < a.size(); ++i)
			out[i] = a[i];
	}
	return out;
}

// Element-wise modulus / absolute value.
template <typename T> Array<real_type_t<T>> abs(const Array<T> &a) {
	using R = real_type_t<T>;
	Array<R> out(a.shape());
	if constexpr (is_complex_v<T>) {
		for (size_t i = 0; i < a.size(); ++i)
			out[i] = static_cast<R>(std::abs(a[i]));
	} else if constexpr (std::is_same_v<T, bool>) {
		for (size_t i = 0; i < a.size(); ++i)
			out[i] = a[i];
	} else {
		for (size_t i = 0; i < a.size(); ++i)
			out[i] = static_cast<R>(std::abs(a[i]));
	}
	return out;
}

// Element-wise argument / phase in radians, in (-pi, pi].
template <typename T> Array<angle_type_t<T>> angle(const Array<T> &a) {
	using R = angle_type_t<T>;
	Array<R> out(a.shape());
	if constexpr (is_complex_v<T>) {
		for (size_t i = 0; i < a.size(); ++i)
			out[i] = static_cast<R>(std::arg(a[i]));
	} else {
		for (size_t i = 0; i < a.size(); ++i) {
			const R v = static_cast<R>(a[i]);
			out[i] = (v < R{}) ? pi_v<R> : R{};
		}
	}
	return out;
}

} // namespace cplx
} // namespace dracolix
