#pragma once

// Compile-time physical units and dimensional analysis.
//
// A Quantity<Dim, T> carries a scalar value and a Dimension encoded in its
// type, so adding metres to seconds is a compile error while multiplication
// and division combine dimensions automatically. Only dimensions are checked;
// unit conversions are ordinary multiplications by scaled unit constants, e.g.
//
//   auto v = 36.0 * units::km / units::h;   // v.value() == 10 (m/s)
//
// Licensed under GPL-3.0+

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace dracolix::units {

// ---------------------------------------------------------------------------
// Dimensions: exponents of the seven SI base dimensions
// ---------------------------------------------------------------------------
template <int L, int M, int Ti, int I, int Th, int N, int J> struct Dimension {
	static constexpr int length = L;
	static constexpr int mass = M;
	static constexpr int time = Ti;
	static constexpr int current = I;
	static constexpr int temperature = Th;
	static constexpr int amount = N;
	static constexpr int luminous_intensity = J;

	static constexpr bool is_dimensionless() {
		return L == 0 && M == 0 && Ti == 0 && I == 0 && Th == 0 && N == 0 &&
			   J == 0;
	}
};

using Dimensionless = Dimension<0, 0, 0, 0, 0, 0, 0>;

// Base dimensions
using Length = Dimension<1, 0, 0, 0, 0, 0, 0>;
using Mass = Dimension<0, 1, 0, 0, 0, 0, 0>;
using Time = Dimension<0, 0, 1, 0, 0, 0, 0>;
using Current = Dimension<0, 0, 0, 1, 0, 0, 0>;
using Temperature = Dimension<0, 0, 0, 0, 1, 0, 0>;
using Amount = Dimension<0, 0, 0, 0, 0, 1, 0>;
using LuminousIntensity = Dimension<0, 0, 0, 0, 0, 0, 1>;

// Derived dimensions
using Area = Dimension<2, 0, 0, 0, 0, 0, 0>;
using Volume = Dimension<3, 0, 0, 0, 0, 0, 0>;
using Velocity = Dimension<1, 0, -1, 0, 0, 0, 0>;
using Acceleration = Dimension<1, 0, -2, 0, 0, 0, 0>;
using Frequency = Dimension<0, 0, -1, 0, 0, 0, 0>;
using Force = Dimension<1, 1, -2, 0, 0, 0, 0>;
using Pressure = Dimension<-1, 1, -2, 0, 0, 0, 0>;
using Energy = Dimension<2, 1, -2, 0, 0, 0, 0>;
using Action = Dimension<2, 1, -1, 0, 0, 0, 0>;
using Power = Dimension<2, 1, -3, 0, 0, 0, 0>;
using Charge = Dimension<0, 0, 1, 1, 0, 0, 0>;
using Voltage = Dimension<2, 1, -3, -1, 0, 0, 0>;
using Capacitance = Dimension<-2, -1, 4, 2, 0, 0, 0>;
using Resistance = Dimension<2, 1, -3, -2, 0, 0, 0>;
using Conductance = Dimension<-2, -1, 3, 2, 0, 0, 0>;
using MagneticFlux = Dimension<2, 1, -2, -1, 0, 0, 0>;
using MagneticFluxDensity = Dimension<0, 1, -2, -1, 0, 0, 0>;
using Inductance = Dimension<2, 1, -2, -2, 0, 0, 0>;
using LuminousFlux = Dimension<0, 0, 0, 0, 0, 0, 1>;
using Illuminance = Dimension<-2, 0, 0, 0, 0, 0, 1>;
using Density = Dimension<-3, 1, 0, 0, 0, 0, 0>;
using DynamicViscosity = Dimension<-1, 1, -1, 0, 0, 0, 0>;
using KinematicViscosity = Dimension<2, 0, -1, 0, 0, 0, 0>;
using MolarConcentration = Dimension<-3, 0, 0, 0, 0, 1, 0>;
using Entropy = Dimension<2, 1, -2, 0, -1, 0, 0>;
using CatalyticActivity = Dimension<0, 0, -1, 0, 0, 1, 0>;
using AbsorbedDose = Dimension<2, 0, -2, 0, 0, 0, 0>;

// ---------------------------------------------------------------------------
// Dimension arithmetic (types)
// ---------------------------------------------------------------------------
template <typename A, typename B>
using dim_mul =
	Dimension<A::length + B::length, A::mass + B::mass, A::time + B::time,
			  A::current + B::current, A::temperature + B::temperature,
			  A::amount + B::amount,
			  A::luminous_intensity + B::luminous_intensity>;

template <typename A, typename B>
using dim_div =
	Dimension<A::length - B::length, A::mass - B::mass, A::time - B::time,
			  A::current - B::current, A::temperature - B::temperature,
			  A::amount - B::amount,
			  A::luminous_intensity - B::luminous_intensity>;

template <typename D, int N>
using dim_pow =
	Dimension<D::length * N, D::mass * N, D::time * N, D::current * N,
			  D::temperature * N, D::amount * N, D::luminous_intensity * N>;

template <typename D>
using dim_sqrt =
	Dimension<D::length / 2, D::mass / 2, D::time / 2, D::current / 2,
			  D::temperature / 2, D::amount / 2, D::luminous_intensity / 2>;

template <typename D> using dim_inv = dim_div<Dimensionless, D>;

// ---------------------------------------------------------------------------
// Quantity
// ---------------------------------------------------------------------------
template <typename Dim, typename T = double> class Quantity {
	static_assert(std::is_arithmetic_v<T>,
				  "Quantity value type must be arithmetic");

  public:
	using dimension = Dim;
	using value_type = T;

	constexpr Quantity() = default;
	constexpr explicit Quantity(T value) : value_(value) {}

	constexpr T value() const { return value_; }

	constexpr Quantity operator-() const { return Quantity(-value_); }

	constexpr Quantity &operator+=(const Quantity &o) {
		value_ += o.value_;
		return *this;
	}
	constexpr Quantity &operator-=(const Quantity &o) {
		value_ -= o.value_;
		return *this;
	}
	constexpr Quantity &operator*=(T s) {
		value_ *= s;
		return *this;
	}
	constexpr Quantity &operator/=(T s) {
		value_ /= s;
		return *this;
	}

  private:
	T value_{};
};

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------
template <typename DA, typename TA, typename DB, typename TB>
constexpr auto operator+(const Quantity<DA, TA> &a, const Quantity<DB, TB> &b) {
	static_assert(std::is_same_v<DA, DB>,
				  "cannot add quantities with different dimensions");
	using R = std::common_type_t<TA, TB>;
	return Quantity<DA, R>(static_cast<R>(a.value()) +
						   static_cast<R>(b.value()));
}

template <typename DA, typename TA, typename DB, typename TB>
constexpr auto operator-(const Quantity<DA, TA> &a, const Quantity<DB, TB> &b) {
	static_assert(std::is_same_v<DA, DB>,
				  "cannot subtract quantities with different dimensions");
	using R = std::common_type_t<TA, TB>;
	return Quantity<DA, R>(static_cast<R>(a.value()) -
						   static_cast<R>(b.value()));
}

template <typename DA, typename TA, typename DB, typename TB>
constexpr auto operator*(const Quantity<DA, TA> &a, const Quantity<DB, TB> &b) {
	using R = std::common_type_t<TA, TB>;
	return Quantity<dim_mul<DA, DB>, R>(static_cast<R>(a.value()) *
										static_cast<R>(b.value()));
}

template <typename DA, typename TA, typename DB, typename TB>
constexpr auto operator/(const Quantity<DA, TA> &a, const Quantity<DB, TB> &b) {
	using R = std::common_type_t<TA, TB>;
	return Quantity<dim_div<DA, DB>, R>(static_cast<R>(a.value()) /
										static_cast<R>(b.value()));
}

// Quantity op scalar
template <typename D, typename T, typename S,
		  typename = std::enable_if_t<std::is_arithmetic_v<S>>>
constexpr auto operator*(const Quantity<D, T> &q, S s) {
	using R = std::common_type_t<T, S>;
	return Quantity<D, R>(static_cast<R>(q.value()) * static_cast<R>(s));
}

template <typename D, typename T, typename S,
		  typename = std::enable_if_t<std::is_arithmetic_v<S>>>
constexpr auto operator*(S s, const Quantity<D, T> &q) {
	return q * s;
}

template <typename D, typename T, typename S,
		  typename = std::enable_if_t<std::is_arithmetic_v<S>>>
constexpr auto operator/(const Quantity<D, T> &q, S s) {
	using R = std::common_type_t<T, S>;
	return Quantity<D, R>(static_cast<R>(q.value()) / static_cast<R>(s));
}

// scalar / Quantity -> inverse dimension
template <typename D, typename T, typename S,
		  typename = std::enable_if_t<std::is_arithmetic_v<S>>>
constexpr auto operator/(S s, const Quantity<D, T> &q) {
	using R = std::common_type_t<T, S>;
	return Quantity<dim_inv<D>, R>(static_cast<R>(s) /
								   static_cast<R>(q.value()));
}

// Comparisons (same dimension only)
template <typename D, typename TA, typename TB>
constexpr bool operator==(const Quantity<D, TA> &a, const Quantity<D, TB> &b) {
	return a.value() == b.value();
}
template <typename D, typename TA, typename TB>
constexpr bool operator!=(const Quantity<D, TA> &a, const Quantity<D, TB> &b) {
	return !(a == b);
}
template <typename D, typename TA, typename TB>
constexpr bool operator<(const Quantity<D, TA> &a, const Quantity<D, TB> &b) {
	return a.value() < b.value();
}
template <typename D, typename TA, typename TB>
constexpr bool operator<=(const Quantity<D, TA> &a, const Quantity<D, TB> &b) {
	return a.value() <= b.value();
}
template <typename D, typename TA, typename TB>
constexpr bool operator>(const Quantity<D, TA> &a, const Quantity<D, TB> &b) {
	return a.value() > b.value();
}
template <typename D, typename TA, typename TB>
constexpr bool operator>=(const Quantity<D, TA> &a, const Quantity<D, TB> &b) {
	return a.value() >= b.value();
}

// ---------------------------------------------------------------------------
// Mathematical helpers
// ---------------------------------------------------------------------------
template <typename D, typename T>
constexpr Quantity<D, T> abs(const Quantity<D, T> &q) {
	return Quantity<D, T>(q.value() < T{} ? -q.value() : q.value());
}

template <int N, typename D, typename T>
constexpr auto pow(const Quantity<D, T> &q) {
	T r = T{1};
	if constexpr (N > 0) {
		for (int i = 0; i < N; ++i)
			r *= q.value();
	} else if constexpr (N < 0) {
		for (int i = 0; i < -N; ++i)
			r *= q.value();
		r = T{1} / r;
	}
	return Quantity<dim_pow<D, N>, T>(r);
}

template <typename D, typename T>
inline Quantity<dim_sqrt<D>, T> sqrt(const Quantity<D, T> &q) {
	static_assert(D::length % 2 == 0 && D::mass % 2 == 0 && D::time % 2 == 0 &&
					  D::current % 2 == 0 && D::temperature % 2 == 0 &&
					  D::amount % 2 == 0 && D::luminous_intensity % 2 == 0,
				  "sqrt requires even dimension exponents");
	return Quantity<dim_sqrt<D>, T>(std::sqrt(q.value()));
}

template <typename D, typename T>
inline auto hypot(const Quantity<D, T> &a, const Quantity<D, T> &b) {
	return Quantity<D, T>(std::hypot(a.value(), b.value()));
}

// Angles are dimensionless; trig works on any dimensionless Quantity and
// returns a plain scalar. Inverse trig returns a dimensionless Quantity.
inline constexpr double pi = 3.141592653589793238462643383279502884;

template <typename T> inline T sin(const Quantity<Dimensionless, T> &a) {
	return std::sin(a.value());
}
template <typename T> inline T cos(const Quantity<Dimensionless, T> &a) {
	return std::cos(a.value());
}
template <typename T> inline T tan(const Quantity<Dimensionless, T> &a) {
	return std::tan(a.value());
}
template <typename T> inline Quantity<Dimensionless, T> asin(T v) {
	return Quantity<Dimensionless, T>(std::asin(v));
}
template <typename T> inline Quantity<Dimensionless, T> acos(T v) {
	return Quantity<Dimensionless, T>(std::acos(v));
}
template <typename T> inline Quantity<Dimensionless, T> atan(T v) {
	return Quantity<Dimensionless, T>(std::atan(v));
}
template <typename T>
inline Quantity<Dimensionless, T> atan2(const Quantity<Dimensionless, T> &y,
										const Quantity<Dimensionless, T> &x) {
	return Quantity<Dimensionless, T>(std::atan2(y.value(), x.value()));
}

// ---------------------------------------------------------------------------
// Units
// ---------------------------------------------------------------------------
// SI base units
inline constexpr Quantity<Length> metre{1.0};
inline constexpr Quantity<Length> meter{1.0};
inline constexpr Quantity<Length> m{1.0};
inline constexpr Quantity<Mass> kilogram{1.0};
inline constexpr Quantity<Mass> kg{1.0};
inline constexpr Quantity<Mass> gram{1e-3};
inline constexpr Quantity<Mass> g{1e-3};
inline constexpr Quantity<Time> second{1.0};
inline constexpr Quantity<Time> s{1.0};
inline constexpr Quantity<Current> ampere{1.0};
inline constexpr Quantity<Current> A{1.0};
inline constexpr Quantity<Temperature> kelvin{1.0};
inline constexpr Quantity<Temperature> K{1.0};
inline constexpr Quantity<Amount> mole{1.0};
inline constexpr Quantity<Amount> mol{1.0};
inline constexpr Quantity<LuminousIntensity> candela{1.0};
inline constexpr Quantity<LuminousIntensity> cd{1.0};

// Scaled length units
inline constexpr Quantity<Length> kilometre{1e3};
inline constexpr Quantity<Length> kilometer{1e3};
inline constexpr Quantity<Length> km{1e3};
inline constexpr Quantity<Length> centimetre{1e-2};
inline constexpr Quantity<Length> cm{1e-2};
inline constexpr Quantity<Length> millimetre{1e-3};
inline constexpr Quantity<Length> mm{1e-3};
inline constexpr Quantity<Length> micrometre{1e-6};
inline constexpr Quantity<Length> um{1e-6};
inline constexpr Quantity<Length> nanometre{1e-9};
inline constexpr Quantity<Length> nm{1e-9};

// Scaled time units
inline constexpr Quantity<Time> millisecond{1e-3};
inline constexpr Quantity<Time> ms{1e-3};
inline constexpr Quantity<Time> microsecond{1e-6};
inline constexpr Quantity<Time> us{1e-6};
inline constexpr Quantity<Time> nanosecond{1e-9};
inline constexpr Quantity<Time> ns{1e-9};
inline constexpr Quantity<Time> minute{60.0};
inline constexpr Quantity<Time> hour{3600.0};
inline constexpr Quantity<Time> h{3600.0};
inline constexpr Quantity<Time> day{86400.0};

// Volume
inline constexpr Quantity<Volume> litre{1e-3};
inline constexpr Quantity<Volume> liter{1e-3};
inline constexpr Quantity<Volume> L{1e-3};

// Angles (dimensionless)
inline constexpr Quantity<Dimensionless> radian{1.0};
inline constexpr Quantity<Dimensionless> rad{1.0};
inline constexpr Quantity<Dimensionless> degree{pi / 180.0};
inline constexpr Quantity<Dimensionless> deg{pi / 180.0};

// Derived SI units
inline constexpr Quantity<Frequency> hertz{1.0};
inline constexpr Quantity<Frequency> Hz{1.0};
inline constexpr Quantity<Force> newton{1.0};
inline constexpr Quantity<Force> N{1.0};
inline constexpr Quantity<Pressure> pascal{1.0};
inline constexpr Quantity<Pressure> Pa{1.0};
inline constexpr Quantity<Energy> joule{1.0};
inline constexpr Quantity<Energy> J{1.0};
inline constexpr Quantity<Power> watt{1.0};
inline constexpr Quantity<Power> W{1.0};
inline constexpr Quantity<Charge> coulomb{1.0};
inline constexpr Quantity<Charge> C{1.0};
inline constexpr Quantity<Voltage> volt{1.0};
inline constexpr Quantity<Voltage> V{1.0};
inline constexpr Quantity<Capacitance> farad{1.0};
inline constexpr Quantity<Capacitance> F{1.0};
inline constexpr Quantity<Resistance> ohm{1.0};
inline constexpr Quantity<Conductance> siemens{1.0};
inline constexpr Quantity<Conductance> S{1.0};
inline constexpr Quantity<MagneticFlux> weber{1.0};
inline constexpr Quantity<MagneticFlux> Wb{1.0};
inline constexpr Quantity<MagneticFluxDensity> tesla{1.0};
inline constexpr Quantity<Inductance> henry{1.0};
inline constexpr Quantity<Inductance> H{1.0};
inline constexpr Quantity<LuminousFlux> lumen{1.0};
inline constexpr Quantity<LuminousFlux> lm{1.0};
inline constexpr Quantity<Illuminance> lux{1.0};
inline constexpr Quantity<Illuminance> lx{1.0};
inline constexpr Quantity<Frequency> becquerel{1.0};
inline constexpr Quantity<Frequency> Bq{1.0};
inline constexpr Quantity<AbsorbedDose> gray{1.0};
inline constexpr Quantity<AbsorbedDose> Gy{1.0};
inline constexpr Quantity<AbsorbedDose> sievert{1.0};
inline constexpr Quantity<AbsorbedDose> Sv{1.0};
inline constexpr Quantity<CatalyticActivity> katal{1.0};
inline constexpr Quantity<CatalyticActivity> kat{1.0};

// Convenience for affine temperature scales (absolute temperatures only).
inline constexpr Quantity<Temperature> celsius_to_kelvin(double celsius) {
	return Quantity<Temperature>(celsius + 273.15);
}
inline constexpr Quantity<Temperature>
kelvin_to_celsius(const Quantity<Temperature> &k) {
	return Quantity<Temperature>(k.value() - 273.15);
}

} // namespace dracolix::units
