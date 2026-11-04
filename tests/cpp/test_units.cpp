// Physical units and dimensional analysis.
// Licensed under GPL-3.0+

#include "../../core/include/dracolix/units.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <type_traits>

using namespace dracolix;
namespace u = dracolix::units;

namespace {

bool close(double a, double b, double tol = 1e-12) {
	return std::abs(a - b) <= tol;
}

// Dimension arithmetic is encoded in the types.
static_assert(std::is_same_v<decltype(u::m / u::s), u::Quantity<u::Velocity>>);
static_assert(std::is_same_v<decltype(u::m * u::m), u::Quantity<u::Area>>);
static_assert(std::is_same_v<decltype(u::m * u::m * u::m),
							 u::Quantity<u::Volume>>);
static_assert(std::is_same_v<decltype(u::m / (u::s * u::s)),
							 u::Quantity<u::Acceleration>>);
static_assert(std::is_same_v<decltype(1.0 / u::s), u::Quantity<u::Frequency>>);
static_assert(
	std::is_same_v<decltype(u::kg * u::m / (u::s * u::s)), u::Quantity<u::Force>>);
static_assert(std::is_same_v<decltype(u::N * u::m), u::Quantity<u::Energy>>);
static_assert(
	std::is_same_v<decltype(u::J / u::s), u::Quantity<u::Power>>);
static_assert(std::is_same_v<decltype(u::A * u::s), u::Quantity<u::Charge>>);
static_assert(
	std::is_same_v<decltype(u::W / u::A), u::Quantity<u::Voltage>>);
static_assert(std::is_same_v<decltype(u::pow<2>(u::m)),
							 u::Quantity<u::Area>>);
static_assert(std::is_same_v<decltype(u::pow<3>(u::m)),
							 u::Quantity<u::Volume>>);
static_assert(std::is_same_v<decltype(u::pow<-1>(u::s)),
							 u::Quantity<u::Frequency>>);
static_assert(std::is_same_v<decltype(u::sqrt(u::m * u::m)),
							 u::Quantity<u::Length>>);
static_assert(u::Dimensionless::is_dimensionless());
static_assert(!u::Length::is_dimensionless());

void test_base_units() {
	auto d = 5.0 * u::km;
	static_assert(std::is_same_v<decltype(d), u::Quantity<u::Length>>);
	assert(close(d.value(), 5000.0));

	auto t = 250.0 * u::ms;
	assert(close(t.value(), 0.25));

	auto mass = 2.0 * u::g;
	assert(close(mass.value(), 0.002));

	auto volume = 2.0 * u::L;
	assert(close(volume.value(), 0.002));
	static_assert(std::is_same_v<decltype(volume), u::Quantity<u::Volume>>);

	assert(close(u::celsius_to_kelvin(0.0).value(), 273.15));
	assert(close(u::kelvin_to_celsius(u::celsius_to_kelvin(100.0)).value(),
				 100.0));
	std::cout << " base units ok\n";
}

void test_conversions() {
	// 36 km/h == 10 m/s
	auto v = 36.0 * u::km / u::h;
	assert(close(v.value(), 10.0));

	// 1 litre == 1e-3 m^3
	auto vol = 1.0 * u::L;
	assert(close(vol.value(), 1e-3));

	// 1 degree == pi/180 rad
	assert(close((1.0 * u::deg).value(), u::pi / 180.0));

	// One newton expressed in base units.
	auto F = 1.0 * u::kg * u::m / (u::s * u::s);
	assert(F == 1.0 * u::N);

	// One joule expressed as N*m.
	auto E = 1.0 * u::N * u::m;
	assert(E == 1.0 * u::J);

	// One watt as J/s.
	auto P = 1.0 * u::J / u::s;
	assert(P == 1.0 * u::W);

	// Electrical identities.
	assert(1.0 * u::V * u::A == 1.0 * u::W);
	assert(1.0 * u::A * u::s == 1.0 * u::C);
	assert(1.0 * u::C / u::V == 1.0 * u::F);
	assert(1.0 * u::V / u::A == 1.0 * u::ohm);
	assert(1.0 * u::Wb / (u::m * u::m) == 1.0 * u::tesla);
	std::cout << " conversions ok\n";
}

void test_arithmetic() {
	u::Quantity<u::Length> a(3.0), b(4.0);
	assert((a + b).value() == 7.0);
	assert((a - b).value() == -1.0);
	assert((a * 2.0).value() == 6.0);
	assert((2.0 * a).value() == 6.0);
	assert((a / 2.0).value() == 1.5);
	assert((-a).value() == -3.0);
	assert(a < b && b > a && a <= b && b >= a && a != b);
	assert(a == u::Quantity<u::Length>(3.0));

	auto e = u::pow<2>(a);
	assert(close(e.value(), 9.0));
	assert(e == 9.0 * (u::m * u::m));

	auto inv = u::pow<-2>(b);
	assert(close(inv.value(), 1.0 / 16.0));

	auto area = u::sqrt(u::Quantity<u::Area>(16.0));
	assert(close(area.value(), 4.0));
	static_assert(std::is_same_v<decltype(area), u::Quantity<u::Length>>);

	assert(close(u::abs(u::Quantity<u::Length>(-2.5)).value(), 2.5));
	assert(close(u::hypot(a, b).value(), 5.0));

	// Compound assignment.
	u::Quantity<u::Length> c(1.0);
	c += a;
	c -= b;
	c *= 2.0;
	c /= 2.0;
	assert(c.value() == 0.0);
	std::cout << " arithmetic ok\n";
}

void test_trig() {
	assert(close(u::sin(90.0 * u::deg), 1.0));
	assert(close(u::cos(u::pi * u::rad), -1.0));
	assert(close(u::tan(45.0 * u::deg), 1.0, 1e-12));
	const double angle = u::atan2(1.0 * u::rad, 1.0 * u::rad).value();
	assert(close(angle, u::pi / 4.0));
	assert(close(u::asin(1.0).value(), u::pi / 2.0));
	std::cout << " trig ok\n";
}

} // namespace

int main() {
	test_base_units();
	test_conversions();
	test_arithmetic();
	test_trig();
	std::cout << "All unit tests passed\n";
	return 0;
}
