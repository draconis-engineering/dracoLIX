// Scientific constants and their dimensions.
// Licensed under GPL-3.0+

#include "../../core/include/dracolix/constants.hpp"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>

using namespace dracolix;
namespace u = dracolix::units;
namespace k = dracolix::constants;

namespace {

bool close(double a, double b, double tol = 1e-12) {
	return std::abs(a - b) <= tol;
}
bool rel_close(double a, double b, double tol = 1e-6) {
	return std::abs(a - b) <= tol * std::max(std::abs(a), std::abs(b));
}

template <typename T> using bare = std::remove_const_t<T>;

static_assert(std::is_same_v<bare<decltype(k::speed_of_light)>,
							 u::Quantity<u::Velocity>>);
static_assert(std::is_same_v<bare<decltype(k::planck_constant)>,
							 u::Quantity<u::Action>>);
static_assert(std::is_same_v<bare<decltype(k::elementary_charge)>,
							 u::Quantity<u::Charge>>);
static_assert(std::is_same_v<bare<decltype(k::boltzmann_constant)>,
							 u::Quantity<u::Entropy>>);
static_assert(std::is_same_v<bare<decltype(k::gravitational_constant)>,
							 u::Quantity<u::Dimension<3, -1, -2, 0, 0, 0, 0>>>);
static_assert(std::is_same_v<bare<decltype(k::standard_gravity)>,
							 u::Quantity<u::Acceleration>>);
static_assert(std::is_same_v<bare<decltype(k::standard_atmosphere)>,
							 u::Quantity<u::Pressure>>);

void test_math() {
	assert(close(k::pi, 3.14159265358979323846, 1e-15));
	assert(close(k::tau, 2.0 * k::pi));
	assert(close(k::euler, 2.71828182845904523536, 1e-15));
	assert(close(k::golden_ratio * k::golden_ratio,
				 k::golden_ratio + 1.0, 1e-15));
	assert(close(k::sqrt2 * k::sqrt2, 2.0, 1e-15));
	std::cout << " math constants ok\n";
}

void test_exact_si() {
	assert(k::speed_of_light.value() == 299792458.0);
	assert(k::planck_constant.value() == 6.62607015e-34);
	assert(k::elementary_charge.value() == 1.602176634e-19);
	assert(k::boltzmann_constant.value() == 1.380649e-23);
	assert(k::avogadro_constant.value() == 6.02214076e23);

	assert(close(k::hbar.value(), k::h.value() / (2.0 * k::pi), 1e-45));
	assert(rel_close(k::R.value(), k::N_A.value() * k::k_B.value()));

	assert(k::standard_gravity.value() == 9.80665);
	assert(k::standard_atmosphere.value() == 101325.0);
	assert(k::zero_celsius_in_kelvin.value() == 273.15);
	std::cout << " exact SI constants ok\n";
}

void test_physical_relations() {
	// c = 1 / sqrt(mu_0 * eps_0)
	const auto c2 = k::speed_of_light * k::speed_of_light;
	const auto inv_mu_eps = 1.0 / (k::mu_0 * k::epsilon_0);
	assert(rel_close(c2.value(), inv_mu_eps.value(), 1e-7));

	// Energy of a photon: E = h * nu.
	const auto nu = 5.0e14 * u::Hz;
	const auto energy = k::h * nu;
	assert(rel_close(energy.value(), 3.313035075e-19, 1e-9));

	// Mass-energy: E = m c^2 for one kilogram.
	const auto rest = 1.0 * u::kg * k::c * k::c;
	assert(rel_close(rest.value(), 8.9875517873681764e16, 1e-9));

	// Gravitational force between two 1 kg masses 1 m apart.
	const auto F = k::G * (1.0 * u::kg) * (1.0 * u::kg) /
				   (1.0 * u::m * (1.0 * u::m));
	assert(rel_close(F.value(), 6.67430e-11, 1e-9));

	assert(k::electron_mass < k::proton_mass);
	assert(k::neutron_mass > k::proton_mass);
	assert(k::u.value() < k::proton_mass.value());
	assert(k::fine_structure_constant > 0.007);
	assert(k::fine_structure_constant < 0.008);
	std::cout << " physical relations ok\n";
}

} // namespace

int main() {
	test_math();
	test_exact_si();
	test_physical_relations();
	std::cout << "All constant tests passed\n";
	return 0;
}
