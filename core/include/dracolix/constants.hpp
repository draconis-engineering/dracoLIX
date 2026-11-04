#pragma once

// Mathematical and physical constants as dimension-checked quantities.
//
// Values follow the CODATA 2018 / SI 2019 definitions. Mathematical constants
// are plain doubles; physical constants carry their units, so e.g.
//
//   constants::speed_of_light.value()   // 299792458 (m/s)
//   (constants::planck_constant / (2.0 * constants::pi)).value()
//
// Licensed under GPL-3.0+

#include "units.hpp"

namespace dracolix::constants {

// ---------------------------------------------------------------------------
// Mathematical constants
// ---------------------------------------------------------------------------
inline constexpr double pi = 3.141592653589793238462643383279502884;
inline constexpr double tau = 2.0 * pi;
inline constexpr double euler = 2.718281828459045235360287471352662498;
inline constexpr double sqrt2 = 1.414213562373095048801688724209698079;
inline constexpr double golden_ratio =
	1.618033988749894848204586834365638118;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------
// Exact by definition in SI 2019
inline constexpr auto speed_of_light = 299792458.0 * units::m / units::s;
inline constexpr auto c = speed_of_light;

inline constexpr auto planck_constant = 6.62607015e-34 * units::J * units::s;
inline constexpr auto h = planck_constant;
inline constexpr auto reduced_planck_constant =
	6.62607015e-34 / (2.0 * pi) * units::J * units::s;
inline constexpr auto hbar = reduced_planck_constant;

inline constexpr auto elementary_charge = 1.602176634e-19 * units::C;
inline constexpr auto e_charge = elementary_charge;

inline constexpr auto boltzmann_constant = 1.380649e-23 * units::J / units::K;
inline constexpr auto k_B = boltzmann_constant;

inline constexpr auto avogadro_constant = 6.02214076e23 / units::mol;
inline constexpr auto N_A = avogadro_constant;

inline constexpr auto molar_gas_constant =
	8.31446261815324 * units::J / (units::mol * units::K);
inline constexpr auto R = molar_gas_constant;

// Electromagnetism
inline constexpr auto vacuum_permeability = 1.25663706212e-6 * units::N /
											 (units::A * units::A);
inline constexpr auto mu_0 = vacuum_permeability;
inline constexpr auto vacuum_permittivity = 8.8541878128e-12 * units::F /
											 units::m;
inline constexpr auto epsilon_0 = vacuum_permittivity;

// Gravitation and mechanics
inline constexpr auto gravitational_constant =
	6.67430e-11 * units::m * units::m * units::m / (units::kg * units::s *
													 units::s);
inline constexpr auto G = gravitational_constant;
inline constexpr auto standard_gravity = 9.80665 * units::m /
										  (units::s * units::s);
inline constexpr auto g_0 = standard_gravity;

// Particle masses (CODATA 2018)
inline constexpr auto electron_mass = 9.1093837015e-31 * units::kg;
inline constexpr auto proton_mass = 1.67262192369e-27 * units::kg;
inline constexpr auto neutron_mass = 1.67492749804e-27 * units::kg;
inline constexpr auto atomic_mass_unit = 1.66053906660e-27 * units::kg;
inline constexpr auto u = atomic_mass_unit;

// Other physical constants
inline constexpr auto fine_structure_constant = 7.2973525693e-3;
inline constexpr auto alpha = fine_structure_constant;
inline constexpr auto rydberg_constant = 10973731.568160 / units::m;
inline constexpr auto stefan_boltzmann_constant =
	5.670374419e-8 * units::W / (units::m * units::m * units::K * units::K *
								   units::K * units::K);
inline constexpr auto sigma = stefan_boltzmann_constant;

// Convenience (non-SI but universally used)
inline constexpr auto standard_atmosphere = 101325.0 * units::Pa;
inline constexpr auto atm = standard_atmosphere;
inline constexpr auto zero_celsius_in_kelvin = units::celsius_to_kelvin(0.0);

} // namespace dracolix::constants
