// Finite-difference PDE tests: heat, wave, advection, Poisson (1-D and 2-D).
// Licensed under GPL-3.0+

#include "../../core/include/dracolix/dracolix.hpp"
#include "../../core/include/dracolix/pde.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace dracolix;
using namespace dracolix::pde;

namespace {

constexpr double PI = 3.14159265358979323846;

double max_error(const Array<double> &a, const Array<double> &b) {
	assert(a.size() == b.size());
	double m = 0.0;
	for (size_t i = 0; i < a.size(); ++i)
		m = std::max(m, std::abs(a[i] - b[i]));
	return m;
}

template <typename F> double max_error_fn(const Array<double> &a, F f) {
	double m = 0.0;
	for (size_t i = 0; i < a.size(); ++i)
		m = std::max(m, std::abs(a[i] - f(i)));
	return m;
}

template <typename F> double max_error_fn2d(const Array<double> &a, F f) {
	double m = 0.0;
	for (size_t i = 0; i < a.shape()[0]; ++i)
		for (size_t j = 0; j < a.shape()[1]; ++j)
			m = std::max(m, std::abs(a.at(i, j) - f(i, j)));
	return m;
}

void test_poisson_1d() {
	const size_t n = 101;
	Grid1D g{0.0, 1.0, n, BC::Dirichlet};
	Array<double> f(n);
	for (size_t i = 0; i < n; ++i)
		f[i] = PI * PI * std::sin(PI * g.x(i));

	auto u = poisson_1d(g, f);
	const double err =
		max_error_fn(u, [&](size_t i) { return std::sin(PI * g.x(i)); });
	assert(err < 1e-3);
	assert(std::abs(u[0]) < 1e-14 && std::abs(u[n - 1]) < 1e-14);

	// Nonzero Dirichlet data: -u'' = 0, u(0)=1, u(1)=3 => u = 1 + 2x.
	Array<double> zero(n);
	auto lin = poisson_1d(g, zero, 1.0, 3.0);
	const double lerr = max_error_fn(
		lin, [&](size_t i) { return 1.0 + 2.0 * g.x(i); });
	assert(lerr < 1e-10);
	std::cout << " poisson_1d ok (err " << err << ")\n";
}

void test_heat_explicit() {
	const size_t n = 101;
	Grid1D g{0.0, 1.0, n, BC::Dirichlet};
	const double alpha = 1.0;
	const double t_end = 0.1;
	const double dx = g.dx();
	const double dt = 0.4 * dx * dx / alpha; // r = 0.4, stable

	Array<double> u0(n);
	for (size_t i = 0; i < n; ++i)
		u0[i] = std::sin(PI * g.x(i));

	auto sol = heat_1d(g, u0, alpha, t_end, dt);
	const double decay = std::exp(-alpha * PI * PI * t_end);
	const double err = max_error_fn(
		sol.u.back(), [&](size_t i) { return decay * std::sin(PI * g.x(i)); });
	assert(std::abs(sol.t.back() - t_end) < 1e-12);
	assert(sol.t.size() == sol.u.size());
	assert(err < 2e-3);

	// Instability is rejected up front.
	bool threw = false;
	try {
		heat_1d(g, u0, alpha, t_end, 2.0 * dx * dx / alpha);
	} catch (const std::invalid_argument &) {
		threw = true;
	}
	assert(threw);
	std::cout << " heat_1d explicit ok (err " << err << ")\n";
}

void test_heat_cn() {
	const size_t n = 101;
	Grid1D g{0.0, 1.0, n, BC::Dirichlet};
	const double alpha = 1.0;
	const double t_end = 0.1;

	Array<double> u0(n);
	for (size_t i = 0; i < n; ++i)
		u0[i] = std::sin(PI * g.x(i));

	auto sol = heat_1d_cn(g, u0, alpha, t_end, 0.002);
	const double decay = std::exp(-alpha * PI * PI * t_end);
	const double err = max_error_fn(
		sol.u.back(), [&](size_t i) { return decay * std::sin(PI * g.x(i)); });
	assert(err < 1e-3);
	assert(std::abs(sol.t.back() - t_end) < 1e-12);

	// Crank-Nicolson allows large steps (unconditionally stable).
	auto coarse = heat_1d_cn(g, u0, alpha, t_end, 0.01);
	const double cerr = max_error_fn(
		coarse.u.back(),
		[&](size_t i) { return decay * std::sin(PI * g.x(i)); });
	assert(cerr < 5e-3);
	std::cout << " heat_1d Crank-Nicolson ok (err " << err << ", coarse "
			  << cerr << ")\n";
}

void test_heat_periodic() {
	const size_t n = 100;
	Grid1D g{0.0, 1.0, n, BC::Periodic};
	const double alpha = 1.0;
	const double t_end = 0.05;
	const double dx = g.dx();
	const double dt = 0.4 * dx * dx / alpha;

	Array<double> u0(n);
	for (size_t i = 0; i < n; ++i)
		u0[i] = std::cos(2.0 * PI * g.x(i));

	auto sol = heat_1d(g, u0, alpha, t_end, dt);
	const double decay = std::exp(-alpha * 4.0 * PI * PI * t_end);
	const double err = max_error_fn(
		sol.u.back(),
		[&](size_t i) { return decay * std::cos(2.0 * PI * g.x(i)); });
	assert(err < 2e-3);
	std::cout << " heat_1d periodic ok (err " << err << ")\n";
}

void test_wave() {
	const size_t n = 201;
	Grid1D g{0.0, 1.0, n, BC::Dirichlet};
	const double c = 1.0;
	const double dx = g.dx();
	const double dt = 0.5 * dx / c; // CFL 0.5
	const double t_end = 0.5;

	Array<double> u0(n), v0(n);
	for (size_t i = 0; i < n; ++i) {
		u0[i] = std::sin(PI * g.x(i));
		v0[i] = 0.0;
	}

	auto sol = wave_1d(g, u0, v0, c, t_end, dt);
	const double err = max_error_fn(sol.u.back(), [&](size_t i) {
		return std::cos(PI * c * t_end) * std::sin(PI * g.x(i));
	});
	assert(std::abs(sol.t.back() - t_end) < 1e-12);
	assert(err < 1e-2);

	// Nonzero initial velocity: u = sin(pi x) cos(pi t) + (1/pi) sin(pi x)
	// sin(pi t) is the exact solution for v0 = sin(pi x).
	for (size_t i = 0; i < n; ++i)
		v0[i] = std::sin(PI * g.x(i));
	auto sol2 = wave_1d(g, u0, v0, c, t_end, dt);
	const double err2 = max_error_fn(sol2.u.back(), [&](size_t i) {
		const double s = std::sin(PI * g.x(i));
		return s * (std::cos(PI * t_end) +
					std::sin(PI * t_end) / PI);
	});
	assert(err2 < 2e-2);

	bool threw = false;
	try {
		wave_1d(g, u0, v0, c, t_end, 2.0 * dx / c);
	} catch (const std::invalid_argument &) {
		threw = true;
	}
	assert(threw);
	std::cout << " wave_1d ok (err " << err << ", with v0 " << err2 << ")\n";
}

void test_advection() {
	const size_t n = 200;
	Grid1D g{0.0, 1.0, n, BC::Periodic};
	const double a = 1.0;
	const double dx = g.dx();
	const double dt = 0.5 * dx / a; // CFL 0.5
	const double t_end = 0.2;

	Array<double> u0(n);
	for (size_t i = 0; i < n; ++i)
		u0[i] = std::sin(2.0 * PI * g.x(i));

	auto sol = advection_1d(g, u0, a, t_end, dt);
	const double err = max_error_fn(sol.u.back(), [&](size_t i) {
		return std::sin(2.0 * PI * (g.x(i) - a * t_end));
	});
	assert(std::abs(sol.t.back() - t_end) < 1e-12);
	assert(err < 6e-2);

	// Negative velocity travels the other way.
	auto sol_neg = advection_1d(g, u0, -a, t_end, dt);
	const double err_neg = max_error_fn(sol_neg.u.back(), [&](size_t i) {
		return std::sin(2.0 * PI * (g.x(i) + a * t_end));
	});
	assert(err_neg < 6e-2);

	bool threw = false;
	try {
		advection_1d(g, u0, a, t_end, 2.0 * dx / a);
	} catch (const std::invalid_argument &) {
		threw = true;
	}
	assert(threw);
	std::cout << " advection_1d ok (err " << err << ", negative " << err_neg
			  << ")\n";
}

void test_poisson_2d() {
	const size_t nx = 41, ny = 41;
	Grid2D g{0.0, 1.0, 0.0, 1.0, nx, ny};
	Array<double> f({nx, ny});
	for (size_t i = 0; i < nx; ++i)
		for (size_t j = 0; j < ny; ++j)
			f.at(i, j) = 2.0 * PI * PI * std::sin(PI * g.x(i)) *
						 std::sin(PI * g.y(j));

	auto u = poisson_2d(g, f);
	const double err = max_error_fn2d(u, [&](size_t i, size_t j) {
		return std::sin(PI * g.x(i)) * std::sin(PI * g.y(j));
	});
	assert(err < 5e-3);
	for (size_t i = 0; i < nx; ++i) {
		assert(std::abs(u.at(i, 0)) < 1e-12);
		assert(std::abs(u.at(i, ny - 1)) < 1e-12);
	}

	// Constant Dirichlet data with f = 0: the exact solution is constant.
	Array<double> zero({nx, ny});
	auto ones = poisson_2d(g, zero, 1e-12, 100000, 1.8, 1.0, 1.0, 1.0,
						   1.0);
	for (size_t i = 0; i < nx; ++i)
		for (size_t j = 0; j < ny; ++j)
			assert(std::abs(ones.at(i, j) - 1.0) < 1e-9);

	bool threw = false;
	try {
		Array<double> bad({nx + 1, ny});
		(void)poisson_2d(g, bad);
	} catch (const std::invalid_argument &) {
		threw = true;
	}
	assert(threw);
	std::cout << " poisson_2d ok (err " << err << ")\n";
}

} // namespace

int main() {
	test_poisson_1d();
	test_heat_explicit();
	test_heat_cn();
	test_heat_periodic();
	test_wave();
	test_advection();
	test_poisson_2d();
	std::cout << "All PDE tests passed\n";
	return 0;
}
