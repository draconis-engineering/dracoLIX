
#include "../../core/include/dracolix/dracolix.hpp"
#include "../../core/include/dracolix/numerics.hpp"
#include "../../core/include/dracolix/ode.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace dracolix;

void test_roots() {
	double r = numerics::bisect([](double x) { return x * x - 2; }, 0, 2);
	assert(std::abs(r - std::sqrt(2)) < 1e-7);
	double rn = numerics::newton([](double x) { return x * x - 2; },
								 [](double x) { return 2 * x; }, 1.5);
	assert(std::abs(rn - std::sqrt(2)) < 1e-8);
	std::cout << " roots ok\n";
}

void test_interp_diff_integ() {
	std::vector<double> xs = {0, 1, 2}, ys = {0, 1, 4};
	assert(std::abs(numerics::interp_linear(xs, ys, 1.5) - 2.5) < 1e-9);
	double d = numerics::derivative([](double x) { return x * x; }, 2.0);
	assert(std::abs(d - 4.0) < 1e-4);
	double I =
		numerics::integrate_simpson([](double x) { return x * x; }, 0, 1, 100);
	assert(std::abs(I - 1.0 / 3) < 1e-6);
	double It = numerics::integrate_trapezoidal([](double x) { return x * x; },
												0, 1, 1000);
	assert(std::abs(It - 1.0 / 3) < 1e-3);
	std::cout << " interp/diff/integ ok\n";
}

void test_ode() {
	// dy/dt = y, y0=1 => y=exp(t), t=1 => e
	auto f = [](double, const ode::State &y) { return ode::State{y[0]}; };
	auto traj = ode::rk4(f, 0, {1.0}, 1.0, 0.1);
	assert(std::abs(traj.back()[0] - std::exp(1.0)) < 1e-3);
	auto traj2 = ode::euler(f, 0, {1.0}, 1.0, 0.01);
	assert(std::abs(traj2.back()[0] - std::exp(1.0)) < 0.05);
	auto traj3 = ode::adaptive_rk45(f, 0, {1.0}, 1.0, 1e-8, 0.2);
	assert(std::abs(traj3.back()[0] - std::exp(1.0)) < 1e-5);
	// harmonic: y''=-y => state [y, v]
	auto f2 = [](double, const ode::State &y) {
		return ode::State{y[1], -y[0]};
	};
	auto th = ode::rk4(f2, 0, {1.0, 0.0}, 6.283185307, 0.1);
	assert(std::abs(th.back()[0] - 1.0) < 1e-2);
	std::cout << " ode ok\n";
}

void test_stiff_stability() {
	// y' = -1000 y, y(0)=1. Explicit RK4 with dt=0.01 is wildly unstable;
	// the implicit methods must stay bounded and decay to ~0.
	auto f = [](double, const ode::State &y) {
		return ode::State{-1000.0 * y[0]};
	};
	auto be = ode::backward_euler(f, 0, {1.0}, 1.0, 0.01);
	auto bd = ode::bdf2(f, 0, {1.0}, 1.0, 0.01);
	for (const auto *traj : {&be, &bd}) {
		for (const auto &y : *traj) {
			assert(std::isfinite(y[0]));
			assert(std::abs(y[0]) < 10.0); // no blow-up
		}
		assert(traj->back()[0] >= 0.0);
		assert(traj->back()[0] < 1e-6);
	}

	// Stiff but non-decaying solution: y' = -1000 (y - cos t) - sin t, y = cos t.
	auto fs = [](double t, const ode::State &y) {
		return ode::State{-1000.0 * (y[0] - std::cos(t)) - std::sin(t)};
	};
	auto track = ode::backward_euler(fs, 0, {1.0}, 2.0, 0.01);
	assert(std::abs(track.back()[0] - std::cos(2.0)) < 0.02);

	// An analytic Jacobian must agree with the finite-difference one.
	ode::Jacobian jac = [](double, const ode::State &) {
		return std::vector<double>{-1000.0};
	};
	auto be_j = ode::backward_euler(f, 0, {1.0}, 1.0, 0.01, jac);
	assert(std::abs(be.back()[0] - be_j.back()[0]) < 1e-6);

	std::cout << " stiff stability ok\n";
}

void test_implicit_convergence_order() {
	// y' = -2 y, y(0)=1 => y(1) = exp(-2).
	auto f = [](double, const ode::State &y) {
		return ode::State{-2.0 * y[0]};
	};
	auto err = [&](double dt, bool use_bdf2) {
		auto traj = use_bdf2 ? ode::bdf2(f, 0, {1.0}, 1.0, dt)
							 : ode::backward_euler(f, 0, {1.0}, 1.0, dt);
		return std::abs(traj.back()[0] - std::exp(-2.0));
	};

	const double e1 = err(0.1, false), e2 = err(0.05, false),
				 e3 = err(0.025, false);
	const double p1 = std::log(e1 / e2) / std::log(2.0);
	const double p2 = std::log(e2 / e3) / std::log(2.0);
	assert(p1 > 0.85 && p1 < 1.15);
	assert(p2 > 0.85 && p2 < 1.15);
	assert(e1 > e2 && e2 > e3);

	const double b1 = err(0.1, true), b2 = err(0.05, true),
				 b3 = err(0.025, true);
	const double q1 = std::log(b1 / b2) / std::log(2.0);
	const double q2 = std::log(b2 / b3) / std::log(2.0);
	assert(q1 > 1.5 && q1 < 2.8);
	assert(q2 > 1.5 && q2 < 2.8);
	assert(b1 < e1 && b2 < e2 && b3 < e3);

	std::cout << " implicit convergence order ok (BE p=" << p1 << ","
			  << p2 << "; BDF2 p=" << q1 << "," << q2 << ")\n";
}

void test_solve_ivp() {
	auto f = [](double, const ode::State &y) { return ode::State{y[0]}; };
	const double exact = std::exp(1.0);

	struct Case {
		ode::Method method;
		double step;
		const char *name;
	};
	const Case cases[] = {
		{ode::Method::Euler, 0.01, "euler"},
		{ode::Method::RK4, 0.1, "rk4"},
		{ode::Method::AdaptiveRK45, 0.2, "rk45"},
		{ode::Method::BackwardEuler, 0.01, "backward_euler"},
		{ode::Method::BDF2, 0.01, "bdf2"},
	};

	for (const auto &c : cases) {
		auto sol = ode::solve_ivp(f, 0.0, {1.0}, 1.0, c.method, c.step);
		assert(sol.success);
		assert(sol.t.size() == sol.y.size());
		assert(!sol.t.empty());
		assert(std::abs(sol.t.front() - 0.0) < 1e-15);
		assert(std::abs(sol.t.back() - 1.0) < 1e-12);
		for (size_t i = 1; i < sol.t.size(); ++i)
			assert(sol.t[i] > sol.t[i - 1]);
		assert(std::abs(sol.y.back()[0] - exact) < 0.02);
		assert(std::string(ode::method_name(c.method)) == c.name);
		assert(ode::parse_method(c.name) == c.method);
	}

	// Fixed-step trajectories are byte-for-byte the dedicated solvers.
	auto s_e = ode::solve_ivp(f, 0, {1.0}, 1.0, ode::Method::Euler, 0.1);
	auto t_e = ode::euler(f, 0, {1.0}, 1.0, 0.1);
	assert(s_e.y.size() == t_e.size());
	for (size_t i = 0; i < t_e.size(); ++i)
		assert(s_e.y[i][0] == t_e[i][0]);

	// Zero-length interval.
	auto s0 = ode::solve_ivp(f, 0.5, {2.0}, 0.5);
	assert(s0.success && s0.t.size() == 1 && s0.y.size() == 1);
	assert(s0.t[0] == 0.5 && s0.y[0][0] == 2.0);

	// Method aliases.
	assert(ode::parse_method("implicit_euler") == ode::Method::BackwardEuler);
	assert(ode::parse_method("adaptive_rk45") == ode::Method::AdaptiveRK45);
	bool threw = false;
	try {
		ode::parse_method("nope");
	} catch (const std::invalid_argument &) {
		threw = true;
	}
	assert(threw);

	std::cout << " solve_ivp ok\n";
}

int main() {
	test_roots();
	test_interp_diff_integ();
	test_ode();
	test_stiff_stability();
	test_implicit_convergence_order();
	test_solve_ivp();
	std::cout << "All numerics/ode tests passed\n";
	return 0;
}
