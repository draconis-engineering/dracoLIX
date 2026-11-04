// Optimization: golden-section, Nelder-Mead and BFGS.
// Licensed under GPL-3.0+

#include "../../core/include/dracolix/optimize.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace dracolix::optimize;

namespace {
bool close(double a, double b, double tol) { return std::abs(a - b) < tol; }

void test_golden_section() {
	auto f = [](double x) { return (x - 3.0) * (x - 3.0) + 1.0; };
	auto r = golden_section(f, 0.0, 10.0);
	assert(r.converged);
	assert(close(r.x, 3.0, 1e-6));
	assert(close(r.fx, 1.0, 1e-10));

	auto g = [](double x) { return std::cos(x); };
	auto rg = golden_section(g, 0.0, 4.0);
	assert(close(rg.x, 3.14159265358979323846, 1e-6));
	std::cout << " golden_section ok\n";
}

void test_nelder_mead_quadratic() {
	auto f = [](const std::vector<double> &x) {
		const double a = x[0] - 1.0, b = x[1] + 2.0, c = x[2] - 0.5;
		return a * a + 4.0 * b * b + c * c;
	};
	auto r = nelder_mead(f, {0.0, 0.0, 0.0}, 1e-10, 20000);
	assert(r.converged);
	assert(close(r.x[0], 1.0, 1e-6));
	assert(close(r.x[1], -2.0, 1e-6));
	assert(close(r.x[2], 0.5, 1e-6));
	assert(r.fx < 1e-12);
	std::cout << " nelder_mead quadratic ok (" << r.iterations << " it)\n";
}

void test_nelder_mead_rosenbrock() {
	auto rosen = [](const std::vector<double> &x) {
		const double a = 1.0 - x[0];
		const double b = x[1] - x[0] * x[0];
		return a * a + 100.0 * b * b;
	};
	auto r = nelder_mead(rosen, {-1.2, 1.0}, 1e-9, 50000);
	assert(close(r.x[0], 1.0, 1e-3));
	assert(close(r.x[1], 1.0, 1e-3));
	assert(r.fx < 1e-6);
	std::cout << " nelder_mead rosenbrock ok (" << r.iterations << " it, fx "
			  << r.fx << ")\n";
}

void test_bfgs_rosenbrock() {
	auto rosen = [](const std::vector<double> &x) {
		const double a = 1.0 - x[0];
		const double b = x[1] - x[0] * x[0];
		return a * a + 100.0 * b * b;
	};
	auto grad = [](const std::vector<double> &x) {
		const double a = 1.0 - x[0];
		const double b = x[1] - x[0] * x[0];
		return std::vector<double>{-2.0 * a - 400.0 * x[0] * b, 200.0 * b};
	};

	auto r = bfgs(rosen, grad, {-1.2, 1.0}, 1e-8, 5000);
	assert(r.converged);
	assert(close(r.x[0], 1.0, 1e-5));
	assert(close(r.x[1], 1.0, 1e-5));
	assert(r.fx < 1e-12);

	auto rn = bfgs(rosen, {-1.2, 1.0}, 1e-7, 5000);
	assert(rn.converged);
	assert(close(rn.x[0], 1.0, 1e-4));
	assert(close(rn.x[1], 1.0, 1e-4));
	assert(rn.fx < 1e-8);
	std::cout << " bfgs rosenbrock ok (" << r.iterations << " it analytic, "
			  << rn.iterations << " it numeric)\n";
}

void test_numerical_gradient() {
	auto f = [](const std::vector<double> &x) {
		return 3.0 * x[0] * x[0] + 2.0 * x[1] + x[0] * x[1];
	};
	auto g = numerical_gradient(f, {1.0, 2.0});
	assert(close(g[0], 8.0, 1e-5));
	assert(close(g[1], 3.0, 1e-5));
	std::cout << " numerical_gradient ok\n";
}

void test_validation() {
	bool threw = false;
	try {
		golden_section([](double x) { return x; }, 1.0, 0.0);
	} catch (const std::invalid_argument &) {
		threw = true;
	}
	assert(threw);
	threw = false;
	try {
		nelder_mead([](const std::vector<double> &) { return 0.0; }, {});
	} catch (const std::invalid_argument &) {
		threw = true;
	}
	assert(threw);
	std::cout << " validation ok\n";
}
} // namespace

int main() {
	test_golden_section();
	test_nelder_mead_quadratic();
	test_nelder_mead_rosenbrock();
	test_bfgs_rosenbrock();
	test_numerical_gradient();
	test_validation();
	std::cout << "All optimization tests passed\n";
	return 0;
}
