// Linear least squares and polynomial fitting.
// Licensed under GPL-3.0+

#include "../../core/include/dracolix/least_squares.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace dracolix;
using namespace dracolix::least_squares;

namespace {

bool close(double a, double b, double tol = 1e-9) {
	return std::abs(a - b) <= tol;
}

void test_exact_square() {
	Array<double> A({2, 2});
	A.at(0, 0) = 3.0;
	A.at(0, 1) = 1.0;
	A.at(1, 0) = 1.0;
	A.at(1, 1) = 2.0;
	Array<double> b({2});
	b[0] = 9.0;
	b[1] = 8.0;

	auto r = lstsq(A, b);
	// 3x + y = 9, x + 2y = 8  =>  x = 2, y = 3
	assert(r.x.size() == 2);
	assert(close(r.x[0], 2.0, 1e-10));
	assert(close(r.x[1], 3.0, 1e-10));
	assert(r.residual < 1e-10);
	assert(r.rank == 2);
	std::cout << " exact square ok (res " << r.residual << ")\n";
}

void test_overdetermined_line() {
	// Fit y = 2 + 3x through exactly collinear samples.
	const double xs[] = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
	Array<double> A({6, 2});
	Array<double> b({6});
	for (size_t i = 0; i < 6; ++i) {
		A.at(i, 0) = 1.0;
		A.at(i, 1) = xs[i];
		b[i] = 2.0 + 3.0 * xs[i];
	}

	auto r = lstsq(A, b);
	assert(close(r.x[0], 2.0, 1e-10));
	assert(close(r.x[1], 3.0, 1e-10));
	assert(r.residual < 1e-10);
	assert(r.rank == 2);
	std::cout << " overdetermined line ok\n";
}

void test_residual_and_normal_equations() {
	// Inconsistent overdetermined system: fit a line to noisy samples.
	const double xs[] = {0.0, 1.0, 2.0, 3.0};
	const double ys[] = {0.1, 2.2, 3.7, 6.1};
	Array<double> A({4, 2});
	Array<double> b({4});
	for (size_t i = 0; i < 4; ++i) {
		A.at(i, 0) = 1.0;
		A.at(i, 1) = xs[i];
		b[i] = ys[i];
	}

	auto r = lstsq(A, b);
	assert(r.residual > 0.0);

	// Recompute ||A x - b|| independently.
	double res = 0.0;
	double ata_grad[2] = {0.0, 0.0};
	for (size_t i = 0; i < 4; ++i) {
		const double pred = r.x[0] + r.x[1] * xs[i];
		const double e = pred - ys[i];
		res += e * e;
		ata_grad[0] += e;
		ata_grad[1] += e * xs[i];
	}
	assert(close(r.residual, std::sqrt(res), 1e-10));
	// Normal equations: A^T (A x - b) = 0 at the minimizer.
	assert(std::abs(ata_grad[0]) < 1e-9);
	assert(std::abs(ata_grad[1]) < 1e-9);

	// The least-squares line is the sample mean at x = 0 and has the usual
	// regression slope.
	const double mean_x = 1.5, mean_y = 3.025;
	double sxy = 0.0, sxx = 0.0;
	for (size_t i = 0; i < 4; ++i) {
		sxy += (xs[i] - mean_x) * (ys[i] - mean_y);
		sxx += (xs[i] - mean_x) * (xs[i] - mean_x);
	}
	assert(close(r.x[1], sxy / sxx, 1e-10));
	assert(close(r.x[0], mean_y - r.x[1] * mean_x, 1e-10));
	std::cout << " residual / normal equations ok (res " << r.residual
			  << ")\n";
}

void test_polyfit_polyval() {
	// y = 1 - 2x + 0.5 x^2, coefficients constant-first.
	Array<double> x({5});
	Array<double> y({5});
	for (size_t i = 0; i < 5; ++i) {
		x[i] = -1.0 + (double)i;
		y[i] = 1.0 - 2.0 * x[i] + 0.5 * x[i] * x[i];
	}

	auto c = polyfit(x, y, 2);
	assert(c.size() == 3);
	assert(close(c[0], 1.0, 1e-9));
	assert(close(c[1], -2.0, 1e-9));
	assert(close(c[2], 0.5, 1e-9));

	auto p = polyval(c, x);
	assert(p.size() == x.size());
	for (size_t i = 0; i < x.size(); ++i)
		assert(close(p[i], y[i], 1e-9));

	// Degree-0 fit gives the mean.
	auto mean_fit = polyfit(x, y, 0);
	double mean = 0.0;
	for (size_t i = 0; i < y.size(); ++i)
		mean += y[i];
	mean /= (double)y.size();
	assert(mean_fit.size() == 1);
	assert(close(mean_fit[0], mean, 1e-10));

	// Horner evaluation of a constant polynomial.
	Array<double> one({1});
	one[0] = 7.0;
	Array<double> xs({3});
	xs[0] = -2.0;
	xs[1] = 0.0;
	xs[2] = 5.0;
	auto seven = polyval(one, xs);
	for (size_t i = 0; i < seven.size(); ++i)
		assert(close(seven[i], 7.0));
	std::cout << " polyfit/polyval ok\n";
}

void test_errors() {
	auto expect = [](auto fn) {
		try {
			fn();
		} catch (const std::exception &) {
			return true;
		}
		return false;
	};

	Array<double> A({3, 2});
	Array<double> b({3});
	assert(expect([&] { lstsq(A, Array<double>({2})); })); // b wrong length
	assert(expect([&] { lstsq(Array<double>({2}), b); })); // A not 2-D

	// Underdetermined systems are rejected.
	Array<double> wide({2, 3});
	assert(expect([&] { lstsq(wide, Array<double>({2})); }));

	// Duplicated columns make QR fail (rank-deficient).
	Array<double> dup({3, 2});
	for (size_t i = 0; i < 3; ++i) {
		dup.at(i, 0) = 1.0;
		dup.at(i, 1) = 1.0;
	}
	assert(expect([&] { lstsq(dup, b); }));

	// polyfit / polyval shape validation.
	Array<double> x({3});
	assert(expect([&] { polyfit(x, Array<double>({4}), 1); }));
	assert(expect([&] { polyfit(x, Array<double>({3}), 5); }));
	assert(expect([&] { polyval(Array<double>({2, 2}), x); }));
	assert(expect([&] { polyval(Array<double>({0}), x); }));
	std::cout << " error handling ok\n";
}

} // namespace

int main() {
	test_exact_square();
	test_overdetermined_line();
	test_residual_and_normal_equations();
	test_polyfit_polyval();
	test_errors();
	std::cout << "All least-squares tests passed\n";
	return 0;
}
