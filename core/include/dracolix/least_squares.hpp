#pragma once

// Linear least squares and polynomial fitting.
//
// ``lstsq`` solves min ||A x - b||_2 for an overdetermined full-rank system
// using the QR factorization already in the core. ``polyfit`` is the usual
// Vandermonde convenience wrapper around it.
//
// Licensed under GPL-3.0+

#include "array.hpp"
#include "decomp.hpp"
#include "linalg.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace dracolix::least_squares {

struct LstsqResult {
	Array<double> x;
	double residual = 0.0;
	size_t rank = 0;
};

// Solve min ||A x - b||_2 for A (m x n), b (m), m >= n.
inline LstsqResult lstsq(const Array<double> &A, const Array<double> &b) {
	if (A.ndim() != 2)
		throw std::invalid_argument("lstsq: A must be 2-D");
	const size_t m = A.shape()[0], n = A.shape()[1];
	if (b.ndim() != 1 || b.size() != m)
		throw std::invalid_argument("lstsq: b must be a 1-D array of length m");
	if (m < n)
		throw std::invalid_argument("lstsq: requires m >= n (overdetermined)");

	auto qr = decomp::qr(A);
	const Array<double> &Q = qr.Q;
	const Array<double> &R = qr.R;

	std::vector<double> qtb(n, 0.0);
	for (size_t i = 0; i < n; ++i) {
		double s = 0.0;
		for (size_t k = 0; k < m; ++k)
			s += Q.at(k, i) * b[k];
		qtb[i] = s;
	}

	double max_diag = 0.0;
	for (size_t i = 0; i < n; ++i)
		max_diag = std::max(max_diag, std::abs(R.at(i, i)));
	const double rank_tol = 1e-12 * std::max(1.0, max_diag);

	Array<double> x(n);
	size_t rank = 0;
	for (size_t ii = n; ii-- > 0;) {
		double s = qtb[ii];
		for (size_t k = ii + 1; k < n; ++k)
			s -= R.at(ii, k) * x[k];
		const double d = R.at(ii, ii);
		if (std::abs(d) <= rank_tol)
			throw std::runtime_error("lstsq: rank-deficient system");
		++rank;
		x[ii] = s / d;
	}

	double res2 = 0.0;
	for (size_t i = 0; i < m; ++i) {
		double s = -b[i];
		for (size_t k = 0; k < n; ++k)
			s += A.at(i, k) * x[k];
		res2 += s * s;
	}
	return {std::move(x), std::sqrt(res2), rank};
}

// Least-squares polynomial fit of degree ``degree``: coefficients are ordered
// from the constant term upwards (c0 + c1 x + ... + c_degree x^degree).
inline Array<double> polyfit(const Array<double> &x, const Array<double> &y,
							 size_t degree) {
	if (x.ndim() != 1 || y.ndim() != 1 || x.size() != y.size())
		throw std::invalid_argument(
			"polyfit: x and y must be 1-D arrays of the same length");
	const size_t m = x.size();
	if (m < degree + 1)
		throw std::invalid_argument("polyfit: need at least degree+1 points");
	Array<double> A({m, degree + 1});
	for (size_t i = 0; i < m; ++i) {
		double p = 1.0;
		for (size_t j = 0; j <= degree; ++j) {
			A.at(i, j) = p;
			p *= x[i];
		}
	}
	return lstsq(A, y).x;
}

// Evaluate a polynomial (constant term first) at every element of x.
inline Array<double> polyval(const Array<double> &coeffs,
							 const Array<double> &x) {
	if (coeffs.ndim() != 1)
		throw std::invalid_argument("polyval: coeffs must be 1-D");
	if (x.ndim() != 1)
		throw std::invalid_argument("polyval: x must be 1-D");
	if (coeffs.size() == 0)
		throw std::invalid_argument("polyval: empty coefficient array");
	const size_t p = coeffs.size();
	Array<double> out(x.shape());
	for (size_t i = 0; i < x.size(); ++i) {
		double s = coeffs[p - 1];
		for (size_t j = p - 1; j > 0; --j)
			s = s * x[i] + coeffs[j - 1];
		out[i] = s;
	}
	return out;
}

} // namespace dracolix::least_squares
