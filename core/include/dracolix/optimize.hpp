#pragma once

// Unconstrained optimization: golden-section (1-D), Nelder-Mead (derivative
// free, N-D) and BFGS (gradient based, N-D with numerical gradients as a
// fallback).
//
// Licensed under GPL-3.0+

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dracolix::optimize {

struct ScalarOptimizeResult {
	double x = 0.0;
	double fx = std::numeric_limits<double>::infinity();
	size_t iterations = 0;
	bool converged = false;
	std::string message;
};

struct OptimizeResult {
	std::vector<double> x;
	double fx = std::numeric_limits<double>::infinity();
	size_t iterations = 0;
	bool converged = false;
	std::string message;
};

using Cost = std::function<double(const std::vector<double> &)>;
using Gradient =
	std::function<std::vector<double>(const std::vector<double> &)>;

namespace detail {

inline double dot(const std::vector<double> &a, const std::vector<double> &b) {
	double s = 0.0;
	for (size_t i = 0; i < a.size(); ++i)
		s += a[i] * b[i];
	return s;
}

inline double norm2(const std::vector<double> &a) {
	return std::sqrt(dot(a, a));
}

inline double norm_inf(const std::vector<double> &a) {
	double m = 0.0;
	for (double v : a)
		m = std::max(m, std::abs(v));
	return m;
}

inline std::vector<double> add_scaled(const std::vector<double> &a,
									  const std::vector<double> &b, double s) {
	std::vector<double> r(a.size());
	for (size_t i = 0; i < a.size(); ++i)
		r[i] = a[i] + s * b[i];
	return r;
}

inline std::vector<double> matvec(const std::vector<double> &A,
								  const std::vector<double> &x) {
	const size_t n = x.size();
	std::vector<double> y(n, 0.0);
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < n; ++j)
			y[i] += A[i * n + j] * x[j];
	return y;
}

} // namespace detail

// Central finite-difference gradient.
inline std::vector<double> numerical_gradient(const Cost &f,
											  const std::vector<double> &x,
											  double h = 1e-6) {
	const size_t n = x.size();
	std::vector<double> g(n, 0.0);
	std::vector<double> xp = x, xm = x;
	for (size_t i = 0; i < n; ++i) {
		const double hi = h * std::max(1.0, std::abs(x[i]));
		xp[i] = x[i] + hi;
		xm[i] = x[i] - hi;
		g[i] = (f(xp) - f(xm)) / (2.0 * hi);
		xp[i] = x[i];
		xm[i] = x[i];
	}
	return g;
}

// Golden-section search for a unimodal function on [a, b].
inline ScalarOptimizeResult
golden_section(const std::function<double(double)> &f, double a, double b,
			   double tol = 1e-8, int max_iter = 1000) {
	if (!(b > a))
		throw std::invalid_argument("golden_section: need b > a");
	if (tol <= 0)
		throw std::invalid_argument("golden_section: need tol > 0");
	const double phi = 0.5 * (std::sqrt(5.0) - 1.0);
	double c = b - phi * (b - a);
	double d = a + phi * (b - a);
	double fc = f(c), fd = f(d);
	size_t it = 0;
	for (; it < (size_t)max_iter; ++it) {
		if (std::abs(b - a) <= tol * (1.0 + std::abs(0.5 * (a + b))))
			break;
		if (fc < fd) {
			b = d;
			d = c;
			fd = fc;
			c = b - phi * (b - a);
			fc = f(c);
		} else {
			a = c;
			c = d;
			fc = fd;
			d = a + phi * (b - a);
			fd = f(d);
		}
	}
	ScalarOptimizeResult res;
	res.x = 0.5 * (a + b);
	res.fx = f(res.x);
	res.iterations = it;
	res.converged = it < (size_t)max_iter;
	if (!res.converged)
		res.message = "golden_section: max_iter reached";
	return res;
}

// Nelder-Mead simplex method. ``step`` controls the initial simplex size.
inline OptimizeResult nelder_mead(const Cost &f, std::vector<double> x0,
								  double tol = 1e-8, int max_iter = 10000,
								  double step = 0.05) {
	if (x0.empty())
		throw std::invalid_argument("nelder_mead: need a non-empty x0");
	if (tol <= 0)
		throw std::invalid_argument("nelder_mead: need tol > 0");
	if (!(step > 0))
		throw std::invalid_argument("nelder_mead: need step > 0");
	const size_t n = x0.size();
	std::vector<std::vector<double>> simplex(n + 1, x0);
	std::vector<double> fs(n + 1, 0.0);
	fs[0] = f(x0);
	for (size_t i = 0; i < n; ++i) {
		const double d = step * std::max(1.0, std::abs(x0[i]));
		simplex[i + 1][i] += d;
		fs[i + 1] = f(simplex[i + 1]);
	}

	constexpr double alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5;
	OptimizeResult res;
	size_t it = 0;
	for (; it < (size_t)max_iter; ++it) {
		size_t best = 0, worst = 0;
		for (size_t i = 1; i <= n; ++i) {
			if (fs[i] < fs[best])
				best = i;
			if (fs[i] > fs[worst])
				worst = i;
		}
		size_t second = (worst == 0) ? 1 : 0;
		for (size_t i = 0; i <= n; ++i) {
			if (i == worst)
				continue;
			if (fs[i] > fs[second])
				second = i;
		}

		double spread = 0.0, fspread = 0.0;
		for (size_t i = 0; i <= n; ++i) {
			std::vector<double> d(n);
			for (size_t k = 0; k < n; ++k)
				d[k] = simplex[i][k] - simplex[best][k];
			spread = std::max(spread, detail::norm2(d));
			fspread = std::max(fspread, std::abs(fs[i] - fs[best]));
		}
		if (spread <= tol * (1.0 + detail::norm2(simplex[best])) &&
			fspread <= tol * (1.0 + std::abs(fs[best])))
			break;

		std::vector<double> centroid(n, 0.0);
		for (size_t i = 0; i <= n; ++i) {
			if (i == worst)
				continue;
			for (size_t k = 0; k < n; ++k)
				centroid[k] += simplex[i][k] / static_cast<double>(n);
		}

		std::vector<double> xr(n);
		for (size_t k = 0; k < n; ++k)
			xr[k] = centroid[k] + alpha * (centroid[k] - simplex[worst][k]);
		const double fr = f(xr);
		if (fr < fs[best]) {
			std::vector<double> xe(n);
			for (size_t k = 0; k < n; ++k)
				xe[k] = centroid[k] + gamma * (xr[k] - centroid[k]);
			const double fe = f(xe);
			if (fe < fr) {
				simplex[worst] = std::move(xe);
				fs[worst] = fe;
			} else {
				simplex[worst] = std::move(xr);
				fs[worst] = fr;
			}
		} else if (fr < fs[second]) {
			simplex[worst] = std::move(xr);
			fs[worst] = fr;
		} else {
			bool shrink = false;
			if (fr < fs[worst]) {
				std::vector<double> xc(n);
				for (size_t k = 0; k < n; ++k)
					xc[k] = centroid[k] + rho * (xr[k] - centroid[k]);
				const double fc = f(xc);
				if (fc <= fr) {
					simplex[worst] = std::move(xc);
					fs[worst] = fc;
				} else {
					shrink = true;
				}
			} else {
				std::vector<double> xc(n);
				for (size_t k = 0; k < n; ++k)
					xc[k] =
						centroid[k] + rho * (simplex[worst][k] - centroid[k]);
				const double fc = f(xc);
				if (fc < fs[worst]) {
					simplex[worst] = std::move(xc);
					fs[worst] = fc;
				} else {
					shrink = true;
				}
			}
			if (shrink) {
				for (size_t i = 0; i <= n; ++i) {
					if (i == best)
						continue;
					for (size_t k = 0; k < n; ++k)
						simplex[i][k] =
							simplex[best][k] +
							sigma * (simplex[i][k] - simplex[best][k]);
					fs[i] = f(simplex[i]);
				}
			}
		}
	}

	size_t best = 0;
	for (size_t i = 1; i <= n; ++i)
		if (fs[i] < fs[best])
			best = i;
	res.x = simplex[best];
	res.fx = fs[best];
	res.iterations = it;
	res.converged = it < (size_t)max_iter;
	if (!res.converged)
		res.message = "nelder_mead: max_iter reached";
	return res;
}

// BFGS quasi-Newton method. If ``grad`` is empty, gradients are approximated
// by central differences.
inline OptimizeResult bfgs(const Cost &f, const Gradient &grad,
						   std::vector<double> x0, double tol = 1e-6,
						   int max_iter = 1000) {
	if (x0.empty())
		throw std::invalid_argument("bfgs: need a non-empty x0");
	if (tol <= 0)
		throw std::invalid_argument("bfgs: need tol > 0");
	const size_t n = x0.size();
	auto eval_grad = [&](const std::vector<double> &x) {
		return grad ? grad(x) : numerical_gradient(f, x);
	};

	std::vector<double> x = x0;
	double fx = f(x);
	std::vector<double> g = eval_grad(x);
	if (g.size() != n)
		throw std::runtime_error("bfgs: gradient has the wrong size");

	// Inverse-Hessian approximation, initialised to the identity.
	std::vector<double> H(n * n, 0.0);
	for (size_t i = 0; i < n; ++i)
		H[i * n + i] = 1.0;

	OptimizeResult res;
	size_t it = 0;
	for (; it < (size_t)max_iter; ++it) {
		if (detail::norm_inf(g) <= tol)
			break;

		std::vector<double> p = detail::matvec(H, g);
		for (double &v : p)
			v = -v;
		double slope = detail::dot(g, p);
		if (!(slope < 0.0)) {
			// Reset to steepest descent if H stopped being positive definite.
			for (size_t i = 0; i < n; ++i)
				for (size_t j = 0; j < n; ++j)
					H[i * n + j] = (i == j) ? 1.0 : 0.0;
			for (size_t i = 0; i < n; ++i)
				p[i] = -g[i];
			slope = detail::dot(g, p);
		}

		double alpha = 1.0;
		std::vector<double> xn(n);
		double fn = fx;
		bool accepted = false;
		for (int ls = 0; ls < 60; ++ls) {
			for (size_t i = 0; i < n; ++i)
				xn[i] = x[i] + alpha * p[i];
			fn = f(xn);
			if (fn <= fx + 1e-4 * alpha * slope) {
				accepted = true;
				break;
			}
			alpha *= 0.5;
		}
		if (!accepted) {
			res.message = "bfgs: line search failed";
			break;
		}

		std::vector<double> gn = eval_grad(xn);
		if (gn.size() != n)
			throw std::runtime_error("bfgs: gradient has the wrong size");
		std::vector<double> s(n), y(n);
		for (size_t i = 0; i < n; ++i) {
			s[i] = xn[i] - x[i];
			y[i] = gn[i] - g[i];
		}
		const double ys = detail::dot(y, s);
		if (ys > 1e-14) {
			std::vector<double> Hy = detail::matvec(H, y);
			const double yHy = detail::dot(y, Hy);
			const double coeff = (ys + yHy) / (ys * ys);
			for (size_t i = 0; i < n; ++i)
				for (size_t j = 0; j < n; ++j)
					H[i * n + j] += coeff * s[i] * s[j] -
									(Hy[i] * s[j] + s[i] * Hy[j]) / ys;
		}
		x = std::move(xn);
		fx = fn;
		g = std::move(gn);

		if (detail::norm2(s) <= tol * (1.0 + detail::norm2(x)))
			break;
	}

	res.x = std::move(x);
	res.fx = fx;
	res.iterations = it;
	res.converged = detail::norm_inf(g) <= tol ||
					detail::norm2(g) <= tol * (1.0 + std::abs(fx));
	if (!res.converged && res.message.empty())
		res.message = "bfgs: max_iter reached";
	return res;
}

inline OptimizeResult bfgs(const Cost &f, std::vector<double> x0,
						   double tol = 1e-6, int max_iter = 1000) {
	return bfgs(f, Gradient{}, std::move(x0), tol, max_iter);
}

} // namespace dracolix::optimize
