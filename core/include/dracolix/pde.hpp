#pragma once

// Finite-difference PDE infrastructure on uniform grids.
//
// 1-D: heat (explicit FTCS and Crank-Nicolson), wave (explicit leapfrog),
//      advection (first-order upwind), Poisson (tridiagonal direct solve).
// 2-D: Poisson (five-point stencil, SOR).
//
// All routines use dracolix::Array<double> and stay header-only. Dirichlet
// boundaries are homogeneous by default (explicit boundary values are accepted
// where noted); periodic grids wrap around and do not duplicate the endpoint.
//
// Licensed under GPL-3.0+

#include "array.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace dracolix::pde {

// ---------------------------------------------------------------------------
// Grids and boundary conditions
// ---------------------------------------------------------------------------
enum class BC { Dirichlet, Periodic };

struct Grid1D {
	double x0 = 0.0;
	double x1 = 1.0;
	size_t n = 101;
	BC bc = BC::Dirichlet;

	double length() const { return x1 - x0; }

	// Dirichlet grids include both endpoints; periodic grids span [x0, x1)
	// with n distinct points and no duplicated endpoint.
	double dx() const {
		if (n < 2)
			throw std::invalid_argument("Grid1D: n must be >= 2");
		if (bc == BC::Periodic)
			return length() / static_cast<double>(n);
		return length() / static_cast<double>(n - 1);
	}

	double x(size_t i) const { return x0 + static_cast<double>(i) * dx(); }
};

struct Grid2D {
	double x0 = 0.0;
	double x1 = 1.0;
	double y0 = 0.0;
	double y1 = 1.0;
	size_t nx = 41;
	size_t ny = 41;

	double dx() const {
		if (nx < 2)
			throw std::invalid_argument("Grid2D: nx must be >= 2");
		return (x1 - x0) / static_cast<double>(nx - 1);
	}
	double dy() const {
		if (ny < 2)
			throw std::invalid_argument("Grid2D: ny must be >= 2");
		return (y1 - y0) / static_cast<double>(ny - 1);
	}
	double x(size_t i) const { return x0 + static_cast<double>(i) * dx(); }
	double y(size_t j) const { return y0 + static_cast<double>(j) * dy(); }
};

// Time history for a 1-D field.
struct PdeSolution1D {
	std::vector<double> t;
	std::vector<Array<double>> u;
};

namespace detail {

// Thomas algorithm for a tridiagonal system. a[0] and c[n-1] are unused.
inline std::vector<double> solve_tridiagonal(std::vector<double> a,
											 std::vector<double> b,
											 std::vector<double> c,
											 std::vector<double> d) {
	const size_t n = b.size();
	if (a.size() != n || c.size() != n || d.size() != n)
		throw std::invalid_argument("tridiagonal: size mismatch");
	if (n == 0)
		return {};
	for (size_t i = 1; i < n; ++i) {
		const double m = a[i] / b[i - 1];
		b[i] -= m * c[i - 1];
		d[i] -= m * d[i - 1];
	}
	std::vector<double> x(n);
	x[n - 1] = d[n - 1] / b[n - 1];
	for (size_t i = n - 1; i-- > 0;)
		x[i] = (d[i] - c[i] * x[i + 1]) / b[i];
	return x;
}

inline void check_field_size(const Grid1D &g, const Array<double> &u,
							 const char *name) {
	if (u.ndim() != 1 || u.size() != g.n)
		throw std::invalid_argument(std::string(name) +
									": expected a 1-D field of length " +
									std::to_string(g.n));
}

} // namespace detail

// ---------------------------------------------------------------------------
// Heat equation: u_t = alpha u_xx
// ---------------------------------------------------------------------------
// Explicit FTCS. Stable when alpha*dt/dx^2 <= 1/2.
inline PdeSolution1D heat_1d(const Grid1D &g, const Array<double> &u0,
							 double alpha, double t_end, double dt,
							 double left = 0.0, double right = 0.0) {
	detail::check_field_size(g, u0, "heat_1d");
	if (alpha < 0)
		throw std::invalid_argument("heat_1d: alpha must be >= 0");
	if (dt <= 0 || t_end < 0)
		throw std::invalid_argument("heat_1d: need dt > 0 and t_end >= 0");
	const size_t n = g.n;
	const double dx = g.dx();
	PdeSolution1D sol;
	sol.t.push_back(0.0);
	sol.u.push_back(u0);
	Array<double> u = u0;
	double t = 0.0;

	auto step = [&](double h) {
		const double r = alpha * h / (dx * dx);
		if (r > 0.5 + 1e-12)
			throw std::invalid_argument(
				"heat_1d: unstable, require alpha*dt/dx^2 <= 0.5");
		Array<double> un(n);
		if (g.bc == BC::Periodic) {
			for (size_t i = 0; i < n; ++i) {
				const size_t ip = (i + 1) % n;
				const size_t im = (i + n - 1) % n;
				un[i] = u[i] + r * (u[ip] - 2.0 * u[i] + u[im]);
			}
		} else {
			un[0] = left;
			un[n - 1] = right;
			for (size_t i = 1; i + 1 < n; ++i)
				un[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
		}
		u = std::move(un);
	};

	if (g.bc == BC::Dirichlet) {
		u[0] = left;
		u[n - 1] = right;
	}
	while (t < t_end - 1e-12) {
		const double h = std::min(dt, t_end - t);
		step(h);
		t += h;
		sol.t.push_back(t);
		sol.u.push_back(u);
	}
	return sol;
}

// Crank-Nicolson (unconditionally stable for the heat equation). Dirichlet
// boundaries only; the tridiagonal solve is direct.
inline PdeSolution1D heat_1d_cn(const Grid1D &g, const Array<double> &u0,
								double alpha, double t_end, double dt,
								double left = 0.0, double right = 0.0) {
	detail::check_field_size(g, u0, "heat_1d_cn");
	if (g.bc != BC::Dirichlet)
		throw std::invalid_argument(
			"heat_1d_cn: only Dirichlet boundaries are supported");
	if (alpha < 0)
		throw std::invalid_argument("heat_1d_cn: alpha must be >= 0");
	if (dt <= 0 || t_end < 0)
		throw std::invalid_argument("heat_1d_cn: need dt > 0 and t_end >= 0");
	const size_t n = g.n;
	const double dx = g.dx();
	PdeSolution1D sol;
	sol.t.push_back(0.0);
	sol.u.push_back(u0);
	Array<double> u = u0;
	u[0] = left;
	u[n - 1] = right;
	double t = 0.0;

	auto step = [&](double h) {
		const double r = alpha * h / (dx * dx);
		const size_t m = n - 2; // interior unknowns
		std::vector<double> a(m, -0.5 * r), b(m, 1.0 + r), c(m, -0.5 * r),
			d(m, 0.0);
		for (size_t k = 0; k < m; ++k) {
			const size_t i = k + 1;
			d[k] = 0.5 * r * u[i - 1] + (1.0 - r) * u[i] + 0.5 * r * u[i + 1];
		}
		// Move known boundary contributions to the RHS.
		d.front() += 0.5 * r * left;
		d.back() += 0.5 * r * right;
		auto x = detail::solve_tridiagonal(std::move(a), std::move(b),
										   std::move(c), std::move(d));
		Array<double> un(n);
		un[0] = left;
		un[n - 1] = right;
		for (size_t k = 0; k < m; ++k)
			un[k + 1] = x[k];
		u = std::move(un);
	};

	while (t < t_end - 1e-12) {
		const double h = std::min(dt, t_end - t);
		step(h);
		t += h;
		sol.t.push_back(t);
		sol.u.push_back(u);
	}
	return sol;
}

// ---------------------------------------------------------------------------
// Wave equation: u_tt = c^2 u_xx
// ---------------------------------------------------------------------------
// Explicit leapfrog with a second-order startup step. Homogeneous Dirichlet
// boundaries; requires the CFL condition c*dt/dx <= 1.
inline PdeSolution1D wave_1d(const Grid1D &g, const Array<double> &u0,
							 const Array<double> &v0, double c, double t_end,
							 double dt) {
	detail::check_field_size(g, u0, "wave_1d");
	detail::check_field_size(g, v0, "wave_1d");
	if (g.bc != BC::Dirichlet)
		throw std::invalid_argument(
			"wave_1d: only Dirichlet boundaries are supported");
	if (c < 0)
		throw std::invalid_argument("wave_1d: c must be >= 0");
	if (dt <= 0 || t_end < 0)
		throw std::invalid_argument("wave_1d: need dt > 0 and t_end >= 0");
	const size_t n = g.n;
	const double dx = g.dx();
	PdeSolution1D sol;
	Array<double> um = u0;
	um[0] = um[n - 1] = 0.0;
	sol.t.push_back(0.0);
	sol.u.push_back(um);
	double t = 0.0;
	if (n < 3 || t_end == 0.0)
		return sol;

	// Startup: u^1 = u^0 + h v^0 + (c h/dx)^2 / 2 * L u^0.
	const double h0 = std::min(dt, t_end);
	const double r0 = c * h0 / dx;
	if (r0 > 1.0 + 1e-12)
		throw std::invalid_argument("wave_1d: CFL violated, need c*dt/dx <= 1");
	Array<double> u(n);
	u[0] = u[n - 1] = 0.0;
	for (size_t i = 1; i + 1 < n; ++i)
		u[i] = um[i] + h0 * v0[i] +
			   0.5 * r0 * r0 * (um[i + 1] - 2.0 * um[i] + um[i - 1]);

	t = h0;
	sol.t.push_back(t);
	sol.u.push_back(u);

	while (t < t_end - 1e-12) {
		const double h = std::min(dt, t_end - t);
		const double r = c * h / dx;
		if (r > 1.0 + 1e-12)
			throw std::invalid_argument(
				"wave_1d: CFL violated, need c*dt/dx <= 1");
		Array<double> un(n);
		un[0] = un[n - 1] = 0.0;
		for (size_t i = 1; i + 1 < n; ++i)
			un[i] =
				2.0 * u[i] - um[i] + r * r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
		um = std::move(u);
		u = std::move(un);
		t += h;
		sol.t.push_back(t);
		sol.u.push_back(u);
	}
	return sol;
}

// ---------------------------------------------------------------------------
// Advection equation: u_t + a u_x = 0
// ---------------------------------------------------------------------------
// First-order upwind. Stable when |a|*dt/dx <= 1. Periodic grids wrap around;
// Dirichlet grids hold both boundaries at zero.
inline PdeSolution1D advection_1d(const Grid1D &g, const Array<double> &u0,
								  double a, double t_end, double dt) {
	detail::check_field_size(g, u0, "advection_1d");
	if (dt <= 0 || t_end < 0)
		throw std::invalid_argument("advection_1d: need dt > 0 and t_end >= 0");
	const size_t n = g.n;
	const double dx = g.dx();
	PdeSolution1D sol;
	sol.t.push_back(0.0);
	Array<double> u = u0;
	if (g.bc == BC::Dirichlet)
		u[0] = u[n - 1] = 0.0;
	sol.u.push_back(u);
	double t = 0.0;

	auto step = [&](double h) {
		const double C = a * h / dx;
		if (std::abs(C) > 1.0 + 1e-12)
			throw std::invalid_argument(
				"advection_1d: CFL violated, need |a|*dt/dx <= 1");
		Array<double> un(n);
		if (g.bc == BC::Periodic) {
			for (size_t i = 0; i < n; ++i) {
				if (a >= 0.0) {
					const size_t im = (i + n - 1) % n;
					un[i] = u[i] - C * (u[i] - u[im]);
				} else {
					const size_t ip = (i + 1) % n;
					un[i] = u[i] - C * (u[ip] - u[i]);
				}
			}
		} else {
			un[0] = un[n - 1] = 0.0;
			for (size_t i = 1; i + 1 < n; ++i) {
				if (a >= 0.0)
					un[i] = u[i] - C * (u[i] - u[i - 1]);
				else
					un[i] = u[i] - C * (u[i + 1] - u[i]);
			}
		}
		u = std::move(un);
	};

	while (t < t_end - 1e-12) {
		const double h = std::min(dt, t_end - t);
		step(h);
		t += h;
		sol.t.push_back(t);
		sol.u.push_back(u);
	}
	return sol;
}

// ---------------------------------------------------------------------------
// Poisson equation: -u'' = f
// ---------------------------------------------------------------------------
// Dirichlet boundaries u(x0)=left, u(x1)=right. f is a field on the full grid;
// only interior values are used.
inline Array<double> poisson_1d(const Grid1D &g, const Array<double> &f,
								double left = 0.0, double right = 0.0) {
	detail::check_field_size(g, f, "poisson_1d");
	if (g.bc != BC::Dirichlet)
		throw std::invalid_argument(
			"poisson_1d: only Dirichlet boundaries are supported");
	const size_t n = g.n;
	if (n < 3)
		throw std::invalid_argument("poisson_1d: need at least 3 points");
	const double dx2 = g.dx() * g.dx();
	const size_t m = n - 2;
	std::vector<double> a(m, -1.0), b(m, 2.0), c(m, -1.0), d(m, 0.0);
	for (size_t k = 0; k < m; ++k)
		d[k] = dx2 * f[k + 1];
	d.front() += left;
	d.back() += right;
	auto x = detail::solve_tridiagonal(std::move(a), std::move(b), std::move(c),
									   std::move(d));
	Array<double> u(n);
	u[0] = left;
	u[n - 1] = right;
	for (size_t k = 0; k < m; ++k)
		u[k + 1] = x[k];
	return u;
}

// ---------------------------------------------------------------------------
// Poisson equation in 2-D: -(u_xx + u_yy) = f
// ---------------------------------------------------------------------------
// Five-point stencil solved with red-black-free SOR (lexicographic sweeps).
// Dirichlet boundary values are supplied per side; f is (nx, ny) and only
// interior values are used. Throws if SOR does not converge within max_iter.
inline Array<double> poisson_2d(const Grid2D &g, const Array<double> &f,
								double tol = 1e-10, int max_iter = 100000,
								double omega = 1.8, double left = 0.0,
								double right = 0.0, double bottom = 0.0,
								double top = 0.0) {
	if (f.ndim() != 2 || f.shape()[0] != g.nx || f.shape()[1] != g.ny)
		throw std::invalid_argument("poisson_2d: f must have shape (nx, ny)");
	if (tol <= 0 || max_iter <= 0 || omega <= 0 || omega >= 2)
		throw std::invalid_argument(
			"poisson_2d: require tol>0, max_iter>0, 0<omega<2");
	const size_t nx = g.nx, ny = g.ny;
	if (nx < 3 || ny < 3)
		throw std::invalid_argument("poisson_2d: need nx,ny >= 3");
	const double dx2 = g.dx() * g.dx();
	const double dy2 = g.dy() * g.dy();
	const double den = 2.0 / dx2 + 2.0 / dy2;

	Array<double> u({nx, ny});
	for (size_t i = 0; i < nx; ++i) {
		u.at(i, 0) = bottom;
		u.at(i, ny - 1) = top;
	}
	for (size_t j = 0; j < ny; ++j) {
		u.at(0, j) = left;
		u.at(nx - 1, j) = right;
	}

	bool converged = false;
	for (int it = 0; it < max_iter; ++it) {
		double maxdiff = 0.0;
		for (size_t i = 1; i + 1 < nx; ++i) {
			for (size_t j = 1; j + 1 < ny; ++j) {
				const double gs =
					((u.at(i - 1, j) + u.at(i + 1, j)) / dx2 +
					 (u.at(i, j - 1) + u.at(i, j + 1)) / dy2 + f.at(i, j)) /
					den;
				const double nv = u.at(i, j) + omega * (gs - u.at(i, j));
				maxdiff = std::max(maxdiff, std::abs(nv - u.at(i, j)));
				u.at(i, j) = nv;
			}
		}
		if (maxdiff <= tol) {
			converged = true;
			break;
		}
	}
	if (!converged)
		throw std::runtime_error("poisson_2d: SOR did not converge");
	return u;
}

} // namespace dracolix::pde
