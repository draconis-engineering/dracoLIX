#pragma once

// ODE solvers
// Explicit: Euler, RK4, adaptive RK45 (step doubling)
// Implicit/stiff: backward Euler, BDF2 (Newton iteration with an optional
// analytic Jacobian, finite-difference Jacobian otherwise)
// Unified entry point: solve_ivp()
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

namespace dracolix::ode {

using State = std::vector<double>;
using RHS = std::function<State(double t, const State &y)>;

// Row-major n x n Jacobian df/dy at (t, y).
using Jacobian = std::function<std::vector<double>(double t, const State &y)>;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
// Utility function for state addition
inline State add_state(const State &a, const State &b, double s = 1.0) {
	State r(a.size());
	for (size_t i = 0; i < a.size(); ++i)
		r[i] = a[i] + s * b[i];
	return r;
}

// Utility function for state scaling
inline State scale_state(const State &a, double s) {
	State r(a.size());
	for (size_t i = 0; i < a.size(); ++i)
		r[i] = a[i] * s;
	return r;
}

// ---------------------------------------------------------------------------
// Newton options
// ---------------------------------------------------------------------------
struct NewtonOptions {
	int max_iter = 50;
	double tol = 1e-10;
};

namespace detail {

inline double norm_inf(const State &v) {
	double m = 0.0;
	for (double x : v)
		m = std::max(m, std::abs(x));
	return m;
}

// Solve A x = b for a dense row-major n x n A using Gaussian elimination with
// partial pivoting. A is copied; throws if the matrix is singular.
inline State solve_dense(std::vector<double> A, State b) {
	const size_t n = b.size();
	if (A.size() != n * n)
		throw std::invalid_argument("solve_dense: shape mismatch");
	for (size_t col = 0; col < n; ++col) {
		size_t piv = col;
		double best = std::abs(A[col * n + col]);
		for (size_t row = col + 1; row < n; ++row) {
			double v = std::abs(A[row * n + col]);
			if (v > best) {
				best = v;
				piv = row;
			}
		}
		if (best <= std::numeric_limits<double>::min())
			throw std::runtime_error("solve_dense: singular Jacobian");
		if (piv != col) {
			for (size_t k = 0; k < n; ++k)
				std::swap(A[col * n + k], A[piv * n + k]);
			std::swap(b[col], b[piv]);
		}
		const double d = A[col * n + col];
		for (size_t row = col + 1; row < n; ++row) {
			const double f = A[row * n + col] / d;
			if (f == 0.0)
				continue;
			A[row * n + col] = 0.0;
			for (size_t k = col + 1; k < n; ++k)
				A[row * n + k] -= f * A[col * n + k];
			b[row] -= f * b[col];
		}
	}
	State x(n);
	for (size_t i = n; i-- > 0;) {
		double s = b[i];
		for (size_t k = i + 1; k < n; ++k)
			s -= A[i * n + k] * x[k];
		x[i] = s / A[i * n + i];
	}
	return x;
}

// Finite-difference Jacobian of the RHS (central differences where possible).
inline std::vector<double> fd_jacobian(RHS f, double t, const State &y) {
	const size_t n = y.size();
	std::vector<double> J(n * n, 0.0);
	const double eps = std::sqrt(std::numeric_limits<double>::epsilon());
	State yp = y, ym = y;
	for (size_t j = 0; j < n; ++j) {
		const double dy = eps * std::max(1.0, std::abs(y[j]));
		yp[j] = y[j] + dy;
		ym[j] = y[j] - dy;
		State fp = f(t, yp);
		State fm = f(t, ym);
		if (fp.size() != n || fm.size() != n)
			throw std::runtime_error("ODE RHS returned a wrong-sized state");
		for (size_t i = 0; i < n; ++i)
			J[i * n + j] = (fp[i] - fm[i]) / (2.0 * dy);
		yp[j] = y[j];
		ym[j] = y[j];
	}
	return J;
}

inline std::vector<double> rhs_jacobian(RHS f, const Jacobian &jac, double t,
										const State &y) {
	if (jac)
		return jac(t, y);
	return fd_jacobian(f, t, y);
}

// Solve one implicit step
//   y = a*y_n + b*y_{n-1} + c*h*f(t, y)
// with Newton's method. The iteration matrix is I - c*h*J_f.
inline State solve_implicit(RHS f, const Jacobian &jac, double t, State y,
							const State &yn, const State &ynm1, double a,
							double b, double c, double h,
							const NewtonOptions &opts) {
	const size_t n = y.size();
	const double scale = 1.0 + norm_inf(y);

	for (int iter = 0; iter < opts.max_iter; ++iter) {
		State fy = f(t, y);
		if (fy.size() != n)
			throw std::runtime_error("ODE RHS returned a wrong-sized state");
		State G(n);
		for (size_t i = 0; i < n; ++i)
			G[i] = y[i] - a * yn[i] - b * ynm1[i] - c * h * fy[i];
		const double res = norm_inf(G);
		if (res <= opts.tol * scale)
			return y;

		std::vector<double> Jf = rhs_jacobian(f, jac, t, y);
		if (Jf.size() != n * n)
			throw std::runtime_error("ODE Jacobian returned a wrong size");
		std::vector<double> JG(n * n, 0.0);
		for (size_t i = 0; i < n; ++i) {
			for (size_t j = 0; j < n; ++j)
				JG[i * n + j] = -c * h * Jf[i * n + j];
			JG[i * n + i] += 1.0;
		}
		State rhs(n);
		for (size_t i = 0; i < n; ++i)
			rhs[i] = -G[i];
		State delta = solve_dense(JG, rhs);

		// Damped Newton: accept the first step that reduces the residual.
		double lambda = 1.0;
		bool accepted = false;
		for (int ls = 0; ls < 10; ++ls) {
			State trial(n);
			for (size_t i = 0; i < n; ++i)
				trial[i] = y[i] + lambda * delta[i];
			State ft = f(t, trial);
			if (ft.size() != n)
				throw std::runtime_error(
					"ODE RHS returned a wrong-sized state");
			State Gt(n);
			for (size_t i = 0; i < n; ++i)
				Gt[i] = trial[i] - a * yn[i] - b * ynm1[i] - c * h * ft[i];
			if (norm_inf(Gt) < res || norm_inf(Gt) <= opts.tol * scale) {
				y = std::move(trial);
				accepted = true;
				break;
			}
			lambda *= 0.5;
		}
		if (!accepted)
			throw std::runtime_error(
				"ODE implicit solver: Newton line search failed");
	}
	throw std::runtime_error(
		"ODE implicit solver: Newton did not converge in max_iter");
}

inline State euler_step(RHS f, double t, const State &y, double h) {
	State dy = f(t, y);
	State out(y.size());
	for (size_t i = 0; i < y.size(); ++i)
		out[i] = y[i] + h * dy[i];
	return out;
}

inline State rk4_step(RHS f, double t, const State &y, double h) {
	State k1 = f(t, y);
	State k2 = f(t + h / 2, add_state(y, scale_state(k1, h / 2)));
	State k3 = f(t + h / 2, add_state(y, scale_state(k2, h / 2)));
	State k4 = f(t + h, add_state(y, scale_state(k3, h)));
	State out(y.size());
	for (size_t i = 0; i < y.size(); ++i)
		out[i] = y[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
	return out;
}

// Fixed step sizes: dt everywhere except the final (possibly shorter) step.
inline std::vector<double> fixed_times(double t0, double t1, double dt) {
	std::vector<double> ts;
	ts.push_back(t0);
	double t = t0;
	while (t < t1 - 1e-12) {
		t += std::min(dt, t1 - t);
		ts.push_back(t);
	}
	return ts;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Fixed-step explicit methods
// ---------------------------------------------------------------------------
// Fixed-step Euler
// Integrates the ODE using the fixed-step Euler method
// Returns the trajectory of the solution as a vector of State objects
inline std::vector<State> euler(RHS f, double t0, State y0, double t1,
								double dt) {
	if (dt <= 0)
		throw std::invalid_argument("dt>0");
	std::vector<State> traj;
	traj.reserve(size_t((t1 - t0) / dt) + 1);
	double t = t0;
	State y = y0;
	traj.push_back(y);
	while (t < t1 - 1e-12) {
		double h = std::min(dt, t1 - t);
		State dy = f(t, y);
		for (size_t i = 0; i < y.size(); ++i)
			y[i] += h * dy[i];
		t += h;
		traj.push_back(y);
	}
	return traj;
}

// Classical Runge Kutta 4
// Integrates the ODE using the classical RK4 method
// Returns the trajectory of the solution as a vector of State objects
inline std::vector<State> rk4(RHS f, double t0, State y0, double t1,
							  double dt) {
	if (dt <= 0)
		throw std::invalid_argument("dt>0");
	std::vector<State> traj;
	traj.reserve(size_t((t1 - t0) / dt) + 1);
	double t = t0;
	State y = y0;
	traj.push_back(y);
	while (t < t1 - 1e-12) {
		double h = std::min(dt, t1 - t);
		State k1 = f(t, y);
		State k2 = f(t + h / 2, add_state(y, scale_state(k1, h / 2)));
		State k3 = f(t + h / 2, add_state(y, scale_state(k2, h / 2)));
		State k4 = f(t + h, add_state(y, scale_state(k3, h)));
		for (size_t i = 0; i < y.size(); ++i)
			y[i] += h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
		t += h;
		traj.push_back(y);
	}
	return traj;
}

// ---------------------------------------------------------------------------
// Implicit / stiff methods
// ---------------------------------------------------------------------------
// Backward (implicit) Euler, first-order. A-stable, unconditionally stable for
// linear problems, at the cost of a nonlinear solve per step.
inline std::vector<State> backward_euler(RHS f, double t0, State y0, double t1,
										 double dt,
										 const Jacobian &jac = Jacobian{},
										 const NewtonOptions &opts = {}) {
	if (dt <= 0)
		throw std::invalid_argument("dt>0");
	std::vector<State> traj;
	traj.reserve(size_t((t1 - t0) / dt) + 1);
	double t = t0;
	State y = y0;
	traj.push_back(y);
	while (t < t1 - 1e-12) {
		double h = std::min(dt, t1 - t);
		State yn = y;
		// y = yn + h f(t+h, y)  (a=1, b=0, c=1)
		y = detail::solve_implicit(f, jac, t + h, y, yn, yn, 1.0, 0.0, 1.0, h,
								   opts);
		t += h;
		traj.push_back(y);
	}
	return traj;
}

// Variable-step BDF2 (second order, A-stable). The first step uses backward
// Euler; after that the two-step formula
//   y_{n+1} = a y_n + b y_{n-1} + c h f(t_{n+1}, y_{n+1})
// is used, with coefficients depending on the step-size ratio.
inline std::vector<State> bdf2(RHS f, double t0, State y0, double t1, double dt,
							   const Jacobian &jac = Jacobian{},
							   const NewtonOptions &opts = {}) {
	if (dt <= 0)
		throw std::invalid_argument("dt>0");
	std::vector<State> traj;
	traj.reserve(size_t((t1 - t0) / dt) + 1);
	double t = t0;
	State ynm1 = y0; // y_{n-1}
	State yn = y0;	 // y_n
	traj.push_back(y0);
	double hprev = 0.0;
	bool first = true;
	while (t < t1 - 1e-12) {
		double h = std::min(dt, t1 - t);
		State ynew;
		if (first) {
			ynew = detail::solve_implicit(f, jac, t + h, yn, yn, ynm1, 1.0, 0.0,
										  1.0, h, opts);
			first = false;
		} else {
			const double w = h / hprev;
			const double den = 1.0 + 2.0 * w;
			const double a = (1.0 + w) * (1.0 + w) / den;
			const double b = -(w * w) / den;
			const double c = (1.0 + w) / den;
			ynew = detail::solve_implicit(f, jac, t + h, yn, yn, ynm1, a, b, c,
										  h, opts);
		}
		ynm1 = yn;
		yn = ynew;
		hprev = h;
		t += h;
		traj.push_back(yn);
	}
	return traj;
}

// ---------------------------------------------------------------------------
// Adaptive RK45
// ---------------------------------------------------------------------------
struct OdeSolution {
	std::vector<double> t;
	std::vector<State> y;
	bool success = true;
	std::string message;
};

// Adaptive Runge Kutta 45 with step doubling (embedded error estimate). This
// variant also records the accepted time points.
inline OdeSolution adaptive_rk45_solution(RHS f, double t0, State y0, double t1,
										  double tol = 1e-6, double h0 = 0.1) {
	if (h0 <= 0)
		throw std::invalid_argument("h0>0");
	OdeSolution sol;
	sol.t.push_back(t0);
	sol.y.push_back(y0);
	double t = t0;
	State y = y0;
	double h = h0;

	auto rk_step = [&](double tt, const State &yy, double hh) {
		State k1 = f(tt, yy);
		State k2 = f(tt + hh / 2, add_state(yy, scale_state(k1, hh / 2)));
		State k3 = f(tt + hh / 2, add_state(yy, scale_state(k2, hh / 2)));
		State k4 = f(tt + hh, add_state(yy, scale_state(k3, hh)));
		State y4(yy.size());
		for (size_t i = 0; i < yy.size(); ++i)
			y4[i] = yy[i] + hh / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);

		State y_half = yy;
		double th = tt;
		for (int s = 0; s < 2; ++s) {
			double hs = hh / 2;
			State k1h = f(th, y_half);
			State k2h =
				f(th + hs / 2, add_state(y_half, scale_state(k1h, hs / 2)));
			State k3h =
				f(th + hs / 2, add_state(y_half, scale_state(k2h, hs / 2)));
			State k4h = f(th + hs, add_state(y_half, scale_state(k3h, hs)));
			for (size_t i = 0; i < y_half.size(); ++i)
				y_half[i] +=
					hs / 6 * (k1h[i] + 2 * k2h[i] + 2 * k3h[i] + k4h[i]);
			th += hs;
		}
		return std::pair<State, State>(y4, y_half);
	};

	while (t < t1 - 1e-12) {
		if (t + h > t1)
			h = t1 - t;
		auto [y4, y_half] = rk_step(t, y, h);
		double err = 0;
		for (size_t i = 0; i < y.size(); ++i)
			err += std::pow(y4[i] - y_half[i], 2);
		err = std::sqrt(err);
		if (err < tol || h < 1e-12) {
			y = y_half;
			t += h;
			sol.t.push_back(t);
			sol.y.push_back(y);
			if (err < tol / 10)
				h *= 1.5;
		} else {
			h *= 0.5;
		}
		if (sol.y.size() > 100000) {
			sol.success = false;
			sol.message = "adaptive_rk45: step limit exceeded";
			break;
		}
	}
	return sol;
}

// Backwards-compatible trajectory-only adaptive RK45.
inline std::vector<State> adaptive_rk45(RHS f, double t0, State y0, double t1,
										double tol = 1e-6, double h0 = 0.1) {
	return adaptive_rk45_solution(f, t0, y0, t1, tol, h0).y;
}

// ---------------------------------------------------------------------------
// Unified entry point
// ---------------------------------------------------------------------------
enum class Method { Euler, RK4, AdaptiveRK45, BackwardEuler, BDF2 };

inline const char *method_name(Method m) {
	switch (m) {
	case Method::Euler:
		return "euler";
	case Method::RK4:
		return "rk4";
	case Method::AdaptiveRK45:
		return "rk45";
	case Method::BackwardEuler:
		return "backward_euler";
	case Method::BDF2:
		return "bdf2";
	}
	return "unknown";
}

inline Method parse_method(const std::string &name) {
	if (name == "euler")
		return Method::Euler;
	if (name == "rk4")
		return Method::RK4;
	if (name == "rk45" || name == "adaptive_rk45")
		return Method::AdaptiveRK45;
	if (name == "backward_euler" || name == "implicit_euler" || name == "be")
		return Method::BackwardEuler;
	if (name == "bdf2")
		return Method::BDF2;
	throw std::invalid_argument("unknown ODE method: " + name);
}

// Integrate y' = f(t, y) from t0 to t1.
//
// ``step`` is the fixed step size for fixed-step methods, or the initial step
// for the adaptive method. When ``step <= 0`` a default is chosen
// ((t1-t0)/100 for fixed methods, h0=0.1 for adaptive). ``jac`` is optional;
// the implicit methods fall back to finite differences.
inline OdeSolution solve_ivp(RHS f, double t0, State y0, double t1,
							 Method method = Method::AdaptiveRK45,
							 double step = 0.0, double tol = 1e-6,
							 const Jacobian &jac = Jacobian{},
							 const NewtonOptions &newton = {}) {
	if (t1 < t0)
		throw std::invalid_argument("solve_ivp: t1 must be >= t0");
	if (t1 == t0) {
		OdeSolution sol;
		sol.t.push_back(t0);
		sol.y.push_back(std::move(y0));
		return sol;
	}
	const double span = t1 - t0;
	OdeSolution sol;
	if (method == Method::AdaptiveRK45) {
		const double h0 = step > 0 ? step : 0.1;
		return adaptive_rk45_solution(f, t0, std::move(y0), t1, tol, h0);
	}
	const double dt = step > 0 ? step : span / 100.0;
	sol.t = detail::fixed_times(t0, t1, dt);
	switch (method) {
	case Method::Euler:
		sol.y = euler(f, t0, std::move(y0), t1, dt);
		break;
	case Method::RK4:
		sol.y = rk4(f, t0, std::move(y0), t1, dt);
		break;
	case Method::BackwardEuler:
		sol.y = backward_euler(f, t0, std::move(y0), t1, dt, jac, newton);
		break;
	case Method::BDF2:
		sol.y = bdf2(f, t0, std::move(y0), t1, dt, jac, newton);
		break;
	default:
		throw std::logic_error("solve_ivp: unreachable method");
	}
	if (sol.t.size() != sol.y.size()) {
		sol.success = false;
		sol.message = "solve_ivp: time/state size mismatch";
	}
	return sol;
}

} // namespace dracolix::ode
