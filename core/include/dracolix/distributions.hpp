#pragma once

// Common probability distributions: pdf/pmf, cdf, quantile and sampling.
//
// Every distribution is a small value type with validation in its
// constructor. Sampling takes an explicit random::Rng so results are
// reproducible.
//
// Licensed under GPL-3.0+

#include "array.hpp"
#include "random.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dracolix::distributions {

namespace detail {
constexpr double kSqrt2 = 1.41421356237309504880168872420969808;
constexpr double kInvSqrt2Pi = 0.398942280401432677939946059934381868;
constexpr double kLogSqrt2Pi = 0.918938533204672741780329736405617639;

template <typename Dist, typename Sampler>
Array<double> sample_array(const Dist &, const std::vector<size_t> &shape,
						   random::Rng &rng, Sampler &&sampler) {
	Array<double> out(shape);
	for (size_t i = 0; i < out.size(); ++i)
		out[i] = static_cast<double>(sampler(rng));
	return out;
}
} // namespace detail

// ---------------------------------------------------------------------------
// Normal
// ---------------------------------------------------------------------------
struct Normal {
	double mu;
	double sigma;

	Normal(double mu_ = 0.0, double sigma_ = 1.0) : mu(mu_), sigma(sigma_) {
		if (!(sigma > 0.0))
			throw std::invalid_argument("Normal: sigma must be > 0");
	}

	double mean() const { return mu; }
	double variance() const { return sigma * sigma; }

	double pdf(double x) const {
		const double z = (x - mu) / sigma;
		return detail::kInvSqrt2Pi / sigma * std::exp(-0.5 * z * z);
	}

	double logpdf(double x) const {
		const double z = (x - mu) / sigma;
		return -0.5 * z * z - detail::kLogSqrt2Pi - std::log(sigma);
	}

	double cdf(double x) const {
		const double z = (x - mu) / (sigma * detail::kSqrt2);
		return 0.5 * std::erfc(-z);
	}

	double quantile(double p) const {
		if (p < 0.0 || p > 1.0)
			throw std::invalid_argument("Normal::quantile: need 0 <= p <= 1");
		if (p == 0.0)
			return -std::numeric_limits<double>::infinity();
		if (p == 1.0)
			return std::numeric_limits<double>::infinity();
		// Bisection on the standard normal CDF; robust for extreme tails.
		double lo = -40.0, hi = 40.0;
		for (int i = 0; i < 200; ++i) {
			const double mid = 0.5 * (lo + hi);
			if (0.5 * std::erfc(-mid / detail::kSqrt2) < p)
				lo = mid;
			else
				hi = mid;
		}
		return mu + sigma * 0.5 * (lo + hi);
	}

	double sample(random::Rng &rng) const { return rng.normal(mu, sigma); }

	Array<double> sample(const std::vector<size_t> &shape,
						 random::Rng &rng) const {
		return detail::sample_array(*this, shape, rng, [&](random::Rng &r) {
			return r.normal(mu, sigma);
		});
	}
};

// ---------------------------------------------------------------------------
// Continuous uniform
// ---------------------------------------------------------------------------
struct Uniform {
	double a;
	double b;

	Uniform(double a_ = 0.0, double b_ = 1.0) : a(a_), b(b_) {
		if (!(b > a))
			throw std::invalid_argument("Uniform: need b > a");
	}

	double mean() const { return 0.5 * (a + b); }
	double variance() const { return (b - a) * (b - a) / 12.0; }

	double pdf(double x) const {
		return (x < a || x > b) ? 0.0 : 1.0 / (b - a);
	}

	double cdf(double x) const {
		if (x <= a)
			return 0.0;
		if (x >= b)
			return 1.0;
		return (x - a) / (b - a);
	}

	double quantile(double p) const {
		if (p < 0.0 || p > 1.0)
			throw std::invalid_argument("Uniform::quantile: need 0 <= p <= 1");
		return a + p * (b - a);
	}

	double sample(random::Rng &rng) const { return rng.uniform(a, b); }

	Array<double> sample(const std::vector<size_t> &shape,
						 random::Rng &rng) const {
		return detail::sample_array(
			*this, shape, rng, [&](random::Rng &r) { return r.uniform(a, b); });
	}
};

// ---------------------------------------------------------------------------
// Exponential
// ---------------------------------------------------------------------------
struct Exponential {
	double rate;

	explicit Exponential(double rate_ = 1.0) : rate(rate_) {
		if (!(rate > 0.0))
			throw std::invalid_argument("Exponential: rate must be > 0");
	}

	double mean() const { return 1.0 / rate; }
	double variance() const { return 1.0 / (rate * rate); }

	double pdf(double x) const {
		return x < 0.0 ? 0.0 : rate * std::exp(-rate * x);
	}

	double cdf(double x) const {
		return x <= 0.0 ? 0.0 : 1.0 - std::exp(-rate * x);
	}

	double quantile(double p) const {
		if (p < 0.0 || p > 1.0)
			throw std::invalid_argument(
				"Exponential::quantile: need 0 <= p <= 1");
		if (p == 1.0)
			return std::numeric_limits<double>::infinity();
		return -std::log1p(-p) / rate;
	}

	double sample(random::Rng &rng) const { return rng.exponential(rate); }

	Array<double> sample(const std::vector<size_t> &shape,
						 random::Rng &rng) const {
		return detail::sample_array(*this, shape, rng, [&](random::Rng &r) {
			return r.exponential(rate);
		});
	}
};

// ---------------------------------------------------------------------------
// Bernoulli
// ---------------------------------------------------------------------------
struct Bernoulli {
	double p;

	explicit Bernoulli(double p_ = 0.5) : p(p_) {
		if (p < 0.0 || p > 1.0)
			throw std::invalid_argument("Bernoulli: need 0 <= p <= 1");
	}

	double mean() const { return p; }
	double variance() const { return p * (1.0 - p); }

	double pmf(int64_t k) const {
		if (k == 0)
			return 1.0 - p;
		if (k == 1)
			return p;
		return 0.0;
	}

	double cdf(int64_t k) const {
		if (k < 0)
			return 0.0;
		if (k == 0)
			return 1.0 - p;
		return 1.0;
	}

	int64_t quantile(double q) const {
		if (q < 0.0 || q > 1.0)
			throw std::invalid_argument(
				"Bernoulli::quantile: need 0 <= q <= 1");
		return q <= 1.0 - p ? 0 : 1;
	}

	double sample(random::Rng &rng) const {
		return rng.bernoulli(p) ? 1.0 : 0.0;
	}

	Array<double> sample(const std::vector<size_t> &shape,
						 random::Rng &rng) const {
		return detail::sample_array(*this, shape, rng, [&](random::Rng &r) {
			return r.bernoulli(p) ? 1.0 : 0.0;
		});
	}
};

// ---------------------------------------------------------------------------
// Poisson
// ---------------------------------------------------------------------------
struct Poisson {
	double lambda;

	explicit Poisson(double lambda_ = 1.0) : lambda(lambda_) {
		if (!(lambda > 0.0))
			throw std::invalid_argument("Poisson: lambda must be > 0");
	}

	double mean() const { return lambda; }
	double variance() const { return lambda; }

	double pmf(int64_t k) const {
		if (k < 0)
			return 0.0;
		return std::exp(-lambda + static_cast<double>(k) * std::log(lambda) -
						std::lgamma(static_cast<double>(k) + 1.0));
	}

	double cdf(int64_t k) const {
		if (k < 0)
			return 0.0;
		const int64_t kmax = k > 1000000 ? 1000000 : k;
		double sum = 0.0;
		double term = std::exp(-lambda);
		for (int64_t i = 0; i <= kmax; ++i) {
			sum += term;
			term *= lambda / static_cast<double>(i + 1);
		}
		return std::min(1.0, sum);
	}

	int64_t quantile(double q) const {
		if (q < 0.0 || q > 1.0)
			throw std::invalid_argument("Poisson::quantile: need 0 <= q <= 1");
		if (q == 0.0)
			return 0;
		double sum = std::exp(-lambda);
		double term = sum;
		int64_t k = 0;
		while (sum < q && k < 1000000) {
			++k;
			term *= lambda / static_cast<double>(k);
			sum += term;
		}
		return k;
	}

	double sample(random::Rng &rng) const {
		return static_cast<double>(rng.poisson(lambda));
	}

	Array<double> sample(const std::vector<size_t> &shape,
						 random::Rng &rng) const {
		return detail::sample_array(*this, shape, rng, [&](random::Rng &r) {
			return static_cast<double>(r.poisson(lambda));
		});
	}
};

} // namespace dracolix::distributions
