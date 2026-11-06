#pragma once

// Deterministic random number generation.
//
// A small seedable wrapper around std::mt19937_64 plus Array helpers. The
// engine is explicit (no hidden global state) so simulations are reproducible.
//
// Licensed under GPL-3.0+

#include "array.hpp"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace dracolix::random {

class Rng {
  public:
	explicit Rng(uint64_t seed = 0) : gen_(seed) {}

	void seed(uint64_t s) { gen_.seed(s); }

	uint64_t next_u64() { return gen_(); }

	// Uniform in [0, 1) with 53 bits of resolution.
	double uniform() {
		return static_cast<double>(gen_() >> 11) * (1.0 / 9007199254740992.0);
	}

	// Uniform in [a, b).
	double uniform(double a, double b) {
		if (!(b >= a))
			throw std::invalid_argument("Rng::uniform: need b >= a");
		return a + (b - a) * uniform();
	}

	// Uniform integer in [low, high).
	int64_t randint(int64_t low, int64_t high) {
		if (high <= low)
			throw std::invalid_argument("Rng::randint: need high > low");
		std::uniform_int_distribution<int64_t> dist(low, high - 1);
		return dist(gen_);
	}

	// Normal (Gaussian) sample.
	double normal(double mu = 0.0, double sigma = 1.0) {
		if (!(sigma > 0.0))
			throw std::invalid_argument("Rng::normal: need sigma > 0");
		std::normal_distribution<double> dist(mu, sigma);
		return dist(gen_);
	}

	// Bernoulli trial: true with probability p.
	bool bernoulli(double p) {
		if (p < 0.0 || p > 1.0)
			throw std::invalid_argument("Rng::bernoulli: need 0 <= p <= 1");
		return uniform() < p;
	}

	// Exponential sample with the given rate (mean = 1/rate).
	double exponential(double rate = 1.0) {
		if (!(rate > 0.0))
			throw std::invalid_argument("Rng::exponential: need rate > 0");
		std::exponential_distribution<double> dist(rate);
		return dist(gen_);
	}

	// Poisson sample.
	int64_t poisson(double lambda) {
		if (!(lambda > 0.0))
			throw std::invalid_argument("Rng::poisson: need lambda > 0");
		std::poisson_distribution<int64_t> dist(lambda);
		return dist(gen_);
	}

  private:
	std::mt19937_64 gen_;
};

// ---------------------------------------------------------------------------
// Array helpers
// ---------------------------------------------------------------------------
inline Array<double> uniform(const std::vector<size_t> &shape, double a,
							 double b, Rng &rng) {
	Array<double> out(shape);
	for (size_t i = 0; i < out.size(); ++i)
		out[i] = rng.uniform(a, b);
	return out;
}

inline Array<double> uniform(const std::vector<size_t> &shape, Rng &rng) {
	return uniform(shape, 0.0, 1.0, rng);
}

inline Array<double> normal(const std::vector<size_t> &shape, double mu,
							double sigma, Rng &rng) {
	Array<double> out(shape);
	for (size_t i = 0; i < out.size(); ++i)
		out[i] = rng.normal(mu, sigma);
	return out;
}

inline Array<double> normal(const std::vector<size_t> &shape, Rng &rng) {
	return normal(shape, 0.0, 1.0, rng);
}

inline Array<int64_t> randint(const std::vector<size_t> &shape, int64_t low,
							  int64_t high, Rng &rng) {
	Array<int64_t> out(shape);
	for (size_t i = 0; i < out.size(); ++i)
		out[i] = rng.randint(low, high);
	return out;
}

} // namespace dracolix::random
