// Probability distributions: pdf/pmf, cdf, quantile and sampling.
// Licensed under GPL-3.0+

#include "../../core/include/dracolix/distributions.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace dracolix;
using namespace dracolix::distributions;
using dracolix::random::Rng;

namespace {
constexpr double PI = 3.14159265358979323846;

bool close(double a, double b, double tol = 1e-9) {
	return std::abs(a - b) <= tol;
}

void test_normal() {
	Normal n(0.0, 1.0);
	assert(close(n.pdf(0.0), 1.0 / std::sqrt(2.0 * PI)));
	assert(close(n.cdf(0.0), 0.5));
	assert(close(n.cdf(1.96), 0.9750021048517795, 1e-6));
	assert(close(n.quantile(0.5), 0.0, 1e-12));
	assert(close(n.quantile(0.975), 1.959963984540054, 1e-6));
	assert(close(n.logpdf(1.0), std::log(n.pdf(1.0))));
	assert(close(n.mean(), 0.0));
	assert(close(n.variance(), 1.0));

	for (double x : {-3.0, -0.5, 0.0, 0.7, 2.5}) {
		assert(close(n.cdf(n.quantile(n.cdf(x))), n.cdf(x), 1e-8));
	}

	Normal m(2.0, 3.0);
	assert(close(m.cdf(m.quantile(0.3)), 0.3, 1e-10));

	Rng rng(11);
	auto s = n.sample({10000}, rng);
	double sum = 0.0;
	for (size_t i = 0; i < s.size(); ++i)
		sum += s[i];
	assert(std::abs(sum / (double)s.size()) < 0.05);
	std::cout << " normal ok\n";
}

void test_uniform() {
	Uniform u(0.0, 2.0);
	assert(close(u.pdf(1.0), 0.5));
	assert(close(u.pdf(-1.0), 0.0));
	assert(close(u.pdf(3.0), 0.0));
	assert(close(u.cdf(-1.0), 0.0));
	assert(close(u.cdf(1.0), 0.5));
	assert(close(u.cdf(3.0), 1.0));
	assert(close(u.quantile(0.25), 0.5));
	assert(close(u.mean(), 1.0));
	assert(close(u.variance(), 1.0 / 3.0));

	Rng rng(3);
	auto s = u.sample({1000}, rng);
	for (size_t i = 0; i < s.size(); ++i)
		assert(s[i] >= 0.0 && s[i] < 2.0);
	std::cout << " uniform ok\n";
}

void test_exponential() {
	Exponential e(2.0);
	assert(close(e.pdf(0.0), 2.0));
	assert(close(e.cdf(0.0), 0.0));
	assert(close(e.cdf(1.0), 1.0 - std::exp(-2.0)));
	assert(close(e.quantile(0.5), std::log(2.0) / 2.0));
	assert(close(e.mean(), 0.5));
	assert(close(e.variance(), 0.25));
	assert(std::isinf(e.quantile(1.0)));

	Rng rng(8);
	double sum = 0.0;
	for (int i = 0; i < 50000; ++i)
		sum += e.sample(rng);
	assert(std::abs(sum / 50000.0 - 0.5) < 0.02);
	std::cout << " exponential ok\n";
}

void test_bernoulli() {
	Bernoulli b(0.3);
	assert(close(b.pmf(0), 0.7));
	assert(close(b.pmf(1), 0.3));
	assert(close(b.pmf(2), 0.0));
	assert(close(b.cdf(-1), 0.0));
	assert(close(b.cdf(0), 0.7));
	assert(close(b.cdf(1), 1.0));
	assert(b.quantile(0.5) == 0);
	assert(b.quantile(0.8) == 1);
	assert(close(b.mean(), 0.3));
	assert(close(b.variance(), 0.21));

	Rng rng(4);
	auto s = b.sample({5000}, rng);
	double ones = 0.0;
	for (size_t i = 0; i < s.size(); ++i) {
		assert(s[i] == 0.0 || s[i] == 1.0);
		ones += s[i];
	}
	assert(std::abs(ones / 5000.0 - 0.3) < 0.03);
	std::cout << " bernoulli ok\n";
}

void test_poisson() {
	Poisson p(2.0);
	assert(close(p.pmf(0), std::exp(-2.0)));
	assert(close(p.pmf(1), 2.0 * std::exp(-2.0)));
	assert(close(p.pmf(2), 2.0 * std::exp(-2.0)));
	assert(close(p.pmf(-1), 0.0));
	assert(close(p.cdf(-1), 0.0));
	assert(close(p.cdf(2), 5.0 * std::exp(-2.0)));
	double total = 0.0;
	for (int64_t k = 0; k <= 50; ++k)
		total += p.pmf(k);
	assert(close(total, 1.0, 1e-12));
	assert(p.quantile(0.5) == 2);
	assert(close(p.mean(), 2.0));
	assert(close(p.variance(), 2.0));

	Rng rng(6);
	double sum = 0.0;
	for (int i = 0; i < 50000; ++i)
		sum += p.sample(rng);
	assert(std::abs(sum / 50000.0 - 2.0) < 0.05);
	std::cout << " poisson ok\n";
}

void test_validation() {
	bool threw = false;
	try {
		Normal bad(0.0, 0.0);
	} catch (const std::invalid_argument &) {
		threw = true;
	}
	assert(threw);

	auto expect_throw = [](auto fn) {
		bool t = false;
		try {
			fn();
		} catch (const std::invalid_argument &) {
			t = true;
		}
		return t;
	};
	assert(expect_throw([] { Uniform u(2.0, 1.0); }));
	assert(expect_throw([] { Exponential e(0.0); }));
	assert(expect_throw([] { Bernoulli b(1.5); }));
	assert(expect_throw([] { Poisson p(0.0); }));
	assert(expect_throw([] { Normal n; n.quantile(1.5); }));
	std::cout << " validation ok\n";
}
} // namespace

int main() {
	test_normal();
	test_uniform();
	test_exponential();
	test_bernoulli();
	test_poisson();
	test_validation();
	std::cout << "All distribution tests passed\n";
	return 0;
}
