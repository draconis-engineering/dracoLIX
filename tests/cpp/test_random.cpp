// Random number generation: determinism, ranges, moments and array helpers.
// Licensed under GPL-3.0+

#include "../../core/include/dracolix/random.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace dracolix;
using namespace dracolix::random;

namespace {
void test_reproducibility() {
	Rng a(12345), b(12345);
	for (int i = 0; i < 100; ++i) {
		assert(a.next_u64() == b.next_u64());
	}
	Rng c(1), d(2);
	bool differ = false;
	for (int i = 0; i < 20; ++i)
		if (c.next_u64() != d.next_u64())
			differ = true;
	assert(differ);

	Rng e(7);
	const double x1 = e.uniform();
	e.seed(7);
	assert(e.uniform() == x1);
	std::cout << " reproducibility ok\n";
}

void test_uniform() {
	Rng rng(42);
	double sum = 0.0;
	for (int i = 0; i < 200000; ++i) {
		const double x = rng.uniform();
		assert(x >= 0.0 && x < 1.0);
		sum += x;
	}
	assert(std::abs(sum / 200000.0 - 0.5) < 0.01);

	for (int i = 0; i < 1000; ++i) {
		const double x = rng.uniform(-3.0, 5.0);
		assert(x >= -3.0 && x < 5.0);
	}
	std::cout << " uniform ok\n";
}

void test_integer_and_special() {
	Rng rng(9);
	int counts[6] = {0, 0, 0, 0, 0, 0};
	for (int i = 0; i < 60000; ++i) {
		const int64_t k = rng.randint(0, 6);
		assert(k >= 0 && k < 6);
		++counts[k];
	}
	for (int c : counts)
		assert(std::abs(c - 10000) < 600);

	double ones = 0.0;
	for (int i = 0; i < 50000; ++i)
		ones += rng.bernoulli(0.25) ? 1.0 : 0.0;
	assert(std::abs(ones / 50000.0 - 0.25) < 0.01);

	double esum = 0.0;
	for (int i = 0; i < 100000; ++i) {
		const double x = rng.exponential(2.0);
		assert(x >= 0.0);
		esum += x;
	}
	assert(std::abs(esum / 100000.0 - 0.5) < 0.02);

	double psum = 0.0;
	for (int i = 0; i < 100000; ++i)
		psum += (double)rng.poisson(3.0);
	assert(std::abs(psum / 100000.0 - 3.0) < 0.05);
	std::cout << " integer/bernoulli/exponential/poisson ok\n";
}

void test_normal() {
	Rng rng(2024);
	double sum = 0.0, sum2 = 0.0;
	const int n = 200000;
	for (int i = 0; i < n; ++i) {
		const double x = rng.normal(2.0, 3.0);
		sum += x;
		sum2 += x * x;
	}
	const double mean = sum / n;
	const double var = sum2 / n - mean * mean;
	assert(std::abs(mean - 2.0) < 0.03);
	assert(std::abs(var - 9.0) < 0.15);
	std::cout << " normal ok (mean " << mean << ", var " << var << ")\n";
}

void test_arrays() {
	Rng a(5), b(5);
	auto u = random::uniform({3, 4}, -1.0, 1.0, a);
	assert(u.shape() == std::vector<size_t>({3, 4}));
	for (size_t i = 0; i < u.size(); ++i)
		assert(u[i] >= -1.0 && u[i] < 1.0);

	auto u2 = random::uniform({3, 4}, -1.0, 1.0, b);
	for (size_t i = 0; i < u.size(); ++i)
		assert(u[i] == u2[i]);

	auto g = random::normal({10}, 0.0, 1.0, a);
	assert(g.size() == 10);
	auto k = random::randint({20}, 3, 8, a);
	assert(k.size() == 20);
	for (size_t i = 0; i < k.size(); ++i)
		assert(k[i] >= 3 && k[i] < 8);
	std::cout << " array helpers ok\n";
}
} // namespace

int main() {
	test_reproducibility();
	test_uniform();
	test_integer_and_special();
	test_normal();
	test_arrays();
	std::cout << "All random tests passed\n";
	return 0;
}
