// Time-series primitives.
// Licensed under GPL-3.0+

#include "../../core/include/dracolix/timeseries.hpp"

#include <cassert>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace dracolix;
using namespace dracolix::timeseries;

namespace {

bool close(double a, double b, double tol = 1e-12) {
	return std::abs(a - b) <= tol;
}

Array<double> make(std::initializer_list<double> vals) {
	Array<double> a(vals.size());
	size_t i = 0;
	for (double v : vals)
		a[i++] = v;
	return a;
}

std::vector<double> to_vec(const Array<double> &a) {
	return std::vector<double>(a.data(), a.data() + a.size());
}

void test_diff_cumsum() {
	auto x = make({1.0, 4.0, 9.0, 16.0});
	assert(to_vec(diff(x)) == std::vector<double>({3.0, 5.0, 7.0}));
	assert(to_vec(diff(x, 2)) == std::vector<double>({8.0, 12.0}));
	assert(diff(x, 4).size() == 0);
	assert(diff(x, 10).size() == 0);

	assert(to_vec(cumsum(x)) == std::vector<double>({1.0, 5.0, 14.0, 30.0}));
	assert(cumsum(Array<double>(0)).size() == 0);
	std::cout << " diff / cumsum ok\n";
}

void test_rolling() {
	auto x = make({1.0, 2.0, 3.0, 4.0, 5.0});
	assert(to_vec(rolling_sum(x, 3)) == std::vector<double>({6.0, 9.0, 12.0}));
	assert(to_vec(rolling_mean(x, 3)) == std::vector<double>({2.0, 3.0, 4.0}));
	assert(rolling_mean(x, 5).size() == 1);
	assert(close(rolling_mean(x, 5)[0], 3.0));
	assert(rolling_mean(x, 6).size() == 0);
	assert(rolling_mean(x, 1).size() == 5);

	auto y = make({1.0, 2.0, 3.0, 4.0});
	auto var = rolling_variance(y, 3);
	assert(var.size() == 2);
	assert(close(var[0], 2.0 / 3.0));
	assert(close(var[1], 2.0 / 3.0));
	auto svar = rolling_variance(y, 3, 1);
	assert(close(svar[0], 1.0));
	assert(close(svar[1], 1.0));
	auto sd = rolling_std(y, 3, 1);
	assert(close(sd[0], 1.0) && close(sd[1], 1.0));
	std::cout << " rolling statistics ok\n";
}

void test_ewma() {
	auto x = make({0.0, 1.0, 0.0});
	assert(to_vec(ewma(x, 1.0)) == to_vec(x));
	assert(to_vec(ewma(x, 0.5)) == std::vector<double>({0.0, 0.5, 0.25}));

	auto ones = make({2.0, 2.0, 2.0, 2.0});
	auto e = ewma(ones, 0.3);
	for (size_t i = 0; i < e.size(); ++i)
		assert(close(e[i], 2.0));

	// ewma_span(span) uses alpha = 2/(span+1).
	auto a = ewma_span(x, 3);
	auto b = ewma(x, 0.5);
	for (size_t i = 0; i < a.size(); ++i)
		assert(close(a[i], b[i]));
	assert(ewma(Array<double>(0), 0.5).size() == 0);
	std::cout << " ewma ok\n";
}

void test_autocorrelation_zscore() {
	auto x = make({1.0, 2.0, 3.0, 4.0, 5.0});
	auto ac = autocorrelation(x, 4);
	assert(ac.size() == 5);
	assert(close(ac[0], 1.0));
	assert(close(ac[1], 0.4));
	assert(close(ac[2], -0.1));
	assert(close(ac[3], -0.4));
	assert(close(ac[4], -0.4));

	auto z = zscore(x);
	assert(z.size() == 5);
	const double s2 = std::sqrt(2.0);
	const double expected[] = {-s2, -s2 / 2.0, 0.0, s2 / 2.0, s2};
	for (size_t i = 0; i < z.size(); ++i)
		assert(close(z[i], expected[i]));

	// Standardised data has zero mean and unit population variance.
	auto zc = cumsum(z);
	assert(close(zc[zc.size() - 1], 0.0, 1e-12));
	double var = 0.0;
	for (size_t i = 0; i < z.size(); ++i)
		var += z[i] * z[i];
	assert(close(var / (double)z.size(), 1.0));
	std::cout << " autocorrelation / zscore ok\n";
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
	auto x = make({1.0, 2.0, 3.0});

	assert(expect([&] { diff(x, 0); }));
	assert(expect([&] { rolling_sum(x, 0); }));
	assert(expect([&] { rolling_mean(x, 0); }));
	assert(expect([&] { rolling_variance(x, 3, 3); }));
	assert(expect([&] { ewma(x, 0.0); }));
	assert(expect([&] { ewma(x, 1.5); }));
	assert(expect([&] { ewma_span(x, 0); }));
	assert(expect([&] { autocorrelation(make({1.0}), 0); }));
	assert(expect([&] { autocorrelation(x, 3); }));
	assert(expect([&] { autocorrelation(make({2.0, 2.0, 2.0}), 1); }));
	assert(expect([&] { zscore(make({2.0, 2.0, 2.0})); }));
	assert(expect([&] {
		Array<double> two({2, 2});
		(void)diff(two);
	}));
	std::cout << " error handling ok\n";
}

} // namespace

int main() {
	test_diff_cumsum();
	test_rolling();
	test_ewma();
	test_autocorrelation_zscore();
	test_errors();
	std::cout << "All time-series tests passed\n";
	return 0;
}
