#pragma once

// Time-series primitives: differencing, cumulative sums, rolling statistics,
// exponential smoothing, autocorrelation and standardisation.
//
// Every function takes and returns a 1-D Array<double>. Rolling functions emit
// only full windows (length n - window + 1), and ``autocorrelation`` is
// normalised by the sample variance so lag 0 is 1.
//
// Licensed under GPL-3.0+

#include "array.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace dracolix::timeseries {

namespace detail {

inline void require_1d(const Array<double> &x, const char *fn) {
	if (x.ndim() != 1)
		throw std::invalid_argument(std::string(fn) + ": requires 1-D input");
}

inline double mean_of(const Array<double> &x) {
	double s = 0.0;
	for (size_t i = 0; i < x.size(); ++i)
		s += x[i];
	return s / static_cast<double>(x.size());
}

} // namespace detail

// x[i + lag] - x[i]
inline Array<double> diff(const Array<double> &x, size_t lag = 1) {
	detail::require_1d(x, "diff");
	if (lag == 0)
		throw std::invalid_argument("diff: lag must be >= 1");
	if (lag >= x.size())
		return Array<double>(0);
	Array<double> out(x.size() - lag);
	for (size_t i = 0; i < out.size(); ++i)
		out[i] = x[i + lag] - x[i];
	return out;
}

inline Array<double> cumsum(const Array<double> &x) {
	detail::require_1d(x, "cumsum");
	Array<double> out(x.size());
	double acc = 0.0;
	for (size_t i = 0; i < x.size(); ++i) {
		acc += x[i];
		out[i] = acc;
	}
	return out;
}

// Trailing rolling sum, one value per complete window.
inline Array<double> rolling_sum(const Array<double> &x, size_t window) {
	detail::require_1d(x, "rolling_sum");
	if (window == 0)
		throw std::invalid_argument("rolling_sum: window must be >= 1");
	if (window > x.size())
		return Array<double>(0);
	Array<double> out(x.size() - window + 1);
	double acc = 0.0;
	for (size_t i = 0; i < x.size(); ++i) {
		acc += x[i];
		if (i >= window)
			acc -= x[i - window];
		if (i + 1 >= window)
			out[i + 1 - window] = acc;
	}
	return out;
}

inline Array<double> rolling_mean(const Array<double> &x, size_t window) {
	auto s = rolling_sum(x, window);
	for (size_t i = 0; i < s.size(); ++i)
		s[i] /= static_cast<double>(window);
	return s;
}

// Trailing rolling variance with ``ddof`` degrees of freedom (0 = population,
// 1 = sample). Uses a two-pass computation per window for stability.
inline Array<double> rolling_variance(const Array<double> &x, size_t window,
									  size_t ddof = 0) {
	detail::require_1d(x, "rolling_variance");
	if (window == 0)
		throw std::invalid_argument("rolling_variance: window must be >= 1");
	if (window <= ddof)
		throw std::invalid_argument(
			"rolling_variance: window must be greater than ddof");
	if (window > x.size())
		return Array<double>(0);
	const size_t m = x.size() - window + 1;
	Array<double> out(m);
	for (size_t k = 0; k < m; ++k) {
		double mean = 0.0;
		for (size_t i = 0; i < window; ++i)
			mean += x[k + i];
		mean /= static_cast<double>(window);
		double acc = 0.0;
		for (size_t i = 0; i < window; ++i) {
			const double d = x[k + i] - mean;
			acc += d * d;
		}
		out[k] = acc / static_cast<double>(window - ddof);
	}
	return out;
}

inline Array<double> rolling_std(const Array<double> &x, size_t window,
								 size_t ddof = 0) {
	auto v = rolling_variance(x, window, ddof);
	for (size_t i = 0; i < v.size(); ++i)
		v[i] = std::sqrt(v[i]);
	return v;
}

// Exponentially weighted moving average, y[0] = x[0] and
// y[i] = alpha*x[i] + (1-alpha)*y[i-1].
inline Array<double> ewma(const Array<double> &x, double alpha) {
	detail::require_1d(x, "ewma");
	if (!(alpha > 0.0) || alpha > 1.0)
		throw std::invalid_argument("ewma: alpha must be in (0, 1]");
	Array<double> out(x.size());
	if (x.size() == 0)
		return out;
	out[0] = x[0];
	for (size_t i = 1; i < x.size(); ++i)
		out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1];
	return out;
}

// Span-based smoothing: alpha = 2 / (span + 1).
inline Array<double> ewma_span(const Array<double> &x, size_t span) {
	if (span == 0)
		throw std::invalid_argument("ewma_span: span must be >= 1");
	return ewma(x, 2.0 / (static_cast<double>(span) + 1.0));
}

// Normalised autocorrelation for lags 0..max_lag (input length n).
inline Array<double> autocorrelation(const Array<double> &x, size_t max_lag) {
	detail::require_1d(x, "autocorrelation");
	if (x.size() < 2)
		throw std::invalid_argument("autocorrelation: need at least 2 samples");
	if (max_lag >= x.size())
		throw std::invalid_argument(
			"autocorrelation: max_lag must be smaller than the input length");
	const double mean = detail::mean_of(x);
	double denom = 0.0;
	for (size_t i = 0; i < x.size(); ++i) {
		const double d = x[i] - mean;
		denom += d * d;
	}
	if (denom <= 0.0)
		throw std::invalid_argument("autocorrelation: input has zero variance");

	Array<double> out(max_lag + 1);
	for (size_t lag = 0; lag <= max_lag; ++lag) {
		double num = 0.0;
		for (size_t i = 0; i + lag < x.size(); ++i)
			num += (x[i] - mean) * (x[i + lag] - mean);
		out[lag] = num / denom;
	}
	return out;
}

// Standard score (population standard deviation).
inline Array<double> zscore(const Array<double> &x) {
	detail::require_1d(x, "zscore");
	if (x.size() == 0)
		return Array<double>(0);
	const double mean = detail::mean_of(x);
	double acc = 0.0;
	for (size_t i = 0; i < x.size(); ++i) {
		const double d = x[i] - mean;
		acc += d * d;
	}
	const double sd = std::sqrt(acc / static_cast<double>(x.size()));
	if (sd <= 0.0)
		throw std::invalid_argument("zscore: input has zero variance");
	Array<double> out(x.size());
	for (size_t i = 0; i < x.size(); ++i)
		out[i] = (x[i] - mean) / sd;
	return out;
}

} // namespace dracolix::timeseries
