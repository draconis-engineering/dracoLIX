#pragma once

// Signal processing primitives: DFT/FFT, inverse transforms, convolution,
// correlation, windows and power spectra.
//
// The radix-2 FFT requires a power-of-two length; the direct DFT works for any
// length. ``fft_real`` zero-pads real input to the next power of two. All
// transforms use the sign convention X[k] = sum_n x[n] exp(-2*pi*i*k*n/N).
//
// Licensed under GPL-3.0+

#include "array.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace dracolix::signal {

using Complex = std::complex<double>;

namespace detail {

inline constexpr double kPi = 3.141592653589793238462643383279502884;

inline void require_1d(const Array<double> &a, const char *fn) {
	if (a.ndim() != 1)
		throw std::invalid_argument(std::string(fn) + ": requires 1-D input");
}

inline void require_1d(const Array<Complex> &a, const char *fn) {
	if (a.ndim() != 1)
		throw std::invalid_argument(std::string(fn) + ": requires 1-D input");
}

} // namespace detail

inline bool is_power_of_two(size_t n) {
	return n != 0 && (n & (n - 1)) == 0;
}

inline size_t next_power_of_two(size_t n) {
	if (n <= 1)
		return 1;
	size_t p = 1;
	while (p < n)
		p <<= 1;
	return p;
}

// ---------------------------------------------------------------------------
// Direct transforms (any length)
// ---------------------------------------------------------------------------
inline Array<Complex> dft(const Array<Complex> &x) {
	detail::require_1d(x, "dft");
	const size_t n = x.size();
	Array<Complex> out(n);
	for (size_t k = 0; k < n; ++k) {
		Complex sum{};
		for (size_t j = 0; j < n; ++j) {
			const double angle =
				-2.0 * detail::kPi * static_cast<double>(k * j) /
				static_cast<double>(n);
			sum += x[j] * Complex(std::cos(angle), std::sin(angle));
		}
		out[k] = sum;
	}
	return out;
}

inline Array<Complex> dft(const Array<double> &x) {
	detail::require_1d(x, "dft");
	const size_t n = x.size();
	Array<Complex> out(n);
	for (size_t k = 0; k < n; ++k) {
		Complex sum{};
		for (size_t j = 0; j < n; ++j) {
			const double angle =
				-2.0 * detail::kPi * static_cast<double>(k * j) /
				static_cast<double>(n);
			sum += x[j] * Complex(std::cos(angle), std::sin(angle));
		}
		out[k] = sum;
	}
	return out;
}

inline Array<Complex> idft(const Array<Complex> &X) {
	detail::require_1d(X, "idft");
	const size_t n = X.size();
	Array<Complex> out(n);
	if (n == 0)
		return out;
	for (size_t k = 0; k < n; ++k) {
		Complex sum{};
		for (size_t j = 0; j < n; ++j) {
			const double angle =
				2.0 * detail::kPi * static_cast<double>(k * j) /
				static_cast<double>(n);
			sum += X[j] * Complex(std::cos(angle), std::sin(angle));
		}
		out[k] = sum / static_cast<double>(n);
	}
	return out;
}

// ---------------------------------------------------------------------------
// Radix-2 FFT
// ---------------------------------------------------------------------------
// In-place iterative Cooley-Tukey FFT. Takes its input by value and returns
// the transformed array.
inline Array<Complex> fft(Array<Complex> x) {
	detail::require_1d(x, "fft");
	const size_t n = x.size();
	if (n <= 1)
		return x;
	if (!is_power_of_two(n))
		throw std::invalid_argument("fft: length must be a power of two");

	// Bit-reversal permutation.
	for (size_t i = 1, j = 0; i < n; ++i) {
		size_t bit = n >> 1;
		for (; j & bit; bit >>= 1)
			j ^= bit;
		j ^= bit;
		if (i < j)
			std::swap(x[i], x[j]);
	}

	for (size_t len = 2; len <= n; len <<= 1) {
		const double angle = -2.0 * detail::kPi / static_cast<double>(len);
		const Complex wlen(std::cos(angle), std::sin(angle));
		for (size_t i = 0; i < n; i += len) {
			Complex w(1.0, 0.0);
			for (size_t j = 0; j < len / 2; ++j) {
				const Complex u = x[i + j];
				const Complex v = x[i + j + len / 2] * w;
				x[i + j] = u + v;
				x[i + j + len / 2] = u - v;
				w *= wlen;
			}
		}
	}
	return x;
}

inline Array<Complex> ifft(Array<Complex> X) {
	detail::require_1d(X, "ifft");
	const size_t n = X.size();
	if (n == 0)
		return X;
	for (size_t i = 0; i < X.size(); ++i)
		X[i] = std::conj(X[i]);
	auto out = fft(std::move(X));
	for (size_t i = 0; i < out.size(); ++i)
		out[i] = std::conj(out[i]) / static_cast<double>(n);
	return out;
}

inline Array<Complex> fft_real(const Array<double> &x) {
	detail::require_1d(x, "fft_real");
	if (x.size() == 0)
		return Array<Complex>(0);
	const size_t n = next_power_of_two(x.size());
	Array<Complex> buf(n);
	for (size_t i = 0; i < x.size(); ++i)
		buf[i] = Complex(x[i], 0.0);
	return fft(std::move(buf));
}

inline Array<double> ifft_real(const Array<Complex> &X) {
	auto full = ifft(X);
	Array<double> out(full.size());
	for (size_t i = 0; i < full.size(); ++i)
		out[i] = full[i].real();
	return out;
}

inline Array<double> magnitude(const Array<Complex> &X) {
	detail::require_1d(X, "magnitude");
	Array<double> out(X.size());
	for (size_t i = 0; i < X.size(); ++i)
		out[i] = std::abs(X[i]);
	return out;
}

// ---------------------------------------------------------------------------
// Convolution and correlation
// ---------------------------------------------------------------------------
enum class ConvolutionMode { Full, Same, Valid };

inline Array<double> convolve(const Array<double> &a, const Array<double> &b,
							  ConvolutionMode mode = ConvolutionMode::Full) {
	detail::require_1d(a, "convolve");
	detail::require_1d(b, "convolve");
	const size_t na = a.size(), nb = b.size();
	if (na == 0 || nb == 0)
		throw std::invalid_argument("convolve: empty input");
	if (mode == ConvolutionMode::Valid && na < nb)
		throw std::invalid_argument(
			"convolve: Valid mode requires a.size() >= b.size()");

	const size_t full = na + nb - 1;
	std::vector<double> tmp(full, 0.0);
	for (size_t i = 0; i < na; ++i) {
		for (size_t j = 0; j < nb; ++j)
			tmp[i + j] += a[i] * b[j];
	}

	size_t start = 0, length = full;
	if (mode == ConvolutionMode::Same) {
		start = (nb - 1) / 2;
		length = na;
	} else if (mode == ConvolutionMode::Valid) {
		start = nb - 1;
		length = na - nb + 1;
	}
	Array<double> out(length);
	for (size_t i = 0; i < length; ++i)
		out[i] = tmp[start + i];
	return out;
}

// FFT-based full convolution (zero-padded to the next power of two).
inline Array<double> convolve_fft(const Array<double> &a,
								  const Array<double> &b) {
	detail::require_1d(a, "convolve_fft");
	detail::require_1d(b, "convolve_fft");
	const size_t na = a.size(), nb = b.size();
	if (na == 0 || nb == 0)
		throw std::invalid_argument("convolve_fft: empty input");
	const size_t full = na + nb - 1;
	const size_t n = next_power_of_two(full);

	Array<Complex> fa(n), fb(n);
	for (size_t i = 0; i < na; ++i)
		fa[i] = Complex(a[i], 0.0);
	for (size_t i = 0; i < nb; ++i)
		fb[i] = Complex(b[i], 0.0);
	fa = fft(std::move(fa));
	fb = fft(std::move(fb));
	for (size_t i = 0; i < n; ++i)
		fa[i] *= fb[i];
	fa = ifft(std::move(fa));

	Array<double> out(full);
	for (size_t i = 0; i < full; ++i)
		out[i] = fa[i].real();
	return out;
}

// Cross-correlation of two real signals: convolve(a, reverse(b)).
inline Array<double> correlate(const Array<double> &a, const Array<double> &b,
							   ConvolutionMode mode = ConvolutionMode::Full) {
	detail::require_1d(a, "correlate");
	detail::require_1d(b, "correlate");
	Array<double> rev(b.size());
	for (size_t i = 0; i < b.size(); ++i)
		rev[i] = b[b.size() - 1 - i];
	return convolve(a, rev, mode);
}

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------
enum class Window { Hann, Hamming, Blackman };

inline Array<double> window(Window w, size_t n) {
	if (n == 0)
		throw std::invalid_argument("window: n must be > 0");
	Array<double> out(n);
	if (n == 1) {
		out[0] = 1.0;
		return out;
	}
	const double denom = static_cast<double>(n - 1);
	for (size_t i = 0; i < n; ++i) {
		const double x = 2.0 * detail::kPi * static_cast<double>(i) / denom;
		switch (w) {
		case Window::Hann:
			out[i] = 0.5 - 0.5 * std::cos(x);
			break;
		case Window::Hamming:
			out[i] = 0.54 - 0.46 * std::cos(x);
			break;
		case Window::Blackman:
			out[i] = 0.42 - 0.5 * std::cos(x) + 0.08 * std::cos(2.0 * x);
			break;
		}
	}
	return out;
}

inline Array<double> hann(size_t n) { return window(Window::Hann, n); }
inline Array<double> hamming(size_t n) { return window(Window::Hamming, n); }
inline Array<double> blackman(size_t n) {
	return window(Window::Blackman, n);
}

// ---------------------------------------------------------------------------
// Frequency axis and power spectrum
// ---------------------------------------------------------------------------
// One-sided frequency axis for an FFT of length ``n``.
inline Array<double> frequencies(size_t n, double sample_rate) {
	if (n == 0)
		throw std::invalid_argument("frequencies: n must be > 0");
	if (!(sample_rate > 0.0))
		throw std::invalid_argument("frequencies: sample_rate must be > 0");
	const size_t m = n / 2 + 1;
	Array<double> out(m);
	for (size_t k = 0; k < m; ++k)
		out[k] = static_cast<double>(k) * sample_rate /
				 static_cast<double>(n);
	return out;
}

struct Spectrum {
	Array<double> frequencies;
	Array<double> power;
};

// Periodogram power spectral density: |FFT(x - mean)|^2 / n on the one-sided
// frequency grid. The input is zero-padded to the next power of two.
inline Spectrum power_spectrum(const Array<double> &x, double sample_rate) {
	detail::require_1d(x, "power_spectrum");
	if (x.size() < 2)
		throw std::invalid_argument("power_spectrum: need at least 2 samples");
	if (!(sample_rate > 0.0))
		throw std::invalid_argument("power_spectrum: sample_rate must be > 0");

	double mean = 0.0;
	for (size_t i = 0; i < x.size(); ++i)
		mean += x[i];
	mean /= static_cast<double>(x.size());

	const size_t n = next_power_of_two(x.size());
	Array<Complex> buf(n);
	for (size_t i = 0; i < x.size(); ++i)
		buf[i] = Complex(x[i] - mean, 0.0);
	buf = fft(std::move(buf));

	Spectrum s{frequencies(n, sample_rate), Array<double>(n / 2 + 1)};
	for (size_t k = 0; k < s.frequencies.size(); ++k)
		s.power[k] = std::norm(buf[k]) / static_cast<double>(n);
	return s;
}

} // namespace dracolix::signal
