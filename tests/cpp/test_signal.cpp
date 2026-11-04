// Signal processing: DFT/FFT, inverse transforms, convolution, windows and
// power spectra.
// Licensed under GPL-3.0+

#include "../../core/include/dracolix/signal.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace dracolix;
using namespace dracolix::signal;

namespace {

constexpr double PI = 3.14159265358979323846;

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

Array<Complex> makec(std::initializer_list<Complex> vals) {
	Array<Complex> a(vals.size());
	size_t i = 0;
	for (const Complex &v : vals)
		a[i++] = v;
	return a;
}

std::vector<double> to_vec(const Array<double> &a) {
	return std::vector<double>(a.data(), a.data() + a.size());
}

void test_sizes() {
	assert(is_power_of_two(1));
	assert(is_power_of_two(2));
	assert(is_power_of_two(8));
	assert(is_power_of_two(1024));
	assert(!is_power_of_two(0));
	assert(!is_power_of_two(3));
	assert(!is_power_of_two(6));
	assert(next_power_of_two(0) == 1);
	assert(next_power_of_two(1) == 1);
	assert(next_power_of_two(2) == 2);
	assert(next_power_of_two(3) == 4);
	assert(next_power_of_two(17) == 32);
	std::cout << " sizes ok\n";
}

void test_dft_delta() {
	auto x = make({1.0, 0.0, 0.0, 0.0});
	auto X = dft(x);
	assert(X.size() == 4);
	for (size_t k = 0; k < 4; ++k) {
		assert(close(X[k].real(), 1.0));
		assert(close(X[k].imag(), 0.0));
	}

	// idft inverts the DFT.
	auto back = idft(X);
	for (size_t i = 0; i < 4; ++i) {
		assert(close(back[i].real(), x[i]));
		assert(close(back[i].imag(), 0.0));
	}
	std::cout << " dft delta / idft ok\n";
}

void test_dft_cosine() {
	const size_t n = 8;
	Array<double> x(n);
	for (size_t i = 0; i < n; ++i)
		x[i] = std::cos(2.0 * PI * 2.0 * (double)i / (double)n);
	auto X = dft(x);
	// A real cosine at bin 2 splits into bins 2 and n-2 with amplitude n/2.
	assert(close(X[2].real(), 4.0, 1e-9));
	assert(close(X[6].real(), 4.0, 1e-9));
	for (size_t k = 0; k < n; ++k) {
		if (k != 2 && k != 6)
			assert(std::abs(X[k]) < 1e-9);
	}
	std::cout << " dft cosine ok\n";
}

void test_fft_matches_dft() {
	const size_t n = 64;
	Array<Complex> x(n);
	Array<double> xr(n);
	for (size_t i = 0; i < n; ++i) {
		const double v =
			std::sin(0.7 * (double)i) + 0.5 * std::cos(1.3 * (double)i);
		x[i] = Complex(v, -0.25 * std::sin(0.31 * (double)i));
		xr[i] = v;
	}
	auto X = fft(x);
	auto Xd = dft(x);
	for (size_t k = 0; k < n; ++k) {
		assert(close(X[k].real(), Xd[k].real(), 1e-9));
		assert(close(X[k].imag(), Xd[k].imag(), 1e-9));
	}

	auto Xr = dft(xr);
	auto Xrf = fft_real(xr);
	assert(Xrf.size() == n);
	for (size_t k = 0; k < n; ++k) {
		assert(close(Xrf[k].real(), Xr[k].real(), 1e-9));
		assert(close(Xrf[k].imag(), Xr[k].imag(), 1e-9));
	}

	// Round trip.
	auto back = ifft(X);
	for (size_t i = 0; i < n; ++i) {
		assert(close(back[i].real(), x[i].real(), 1e-9));
		assert(close(back[i].imag(), x[i].imag(), 1e-9));
	}

	// Parseval: sum |x|^2 == (1/n) sum |X|^2.
	double time_energy = 0.0, freq_energy = 0.0;
	for (size_t i = 0; i < n; ++i) {
		time_energy += std::norm(x[i]);
		freq_energy += std::norm(X[i]);
	}
	assert(close(time_energy, freq_energy / (double)n, 1e-9));

	// fft_real zero-pads to the next power of two.
	auto padded = fft_real(make({1.0, 0.0, 0.0}));
	assert(padded.size() == 4);
	for (size_t k = 0; k < 4; ++k) {
		assert(close(padded[k].real(), 1.0));
		assert(close(padded[k].imag(), 0.0));
	}
	std::cout << " fft matches dft / round trip / parseval ok\n";
}

void test_convolution() {
	auto a = make({1.0, 2.0, 3.0});
	auto b = make({0.0, 1.0, 0.5});

	auto full = convolve(a, b, ConvolutionMode::Full);
	assert(to_vec(full) == std::vector<double>({0.0, 1.0, 2.5, 4.0, 1.5}));

	auto same = convolve(a, b, ConvolutionMode::Same);
	assert(to_vec(same) == std::vector<double>({1.0, 2.5, 4.0}));

	auto valid = convolve(a, b, ConvolutionMode::Valid);
	assert(to_vec(valid) == std::vector<double>({2.5}));

	// A longer signal and a short kernel exercise multi-sample valid output.
	auto long_a = make({1.0, 2.0, 3.0, 4.0});
	auto kern = make({1.0, 1.0});
	auto lv = convolve(long_a, kern, ConvolutionMode::Valid);
	assert(to_vec(lv) == std::vector<double>({3.0, 5.0, 7.0}));

	auto fft_full = convolve_fft(a, b);
	assert(fft_full.size() == full.size());
	for (size_t i = 0; i < full.size(); ++i)
		assert(close(fft_full[i], full[i], 1e-12));

	// Correlating with a unit impulse at lag 0 shifts the signal.
	auto imp = make({1.0, 0.0, 0.0});
	auto corr = correlate(a, imp, ConvolutionMode::Full);
	assert(to_vec(corr) == std::vector<double>({0.0, 0.0, 1.0, 2.0, 3.0}));

	std::cout << " convolution / correlation ok\n";
}

void test_windows() {
	// Odd-length symmetric windows have a unit centre.
	auto h = hann(5);
	assert(h.size() == 5);
	assert(close(h[0], 0.0));
	assert(close(h[1], 0.5));
	assert(close(h[2], 1.0));
	assert(close(h[3], 0.5));
	assert(close(h[4], 0.0));

	auto hm = hamming(5);
	assert(close(hm[0], 0.08));
	assert(close(hm[2], 1.0));
	assert(close(hm[4], 0.08));

	auto bm = blackman(5);
	assert(std::abs(bm[0]) < 1e-15);
	assert(close(bm[2], 1.0));
	assert(std::abs(bm[4]) < 1e-15);

	auto one = hann(1);
	assert(one.size() == 1 && one[0] == 1.0);

	// Even windows are symmetric too.
	auto he = hann(6);
	for (size_t i = 0; i < he.size(); ++i)
		assert(close(he[i], he[he.size() - 1 - i]));

	std::cout << " windows ok\n";
}

void test_spectrum() {
	auto f = frequencies(8, 8.0);
	assert(to_vec(f) == std::vector<double>({0.0, 1.0, 2.0, 3.0, 4.0}));

	// Pure cosine at 4 Hz sampled at 64 Hz.
	const size_t n = 64;
	const double fs = 64.0;
	Array<double> x(n);
	for (size_t i = 0; i < n; ++i)
		x[i] = std::cos(2.0 * PI * 4.0 * (double)i / fs);

	auto s = power_spectrum(x, fs);
	assert(s.frequencies.size() == n / 2 + 1);
	assert(s.power.size() == s.frequencies.size());
	size_t peak = 0;
	for (size_t k = 1; k < s.power.size(); ++k)
		if (s.power[k] > s.power[peak])
			peak = k;
	assert(peak == 4);
	assert(close(s.frequencies[peak], 4.0));
	assert(s.power[peak] > 10.0);
	std::cout << " power spectrum ok (peak " << s.frequencies[peak] << " Hz)\n";
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
	assert(expect([] { fft(makec({1.0, 2.0, 3.0})); }));
	assert(expect([] { ifft(makec({1.0, 2.0, 3.0})); }));
	assert(expect([] { convolve(Array<double>(0), make({1.0})); }));
	assert(expect([] { convolve(make({1.0}), Array<double>(0)); }));
	assert(expect([] {
		convolve(make({1.0}), make({1.0, 2.0}), ConvolutionMode::Valid);
	}));
	assert(expect([] { window(Window::Hann, 0); }));
	assert(expect([] { frequencies(0, 1.0); }));
	assert(expect([] { frequencies(8, 0.0); }));
	assert(expect([] { power_spectrum(make({1.0}), 1.0); }));
	assert(expect([] { power_spectrum(make({1.0, 2.0}), 0.0); }));
	assert(expect([] {
		Array<double> two({2, 2});
		(void)dft(two);
	}));
	std::cout << " error handling ok\n";
}

} // namespace

int main() {
	test_sizes();
	test_dft_delta();
	test_dft_cosine();
	test_fft_matches_dft();
	test_convolution();
	test_windows();
	test_spectrum();
	test_errors();
	std::cout << "All signal tests passed\n";
	return 0;
}
