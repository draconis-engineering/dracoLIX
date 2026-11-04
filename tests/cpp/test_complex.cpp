// Element-wise complex operations: real, imag, conj, abs, angle.
// Licensed under GPL-3.0+

#include "../../core/include/dracolix/complex_ops.hpp"
#include "../../core/include/dracolix/dracolix.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>

using namespace dracolix;

namespace {

constexpr double PI = 3.14159265358979323846;

bool close(double a, double b, double tol = 1e-12) {
	return std::abs(a - b) <= tol;
}

void test_complex_double() {
	Array<std::complex<double>> a(
		std::vector<size_t>{2, 2});
	a[0] = {1.0, 2.0};
	a[1] = {-3.0, 4.0};
	a[2] = {0.0, -1.0};
	a[3] = {-2.0, 0.0};

	auto re = cplx::real(a);
	auto im = cplx::imag(a);
	auto cj = cplx::conj(a);
	auto mag = cplx::abs(a);
	auto ph = cplx::angle(a);

	assert(re.shape() == a.shape());
	assert(re.dtype() == DType::F64);
	assert(im.dtype() == DType::F64);
	assert(mag.dtype() == DType::F64);
	assert(ph.dtype() == DType::F64);
	assert(cj.dtype() == DType::C128);

	assert(close(re[0], 1.0) && close(re[1], -3.0));
	assert(close(im[0], 2.0) && close(im[1], 4.0));
	assert(close(cj[0].real(), 1.0) && close(cj[0].imag(), -2.0));
	assert(close(cj[1].real(), -3.0) && close(cj[1].imag(), -4.0));
	assert(close(mag[0], std::sqrt(5.0)));
	assert(close(mag[1], 5.0));
	assert(close(mag[2], 1.0));
	assert(close(mag[3], 2.0));
	assert(close(ph[0], std::atan2(2.0, 1.0)));
	assert(close(ph[1], std::atan2(4.0, -3.0)));
	assert(close(ph[2], -PI / 2));
	assert(close(ph[3], PI));

	// conj(conj(a)) == a
	auto cc = cplx::conj(cj);
	for (size_t i = 0; i < a.size(); ++i) {
		assert(close(cc[i].real(), a[i].real()));
		assert(close(cc[i].imag(), a[i].imag()));
	}
	std::cout << " complex<double> ok\n";
}

void test_complex_float() {
	Array<std::complex<float>> a(std::vector<size_t>{3});
	a[0] = {3.0f, 4.0f};
	a[1] = {-1.0f, 0.0f};
	a[2] = {0.0f, 2.0f};

	auto re = cplx::real(a);
	auto im = cplx::imag(a);
	auto mag = cplx::abs(a);
	assert(re.dtype() == DType::F32);
	assert(im.dtype() == DType::F32);
	assert(mag.dtype() == DType::F32);
	assert(close(re[0], 3.0, 1e-6));
	assert(close(im[0], 4.0, 1e-6));
	assert(close(mag[0], 5.0, 1e-6));
	assert(close(mag[2], 2.0, 1e-6));
	std::cout << " complex<float> ok\n";
}

void test_real_inputs() {
	Array<double> a(std::vector<size_t>{4});
	a[0] = 0.0;
	a[1] = 3.0;
	a[2] = -2.5;
	a[3] = -0.0;

	auto re = cplx::real(a);
	auto im = cplx::imag(a);
	auto cj = cplx::conj(a);
	auto mag = cplx::abs(a);
	auto ph = cplx::angle(a);

	assert(re.dtype() == DType::F64);
	assert(im.dtype() == DType::F64);
	assert(cj.dtype() == DType::F64);
	assert(mag.dtype() == DType::F64);
	assert(ph.dtype() == DType::F64);

	for (size_t i = 0; i < a.size(); ++i) {
		assert(close(re[i], a[i]));
		assert(close(cj[i], a[i]));
		assert(close(im[i], 0.0));
	}
	assert(close(mag[2], 2.5));
	assert(close(ph[0], 0.0));
	assert(close(ph[1], 0.0));
	assert(close(ph[2], PI));
	assert(close(ph[3], 0.0));
	std::cout << " real<double> ok\n";
}

void test_integer_and_bool() {
	Array<int32_t> ai(std::vector<size_t>{3});
	ai[0] = -7;
	ai[1] = 0;
	ai[2] = 4;
	auto re = cplx::real(ai);
	auto im = cplx::imag(ai);
	auto mag = cplx::abs(ai);
	auto ph = cplx::angle(ai);
	assert(re.dtype() == DType::I32);
	assert(im.dtype() == DType::I32);
	assert(mag.dtype() == DType::I32);
	assert(ph.dtype() == DType::F64);
	assert(re[0] == -7 && im[0] == 0 && mag[0] == 7);
	assert(mag[2] == 4);
	assert(close(ph[0], PI));
	assert(close(ph[1], 0.0));
	assert(close(ph[2], 0.0));

	Array<bool> ab(std::vector<size_t>{2});
	ab[0] = false;
	ab[1] = true;
	auto bm = cplx::abs(ab);
	auto bc = cplx::conj(ab);
	assert(bm.dtype() == DType::Bool);
	assert(bc.dtype() == DType::Bool);
	assert(bm[0] == false && bm[1] == true);
	assert(bc[1] == true);
	std::cout << " integer/bool ok\n";
}

} // namespace

int main() {
	test_complex_double();
	test_complex_float();
	test_real_inputs();
	test_integer_and_bool();
	std::cout << "test_complex: all ok\n";
	return 0;
}
