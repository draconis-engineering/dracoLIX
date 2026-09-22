#include "../../core/include/dracolix/dracolix.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace dracolix;

void test_eig_sym_2x2() {
	Array<double> A({2, 2});
	A.at(0, 0) = 2;
	A.at(0, 1) = 1;
	A.at(1, 0) = 1;
	A.at(1, 1) = 2;
	auto e = eigen::eig_sym(A);
	assert(std::abs(e.values[0] - 3.0) < 1e-6 &&
		   std::abs(e.values[1] - 1.0) < 1e-6);
	for (size_t k = 0; k < 2; ++k) {
		Array<double> v({2});
		v[0] = e.vectors.at(0, k);
		v[1] = e.vectors.at(1, k);
		auto Av = linalg::matvec(A, v);
		for (size_t i = 0; i < 2; ++i)
			assert(std::abs(Av[i] - e.values[k] * v[i]) < 1e-6);
	}
	std::cout << " eig 2x2 ok\n";
}

void test_eig_sym_3x3() {
	Array<double> B({3, 3});
	B.at(0, 0) = 4;
	B.at(0, 1) = 1;
	B.at(0, 2) = 1;
	B.at(1, 0) = 1;
	B.at(1, 1) = 3;
	B.at(1, 2) = 0;
	B.at(2, 0) = 1;
	B.at(2, 1) = 0;
	B.at(2, 2) = 2;
	auto eb = eigen::eig_sym(B);
	Array<double> D({3, 3});
	std::fill(D.data(), D.data() + 9, 0);
	for (size_t i = 0; i < 3; ++i)
		D.at(i, i) = eb.values[i];
	auto Vt = eb.vectors.transpose();
	auto VDVt = linalg::matmul(linalg::matmul(eb.vectors, D), Vt);
	for (size_t i = 0; i < 3; ++i)
		for (size_t j = 0; j < 3; ++j)
			assert(std::abs(VDVt.at(i, j) - B.at(i, j)) < 1e-5);
	std::cout << " eig 3x3 reconstruction ok\n";
}

void test_svd_tall() {
	Array<double> C({3, 2});
	C.at(0, 0) = 1;
	C.at(0, 1) = 0;
	C.at(1, 0) = 0;
	C.at(1, 1) = 1;
	C.at(2, 0) = 1;
	C.at(2, 1) = 1;
	auto s = svd::svd(C);
	assert(s.S[0] > s.S[1] && s.S[1] > 0.9);
	Array<double> Sdiag({s.S.size(), s.S.size()});
	std::fill(Sdiag.data(), Sdiag.data() + s.S.size() * s.S.size(), 0);
	for (size_t i = 0; i < s.S.size(); ++i)
		Sdiag.at(i, i) = s.S[i];
	auto rec = linalg::matmul(linalg::matmul(s.U, Sdiag), s.Vt);
	for (size_t i = 0; i < 3; ++i)
		for (size_t j = 0; j < 2; ++j)
			assert(std::abs(rec.at(i, j) - C.at(i, j)) < 1e-5);
	std::cout << " svd tall ok\n";
}

void test_svd_wide() {
	Array<double> W({2, 3});
	W.at(0, 0) = 1;
	W.at(0, 1) = 2;
	W.at(0, 2) = 3;
	W.at(1, 0) = 4;
	W.at(1, 1) = 5;
	W.at(1, 2) = 6;
	auto sw = svd::svd(W);
	Array<double> Sdiag({sw.S.size(), sw.S.size()});
	std::fill(Sdiag.data(), Sdiag.data() + sw.S.size() * sw.S.size(), 0);
	for (size_t i = 0; i < sw.S.size(); ++i)
		Sdiag.at(i, i) = sw.S[i];
	auto rec = linalg::matmul(linalg::matmul(sw.U, Sdiag), sw.Vt);
	for (size_t i = 0; i < 2; ++i)
		for (size_t j = 0; j < 3; ++j)
			assert(std::abs(rec.at(i, j) - W.at(i, j)) < 1e-4);
	std::cout << " svd wide ok\n";
}

int main() {
	test_eig_sym_2x2();
	test_eig_sym_3x3();
	test_svd_tall();
	test_svd_wide();
	std::cout << "All eigen/svd tests passed\n";
	return 0;
}
