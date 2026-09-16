#include "../../core/include/dracolix/sparse.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace dracolix;

// Reference dense helpers
template <typename T>
Array<T> dense_matmul(const Array<T> &A, const Array<T> &B) {
	size_t m = A.shape()[0], n = A.shape()[1], p = B.shape()[1];
	Array<T> C({m, p});
	for (size_t i = 0; i < m; ++i)
		for (size_t j = 0; j < p; ++j) {
			T acc = T{};
			for (size_t k = 0; k < n; ++k)
				acc += A.at(i, k) * B.at(k, j);
			C.at(i, j) = acc;
		}
	return C;
}

void test_from_coo_csr() {
	// unsorted COO with a duplicate; zero should be dropped
	auto A = CsrMatrix<double>::from_coo(3, 3, {1, 0, 2, 1}, {2, 0, 0, 2},
										 {4.0, 1.0, -1.0, 2.0});
	// row 0: col0=1; row1: col2=4+2=6; row2: col0=-1
	assert(A.rows() == 3 && A.cols() == 3 && A.nnz() == 3);
	assert(A.row_ptr() == std::vector<size_t>({0, 1, 2, 3}));
	assert(A.col_ind() == std::vector<size_t>({0, 2, 0}));
	assert(std::abs(A.values()[0] - 1.0) < 1e-12);
	assert(std::abs(A.values()[1] - 6.0) < 1e-12);
	assert(std::abs(A.at(1, 2) - 6.0) < 1e-12);
	assert(A.at(0, 1) == 0.0); // missing -> zero
	// zero-dropping: entries that sum to zero disappear
	auto B = CsrMatrix<double>::from_coo(1, 1, {0}, {0}, {5.0}, true);
	assert(B.nnz() == 1 && std::abs(B.at(0, 0) - 5.0) < 1e-12);
	// duplicated entries that cancel to zero are dropped too
	auto Z = CsrMatrix<double>::from_coo(2, 2, {0, 1, 1}, {0, 1, 1},
										 {1.0, 5.0, -5.0}, true);
	assert(Z.nnz() == 1 && std::abs(Z.at(0, 0) - 1.0) < 1e-12);
	assert(Z.at(1, 1) == 0.0);
	auto Zk = CsrMatrix<double>::from_coo(2, 2, {0, 1, 1}, {0, 1, 1},
										  {1.0, 5.0, -5.0}, false);
	assert(Zk.nnz() == 2 && Zk.at(1, 1) == 0.0);
	// out of bounds rejected
	bool threw = false;
	try {
		CsrMatrix<double>::from_coo(2, 2, {5}, {0}, {1.0});
	} catch (const std::out_of_range &) {
		threw = true;
	}
	assert(threw);
	std::cout << " csr from_coo OK\n";
}

void test_dense_roundtrip() {
	Array<double> d({3, 4});
	for (size_t i = 0; i < d.size(); ++i)
		d[i] = (i % 3 == 1) ? 0.0 : (double)i;
	auto S = CsrMatrix<double>::from_dense(d);
	auto back = S.to_dense();
	for (size_t i = 0; i < d.size(); ++i)
		assert(std::abs(back[i] - d[i]) < 1e-12);
	assert(S.nnz() == 10); // 12 entries minus 4 zeros in row 1
	std::cout << " csr from_dense/to_dense OK\n";
}

void test_csr_to_csc() {
	Array<double> d({3, 3});
	double vals[9] = {1, 0, 2, 0, 3, 4, 5, 0, 6};
	for (size_t i = 0; i < 9; ++i)
		d[i] = vals[i];
	auto S = CsrMatrix<double>::from_dense(d);
	auto c = S.transpose(); // CSC of A^T
	assert(c.rows() == 3 && c.cols() == 3 && c.nnz() == 6);
	assert(c.to_dense().at(0, 2) == 5.0); // (A^T)[0][2] = A[2][0]
	assert(c.to_dense().at(1, 1) == 3.0);
	assert(c.to_dense().at(2, 2) == 6.0);

	auto same = S.to_csc(); // CSC of A (same orientation)
	auto same_back = same.to_dense();
	for (size_t i = 0; i < 9; ++i)
		assert(std::abs(same_back[i] - vals[i]) < 1e-12);
	assert(same.at(0, 2) == 2.0 && same.at(2, 0) == 5.0);
	// rows ascending within every column: col0 (0,2), col1 (1), col2 (0,1,2)
	assert(same.row_ind() == std::vector<size_t>({0, 2, 1, 0, 1, 2}));
	std::cout << " csr->csc transpose/to_csc OK\n";
}

void test_csc_build_and_convert() {
	// A^T from the COO of A (col-major ordering)
	auto Cs = CscMatrix<double>::from_coo(3, 3, {2, 0, 1}, {0, 1, 2},
										  {7.0, 8.0, 9.0});
	// CSC building by (col, row): col1 has row0=8, col2 has row1=9, col0 has
	// row2=7
	assert(Cs.nnz() == 3);
	assert(Cs.at(2, 0) == 7.0 && Cs.at(0, 1) == 8.0 && Cs.at(1, 2) == 9.0);

	auto back_csr = Cs.to_csr();
	assert(back_csr.rows() == 3 && back_csr.cols() == 3);
	assert(back_csr.at(2, 0) == 7.0 && back_csr.at(0, 1) == 8.0 &&
		   back_csr.at(1, 2) == 9.0);

	auto T = Cs.transpose(); // CSR of A^T
	assert(T.rows() == 3 && T.cols() == 3);
	assert(T.at(0, 2) == 7.0 && T.at(1, 0) == 8.0 && T.at(2, 1) == 9.0);
	std::cout << " csc build/convert OK\n";
}

template <typename T> void test_matvec_random() {
	std::mt19937 rng(1234);
	const size_t m = 40, n = 30;
	Array<T> d({m, n});
	for (size_t i = 0; i < d.size(); ++i)
		d[i] = (T)(rng() % 100);
	for (size_t i = 0; i < d.size(); ++i)
		if ((i % 7) != 0)
			d[i] = T{};

	auto S = CsrMatrix<T>::from_dense(d);
	auto Csc = CscMatrix<T>::from_dense(d);

	Array<T> x({n});
	for (size_t j = 0; j < n; ++j)
		x[j] = (T)(rng() % 50);

	auto y_csr = S.matvec(x);
	auto y_csc = Csc.matvec(x);

	Array<T> yref({m});
	for (size_t i = 0; i < m; ++i) {
		T acc = T{};
		for (size_t j = 0; j < n; ++j)
			acc += d.at(i, j) * x[j];
		yref[i] = acc;
	}
	for (size_t i = 0; i < m; ++i) {
		assert(y_csr[i] == yref[i]);
		assert(y_csc[i] == yref[i]);
	}
	std::cout << " matvec random OK\n";
}

void test_matmul_random() {
	std::mt19937 rng(777);
	const size_t m = 30, n = 25, p = 28;
	auto mk = [&](size_t r, size_t c) {
		Array<double> d({r, c});
		for (size_t i = 0; i < d.size(); ++i)
			d[i] = (double)(rng() % 10);
		for (size_t i = 0; i < d.size(); ++i)
			if ((i % 5) != 0)
				d[i] = 0.0;
		return d;
	};
	Array<double> Ad = mk(m, n), Bd = mk(n, p);
	auto A = CsrMatrix<double>::from_dense(Ad);
	auto B = CsrMatrix<double>::from_dense(Bd);
	auto C = A.matmul(B);
	auto Cref = dense_matmul(Ad, Bd);
	auto Cd = C.to_dense();
	for (size_t k = 0; k < Cd.size(); ++k)
		assert(std::abs(Cd[k] - Cref[k]) < 1e-9);
	std::cout << " sparse matmul random OK\n";
}

void test_matmul_identity() {
	Array<double> d({4, 4});
	for (size_t i = 0; i < 4; ++i)
		d.at(i, i) = 2.0;
	auto A = CsrMatrix<double>::from_dense(d);
	auto C = A.matmul(A);
	auto Cd = C.to_dense();
	for (size_t i = 0; i < 4; ++i)
		for (size_t j = 0; j < 4; ++j)
			assert(std::abs(Cd.at(i, j) - (i == j ? 4.0 : 0.0)) < 1e-12);
	std::cout << " sparse matmul identity OK\n";
}

void test_int_sparse() {
	auto A =
		CsrMatrix<int32_t>::from_coo(2, 2, {0, 1, 0}, {0, 1, 1}, {3, 4, 5});
	assert(A.at(0, 1) == 5 && A.nnz() == 3);
	Array<int32_t> x({2});
	x[0] = 1;
	x[1] = 2;
	auto y = A.matvec(x);
	assert(y[0] == 13 && y[1] == 8);
	auto C = A.matmul(A.transpose_csr());
	assert(C.at(0, 0) == 34 && C.at(1, 1) == 16);
	std::cout << " int32 sparse OK\n";
}

int main() {
	test_from_coo_csr();
	test_dense_roundtrip();
	test_csr_to_csc();
	test_csc_build_and_convert();
	test_matvec_random<double>();
	test_matmul_random();
	test_matmul_identity();
	test_int_sparse();
	std::cout << "All sparse tests passed\n";
	return 0;
}
