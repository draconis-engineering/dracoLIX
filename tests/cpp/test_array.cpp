#include "../../core/include/dracolix/array.hpp"
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>

using namespace dracolix;

void test_layout() {
	Array<double> r({2, 3}, Layout::RowMajor);
	Array<double> c({2, 3}, Layout::ColMajor);
	assert(r.strides()[0] == 3 && r.strides()[1] == 1);
	assert(c.strides()[0] == 1 && c.strides()[1] == 2);
	std::cout << " layout OK\n";
}

void test_view_transpose() {
	Array<double> a({2, 3});
	// fill 0..5 row-major
	for (size_t i = 0; i < 6; ++i)
		a[i] = (double)i;
	// a = [[0 1 2],[3 4 5]]
	auto vt = a.transpose_view();
	assert(vt.shape()[0] == 3 && vt.shape()[1] == 2);
	assert(vt.at(0, 1) == 3);
	assert(vt.at(2, 0) == 2);
	assert(vt.at(2, 1) == 5);
	// mutate via view should reflect
	vt.at(0, 0) = 99;
	assert(a.at(0, 0) == 99);
	a.at(0, 0) = 0; // restore

	auto mt = a.transpose();
	assert(mt.shape()[0] == 3 && mt.shape()[1] == 2);
	assert(mt.at(0, 1) == 3);
	mt.at(0, 0) = 42;
	assert(a.at(0, 0) == 0); // copy not affected
	std::cout << " transpose OK\n";
}

void test_permute() {
	Array<double> a({2, 3, 4});
	for (size_t i = 0; i < a.size(); ++i)
		a[i] = (double)i;
	auto v = a.permute({2, 0, 1}); // shape 4,2,3
	assert(v.shape()[0] == 4 && v.shape()[1] == 2 && v.shape()[2] == 3);
	// check mapping: original (1,2,3) -> permuted (3,1,2)
	assert(v.at(std::vector<size_t>{3, 1, 2}) ==
		   a.at(std::vector<size_t>{1, 2, 3}));
	std::cout << " permute OK\n";
}

void test_slice() {
	Array<double> a({6});
	for (size_t i = 0; i < 6; ++i)
		a[i] = (double)i;
	auto s = a.slice(Slice::range(1, 5, 2)); // 1,3
	assert(s.shape()[0] == 2);
	assert(s.at(std::vector<size_t>{0}) == 1);
	assert(s.at(std::vector<size_t>{1}) == 3);
	s.at(std::vector<size_t>{0}) = 99;
	assert(a[1] == 99);
	a[1] = 1;
	// 2-D slice
	Array<double> b({3, 4});
	for (size_t i = 0; i < 12; ++i)
		b[i] = i;
	auto v = b.slice({Slice::range(0, 2), Slice::all()});
	assert(v.shape()[0] == 2 && v.shape()[1] == 4);
	assert(v.at(1, 3) == 7);
	// negative step
	auto rev = a.slice(Slice::range(5, -1, -1));
	assert(rev.shape()[0] == 6);
	assert(rev.at(std::vector<size_t>{0}) == 5);
	assert(rev.at(std::vector<size_t>{5}) == 0);
	std::cout << " slice OK\n";
}

void test_reductions() {
	Array<double> a({2, 3});
	// [[1 2 3],[4 5 6]]
	a.at(0, 0) = 1;
	a.at(0, 1) = 2;
	a.at(0, 2) = 3;
	a.at(1, 0) = 4;
	a.at(1, 1) = 5;
	a.at(1, 2) = 6;
	assert(a.sum() == 21);
	assert(a.min() == 1);
	assert(a.max() == 6);
	assert(std::abs(a.mean() - 3.5) < 1e-9);
	// axis 0 -> sum over rows => [5,7,9]
	auto s0 = a.sum(0);
	assert(s0.shape()[0] == 3);
	assert(s0[0] == 5 && s0[1] == 7 && s0[2] == 9);
	auto s1 = a.sum(1);
	assert(s1.shape()[0] == 2);
	assert(s1[0] == 6 && s1[1] == 15);
	assert(a.min_axis(0)[0] == 1);
	assert(a.max_axis(1)[1] == 6);
	assert(std::abs(a.mean_axis(0)[0] - 2.5) < 1e-9);
	std::cout << " reductions OK\n";
}

void test_phase033_reductions() {
	Array<double> a({2, 3});
	for (size_t i = 0; i < 6; ++i)
		a[i] = (double)(i + 1); // [[1 2 3],[4 5 6]]
	assert(a.prod() == 720.0);
	auto p0 = a.prod(0); // [4, 10, 18]
	assert(p0[0] == 4 && p0[1] == 10 && p0[2] == 18);
	auto p1 = a.prod(1); // [6, 120]
	assert(p1[0] == 6 && p1[1] == 120);

	assert(a.argmin() == 0);
	assert(a.argmax() == 5);
	auto amin0 = a.argmin(0); // [0, 0, 0]
	assert(amin0.dtype() == DType::I64);
	assert(amin0[0] == 0 && amin0[2] == 0);
	auto amax0 = a.argmax(0); // [1, 1, 1]
	assert(amax0[0] == 1 && amax0[2] == 1);
	auto amin1 = a.argmin(1); // [0, 0]
	assert(amin1[0] == 0 && amin1[1] == 0);
	auto amax1 = a.argmax(1); // [2, 2]
	assert(amax1[0] == 2 && amax1[1] == 2);

	assert(a.any());
	assert(a.all());
	Array<double> z({3});
	z[0] = 0;
	z[1] = 0;
	z[2] = 1;
	assert(z.any());
	assert(!z.all());
	assert(z.any_axis(0)[0]);
	assert(!z.all_axis(0)[0]);

	Array<double> e({0});
	assert(e.prod() == 1.0);
	assert(!e.any());
	assert(e.all());
	std::cout << " 0.3.3 reductions OK\n";
}

void test_phase034_stats() {
	Array<double> a({2, 3});
	for (size_t i = 0; i < 6; ++i)
		a[i] = (double)(i + 1); // [[1 2 3],[4 5 6]]

	assert(std::abs(a.var() - 35.0 / 12.0) < 1e-12);
	assert(std::abs(a.var(1) - 3.5) < 1e-12);
	assert(std::abs(a.std() - std::sqrt(35.0 / 12.0)) < 1e-12);
	assert(std::abs(a.std(1) - std::sqrt(3.5)) < 1e-12);

	auto v0 = a.var_axis(0); // [2.25 2.25 2.25]
	assert(v0.dtype() == DType::F64);
	for (size_t i = 0; i < 3; ++i)
		assert(std::abs(v0[i] - 2.25) < 1e-12);
	auto s1 = a.std_axis(1); // sqrt(2/3) twice
	for (size_t i = 0; i < 2; ++i)
		assert(std::abs(s1[i] - std::sqrt(2.0 / 3.0)) < 1e-12);
	auto vd = a.var_axis(1, 1); // ddof=1 -> 1.0
	assert(std::abs(vd[0] - 1.0) < 1e-12);

	Array<int32_t> i({3});
	i[0] = 2;
	i[1] = 4;
	i[2] = 6;
	assert(std::abs(i.var() - 8.0 / 3.0) < 1e-12);

	Array<bool> b({3});
	b[0] = true;
	b[1] = true;
	b[2] = false;
	assert(std::abs(b.var() - 2.0 / 9.0) < 1e-12);
	auto bv = b.var_axis(0);
	assert(std::abs(bv[0] - 2.0 / 9.0) < 1e-12);

	Array<std::complex<double>> z({2});
	z[0] = {1.0, 1.0};
	z[1] = {3.0, -1.0};
	assert(std::abs(z.var() - 2.0) < 1e-12);
	assert(std::abs(z.std() - std::sqrt(2.0)) < 1e-12);
	auto zv = z.var_axis(0);
	assert(std::abs(zv[0] - 2.0) < 1e-12);

	bool threw = false;
	try {
		a.var(6);
	} catch (const std::invalid_argument &) {
		threw = true;
	}
	assert(threw);
	threw = false;
	try {
		a.std(-1);
	} catch (const std::invalid_argument &) {
		threw = true;
	}
	assert(threw);
	threw = false;
	try {
		a.var_axis(0, 2);
	} catch (const std::invalid_argument &) {
		threw = true;
	}
	assert(threw);

	Array<double> e({0});
	threw = false;
	try {
		e.var();
	} catch (const std::logic_error &) {
		threw = true;
	}
	assert(threw);
	assert(e.size() == 0);
	std::cout << " 0.3.4 stats OK\n";
}

void test_astype() {
	Array<double> a({3});
	a[0] = 1.7;
	a[1] = 2.2;
	a[2] = 3.9;
	auto b = a.astype<int32_t>();
	assert(b[0] == 1 && b[2] == 3);
	assert(b.dtype() == DType::I32);
	auto c = b.astype<double>();
	assert(c[0] == 1.0);
	std::cout << " astype OK\n";
}

void test_broadcast() {
	Array<double> a({2, 1});
	a[0] = 1;
	a[1] = 2;
	Array<double> b({1, 3});
	b[0] = 10;
	b[1] = 20;
	b[2] = 30;
	auto c = a + b;
	assert((c.shape() == std::vector<size_t>{2, 3}));
	assert(c.at(0, 0) == 11 && c.at(0, 2) == 31 && c.at(1, 2) == 32);
	Array<double> row({3});
	row[0] = 100;
	row[1] = 200;
	row[2] = 300;
	Array<double> m({2, 3});
	for (size_t i = 0; i < 6; ++i)
		m[i] = (double)i;
	auto d = m + row;
	assert(d.at(0, 0) == 100 && d.at(1, 2) == 305);
	// 3D
	Array<double> e({2, 3, 1});
	for (size_t i = 0; i < 6; ++i)
		e[i] = (double)i;
	Array<double> f({3});
	f[0] = 10;
	f[1] = 20;
	f[2] = 30;
	auto g = e + f;
	assert((g.shape() == std::vector<size_t>{2, 3, 3}));
	try {
		Array<double> h({2, 3});
		Array<double> k({3, 2});
		auto l = h + k;
		assert(false);
	} catch (const std::invalid_argument &) {
	}
	assert((a * b).at(1, 2) == 60);
	std::cout << " broadcast OK\n";
}

int main() {
	test_layout();
	test_view_transpose();
	test_permute();
	test_slice();
	test_reductions();
	test_phase033_reductions();
	test_phase034_stats();
	test_astype();
	test_broadcast();
	// existing smoke
	Array<double> a({2, 3});
	assert(a.size() == 6 && a.ndim() == 2);
	a.at(0, 0) = 1;
	a.at(1, 2) = 6;
	assert(a.at(0, 0) == 1 && a.at(1, 2) == 6);
	Array<double> b({2, 3});
	b = b + 1.0;
	auto c = a + b;
	(void)c;
	a.reshape({6});
	assert(a.ndim() == 1 && a.size() == 6);
	std::cout << "All Phase1 tests passed\n";
	return 0;
}
