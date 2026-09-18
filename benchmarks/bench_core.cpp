// bench_core.cpp - C++ benchmark suite (Phase 0/3)
// Licensed under GPL-3.0-only
// Covers: elementwise (with broadcasting), reductions, matmul, batched matmul,
// sparse matvec
#include "bench.hpp"
#include "dracolix/dracolix.hpp"
#include <random>

#ifndef __VERSION__
#ifdef _MSC_VER
#define __VERSION__ "MSVC " std::to_string(_MSC_VER)
#else
#define __VERSION__ "Unknown Compiler"
#endif
#endif

using namespace dracolix;
using namespace dracolix::linalg;

// helper: fill with deterministic pseudo-random
template <typename T>
void fill_rand(Array<T> &a, uint64_t seed = 0x9e3779b97f4a7c15ULL) {
	std::mt19937_64 rng(seed);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	for (size_t i = 0; i < a.size(); ++i)
		a[i] = static_cast<T>(dist(rng));
}

int main(int argc, char **argv) {
	size_t iters_small = 200;
	size_t iters_gemm = 20;
	// allow CLI override: bench_core [iters_small] [iters_gemm]
	if (argc > 1)
		iters_small = std::stoul(argv[1]);
	if (argc > 2)
		iters_gemm = std::stoul(argv[2]);

	std::cout << "DracoLIX bench_core — core vs baseline (C++ only)\n";
	std::cout << "build: " << dracolix::version
			  << " | compiler: " << __VERSION__ << "\n\n";

	bench::print_header();

	// --- elementwise (no broadcast) ---
	{
		Array<double> a({1024, 1024}), b({1024, 1024});
		fill_rand(a, 1);
		fill_rand(b, 2);
		auto r = bench::run(
			"add (1M)", "1024x1024", iters_small, 5,
			[&] {
				volatile auto c = a + b;
				(void)c;
			},
			a.size() * 1); // 1 FLOP per element (add)
		bench::print_result(r);
	}
	{
		Array<double> a({2048});
		Array<double> big({2048, 2048});
		fill_rand(a, 3);
		fill_rand(big, 5);
		auto r = bench::run(
			"add broadcast", "2048+2048x2048", 100, 5,
			[&] {
				volatile auto c = big + a;
				(void)c;
			},
			2048 * 2048);
		bench::print_result(r);
	}

	// --- reductions ---
	{
		Array<double> a({1 << 20}); // 1M
		fill_rand(a, 6);
		auto r = bench::run(
			"sum 1M", "1M", iters_small, 5,
			[&] {
				volatile auto s = a.sum();
				(void)s;
			},
			a.size());
		bench::print_result(r);
	}
	{
		Array<double> a({1024, 1024});
		fill_rand(a, 7);
		auto r = bench::run(
			"sum axis=0", "1024x1024", 100, 5,
			[&] {
				volatile auto s = a.sum(0);
				(void)s;
			},
			a.size());
		bench::print_result(r);
	}

	// --- matmul 2-D ---
	for (size_t n : {64, 256, 512}) {
		Array<double> A({n, n}), B({n, n});
		fill_rand(A, n);
		fill_rand(B, n + 100);
		std::string shape = std::to_string(n) + "x" + std::to_string(n);
		size_t flops = 2 * n * n * n; // multiply-add = 2 FLOPs
		auto r = bench::run(
			"matmul", shape, iters_gemm, 2,
			[&] {
				volatile auto C = matmul(A, B);
				(void)C;
			},
			flops);
		bench::print_result(r);
	}

	// --- batched matmul (atleast-3D) ---
	{
		Array<double> A({8, 128, 128}), B({8, 128, 128});
		fill_rand(A, 10);
		fill_rand(B, 11);
		auto r = bench::run(
			"batched matmul", "8x128x128", 20, 2,
			[&] {
				volatile auto C = matmul(A, B);
				(void)C;
			},
			size_t(2) * 8 * 128 * 128 * 128);
		bench::print_result(r);
	}
	{
		Array<double> A({4, 256, 64}), B({4, 64, 256});
		fill_rand(A, 12);
		fill_rand(B, 13);
		auto r = bench::run(
			"batched matmul", "4x256x64*64x256", 20, 2,
			[&] {
				volatile auto C = matmul(A, B);
				(void)C;
			},
			size_t(2) * 4 * 256 * 64 * 256);
		bench::print_result(r);
	}

	// --- sparse matvec ---
	{
		// 1024x1024 with ~1% density
		Array<double> dense({1024, 1024});
		std::mt19937_64 rng(42);
		std::uniform_real_distribution<double> d(-1, 1);
		std::uniform_real_distribution<double> sparsity(0, 1);
		for (size_t i = 0; i < dense.size(); ++i)
			dense[i] = (sparsity(rng) < 0.01 ? d(rng) : 0.0);
		auto csr = CsrMatrix<double>::from_dense(dense);
		Array<double> x({1024});
		fill_rand(x, 20);
		size_t flops = csr.nnz() * 2;
		auto r = bench::run(
			"csr matvec 1%", "1024x1024", 200, 5,
			[&] {
				volatile auto y = csr.matvec(x);
				(void)y;
			},
			flops);
		bench::print_result(r);
		std::cout << "  (csr nnz=" << csr.nnz() << ")\n";
	}

	// --- transpose (view vs copy) ---
	{
		Array<double> A({1024, 1024});
		fill_rand(A, 30);
		auto r1 = bench::run("transpose_view", "1024x1024", 5000, 100, [&] {
			volatile auto v = A.transpose_view();
			(void)v;
		});
		bench::print_result(r1);
		auto r2 = bench::run(
			"transpose copy", "1024x1024", 50, 5,
			[&] {
				volatile auto B = A.transpose();
				(void)B;
			},
			A.size());
		bench::print_result(r2);
	}

	std::cout << "\nDone. Compare with benchmarks/bench_gemm.py for "
				 "NumPy/OpenBLAS baseline.\n";
	return 0;
}
