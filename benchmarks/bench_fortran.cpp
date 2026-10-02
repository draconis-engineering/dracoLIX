#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

extern "C" void matmatmul_c(
    const double* a,
    const double* b,
    double* res,
    int n,
    int m,
    int p
);

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 4000;
    int m = argc > 2 ? std::atoi(argv[2]) : n;
    int p = argc > 3 ? std::atoi(argv[3]) : n;

    std::cout << "Fortran/OpenBLAS GEMM: "
              << n << "x" << m << " * "
              << m << "x" << p << "\n";

    std::vector<double> A(static_cast<size_t>(n) * m);
    std::vector<double> B(static_cast<size_t>(m) * p);
    std::vector<double> C(static_cast<size_t>(n) * p);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    // Warmup
    matmatmul_c(A.data(), B.data(), C.data(), n, m, p);

    auto start = std::chrono::steady_clock::now();

    matmatmul_c(A.data(), B.data(), C.data(), n, m, p);

    auto end = std::chrono::steady_clock::now();

    double seconds =
        std::chrono::duration<double>(end - start).count();

    double flops = 2.0 * n * m * p;
    double gflops = flops / seconds / 1e9;

    std::cout << "Time:   " << seconds << " s\n";
    std::cout << "GFLOP/s: " << gflops << "\n";
    std::cout << "C[0]:   " << C[0] << "\n";
}
