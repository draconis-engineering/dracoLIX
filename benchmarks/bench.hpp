#pragma once
// Benchmark harness - Phase 0 benchmarking infrastructure
// Licensed under GPL-3.0-only
// Header-only, zero external deps (chrono only). Python comparison in bench_gemm.py
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <cmath>

namespace dracolix::bench {

struct Result {
    std::string name;
    std::string shape;
    size_t bytes_processed = 0;
    double elapsed_s = 0;
    size_t iterations = 0;
    double gflops() const {
        // caller sets bytes_processed as flops if needed
        return (elapsed_s > 0) ? (bytes_processed / elapsed_s / 1e9) : 0;
    }
    double gbps() const {
        return (elapsed_s > 0) ? (bytes_processed / elapsed_s / 1e9) : 0;
    }
};

inline double now_s() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

inline Result run(const std::string& name, const std::string& shape,
                  size_t iters, size_t warmup,
                  std::function<void()> fn,
                  size_t flops_per_iter = 0) {
    for (size_t i = 0; i < warmup; ++i) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    Result r;
    r.name = name;
    r.shape = shape;
    r.iterations = iters;
    r.elapsed_s = elapsed;
    r.bytes_processed = flops_per_iter * iters;
    return r;
}

inline void print_header() {
    std::cout << std::left << std::setw(28) << "benchmark"
              << std::setw(18) << "shape"
              << std::right << std::setw(10) << "iters"
              << std::setw(14) << "total(s)"
              << std::setw(14) << "mean(ms)"
              << std::setw(14) << "GFLOP/s"
              << "\n";
    std::cout << std::string(98, '-') << "\n";
}

inline void print_result(const Result& r) {
    double mean_ms = (r.iterations ? r.elapsed_s / r.iterations * 1000.0 : 0);
    std::cout << std::left << std::setw(28) << r.name
              << std::setw(18) << r.shape
              << std::right << std::setw(10) << r.iterations
              << std::setw(14) << std::fixed << std::setprecision(6) << r.elapsed_s
              << std::setw(14) << std::fixed << std::setprecision(3) << mean_ms
              << std::setw(14) << std::fixed << std::setprecision(3) << r.gflops()
              << "\n";
}

} // namespace dracolix::bench
