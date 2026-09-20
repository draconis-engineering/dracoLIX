#pragma once
// SVD prototype via symmetric eigen of A^T A (for m>=n) or A A^T
// Phase 2, real double only.
// Licensed under GPL-3.0-only
#include "array.hpp"
#include "linalg.hpp"
#include "eigen.hpp"
#include <cmath>
#include <algorithm>

namespace dracolix::svd {

struct SvdResult {
    Array<double> U;  // m x r
    Array<double> S;  // r (singular values descending)
    Array<double> Vt; // r x n
};

inline SvdResult svd(const Array<double>& A, double tol = 1e-10) {
    if (A.ndim() != 2) throw std::invalid_argument("svd requires 2-D");
    size_t m = A.shape()[0], n = A.shape()[1];
    bool wide = n > m;
    // For wide matrix, work on A^T to keep eigen size = min(m,n)
    if (wide) {
        auto At = A.transpose();
        auto r = svd(At, tol);
        // r: At = U_r * S * Vt_r  (n x m) ; so A = Vt_r^T * S * U_r^T
        // So swap
        return {r.Vt.transpose(), r.S.clone(), r.U.transpose()};
    }
    // Now m >= n, narrow
    // Compute B = A^T A  (n x n) symmetric
    auto At = A.transpose();
    auto B = linalg::matmul(At, A); // n x n

    auto eig = eigen::eig_sym(B, tol);
    // eig values are sigma^2, may have small negatives due to numerical
    size_t r = 0;
    for (size_t i = 0; i < n; ++i) if (eig.values[i] > tol) ++r;
    // If rank 0, still return empty? Keep at least 0
    Array<double> S({r});
    Array<double> V({n, r});
    for (size_t j = 0; j < r; ++j) {
        double ev = eig.values[j];
        if (ev < 0 && ev > -1e-12) ev = 0;
        S[j] = std::sqrt(std::max(0.0, (double)ev));
        for (size_t i = 0; i < n; ++i) V.at(i, j) = eig.vectors.at(i, j);
    }
    // Vt = V^T
    Array<double> Vt({r, n});
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < n; ++j)
            Vt.at(i, j) = V.at(j, i);

    // U = A * V * inv(S)
    Array<double> U({m, r});
    if (r > 0) {
        auto AV = linalg::matmul(A, V); // m x r
        for (size_t j = 0; j < r; ++j) {
            double s = S[j];
            for (size_t i = 0; i < m; ++i) U.at(i, j) = AV.at(i, j) / s;
        }
    }
    return {std::move(U), std::move(S), std::move(Vt)};
}

} // namespace dracolix::svd
