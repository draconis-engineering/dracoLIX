#pragma once
// Factorizations: LU (with partial pivoting), solve, det, inv, rank
// Licensed under GPL-3.0-only
#include "array.hpp"
#include "linalg.hpp"
#include <cmath>
#include <vector>

namespace dracolix::decomp {

template <typename T>
struct LuResult {
    Array<T> L; // unit lower, n x n
    Array<T> U; // upper, n x n
    std::vector<size_t> perm; // permutation: row i of PA corresponds to original row perm[i]
    int swaps = 0; // number of row swaps (for det sign)
};

template <typename T>
LuResult<T> lu(const Array<T>& A) {
    if (A.ndim() != 2) throw std::invalid_argument("lu requires 2-D");
    size_t n = A.shape()[0];
    if (A.shape()[1] != n) throw std::invalid_argument("lu requires square");
    Array<T> U = A.clone();
    Array<T> L({n,n});
    std::fill(L.data(), L.data()+L.size(), T{});
    for (size_t i=0;i<n;++i) L.at(i,i)=T{1};

    std::vector<size_t> perm(n);
    for (size_t i=0;i<n;++i) perm[i]=i;
    int swaps=0;

    for (size_t k=0;k<n;++k) {
        // pivot: max |U[i,k]| for i>=k
        size_t piv = k;
        double maxv = std::abs((double)U.at(k,k));
        for (size_t i=k+1;i<n;++i) {
            double v = std::abs((double)U.at(i,k));
            if (v > maxv) { maxv=v; piv=i; }
        }
        if (maxv == 0.0) continue; // singular, keep as-is (will be detected on solve)

        if (piv != k) {
            // swap rows k and piv in U, and perm, and L's already computed columns <k
            for (size_t j=0;j<n;++j) std::swap(U.at(k,j), U.at(piv,j));
            for (size_t j=0;j<k;++j) std::swap(L.at(k,j), L.at(piv,j));
            std::swap(perm[k], perm[piv]);
            ++swaps;
        }
        T pivot = U.at(k,k);
        if (pivot == T{}) continue;
        for (size_t i=k+1;i<n;++i) {
            T factor = U.at(i,k) / pivot;
            L.at(i,k) = factor;
            for (size_t j=k;j<n;++j) {
                U.at(i,j) -= factor * U.at(k,j);
            }
            U.at(i,k) = T{}; // explicitly zero lower
        }
    }
    return {std::move(L), std::move(U), std::move(perm), swaps};
}

template <typename T>
Array<T> solve_lu(const LuResult<T>& lu, const Array<T>& b) {
    size_t n = lu.L.shape()[0];
    bool b_is_vec = (b.ndim()==1);
    size_t nrhs = b_is_vec ? 1 : b.shape()[1];
    if (b_is_vec && b.size()!=n) throw std::invalid_argument("solve: b size mismatch");
    if (!b_is_vec && (b.ndim()!=2 || b.shape()[0]!=n)) throw std::invalid_argument("solve: b shape mismatch");

    // permute b: Pb
    Array<T> Pb(b_is_vec ? std::vector<size_t>{n} : std::vector<size_t>{n, nrhs});
    if (b_is_vec) {
        for (size_t i=0;i<n;++i) Pb[i]=b[lu.perm[i]];
    } else {
        for (size_t i=0;i<n;++i) for (size_t j=0;j<nrhs;++j) Pb.at(i,j)=b.at(lu.perm[i], j);
    }

    // forward Ly = Pb
    Array<T> y(Pb.shape());
    if (b_is_vec) {
        for (size_t i=0;i<n;++i) {
            T acc = Pb[i];
            for (size_t j=0;j<i;++j) acc -= lu.L.at(i,j)*y[j];
            // L diag is 1
            y[i]=acc;
        }
    } else {
        for (size_t j=0;j<nrhs;++j) for (size_t i=0;i<n;++i) {
            T acc = Pb.at(i,j);
            for (size_t k=0;k<i;++k) acc -= lu.L.at(i,k)*y.at(k,j);
            y.at(i,j)=acc;
        }
    }

    // back Ux = y
    Array<T> x(y.shape());
    if (b_is_vec) {
        for (int i=(int)n-1; i>=0; --i) {
            T acc = y[i];
            for (size_t j=i+1;j<n;++j) acc -= lu.U.at(i,j)*x[j];
            T diag = lu.U.at(i,i);
            if (diag==T{}) throw std::runtime_error("solve: singular matrix");
            x[i]=acc/diag;
        }
    } else {
        for (size_t j=0;j<nrhs;++j) for (int i=(int)n-1; i>=0; --i) {
            T acc = y.at(i,j);
            for (size_t k=i+1;k<n;++k) acc -= lu.U.at(i,k)*x.at(k,j);
            T diag = lu.U.at(i,i);
            if (diag==T{}) throw std::runtime_error("solve: singular matrix");
            x.at(i,j)=acc/diag;
        }
    }
    return x;
}

template <typename T>
Array<T> solve(const Array<T>& A, const Array<T>& b) {
    auto f = lu(A);
    return solve_lu(f, b);
}

template <typename T>
T determinant(const Array<T>& A) {
    auto f = lu(A);
    T det = (f.swaps %2==0 ? T{1} : T{-1});
    for (size_t i=0;i<A.shape()[0];++i) det *= f.U.at(i,i);
    return det;
}

template <typename T>
Array<T> inverse(const Array<T>& A) {
    size_t n = A.shape()[0];
    if (A.ndim()!=2 || A.shape()[1]!=n) throw std::invalid_argument("inverse requires square 2-D");
    auto f = lu(A);
    Array<T> inv({n,n});
    // solve for each column of identity
    for (size_t j=0;j<n;++j) {
        Array<T> e({n}); std::fill(e.data(), e.data()+n, T{}); e[j]=T{1};
        auto col = solve_lu(f, e);
        for (size_t i=0;i<n;++i) inv.at(i,j)=col[i];
    }
    return inv;
}

template <typename T>
size_t rank(const Array<T>& A, double tol = 1e-9) {
    if (A.ndim()!=2) throw std::invalid_argument("rank requires 2-D");
    auto f = lu(A);
    size_t r=0;
    for (size_t i=0;i<A.shape()[0];++i) if (std::abs((double)f.U.at(i,i)) > tol) ++r;
    return r;
}

} // namespace dracolix::decomp
