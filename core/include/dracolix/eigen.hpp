#pragma once
// Eigenvalues / eigenvectors — symmetric Jacobi method
// Phase 2 prototype, real double only. For general matrices use eig_sym.
// Licensed under GPL-3.0-only
#include "array.hpp"
#include <algorithm>
#include <cmath>

namespace dracolix::eigen {

struct SymEigenResult {
	Array<double> values;  // n
	Array<double> vectors; // n x n, columns are eigenvectors (orthonormal)
};

inline SymEigenResult eig_sym(const Array<double> &A, double tol = 1e-10,
							  int max_sweeps = 100) {
	if (A.ndim() != 2)
		throw std::invalid_argument("eig_sym requires 2-D");
	size_t n = A.shape()[0];
	if (A.shape()[1] != n)
		throw std::invalid_argument("eig_sym requires square");
	// Check approximately symmetric (optional, not strict)
	Array<double> D = A.clone();
	Array<double> V({n, n});
	std::fill(V.data(), V.data() + V.size(), 0.0);
	for (size_t i = 0; i < n; ++i)
		V.at(i, i) = 1.0;

	for (int sweep = 0; sweep < max_sweeps; ++sweep) {
		// find largest off-diagonal
		double max_off = 0.0;
		size_t p = 0, q = 1;
		for (size_t i = 0; i < n; ++i)
			for (size_t j = i + 1; j < n; ++j)
				if (std::abs(D.at(i, j)) > max_off) {
					max_off = std::abs(D.at(i, j));
					p = i;
					q = j;
				}
		if (max_off < tol)
			break;

		double app = D.at(p, p);
		double aqq = D.at(q, q);
		double apq = D.at(p, q);

		double tau = (aqq - app) / (2.0 * apq);
		double t = (tau >= 0 ? 1.0 / (tau + std::sqrt(1 + tau * tau))
							 : -1.0 / (-tau + std::sqrt(1 + tau * tau)));
		double c = 1.0 / std::sqrt(1 + t * t);
		double s = t * c;

		// rotate D: D' = J^T D J
		// J rotates rows/cols p,q
		for (size_t i = 0; i < n; ++i) {
			if (i == p || i == q)
				continue;
			double aip = D.at(i, p);
			double aiq = D.at(i, q);
			D.at(i, p) = D.at(p, i) = c * aip - s * aiq;
			D.at(i, q) = D.at(q, i) = s * aip + c * aiq;
		}
		double app2 = c * c * app - 2 * c * s * apq + s * s * aqq;
		double aqq2 = s * s * app + 2 * c * s * apq + c * c * aqq;
		D.at(p, p) = app2;
		D.at(q, q) = aqq2;
		D.at(p, q) = D.at(q, p) = 0.0;

		// accumulate eigenvectors V = V * J
		for (size_t i = 0; i < n; ++i) {
			double vip = V.at(i, p);
			double viq = V.at(i, q);
			V.at(i, p) = c * vip - s * viq;
			V.at(i, q) = s * vip + c * viq;
		}
	}

	Array<double> vals({n});
	for (size_t i = 0; i < n; ++i)
		vals[i] = D.at(i, i);

	// Sort descending by eigenvalue magnitude? Keep as is but provide sorted
	// For deterministic tests, sort by value descending and permute vectors
	std::vector<size_t> idx(n);
	std::iota(idx.begin(), idx.end(), 0);
	std::sort(idx.begin(), idx.end(),
			  [&](size_t a, size_t b) { return vals[a] > vals[b]; });
	Array<double> svals({n});
	Array<double> svecs({n, n});
	for (size_t j = 0; j < n; ++j) {
		svals[j] = vals[idx[j]];
		for (size_t i = 0; i < n; ++i)
			svecs.at(i, j) = V.at(i, idx[j]);
	}

	return {std::move(svals), std::move(svecs)};
}

} // namespace dracolix::eigen
