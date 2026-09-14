#pragma once
// Sparse matrix support: CSR and CSC compressed storage.
// Phase 2b - representation, conversion, dense interop and sparse matmul.
// Licensed under GPL-3.0-only - see LICENSE
#include "array.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace dracolix {

// Forward declarations (the two storage formats convert to each other).
template <typename T> class CsrMatrix;
template <typename T> class CscMatrix;

// ---------------------------------------------------------------------------
// CsrMatrix<T> - Compressed Sparse Row storage.
//
//   row_ptr  [m+1]        start/end offsets into col_ind/values per row
//   col_ind  [nnz]        column index of each stored entry
//   values   [nnz]        numerical value of each stored entry
//
// Invariants (enforced by every constructors):
//   * columns are sorted ascending within every row
//   * duplicate (row, col) entries are merged (summed)
//   * explicit zeros are dropped (unless drop_zeros=false)
// ---------------------------------------------------------------------------
template <typename T> class CsrMatrix {
public:
  CsrMatrix() = default;
  CsrMatrix(size_t rows, size_t cols)
      : rows_(rows), cols_(cols), row_ptr_(rows + 1, 0) {}

  // ---- metadata ----
  size_t rows() const noexcept { return rows_; }
  size_t cols() const noexcept { return cols_; }
  size_t nnz() const noexcept { return values_.size(); }

  // ---- raw storage (read-only) ----
  const std::vector<size_t> &row_ptr() const noexcept { return row_ptr_; }
  const std::vector<size_t> &col_ind() const noexcept { return col_ind_; }
  const std::vector<T> &values() const noexcept { return values_; }

  // ---- element access: binary search within a row (O(log nnz_per_row)) ----
  T at(size_t i, size_t j) const {
    if (i >= rows_ || j >= cols_)
      throw std::out_of_range("CsrMatrix::at OOB");
    auto begin = col_ind_.begin() + (ptrdiff_t)row_ptr_[i];
    auto end = col_ind_.begin() + (ptrdiff_t)row_ptr_[i + 1];
    auto it = std::lower_bound(begin, end, j);
    if (it != end && *it == j)
      return values_[(size_t)(it - col_ind_.begin())];
    return T{};
  }

  // ---- dense interop ----
  Array<T> to_dense() const;
  static CsrMatrix from_dense(const Array<T> &dense);
  static CsrMatrix from_coo(size_t rows, size_t cols,
                            std::vector<size_t> row_ind,
                            std::vector<size_t> col_ind, std::vector<T> values,
                            bool drop_zeros = true);

// ---- transforms ----
    CscMatrix<T> transpose() const;                          // CSC of A^T
    CscMatrix<T> to_csc() const;                             // CSC of A
    CsrMatrix transpose_csr() const { return transpose().to_csr(); }

  // ---- computation ----
  Array<T> matvec(const Array<T> &x) const;      // y = A @ x
  CsrMatrix matmul(const CsrMatrix<T> &B) const; // C = A @ B

private:
  template <typename U>
  friend class CscMatrix; // transpose()/to_csr() assemble CSR storage
  size_t rows_ = 0;
  size_t cols_ = 0;
  std::vector<size_t> row_ptr_; // rows_+1
  std::vector<size_t> col_ind_; // nnz
  std::vector<T> values_;       // nnz
};

// ---------------------------------------------------------------------------
// CscMatrix<T> - Compressed Sparse Column storage.
//
//   col_ptr  [n+1]        start/end offsets into row_ind/values per column
//   row_ind  [nnz]        row index of each stored entry
//   values   [nnz]        numerical value of each stored entry
//
// The transpose of a CsrMatrix is a CscMatrix (and vice versa), so flipping
// formats is a zero-copy reinterpret for column-oriented kernels.
// ---------------------------------------------------------------------------
template <typename T> class CscMatrix {
public:
  CscMatrix() = default;
  CscMatrix(size_t rows, size_t cols)
      : rows_(rows), cols_(cols), col_ptr_(cols + 1, 0) {}

  size_t rows() const noexcept { return rows_; }
  size_t cols() const noexcept { return cols_; }
  size_t nnz() const noexcept { return values_.size(); }

  const std::vector<size_t> &col_ptr() const noexcept { return col_ptr_; }
  const std::vector<size_t> &row_ind() const noexcept { return row_ind_; }
  const std::vector<T> &values() const noexcept { return values_; }

  // ---- element access: binary search within a column ----
  T at(size_t i, size_t j) const {
    if (i >= rows_ || j >= cols_)
      throw std::out_of_range("CscMatrix::at OOB");
    auto begin = row_ind_.begin() + (ptrdiff_t)col_ptr_[j];
    auto end = row_ind_.begin() + (ptrdiff_t)col_ptr_[j + 1];
    auto it = std::lower_bound(begin, end, i);
    if (it != end && *it == i)
      return values_[(size_t)(it - row_ind_.begin())];
    return T{};
  }

  // ---- dense interop ----
  Array<T> to_dense() const;
  static CscMatrix from_dense(const Array<T> &dense);
  static CscMatrix from_coo(size_t rows, size_t cols,
                            std::vector<size_t> row_ind,
                            std::vector<size_t> col_ind, std::vector<T> values,
                            bool drop_zeros = true);

  // ---- transforms ----
  CsrMatrix<T> transpose() const; // zero-cost format
  CsrMatrix<T> to_csr() const;    // materialize same matrix as CSR

  // ---- computation ----
  Array<T> matvec(const Array<T> &x) const; // y = A @ x

private:
  template <typename U>
  friend class CsrMatrix; // transpose() assembles CSC storage
  size_t rows_ = 0;
  size_t cols_ = 0;
  std::vector<size_t> col_ptr_; // cols_+1
  std::vector<size_t> row_ind_; // nnz
  std::vector<T> values_;       // nnz
};

// ===========================================================================
// Implementation
// ===========================================================================

namespace detail {

// Compress a sorted run of (group, within) COO entries into compressed
// storage. Precondition: group is non-decreasing, within ascending within
// each group (i.e. lexicographic by (group, within)). Merges duplicate
// (group, within) pairs by summation and drops merged-to-zero entries when
// drop_zeros is set. Empty groups are padded so ptr is dense of size n+1.
template <typename T>
void compress_grouped(size_t n_groups, const std::vector<size_t> &group,
                      const std::vector<size_t> &within,
                      const std::vector<T> &values,
                      std::vector<size_t> &within_out,
                      std::vector<T> &value_out, std::vector<size_t> &ptr_out,
                      bool drop_zeros) {
  constexpr size_t SENTINEL = std::numeric_limits<size_t>::max();
  std::vector<size_t> ptr(n_groups + 1, SENTINEL);
  std::vector<size_t> wo;
  std::vector<T> vo;
  wo.reserve(group.size());
  vo.reserve(group.size());

  bool first = true;
  size_t cur = 0;
  for (size_t k = 0; k < group.size(); ++k) {
    size_t g = group[k];
    size_t w = within[k];
    T v = values[k];
    if (!first && g == cur && !wo.empty() && wo.back() == w) {
      vo.back() = vo.back() + v; // merge duplicate entry
      if (drop_zeros && vo.back() == T{}) {
        // A merged sum cancels to zero: drop the entry entirely. ptr[g]
        // still references the group start, which is untouched (a later
        // group's ptr is set fresh from wo.size() on its first entry).
        wo.pop_back();
        vo.pop_back();
      }
      continue;
    }
    if (first || g != cur) {
      ptr[g] = wo.size();
      cur = g;
      first = false;
    }
    if (drop_zeros && v == T{})
      continue;
    wo.push_back(w);
    vo.push_back(v);
  }
  ptr[n_groups] = wo.size();
  for (size_t g = n_groups; g-- > 0;) // pad empty groups (trailing entries)
    if (ptr[g] == SENTINEL)
      ptr[g] = ptr[g + 1];

  within_out.swap(wo);
  value_out.swap(vo);
  ptr_out.swap(ptr);
}

// Build an index permutation sorting (row, col) or (col, row) lexicographic,
// depending on which axis is the storage group (0=row-first, i.e. CSR).
inline std::vector<size_t> coo_order(size_t n, const std::vector<size_t> &a,
                                     const std::vector<size_t> &b,
                                     bool a_first) {
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) {
    size_t ax = a_first ? a[x] : b[x], ay = a_first ? a[y] : b[y];
    size_t bx = a_first ? b[x] : a[x], by = a_first ? b[y] : a[y];
    if (ax != ay)
      return ax < ay;
    return bx < by;
  });
  return order;
}

} // namespace detail

// ---- CsrMatrix: construction
// -------------------------------------------------

template <typename T>
CsrMatrix<T> CsrMatrix<T>::from_coo(size_t rows, size_t cols,
                                    std::vector<size_t> row_ind,
                                    std::vector<size_t> col_ind,
                                    std::vector<T> values, bool drop_zeros) {
  size_t n = row_ind.size();
  if (n != col_ind.size() || n != values.size())
    throw std::invalid_argument("COO triplets must have equal length");
  for (size_t k = 0; k < n; ++k)
    if (row_ind[k] >= rows || col_ind[k] >= cols)
      throw std::out_of_range("CsrMatrix::from_coo index out of bounds");

  auto order = detail::coo_order(n, row_ind, col_ind, true);
  std::vector<size_t> group(n), within(n), cptr;
  std::vector<T> vv(n);
  for (size_t k = 0; k < n; ++k) {
    group[k] = row_ind[order[k]];
    within[k] = col_ind[order[k]];
    vv[k] = values[order[k]];
  }
  std::vector<size_t> ccol;
  detail::compress_grouped(rows, group, within, vv, ccol, values, cptr,
                           drop_zeros);

  CsrMatrix<T> out(rows, cols);
  out.row_ptr_.swap(cptr);
  out.col_ind_.swap(ccol);
  out.values_.swap(values);
  return out;
}

template <typename T>
CsrMatrix<T> CsrMatrix<T>::from_dense(const Array<T> &dense) {
  if (dense.ndim() != 2)
    throw std::invalid_argument("from_dense requires 2-D");
  const size_t m = dense.shape()[0], n = dense.shape()[1];
  if (dense.size() == 0)
    throw std::invalid_argument("from_dense: empty matrix not supported");

  std::vector<size_t> rows, cols;
  std::vector<T> vals;
  rows.reserve(dense.size());
  cols.reserve(dense.size());
  vals.reserve(dense.size());
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < n; ++j) {
      T v = dense.at(i, j);
      if (v != T{}) {
        rows.push_back(i);
        cols.push_back(j);
        vals.push_back(v);
      }
    }
  return from_coo(m, n, std::move(rows), std::move(cols), std::move(vals));
}

// ---- CsrMatrix: dense + transpose
// ---------------------------------------------

template <typename T> Array<T> CsrMatrix<T>::to_dense() const {
  Array<T> out({rows_, cols_});
  std::fill(out.data(), out.data() + out.size(), T{});
  for (size_t i = 0; i < rows_; ++i)
    for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
      out.at(i, col_ind_[k]) = values_[k];
  return out;
}

template <typename T> CscMatrix<T> CsrMatrix<T>::transpose() const {
  // A^T is n x m. In CSC its columns are A's rows, so col_ptr is the
  // compressed row-length histogram of A and row_ind are A's columns.
  CscMatrix<T> out(cols_, rows_);
  out.col_ptr_.assign(rows_ + 1, 0);
  for (size_t i = 0; i < rows_; ++i)
    out.col_ptr_[i + 1] += (row_ptr_[i + 1] - row_ptr_[i]);
  std::partial_sum(out.col_ptr_.begin(), out.col_ptr_.end(),
                   out.col_ptr_.begin());

out.row_ind_.resize(nnz());
  out.values_.resize(nnz());
  std::vector<size_t> cursor(out.col_ptr_.begin(), out.col_ptr_.end() - 1);
  for (size_t i = 0; i < rows_; ++i)
    for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
      size_t p = cursor[i]++;
      out.row_ind_[p] = col_ind_[k];
      out.values_[p] = values_[k];
    }
  return out;
}

template <typename T>
CscMatrix<T> CsrMatrix<T>::to_csc() const {
    // CSC of A (same orientation): col_ptr compresses the column histogram of
    // A; row_ind holds each entry's row. Visiting entries in CSR order keeps
    // rows ascending within every column, satisfying the CSC invariant.
    CscMatrix<T> out(rows_, cols_);
    out.col_ptr_.assign(cols_ + 1, 0);
    for (size_t c : col_ind_) ++out.col_ptr_[c + 1];
    std::partial_sum(out.col_ptr_.begin(), out.col_ptr_.end(), out.col_ptr_.begin());

    out.row_ind_.resize(nnz());
    out.values_.resize(nnz());
    std::vector<size_t> cursor(out.col_ptr_.begin(), out.col_ptr_.end() - 1);
    for (size_t i = 0; i < rows_; ++i)
        for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            size_t p = cursor[col_ind_[k]]++;
            out.row_ind_[p] = i;
            out.values_[p] = values_[k];
        }
    return out;
}

// ---- CsrMatrix: computation
// ----------------------------------------------------

template <typename T> Array<T> CsrMatrix<T>::matvec(const Array<T> &x) const {
  if (x.ndim() != 1 || x.size() != cols_)
    throw std::invalid_argument("CsrMatrix::matvec: x must have length cols");
  Array<T> y({rows_});
  std::fill(y.data(), y.data() + y.size(), T{});
  for (size_t i = 0; i < rows_; ++i) {
    T acc = T{};
    for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
      acc += values_[k] * x[col_ind_[k]];
    y[i] = acc;
  }
  return y;
}

template <typename T>
CsrMatrix<T> CsrMatrix<T>::matmul(const CsrMatrix<T> &B) const {
  if (cols_ != B.rows())
    throw std::invalid_argument("CsrMatrix::matmul: inner dims mismatch");
  const size_t m = rows_, n = cols_, p = B.cols();
  const std::vector<size_t> &bp = B.row_ptr();
  const std::vector<size_t> &bc = B.col_ind();
  const std::vector<T> &bv = B.values();

  // Classic CSR x CSR algorithm: a column workspace plus a per-row marker
  // so each row accumulates into only its touched columns (O(nnz + m*p)).
  std::vector<T> workspace(p, T{});
  std::vector<size_t> marker(p, std::numeric_limits<size_t>::max());

  std::vector<size_t> c_row_ptr(m + 1, 0);
  std::vector<size_t> c_col;
  std::vector<T> c_val;
  c_col.reserve(nnz() * 2);
  c_val.reserve(nnz() * 2);

  for (size_t i = 0; i < m; ++i) {
    for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
      const size_t col_a = col_ind_[k];
      const T a = values_[k];
      for (size_t jj = bp[col_a]; jj < bp[col_a + 1]; ++jj) {
        const size_t j = bc[jj];
        if (marker[j] != i) {
          marker[j] = i;
          workspace[j] = a * bv[jj];
        } else {
          workspace[j] = workspace[j] + a * bv[jj];
        }
      }
    }
    for (size_t j = 0; j < p; ++j) {
      if (marker[j] == i) {
        if (workspace[j] != T{}) {
          c_col.push_back(j);
          c_val.push_back(workspace[j]);
        }
        workspace[j] = T{};
      }
    }
    c_row_ptr[i + 1] = c_col.size();
  }

  CsrMatrix<T> out(m, p);
  out.row_ptr_.swap(c_row_ptr);
  out.col_ind_.swap(c_col);
  out.values_.swap(c_val);
  return out;
}

// ---- CscMatrix: construction
// ----------------------------------------------------

template <typename T>
CscMatrix<T> CscMatrix<T>::from_coo(size_t rows, size_t cols,
                                    std::vector<size_t> row_ind,
                                    std::vector<size_t> col_ind,
                                    std::vector<T> values, bool drop_zeros) {
  size_t n = row_ind.size();
  if (n != col_ind.size() || n != values.size())
    throw std::invalid_argument("COO triplets must have equal length");
  for (size_t k = 0; k < n; ++k)
    if (row_ind[k] >= rows || col_ind[k] >= cols)
      throw std::out_of_range("CscMatrix::from_coo index out of bounds");

  auto order = detail::coo_order(n, col_ind, row_ind, true); // (col, row)
  std::vector<size_t> group(n), within(n), cptr;
  std::vector<T> vv(n);
  for (size_t k = 0; k < n; ++k) {
    group[k] = col_ind[order[k]];
    within[k] = row_ind[order[k]];
    vv[k] = values[order[k]];
  }
  std::vector<size_t> rind;
  detail::compress_grouped(cols, group, within, vv, rind, values, cptr,
                           drop_zeros);

  CscMatrix<T> out(rows, cols);
  out.col_ptr_.swap(cptr);
  out.row_ind_.swap(rind);
  out.values_.swap(values);
  return out;
}

template <typename T>
CscMatrix<T> CscMatrix<T>::from_dense(const Array<T> &dense) {
  if (dense.ndim() != 2)
    throw std::invalid_argument("from_dense requires 2-D");
  const size_t m = dense.shape()[0], n = dense.shape()[1];
  if (dense.size() == 0)
    throw std::invalid_argument("from_dense: empty matrix not supported");

  std::vector<size_t> rows, cols;
  std::vector<T> vals;
  rows.reserve(dense.size());
  cols.reserve(dense.size());
  vals.reserve(dense.size());
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < n; ++j) {
      T v = dense.at(i, j);
      if (v != T{}) {
        rows.push_back(i);
        cols.push_back(j);
        vals.push_back(v);
      }
    }
  return from_coo(m, n, std::move(rows), std::move(cols), std::move(vals));
}

// ---- CscMatrix: dense + transpose
// ------------------------------------------------

template <typename T> Array<T> CscMatrix<T>::to_dense() const {
  Array<T> out({rows_, cols_});
  std::fill(out.data(), out.data() + out.size(), T{});
  for (size_t j = 0; j < cols_; ++j)
    for (size_t k = col_ptr_[j]; k < col_ptr_[j + 1]; ++k)
      out.at(row_ind_[k], j) = values_[k];
  return out;
}

template <typename T> CsrMatrix<T> CscMatrix<T>::transpose() const {
  // A^T is n x m. In CSR its rows are A's columns, so row_ptr is the
  // compressed column-length histogram of A and col_ind are A's rows.
  CsrMatrix<T> out(cols_, rows_);
  out.row_ptr_.assign(cols_ + 1, 0);
  for (size_t j = 0; j < cols_; ++j)
    out.row_ptr_[j + 1] += (col_ptr_[j + 1] - col_ptr_[j]);
  std::partial_sum(out.row_ptr_.begin(), out.row_ptr_.end(),
                   out.row_ptr_.begin());

  out.col_ind_.resize(nnz());
  out.values_.resize(nnz());
  std::vector<size_t> cursor(out.row_ptr_.begin(), out.row_ptr_.end() - 1);
  for (size_t j = 0; j < cols_; ++j)
    for (size_t k = col_ptr_[j]; k < col_ptr_[j + 1]; ++k) {
      size_t p = cursor[j]++;
      out.col_ind_[p] = row_ind_[k];
      out.values_[p] = values_[k];
    }
  return out;
}

// CSC -> CSR in the same orientation (no transpose). Compresses A's stored
// rows into CSR row pointers via a histogram of row_ind.
template <typename T> CsrMatrix<T> CscMatrix<T>::to_csr() const {
  CsrMatrix<T> out(rows_, cols_);
  out.row_ptr_.assign(rows_ + 1, 0);
  for (size_t r : row_ind_)
    ++out.row_ptr_[r + 1];
  std::partial_sum(out.row_ptr_.begin(), out.row_ptr_.end(),
                   out.row_ptr_.begin());

  out.col_ind_.resize(nnz());
  out.values_.resize(nnz());
  std::vector<size_t> cursor(out.row_ptr_.begin(), out.row_ptr_.end() - 1);
  for (size_t j = 0; j < cols_; ++j)
    for (size_t k = col_ptr_[j]; k < col_ptr_[j + 1]; ++k) {
      size_t p = cursor[row_ind_[k]]++;
      out.col_ind_[p] = j;
      out.values_[p] = values_[k];
    }
  return out;
}

// ---- CscMatrix: computation
// ---------------------------------------------------------

template <typename T> Array<T> CscMatrix<T>::matvec(const Array<T> &x) const {
  if (x.ndim() != 1 || x.size() != cols_)
    throw std::invalid_argument("CscMatrix::matvec: x must have length cols");
  Array<T> y({rows_});
  std::fill(y.data(), y.data() + y.size(), T{});
  for (size_t j = 0; j < cols_; ++j) {
    const T xj = x[j];
    for (size_t k = col_ptr_[j]; k < col_ptr_[j + 1]; ++k)
      y[row_ind_[k]] += values_[k] * xj;
  }
  return y;
}

} // namespace dracolix
