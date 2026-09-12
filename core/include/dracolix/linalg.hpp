#pragma once
// Phase 2a - Linear algebra prototype: atleast-3D batched support
// Licensed under GPL-3.0-only - see LICENSE
#include "array.hpp"
#include "dracolix/kernels/gemm.hpp"
#include <cmath>
#include <stdexcept>

namespace dracolix::linalg {

// MatMul: C = A @ B
// - 2-D: (m,n) @ (n,p) -> (m,p)
// - N-D batched: (..., m,n) @ (..., n,p) -> broadcast batch + (m,p)
// Atleast-3D guarantee: 1-D/2-D/3-D are first-class, N>3 works via same logic
template <typename T>
Array<T> matmul(const Array<T>& A, const Array<T>& B);

template <typename T>
Array<T> matvec(const Array<T>& A, const Array<T>& x); // A: (..., MxN), x: (..., N) or (N,) -> (..., M)

template <typename T>
T dot(const Array<T>& a, const Array<T>& b); // both 1-D

template <typename T>
Array<T> dot_batched(const Array<T>& A, const Array<T>& B); // (..., N) @ (..., N) -> (...)

template <typename T>
double norm(const Array<T>& a, int p = 2);

template <typename T>
Array<T> diagonal(const Array<T>& A);

template <typename T>
Array<T> diag(const Array<T>& d);

// ---- internals ----
namespace detail {
inline bool is_contiguous_rowmajor(const std::vector<size_t>& strides, const std::vector<size_t>& shape) {
    if (shape.empty()) return true;
    size_t expect = 1;
    for (int i=(int)shape.size()-1; i>=0; --i) {
        if (strides[i] != expect) return false;
        expect *= shape[i];
    }
    return true;
}
constexpr size_t BLAS_THRESHOLD = 64*64*64;

inline std::vector<size_t> broadcast_batch(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t na=a.size(), nb=b.size();
    size_t n = std::max(na, nb);
    std::vector<size_t> res(n);
    for (int i=(int)n-1, ia=(int)na-1, ib=(int)nb-1; i>=0; --i, --ia, --ib) {
        size_t da = ia>=0 ? a[ia] : 1;
        size_t db = ib>=0 ? b[ib] : 1;
        if (da!=db && da!=1 && db!=1) throw std::invalid_argument("matmul: incompatible batch shapes");
        res[i]=std::max(da,db);
    }
    return res;
}

inline size_t batch_offset(const std::vector<size_t>& shape, const std::vector<size_t>& strides,
                           const std::vector<size_t>& batch_idx, size_t batch_ndim, size_t total_ndim) {
    // shape/strides include last 2 matrix dims; batch dims are leading total_ndim-2
    size_t batch_dims = total_ndim >=2 ? total_ndim-2 : 0;
    size_t offset=0;
    for(size_t i=0;i<batch_dims;++i){
        int orig_d = (int)i - (int)(batch_dims - (shape.size() >=2 ? shape.size()-2 : 0));
        // map: batch_idx[i] -> shape dim
        // shape batch: leading (shape.size()-2) dims
        size_t shape_batch_ndim = shape.size() >=2 ? shape.size()-2 : 0;
        int shape_d = (int)i - (int)(batch_dims - (int)shape_batch_ndim);
        if (shape_d <0) continue;
        if (shape[shape_d]==1) continue;
        offset += batch_idx[i] * strides[shape_d];
    }
    return offset;
}
}

// 2-D specializations (fast path)
template <>
inline Array<double> matmul<double>(const Array<double>& A, const Array<double>& B) {
    if (A.ndim() <2 || B.ndim() <2) throw std::invalid_argument("matmul requires >=2-D");
    if (A.shape().back() != B.shape()[B.ndim()-2]) throw std::invalid_argument("matmul inner dims mismatch");
    if (A.ndim()==2 && B.ndim()==2) {
        if (!detail::is_contiguous_rowmajor(A.strides(), A.shape()) || !detail::is_contiguous_rowmajor(B.strides(), B.shape()))
            throw std::invalid_argument("matmul requires contiguous RowMajor");
        size_t m=A.shape()[0], n=A.shape()[1], p=B.shape()[1];
        Array<double> C({m,p});
        std::fill(C.data(), C.data()+C.size(), 0.0);
        kernels::gemm_f64(A.data(), B.data(), C.data(), m,n,p);
        return C;
    }
    // N-D batched path
    size_t m = A.shape()[A.ndim()-2], n = A.shape()[A.ndim()-1], p = B.shape()[B.ndim()-1];
    if (n != B.shape()[B.ndim()-2]) throw std::invalid_argument("matmul inner dims mismatch (batched)");
    std::vector<size_t> a_batch(A.shape().begin(), A.shape().end()-2);
    std::vector<size_t> b_batch(B.shape().begin(), B.shape().end()-2);
    auto res_batch = detail::broadcast_batch(a_batch, b_batch);
    std::vector<size_t> res_shape = res_batch;
    res_shape.push_back(m); res_shape.push_back(p);
    Array<double> C(res_shape);
    std::fill(C.data(), C.data()+C.size(), 0.0);
    // iterate batches
    size_t batch_total = 1; for(auto s: res_batch) batch_total*=s;
    if (batch_total==0) batch_total=1;
    std::vector<size_t> batch_idx(res_batch.size());
    for(size_t b=0;b<batch_total;++b){
        size_t rem=b;
        for(int d=(int)res_batch.size()-1; d>=0; --d){ batch_idx[d]=rem % res_batch[d]; rem/=res_batch[d]; }
        size_t offA = detail::batch_offset(A.shape(), A.strides(), batch_idx, res_batch.size(), res_shape.size());
        size_t offB = detail::batch_offset(B.shape(), B.strides(), batch_idx, res_batch.size(), res_shape.size());
        size_t offC = 0; // C batch is contiguous with res_batch strides
        // compute C offset: batch_idx * C batch strides
        for(size_t i=0;i<res_batch.size();++i) offC += batch_idx[i] * C.strides()[i];
        // gemm on m x n * n x p
        kernels::gemm_f64(A.data()+offA, B.data()+offB, C.data()+offC, m,n,p);
    }
    return C;
}

template <>
inline Array<float> matmul<float>(const Array<float>& A, const Array<float>& B) {
    if (A.ndim() <2 || B.ndim() <2) throw std::invalid_argument("matmul requires >=2-D");
    if (A.shape().back() != B.shape()[B.ndim()-2]) throw std::invalid_argument("matmul inner dims mismatch");
    if (A.ndim()==2 && B.ndim()==2) {
        size_t m=A.shape()[0], n=A.shape()[1], p=B.shape()[1];
        Array<float> C({m,p});
        std::fill(C.data(), C.data()+C.size(), 0.0f);
        kernels::gemm_f32(A.data(), B.data(), C.data(), m,n,p);
        return C;
    }
    size_t m = A.shape()[A.ndim()-2], n = A.shape()[A.ndim()-1], p = B.shape()[B.ndim()-1];
    std::vector<size_t> a_batch(A.shape().begin(), A.shape().end()-2);
    std::vector<size_t> b_batch(B.shape().begin(), B.shape().end()-2);
    auto res_batch = detail::broadcast_batch(a_batch, b_batch);
    std::vector<size_t> res_shape = res_batch; res_shape.push_back(m); res_shape.push_back(p);
    Array<float> C(res_shape);
    std::fill(C.data(), C.data()+C.size(), 0.0f);
    size_t batch_total=1; for(auto s: res_batch) batch_total*=s; if(batch_total==0) batch_total=1;
    std::vector<size_t> batch_idx(res_batch.size());
    for(size_t b=0;b<batch_total;++b){
        size_t rem=b; for(int d=(int)res_batch.size()-1; d>=0; --d){ batch_idx[d]=rem%res_batch[d]; rem/=res_batch[d]; }
        size_t offA = detail::batch_offset(A.shape(), A.strides(), batch_idx, res_batch.size(), res_shape.size());
        size_t offB = detail::batch_offset(B.shape(), B.strides(), batch_idx, res_batch.size(), res_shape.size());
        size_t offC=0; for(size_t i=0;i<res_batch.size();++i) offC+=batch_idx[i]*C.strides()[i];
        kernels::gemm_f32(A.data()+offA, B.data()+offB, C.data()+offC, m,n,p);
    }
    return C;
}

// Generic for int types and N-D
template <typename T>
Array<T> matmul(const Array<T>& A, const Array<T>& B) {
    if constexpr (std::is_same_v<T,double> || std::is_same_v<T,float>) {
        // int fallback already handled by specializations for 2-D, but N-D generic needs to run
        // if we reach here it's already handled - this is duplicate path
        size_t m=A.shape()[A.ndim()-2], n=A.shape()[A.ndim()-1], p=B.shape()[B.ndim()-1];
        std::vector<size_t> a_batch(A.shape().begin(), A.shape().end()-2);
        std::vector<size_t> b_batch(B.shape().begin(), B.shape().end()-2);
        auto res_batch = detail::broadcast_batch(a_batch, b_batch);
        std::vector<size_t> res_shape=res_batch; res_shape.push_back(m); res_shape.push_back(p);
        Array<T> C(res_shape);
        std::fill(C.data(), C.data()+C.size(), T{});
        size_t batch_total=1; for(auto s:res_batch) batch_total*=s; if(batch_total==0) batch_total=1;
        std::vector<size_t> batch_idx(res_batch.size());
        for(size_t b=0;b<batch_total;++b){
            size_t rem=b; for(int d=(int)res_batch.size()-1; d>=0; --d){ batch_idx[d]=rem%res_batch[d]; rem/=res_batch[d]; }
            size_t offA = detail::batch_offset(A.shape(), A.strides(), batch_idx, res_batch.size(), res_shape.size());
            size_t offB = detail::batch_offset(B.shape(), B.strides(), batch_idx, res_batch.size(), res_shape.size());
            size_t offC=0; for(size_t i=0;i<res_batch.size();++i) offC+=batch_idx[i]*C.strides()[i];
            // naive
            for(size_t i=0;i<m;++i) for(size_t k=0;k<n;++k){ T aik=*(A.data()+offA + i*A.strides()[A.ndim()-2] + k*A.strides()[A.ndim()-1]); for(size_t j=0;j<p;++j) *(C.data()+offC + i*C.strides()[C.ndim()-2]+j*C.strides()[C.ndim()-1]) += aik * *(B.data()+offB + k*B.strides()[B.ndim()-2]+j*B.strides()[B.ndim()-1]); }
        }
        return C;
    } else {
        if (A.ndim()<2 || B.ndim()<2) throw std::invalid_argument("matmul requires >=2-D");
        size_t m=A.shape()[A.ndim()-2], n=A.shape()[A.ndim()-1], p=B.shape()[B.ndim()-1];
        if (n != B.shape()[B.ndim()-2]) throw std::invalid_argument("matmul inner dims");
        std::vector<size_t> a_batch(A.shape().begin(), A.shape().end()-2);
        std::vector<size_t> b_batch(B.shape().begin(), B.shape().end()-2);
        auto res_batch = detail::broadcast_batch(a_batch, b_batch);
        std::vector<size_t> res_shape=res_batch; res_shape.push_back(m); res_shape.push_back(p);
        Array<T> C(res_shape);
        std::fill(C.data(), C.data()+C.size(), T{});
        size_t batch_total=1; for(auto s:res_batch) batch_total*=s; if(batch_total==0) batch_total=1;
        std::vector<size_t> batch_idx(res_batch.size());
        for(size_t b=0;b<batch_total;++b){
            size_t rem=b; for(int d=(int)res_batch.size()-1; d>=0; --d){ batch_idx[d]=rem%res_batch[d]; rem/=res_batch[d]; }
            size_t offA = detail::batch_offset(A.shape(), A.strides(), batch_idx, res_batch.size(), res_shape.size());
            size_t offB = detail::batch_offset(B.shape(), B.strides(), batch_idx, res_batch.size(), res_shape.size());
            size_t offC=0; for(size_t i=0;i<res_batch.size();++i) offC+=batch_idx[i]*C.strides()[i];
            for(size_t i=0;i<m;++i) for(size_t k=0;k<n;++k){ T aik=*(A.data()+offA + i*A.strides()[A.ndim()-2] + k*A.strides()[A.ndim()-1]); for(size_t j=0;j<p;++j) *(C.data()+offC + i*C.strides()[C.ndim()-2]+j*C.strides()[C.ndim()-1]) += aik * *(B.data()+offB + k*B.strides()[B.ndim()-2]+j*B.strides()[B.ndim()-1]); }
        }
        return C;
    }
}

template <typename T>
Array<T> matvec(const Array<T>& A, const Array<T>& x) {
    if (A.ndim()<2) throw std::invalid_argument("matvec: A must be >=2-D");
    size_t n = A.shape().back();
    // x can be 1-D (N) or N-D batched (..., N)
    if (x.ndim()==1) {
        if (x.shape()[0]!=n) throw std::invalid_argument("matvec inner dims");
        if (A.ndim()==2) {
            size_t m=A.shape()[0];
            Array<T> y({m}); std::fill(y.data(), y.data()+m, T{});
            for(size_t i=0;i<m;++i){ T acc=T{}; for(size_t k=0;k<n;++k) acc+=A.at(i,k)*x[k]; y[i]=acc; }
            return y;
        }
        // batched A with unbatched x: broadcast x
        std::vector<size_t> a_batch(A.shape().begin(), A.shape().end()-2);
        size_t m = A.shape()[A.ndim()-2];
        Array<T> y(a_batch); // we will push m? Actually y shape is batch + [m]
        std::vector<size_t> y_shape = a_batch; y_shape.push_back(m);
        Array<T> Y(y_shape); std::fill(Y.data(), Y.data()+Y.size(), T{});
        size_t batch_total=1; for(auto s:a_batch) batch_total*=s;
        std::vector<size_t> batch_idx(a_batch.size());
        for(size_t b=0;b<batch_total;++b){
            size_t rem=b; for(int d=(int)a_batch.size()-1; d>=0; --d){ batch_idx[d]=rem%a_batch[d]; rem/=a_batch[d]; }
            size_t offA=detail::batch_offset(A.shape(), A.strides(), batch_idx, a_batch.size(), A.ndim());
            size_t offY=0; for(size_t i=0;i<a_batch.size();++i) offY+=batch_idx[i]*Y.strides()[i];
            for(size_t i=0;i<m;++i){ T acc=T{}; for(size_t k=0;k<n;++k) acc += *(A.data()+offA + i*A.strides()[A.ndim()-2]+k*A.strides()[A.ndim()-1]) * x[k]; *(Y.data()+offY + i*Y.strides()[Y.ndim()-1]) = acc; }
        }
        return Y;
    } else {
        // both batched
        if (x.shape().back()!=n) throw std::invalid_argument("matvec inner dims (batched)");
        std::vector<size_t> a_batch(A.shape().begin(), A.shape().end()-2);
        std::vector<size_t> x_batch(x.shape().begin(), x.shape().end()-1);
        auto res_batch = detail::broadcast_batch(a_batch, x_batch);
        size_t m = A.shape()[A.ndim()-2];
        std::vector<size_t> y_shape=res_batch; y_shape.push_back(m);
        Array<T> Y(y_shape); std::fill(Y.data(), Y.data()+Y.size(), T{});
        size_t batch_total=1; for(auto s:res_batch) batch_total*=s; if(batch_total==0) batch_total=1;
        std::vector<size_t> batch_idx(res_batch.size());
        for(size_t b=0;b<batch_total;++b){
            size_t rem=b; for(int d=(int)res_batch.size()-1; d>=0; --d){ batch_idx[d]=rem%res_batch[d]; rem/=res_batch[d]; }
            size_t offA=detail::batch_offset(A.shape(), A.strides(), batch_idx, res_batch.size(), A.ndim());
            size_t offX=detail::batch_offset(x.shape(), x.strides(), batch_idx, res_batch.size(), x.ndim());
            size_t offY=0; for(size_t i=0;i<res_batch.size();++i) offY+=batch_idx[i]*Y.strides()[i];
            for(size_t i=0;i<m;++i){ T acc=T{}; for(size_t k=0;k<n;++k) acc += *(A.data()+offA + i*A.strides()[A.ndim()-2]+k*A.strides()[A.ndim()-1]) * *(x.data()+offX + k*x.strides()[x.ndim()-1]); *(Y.data()+offY + i*Y.strides()[Y.ndim()-1])=acc; }
        }
        return Y;
    }
}

template <typename T>
T dot(const Array<T>& a, const Array<T>& b) {
    if (a.ndim()!=1 || b.ndim()!=1) throw std::invalid_argument("dot requires 1-D");
    if (a.shape()[0] != b.shape()[0]) throw std::invalid_argument("dot size mismatch");
    T acc=T{};
    for(size_t i=0;i<a.size();++i) acc += a[i]*b[i];
    return acc;
}

template <typename T>
Array<T> dot_batched(const Array<T>& A, const Array<T>& B) {
    // (..., N) @ (..., N) -> (...) broadcast batch
    if (A.ndim()<1 || B.ndim()<1) throw std::invalid_argument("dot_batched requires >=1-D");
    if (A.shape().back() != B.shape().back()) throw std::invalid_argument("dot_batched inner dims");
    std::vector<size_t> a_batch(A.shape().begin(), A.shape().end()-1);
    std::vector<size_t> b_batch(B.shape().begin(), B.shape().end()-1);
    auto res_batch = detail::broadcast_batch(a_batch, b_batch);
    Array<T> C(res_batch.empty() ? std::vector<size_t>{1} : res_batch);
    // if scalar batch (both 1-D) we return 1-element; dot() is scalar version - this is batched variant
    if (res_batch.empty()) { C[0]=dot(A,B); return C; }
    std::fill(C.data(), C.data()+C.size(), T{});
    size_t batch_total=1; for(auto s:res_batch) batch_total*=s;
    std::vector<size_t> batch_idx(res_batch.size());
    size_t N=A.shape().back();
    for(size_t b=0;b<batch_total;++b){
        size_t rem=b; for(int d=(int)res_batch.size()-1; d>=0; --d){ batch_idx[d]=rem%res_batch[d]; rem/=res_batch[d]; }
        size_t offA=detail::batch_offset(A.shape(), A.strides(), batch_idx, res_batch.size(), A.ndim());
        size_t offB=detail::batch_offset(B.shape(), B.strides(), batch_idx, res_batch.size(), B.ndim());
        size_t offC=0; for(size_t i=0;i<res_batch.size();++i) offC+=batch_idx[i]*C.strides()[i];
        T acc=T{};
        for(size_t k=0;k<N;++k) acc += *(A.data()+offA + k*A.strides()[A.ndim()-1]) * *(B.data()+offB + k*B.strides()[B.ndim()-1]);
        *(C.data()+offC)=acc;
    }
    return C;
}

template <typename T>
double norm(const Array<T>& a, int p) {
    if (a.ndim()!=1) throw std::invalid_argument("norm requires 1-D");
    if (a.size()==0) throw std::logic_error("norm of empty");
    if (p==0) { double m=0; for(size_t i=0;i<a.size();++i) m=std::max(m, std::abs((double)a[i])); return m; }
    else if (p==1) { double s=0; for(size_t i=0;i<a.size();++i) s+=std::abs((double)a[i]); return s; }
    else if (p==2) { double s=0; for(size_t i=0;i<a.size();++i) s+=(double)a[i]*(double)a[i]; return std::sqrt(s); }
    else throw std::invalid_argument("norm: only p=0(inf),1,2");
}

template <typename T>
Array<T> diagonal(const Array<T>& A) {
    if (A.ndim()!=2) throw std::invalid_argument("diagonal requires 2-D");
    size_t d = std::min(A.shape()[0], A.shape()[1]);
    Array<T> out({d});
    for(size_t i=0;i<d;++i) out[i]=A.at(i,i);
    return out;
}

template <typename T>
Array<T> diag(const Array<T>& d) {
    if (d.ndim()!=1) throw std::invalid_argument("diag requires 1-D");
    size_t n=d.size();
    Array<T> M({n,n});
    std::fill(M.data(), M.data()+M.size(), T{});
    for(size_t i=0;i<n;++i) M.at(i,i)=d[i];
    return M;
}

} // namespace dracolix::linalg
