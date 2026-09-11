#pragma once
#include "dtype.hpp"
#include "layout.hpp"
#include "array_view.hpp"
#include <vector>
#include <stdexcept>
#include <numeric>
#include <memory>
#include <cstring>
#include <algorithm>
#include <limits>
#include <type_traits>

namespace dracolix {

template <typename T>
class Array {
public:
    explicit Array(size_t size, Layout layout = Layout::RowMajor)
        : size_(size), ndim_(1), shape_{size}, dtype_(DTypeOf<T>::value), layout_(layout) {
        if (dtype_ == DType::Void) throw std::logic_error("Array<T>: unmapped DType");
        data_.resize(size_);
        compute_strides();
    }

    explicit Array(std::vector<size_t> shape, Layout layout = Layout::RowMajor)
        : shape_(std::move(shape)), dtype_(DTypeOf<T>::value), layout_(layout) {
        if (dtype_ == DType::Void) throw std::logic_error("Array<T>: unmapped DType");
        if (shape_.empty()) throw std::invalid_argument("Array: shape cannot be empty");
        ndim_ = shape_.size();
        size_ = 1;
        for (auto s : shape_) {
            if (s == 0) throw std::invalid_argument("Array: dimension cannot be 0");
            size_ *= s;
        }
        data_.resize(size_);
        compute_strides();
    }

    // Copy/move
    Array(const Array&) = default;
    Array(Array&&) noexcept = default;
    Array& operator=(const Array&) = default;
    Array& operator=(Array&&) noexcept = default;
    ~Array() = default;

    // Flat access (contiguous logical)
    T& operator[](size_t idx) {
        if (idx >= size_) throw std::out_of_range("Array index out of bounds");
        return data_[idx];
    }

    // Const flat access (contiguous logical)
    const T& operator[](size_t idx) const {
        if (idx >= size_) throw std::out_of_range("Array index out of bounds");
        return data_[idx];
    }

    // 2-D accessor
    T& at(size_t i, size_t j) {
        if (ndim_ != 2) throw std::logic_error("at(i,j) requires ndim==2");
        if (i >= shape_[0] || j >= shape_[1]) throw std::out_of_range("at(i,j) out of bounds");
        return data_[i * strides_[0] + j * strides_[1]];
    }
    const T& at(size_t i, size_t j) const { return const_cast<Array*>(this)->at(i,j); }

    // N-D accessor
    T& at(const std::vector<size_t>& indices) {
        if (indices.size() != ndim_) throw std::invalid_argument("at: rank mismatch");
        size_t off = 0;
        for (size_t d=0; d<ndim_; ++d) {
            if (indices[d] >= shape_[d]) throw std::out_of_range("at OOB");
            off += indices[d] * strides_[d];
        }
        return data_[off];
    }
    const T& at(const std::vector<size_t>& indices) const { return const_cast<Array*>(this)->at(indices); }

    // Element-wise ops - return new array
    Array operator+(const Array& o) const { return elementwise(o, [](T a, T b){ return a+b; }); }
    Array operator-(const Array& o) const { return elementwise(o, [](T a, T b){ return a-b; }); }
    Array operator*(const Array& o) const { return elementwise(o, [](T a, T b){ return a*b; }); }
    Array operator/(const Array& o) const { return elementwise(o, [](T a, T b){ return a/b; }); }

    Array operator+(T scalar) const { return scalar_op(scalar, [](T a, T s){ return a+s; }); }
    Array operator-(T scalar) const { return scalar_op(scalar, [](T a, T s){ return a-s; }); }
    Array operator*(T scalar) const { return scalar_op(scalar, [](T a, T s){ return a*s; }); }
    Array operator/(T scalar) const { return scalar_op(scalar, [](T a, T s){ return a/s; }); }

    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }
    size_t size() const noexcept { return size_; }
    size_t ndim() const noexcept { return ndim_; }
    const std::vector<size_t>& shape() const noexcept { return shape_; }
    const std::vector<size_t>& strides() const noexcept { return strides_; }
    DType dtype() const noexcept { return dtype_; }
    Layout layout() const noexcept { return layout_; }

    void reshape(std::vector<size_t> new_shape) {
        size_t new_size = 1;
        for (auto s : new_shape) new_size *= s;
        if (new_size != size_) {
            std::vector<T> nd(new_size, T{});
            size_t keep = std::min(size_, new_size);
            std::copy(data_.begin(), data_.begin() + keep, nd.begin());
            data_.swap(nd);
            size_ = new_size;
        }
        shape_ = std::move(new_shape);
        ndim_ = shape_.size();
        compute_strides();
    }

    // ---- Views ----
    ArrayView<T> view() {
        return ArrayView<T>(data_.data(), shape_, strides_);
    }
    ArrayView<const T> view() const {
        return ArrayView<const T>(data_.data(), shape_, strides_);
    }

    // Slicing: 1-D convenience
    ArrayView<T> slice(Slice s) {
        if (ndim_ != 1) throw std::logic_error("slice(Slice) requires 1-D");
        size_t off; size_t len; int64_t step;
        normalize_slice(s, shape_[0], off, len, step);
        std::vector<size_t> nshape{len};
        std::vector<size_t> nstrides{(size_t)( (int64_t)strides_[0] * step )};
        T* ndata = data_.data() + off * strides_[0];
        return ArrayView<T>(ndata, nshape, nstrides);
    }

    // Slicing: N-D with vector<Slice> (one per dim)
    ArrayView<T> slice(const std::vector<Slice>& slices) {
        if (slices.size() != ndim_) throw std::invalid_argument("slice: rank mismatch");
        std::vector<size_t> nshape; nshape.reserve(ndim_);
        std::vector<size_t> nstrides; nstrides.reserve(ndim_);
        size_t base_off = 0;
        for (size_t d=0; d<ndim_; ++d) {
            size_t off, len; int64_t step;
            normalize_slice(slices[d], shape_[d], off, len, step);
            base_off += off * strides_[d];
            // Drop dimensions where len==1? keep for now - NumPy keeps
            nshape.push_back(len);
            nstrides.push_back((size_t)((int64_t)strides_[d] * step));
        }
        // Remove dimensions with len==0? Keep as shape with 0? For now keep
        T* ndata = data_.data() + base_off;
        return ArrayView<T>(ndata, nshape, nstrides);
    }

    // ---- Transpose ----
    // View transpose (no copy) - only for 2-D
    ArrayView<T> transpose_view() {
        if (ndim_ != 2) throw std::logic_error("transpose_view requires 2-D");
        std::vector<size_t> nshape{shape_[1], shape_[0]};
        std::vector<size_t> nstrides{strides_[1], strides_[0]};
        return ArrayView<T>(data_.data(), nshape, nstrides);
    }
    ArrayView<const T> transpose_view() const {
        if (ndim_ != 2) throw std::logic_error("transpose_view requires 2-D");
        return ArrayView<const T>(data_.data(), {shape_[1], shape_[0]}, {strides_[1], strides_[0]});
    }
    // N-D permute
    ArrayView<T> permute(const std::vector<size_t>& axes) {
        if (axes.size() != ndim_) throw std::invalid_argument("permute: axes size mismatch");
        std::vector<size_t> nshape(ndim_);
        std::vector<size_t> nstrides(ndim_);
        std::vector<bool> seen(ndim_, false);
        for (size_t i=0;i<ndim_;++i) {
            if (axes[i] >= ndim_) throw std::out_of_range("permute axis OOB");
            if (seen[axes[i]]) throw std::invalid_argument("permute: duplicate axis");
            seen[axes[i]]=true;
            nshape[i]=shape_[axes[i]];
            nstrides[i]=strides_[axes[i]];
        }
        return ArrayView<T>(data_.data(), nshape, nstrides);
    }

    // Materialized transpose (copy)
    Array transpose() const {
        if (ndim_ != 2) throw std::logic_error("transpose requires 2-D");
        Array res({shape_[1], shape_[0]});
        for (size_t i=0;i<shape_[0];++i)
            for (size_t j=0;j<shape_[1];++j)
                res.at(j,i) = at(i,j);
        return res;
    }

    Array permuted(const std::vector<size_t>& axes) const {
        auto v = const_cast<Array*>(this)->permute(axes);
        // materialize
        Array res(v.shape());
        // iterate over all indices
        for (size_t flat=0; flat<res.size(); ++flat) {
            // unravel flat to res indices
            size_t rem=flat;
            std::vector<size_t> idx(ndim_);
            for (int d=(int)ndim_-1; d>=0; --d) {
                idx[d]=rem % res.shape()[d];
                rem/=res.shape()[d];
            }
            // map to view indices via inverse permute? simpler: use view at
            res.data()[flat]=v.at(idx);
        }
        return res;
    }

    // ---- Copy / Clone ----
    Array clone() const { return *this; }

    // ---- Type conversion ----
    template <typename U>
    Array<U> astype() const {
        Array<U> out(shape_);
        for (size_t i=0;i<size_;++i) out[i]= static_cast<U>(data_[i]);
        return out;
    }

    // ---- Reductions (global) ----
    T sum() const {
        if (size_==0) throw std::logic_error("sum of empty array");
        T acc = T{};
        for (auto v: data_) acc += v;
        return acc;
    }
    T min() const {
        if (size_==0) throw std::logic_error("min of empty array");
        T m = data_[0];
        for (size_t i=1;i<size_;++i) if (data_[i] < m) m=data_[i];
        return m;
    }
    T max() const {
        if (size_==0) throw std::logic_error("max of empty array");
        T m = data_[0];
        for (size_t i=1;i<size_;++i) if (data_[i] > m) m=data_[i];
        return m;
    }
    double mean() const {
        if (size_==0) throw std::logic_error("mean of empty array");
        if constexpr (std::is_same_v<T,bool>) {
            size_t cnt=0; for(auto v: data_) if(v) ++cnt; return (double)cnt / (double)size_;
        } else {
            double acc=0;
            for (auto v: data_) acc += (double)v;
            return acc / (double)size_;
        }
    }

    // ---- Reductions along axis (returns Array with that axis removed) ----
    Array sum(size_t axis) const { return reduce_axis(axis, [](T a,T b){return a+b;}, T{}); }
    Array min_axis(size_t axis) const {
        if (size_==0) throw std::logic_error("min_axis empty");
        return reduce_axis_minmax(axis, true);
    }
    Array max_axis(size_t axis) const {
        if (size_==0) throw std::logic_error("max_axis empty");
        return reduce_axis_minmax(axis, false);
    }
    // mean along axis returns double array
    Array<double> mean_axis(size_t axis) const {
        auto s = sum(axis);
        auto out = s.template astype<double>();
        double div = (double)shape_[axis];
        for (size_t i=0;i<out.size();++i) out[i]/=div;
        return out;
    }

    // ---- Initializers ----
    static Array zeros(std::vector<size_t> shape) {
        Array a(std::move(shape));
        return a;
    }
    static Array ones(std::vector<size_t> shape) {
        Array a(std::move(shape));
        std::fill(a.data_.begin(), a.data_.end(), T(1));
        return a;
    }

private:
    std::vector<T> data_;
    size_t size_ = 0;
    size_t ndim_ = 0;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    DType dtype_;
    Layout layout_ = Layout::RowMajor;

    void compute_strides() {
        strides_.resize(ndim_);
        if (ndim_==0) return;
        if (layout_==Layout::RowMajor) {
            strides_[ndim_-1]=1;
            for (int i=(int)ndim_-2;i>=0;--i) strides_[i]=strides_[i+1]*shape_[i+1];
        } else {
            strides_[0]=1;
            for (size_t i=1;i<ndim_;++i) strides_[i]=strides_[i-1]*shape_[i-1];
        }
    }

    static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
        size_t na = a.size(), nb = b.size();
        size_t n = std::max(na, nb);
        std::vector<size_t> res(n);
        for (int i=(int)n-1, ia=(int)na-1, ib=(int)nb-1; i>=0; --i, --ia, --ib) {
            size_t da = ia >=0 ? a[ia] : 1;
            size_t db = ib >=0 ? b[ib] : 1;
            if (da != db && da != 1 && db != 1)
                throw std::invalid_argument("broadcast: incompatible shapes");
            res[i] = std::max(da, db);
        }
        return res;
    }

    // compute flat offset for operand given result multi-index
    static size_t broadcast_offset(const std::vector<size_t>& shape, const std::vector<size_t>& strides,
                                   const std::vector<size_t>& res_idx, size_t res_ndim) {
        size_t orig_ndim = shape.size();
        size_t offset = 0;
        // align to trailing dimensions
        for (size_t i=0;i<res_ndim;++i) {
            int orig_d = (int)i - (int)(res_ndim - orig_ndim);
            if (orig_d < 0) continue; // leading 1
            if (shape[orig_d] == 1) continue; // broadcasted dim
            offset += res_idx[i] * strides[orig_d];
        }
        return offset;
    }

    template <typename F>
    Array elementwise(const Array& o, F fn) const {
        if (shape_ == o.shape_) {
            Array res(shape_);
            for (size_t i=0;i<size_;++i) res.data_[i] = fn(data_[i], o.data_[i]);
            return res;
        }
        auto bshape = broadcast_shape(shape_, o.shape_);
        Array res(bshape);
        size_t bndim = bshape.size();
        // precompute strides already in res
        std::vector<size_t> idx(bndim);
        for (size_t flat=0; flat<res.size_; ++flat) {
            // unravel flat to idx
            size_t rem=flat;
            for (int d=(int)bndim-1; d>=0; --d) { idx[d]=rem % bshape[d]; rem/=bshape[d]; }
            size_t off_a = broadcast_offset(shape_, strides_, idx, bndim);
            size_t off_b = broadcast_offset(o.shape_, o.strides_, idx, bndim);
            res.data_[flat] = fn(data_[off_a], o.data_[off_b]);
        }
        return res;
    }
    template <typename F>
    Array scalar_op(T s, F fn) const {
        Array res(shape_);
        for (size_t i=0;i<size_;++i) res.data_[i] = fn(data_[i], s);
        return res;
    }

    Array reduce_axis(size_t axis, T(*op)(T,T), T init) const {
        if (axis >= ndim_) throw std::out_of_range("reduce axis OOB");
        std::vector<size_t> out_shape;
        for (size_t d=0; d<ndim_; ++d) if (d!=axis) out_shape.push_back(shape_[d]);
        if (out_shape.empty()) out_shape={1}; // keep at least 1D for scalar? alternative: return 1-element
        // But for true scalar reduction we want size 1; caller sum(axis) on 1-D -> scalar array
        Array out(out_shape);
        std::fill(out.data_.begin(), out.data_.end(), init);
        // For non-sum, init handling is special; this path only for sum
        // Iterate over all elements and map to out index
        for (size_t flat=0; flat<size_; ++flat) {
            // unravel flat to multi-index
            size_t rem=flat;
            std::vector<size_t> idx(ndim_);
            for (int d=(int)ndim_-1; d>=0; --d) { idx[d]=rem % shape_[d]; rem/=shape_[d]; }
            // compute out flat
            size_t out_flat=0; size_t mult=1;
            for (int d=(int)ndim_-1; d>=0; --d) {
                if ((size_t)d==axis) continue;
                // find position in out_shape
                size_t out_d = d > (int)axis ? d-1 : d;
                // need strides of out
                // compute via out strides
                out_flat += idx[d] * out.strides()[out_d];
            }
            out.data()[out_flat] = op(out.data()[out_flat], data_[flat]);
        }
        return out;
    }

    Array reduce_axis_minmax(size_t axis, bool is_min) const {
        if (axis >= ndim_) throw std::out_of_range("reduce axis OOB");
        std::vector<size_t> out_shape;
        for (size_t d=0; d<ndim_; ++d) if (d!=axis) out_shape.push_back(shape_[d]);
        if (out_shape.empty()) out_shape={1};
        Array out(out_shape);
        // init with first slice
        bool first=true;
        for (size_t flat=0; flat<size_; ++flat) {
            size_t rem=flat;
            std::vector<size_t> idx(ndim_);
            for (int d=(int)ndim_-1; d>=0; --d) { idx[d]=rem % shape_[d]; rem/=shape_[d]; }
            size_t out_flat=0;
            for (int d=(int)ndim_-1, out_d=(int)out.ndim()-1; d>=0; --d) {
                if ((size_t)d==axis) continue;
                out_flat += idx[d] * out.strides()[out_d];
                --out_d;
            }
            if (idx[axis]==0) out.data()[out_flat]=data_[flat];
            else {
                if (is_min) out.data()[out_flat]= std::min(out.data()[out_flat], data_[flat]);
                else out.data()[out_flat]= std::max(out.data()[out_flat], data_[flat]);
            }
        }
        return out;
    }
};

} // namespace dracolix
