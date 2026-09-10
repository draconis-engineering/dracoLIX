#pragma once
#include "dtype.hpp"
#include <vector>
#include <stdexcept>
#include <optional>
#include <algorithm>
#include <cstdint>

namespace dracolix {

// Non-owning view. Strides are in elements (not bytes). Does NOT own memory.
template <typename T>
class ArrayView {
public:
    ArrayView() = default;
    ArrayView(T* data, std::vector<size_t> shape, std::vector<size_t> strides)
        : data_(data), shape_(std::move(shape)), strides_(std::move(strides)) {
        if (shape_.size() != strides_.size())
            throw std::invalid_argument("ArrayView: shape/strides size mismatch");
        ndim_ = shape_.size();
        size_ = 1;
        for (auto s : shape_) size_ *= s;
    }

    // 1-D index via strided calculation (works for contiguous and strided)
    T& operator[](size_t flat) {
        if (flat >= size_) throw std::out_of_range("ArrayView flat index OOB");
        // For flat access we need to map flat -> multidimensional using shape/strides?
        // Simpler: if contiguous (strides are row-major), flat is direct offset.
        // For strided views (slice), flat iteration is still logical, not physical.
        // We implement logical flat via unravel.
        auto idx = unravel(flat);
        return at_indices(idx);
    }
    const T& operator[](size_t flat) const { return const_cast<ArrayView*>(this)->operator[](flat); }

    T& at(size_t i, size_t j) {
        if (ndim_ != 2) throw std::logic_error("ArrayView::at(i,j) requires ndim==2");
        if (i >= shape_[0] || j >= shape_[1]) throw std::out_of_range("at OOB");
        return data_[i * strides_[0] + j * strides_[1]];
    }
    const T& at(size_t i, size_t j) const { return const_cast<ArrayView*>(this)->at(i,j); }

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
    const T& at(const std::vector<size_t>& indices) const { return const_cast<ArrayView*>(this)->at(indices); }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }
    size_t ndim() const noexcept { return ndim_; }
    const std::vector<size_t>& shape() const noexcept { return shape_; }
    const std::vector<size_t>& strides() const noexcept { return strides_; }

    bool is_contiguous() const noexcept {
        // Check if strides match row-major contiguous
        if (ndim_==0) return true;
        size_t expect = 1;
        for (int i=(int)ndim_-1; i>=0; --i) {
            if (strides_[i] != expect) return false;
            expect *= shape_[i];
        }
        return true;
    }

    // Materialize to owning Array (copy)
    // Declaration only - defined after Array is complete (in array.hpp)
    // Array<T> to_array() const;

private:
    T* data_ = nullptr;
    size_t size_ = 0;
    size_t ndim_ = 0;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    std::vector<size_t> unravel(size_t flat) const {
        std::vector<size_t> idx(ndim_);
        for (int d=(int)ndim_-1; d>=0; --d) {
            idx[d] = flat % shape_[d];
            flat /= shape_[d];
        }
        return idx;
    }
    T& at_indices(const std::vector<size_t>& idx) {
        size_t off=0;
        for (size_t d=0; d<ndim_; ++d) off += idx[d]*strides_[d];
        return data_[off];
    }
};

// Slice descriptor similar to Python slice(start, stop, step)
struct Slice {
    std::optional<int64_t> start;
    std::optional<int64_t> stop;
    std::optional<int64_t> step; // default 1

    Slice() = default;
    static Slice all() { return Slice{}; }
    static Slice range(int64_t start, int64_t stop, int64_t step=1) {
        Slice s; s.start=start; s.stop=stop; s.step=step; return s;
    }
    static Slice from(int64_t start) { Slice s; s.start=start; return s; }
    static Slice to(int64_t stop) { Slice s; s.stop=stop; return s; }
};

// Normalize a Slice to [offset, length, step] for dim size
inline void normalize_slice(const Slice& s, size_t dim, size_t& out_offset, size_t& out_len, int64_t& out_step) {
    int64_t step = s.step.value_or(1);
    if (step==0) throw std::invalid_argument("slice step cannot be 0");
    int64_t start, stop;
    if (step > 0) {
        start = s.start.value_or(0);
        stop = s.stop.value_or((int64_t)dim);
        if (start < 0) start += (int64_t)dim;
        if (stop < 0) stop += (int64_t)dim;
        start = std::clamp(start, (int64_t)0, (int64_t)dim);
        stop = std::clamp(stop, (int64_t)0, (int64_t)dim);
        if (stop < start) stop = start;
        out_offset = (size_t)start;
        out_len = (size_t)((stop - start + step -1)/ step);
    } else {
        // negative step
        start = s.start.value_or((int64_t)dim -1);
        stop = s.stop.value_or(-1);
        if (start < 0) start += (int64_t)dim;
        if (stop < -1) stop += (int64_t)dim;
        // clamp
        if (start >= (int64_t)dim) start = (int64_t)dim -1;
        if (start < -1) start = -1;
        if (stop >= (int64_t)dim) stop = (int64_t)dim -1;
        // compute len
        if (start <= stop) out_len=0;
        else out_len = (size_t)((start - stop -1)/ (-step) +1);
        out_offset = (start >=0 ? (size_t)start : 0); // if len==0 offset irrelevant
        if (out_len==0) out_offset=0;
    }
    out_step = step;
}

} // namespace dracolix
