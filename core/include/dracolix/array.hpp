#pragma once
#include "dtype.hpp"
#include <vector>
#include <stdexcept>
#include <numeric>
#include <memory>
#include <cstring>

namespace dracolix {

// Outgoing-only core type: no Python/Julia headers here.
// Owns contiguous row-major memory. Views are non-owning later.
template <typename T>
class Array {
public:
    // 1-D contiguous array
    explicit Array(size_t size)
        : size_(size), ndim_(1), shape_{size}, dtype_(DTypeOf<T>::value) {
        if (dtype_ == DType::Void) throw std::logic_error("Array<T>: unmapped DType");
        data_.resize(size_);
        compute_strides();
    }

    // N-D contiguous array from shape vector
    explicit Array(std::vector<size_t> shape)
        : shape_(std::move(shape)), dtype_(DTypeOf<T>::value) {
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

    // Rule of 0 - vector handles memory; no manual new/delete
    Array(const Array&) = default;
    Array(Array&&) noexcept = default;
    Array& operator=(const Array&) = default;
    Array& operator=(Array&&) noexcept = default;
    ~Array() = default;

    // Element access with bounds check
    T& operator[](size_t idx) {
        if (idx >= size_) throw std::out_of_range("Array index out of bounds");
        return data_[idx];
    }
    const T& operator[](size_t idx) const {
        if (idx >= size_) throw std::out_of_range("Array index out of bounds");
        return data_[idx];
    }

    // 2-D accessor for row-major layout: (i,j) -> i*shape[1]+j
    T& at(size_t i, size_t j) {
        if (ndim_ != 2) throw std::logic_error("at(i,j) requires ndim==2");
        if (i >= shape_[0] || j >= shape_[1]) throw std::out_of_range("at(i,j) out of bounds");
        return data_[i * shape_[1] + j];
    }
    const T& at(size_t i, size_t j) const {
        if (ndim_ != 2) throw std::logic_error("at(i,j) requires ndim==2");
        if (i >= shape_[0] || j >= shape_[1]) throw std::out_of_range("at(i,j) out of bounds");
        return data_[i * shape_[1] + j];
    }

    // Element-wise ops - return new array (no silent aliasing)
    Array operator+(const Array& o) const { return elementwise(o, [](T a, T b){ return a+b; }); }
    Array operator-(const Array& o) const { return elementwise(o, [](T a, T b){ return a-b; }); }
    Array operator*(const Array& o) const { return elementwise(o, [](T a, T b){ return a*b; }); }
    Array operator/(const Array& o) const { return elementwise(o, [](T a, T b){ return a/b; }); }

    Array operator+(T scalar) const { return scalar_op(scalar, [](T a, T s){ return a+s; }); }
    Array operator-(T scalar) const { return scalar_op(scalar, [](T a, T s){ return a-s; }); }
    Array operator*(T scalar) const { return scalar_op(scalar, [](T a, T s){ return a*s; }); }
    Array operator/(T scalar) const { return scalar_op(scalar, [](T a, T s){ return a/s; }); }

    // Raw access for kernels / bindings (outgoing-only: caller must respect shape/strides)
    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }
    size_t size() const noexcept { return size_; }
    size_t ndim() const noexcept { return ndim_; }
    const std::vector<size_t>& shape() const noexcept { return shape_; }
    const std::vector<size_t>& strides() const noexcept { return strides_; } // in elements
    DType dtype() const noexcept { return dtype_; }

    // Reshape without realloc if total size unchanged; otherwise realloc and preserve prefix
    void reshape(std::vector<size_t> new_shape) {
        size_t new_size = 1;
        for (auto s : new_shape) new_size *= s;
        if (new_size != size_) {
            // preserve prefix, zero-init new tail (explicit, not silent)
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

private:
    std::vector<T> data_;
    size_t size_ = 0;
    size_t ndim_ = 0;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_; // in elements, row-major
    DType dtype_;

    void compute_strides() {
        strides_.resize(ndim_);
        if (ndim_ == 0) return;
        strides_[ndim_-1] = 1;
        for (int i = (int)ndim_-2; i >=0; --i) {
            strides_[i] = strides_[i+1] * shape_[i+1];
        }
    }

    template <typename F>
    Array elementwise(const Array& o, F fn) const {
        if (shape_ != o.shape_) throw std::invalid_argument("elementwise: shape mismatch");
        Array res(shape_);
        for (size_t i=0;i<size_;++i) res.data_[i] = fn(data_[i], o.data_[i]);
        return res;
    }
    template <typename F>
    Array scalar_op(T s, F fn) const {
        Array res(shape_);
        for (size_t i=0;i<size_;++i) res.data_[i] = fn(data_[i], s);
        return res;
    }
};

} // namespace dracolix
