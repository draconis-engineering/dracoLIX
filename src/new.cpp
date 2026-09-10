#include <iostream>
#include <stdexcept>

enum DType {
    I4, I8, I16, I32, I64, I128, I256,
    F8, F16, F32, F64, F128, F256, F512,
    C16, C32, C64, C128, C256, C512, C1024,
    Bool, Char, Str,
    Void
};

// Helper trait to automatically map C++ types to your DType enum
template <typename T> struct DTypeMap { static const DType value = Void; };
template <> struct DTypeMap<int> { static const DType value = I32; };
template <> struct DTypeMap<float> { static const DType value = F32; };
template <> struct DTypeMap<double> { static const DType value = F64; };
template <> struct DTypeMap<bool> { static const DType value = Bool; };
template <> struct DTypeMap<char> { static const DType value = Char; };

template <typename T>
class Array {
private:
    T* data;
    int size;
    int* shape;
    int ndim;
    int* strides;
    DType dtype;

protected:
    void computeStrides() {
        if (ndim > 0 && strides != nullptr) {
            strides[ndim - 1] = sizeof(T); // Simple 1D stride for this constructor context
        }
    }

public:
    // Constructor (Defaults to 1D array based on arguments)
    Array(int size) : size(size), ndim(1) {
        data = new T[size]();
        shape = new int[1]{size};
        strides = new int[1];
        dtype = DTypeMap<T>::value;
        computeStrides();
    }

    // Destructor (Safely cleans up all three dynamic arrays)
    ~Array() {
        delete[] data;
        delete[] shape;
        delete[] strides;
    }

    // Copy Constructor (Rule of Three)
    Array(const Array& other) : size(other.size), ndim(other.ndim), dtype(other.dtype) {
        data = new T[size];
        for (int i = 0; i < size; ++i) data[i] = other.data[i];

        shape = new int[ndim];
        for (int i = 0; i < ndim; ++i) shape[i] = other.shape[i];

        strides = new int[ndim];
        for (int i = 0; i < ndim; ++i) strides[i] = other.strides[i];
    }

    // Copy Assignment Operator (Rule of Three)
    Array& operator=(const Array& other) {
        if (this == &other) return *this;

        delete[] data;
        delete[] shape;
        delete[] strides;

        size = other.size;
        ndim = other.ndim;
        dtype = other.dtype;

        data = new T[size];
        for (int i = 0; i < size; ++i) data[i] = other.data[i];

        shape = new int[ndim];
        for (int i = 0; i < ndim; ++i) shape[i] = other.shape[i];

        strides = new int[ndim];
        for (int i = 0; i < ndim; ++i) strides[i] = other.strides[i];

        return *this;
    }

    // Subscript Operators with bounds checking
    T& operator[](int index) {
        if (index < 0 || index >= size) throw std::out_of_range("Index out of bounds");
        return data[index];
    }

    const T& operator[](int index) const {
        if (index < 0 || index >= size) throw std::out_of_range("Index out of bounds");
        return data[index];
    }

    // Getters & Setters
    T get(int index) const { return (*this)[index]; }
    void set(int index, T value) { (*this)[index] = value; }
    int getSize() const { return size; }
    int getElements() const { return size; }
    int* getShape() const { return shape; }

    void print() const {
        for (int i = 0; i < size; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << "\n";
    }

    // Resizing logic
    void reshape(int newSize) {
        if (newSize != size) {
            throw std::runtime_error("Reshape total elements must match current size.");
        }
        shape[0] = newSize;
    }

    void setSize(int newSize) {
        T* newData = new T[newSize]();
        int limit = (newSize < size) ? newSize : size;
        for (int i = 0; i < limit; ++i) newData[i] = data[i];

        delete[] data;
        data = newData;
        size = newSize;
        shape[0] = newSize;
    }
};
