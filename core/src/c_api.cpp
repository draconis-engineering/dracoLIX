#include "dracolix/c_api.h"
#include "dracolix/dracolix.hpp"
#include <vector>
#include <cstring>

using DracoArrayT = dracolix::Array<double>;
struct DracoArray { DracoArrayT impl; DracoArray(DracoArrayT&& v) : impl(std::move(v)) {} };

extern "C" {

DracoArray* dracolix_array_f64_zeros(const size_t* shape, size_t ndim){
    try{
        std::vector<size_t> s(shape, shape+ndim);
        auto* p = new DracoArray(DracoArrayT(s));
        std::fill(p->impl.data(), p->impl.data()+p->impl.size(), 0.0);
        return p;
    }catch(...){ return nullptr; }
}
DracoArray* dracolix_array_f64_ones(const size_t* shape, size_t ndim){
    try{
        std::vector<size_t> s(shape, shape+ndim);
        DracoArrayT a(s); std::fill(a.data(), a.data()+a.size(), 1.0);
        return new DracoArray(std::move(a));
    }catch(...){ return nullptr; }
}
DracoArray* dracolix_array_f64_from_data(const double* data, const size_t* shape, size_t ndim){
    try{
        std::vector<size_t> s(shape, shape+ndim);
        DracoArrayT a(s);
        std::memcpy(a.data(), data, a.size()*sizeof(double));
        return new DracoArray(std::move(a));
    }catch(...){ return nullptr; }
}
void dracolix_array_destroy(DracoArray* arr){ delete arr; }
size_t dracolix_array_ndim(const DracoArray* arr){ return arr ? arr->impl.ndim() : 0; }
const size_t* dracolix_array_shape(const DracoArray* arr){ return arr ? arr->impl.shape().data() : nullptr; }
size_t dracolix_array_size(const DracoArray* arr){ return arr ? arr->impl.size() : 0; }
double* dracolix_array_f64_data(DracoArray* arr){ return arr ? arr->impl.data() : nullptr; }
const double* dracolix_array_f64_data_const(const DracoArray* arr){ return arr ? arr->impl.data() : nullptr; }
int dracolix_array_f64_fill(DracoArray* arr, double v){
    if(!arr) return -1;
    std::fill(arr->impl.data(), arr->impl.data()+arr->impl.size(), v);
    return 0;
}
DracoArray* dracolix_matmul_f64(const DracoArray* a, const DracoArray* b){
    if(!a||!b) return nullptr;
    try{
        auto c = dracolix::linalg::matmul(a->impl, b->impl);
        return new DracoArray(std::move(c));
    }catch(...){ return nullptr; }
}
DracoArray* dracolix_matvec_f64(const DracoArray* a, const DracoArray* x){
    if(!a||!x) return nullptr;
    try{
        auto y = dracolix::linalg::matvec(a->impl, x->impl);
        return new DracoArray(std::move(y));
    }catch(...){ return nullptr; }
}
const char* dracolix_version(){ return dracolix::version; }
}
