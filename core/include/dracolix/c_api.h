#pragma once
// C API for Julia (and other C callers) — simple opaque handle over Array<double>
// Phase 5 minimal interop. Licensed under GPL-3.0-only
#ifdef __cplusplus
extern "C" {
#endif
#include <stddef.h>

typedef struct DracoArray DracoArray;

// lifecycle
DracoArray* dracolix_array_f64_zeros(const size_t* shape, size_t ndim);
DracoArray* dracolix_array_f64_ones(const size_t* shape, size_t ndim);
DracoArray* dracolix_array_f64_from_data(const double* data, const size_t* shape, size_t ndim);
void dracolix_array_destroy(DracoArray* arr);

// inspection
size_t dracolix_array_ndim(const DracoArray* arr);
const size_t* dracolix_array_shape(const DracoArray* arr);
size_t dracolix_array_size(const DracoArray* arr);
double* dracolix_array_f64_data(DracoArray* arr);
const double* dracolix_array_f64_data_const(const DracoArray* arr);

// ops (f64 only for Phase 5 prototype; other dtypes via same pattern)
DracoArray* dracolix_matmul_f64(const DracoArray* a, const DracoArray* b);
DracoArray* dracolix_matvec_f64(const DracoArray* a, const DracoArray* x);
int dracolix_array_f64_fill(DracoArray* arr, double value);

// version
const char* dracolix_version();

#ifdef __cplusplus
}
#endif
