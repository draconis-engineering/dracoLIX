// DracoLIX Python bindings (nanobind).
//
// Zero-NumPy runtime: interop is done through the PEP 3118 buffer protocol.
// We export buffers from dracolix Array (bf_getbuffer) and consume buffers
// from numpy/memoryview (PyObject_GetBuffer).
//
// Licensed under GPL-3.0-only - see LICENSE

#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "dracolix/dracolix.hpp"

#include <Python.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace nb = nanobind;

namespace {

// ---------------------------------------------------------------------------
// Detect the "kind" of a Python scalar (bool / int / float)
// ---------------------------------------------------------------------------
enum class NumKind { None, Bool, Int, Float };

NumKind num_kind(nb::handle h) noexcept {
  if (PyBool_Check(h.ptr()))
    return NumKind::Bool;
  if (PyLong_Check(h.ptr()))
    return NumKind::Int;
  if (PyFloat_Check(h.ptr()))
    return NumKind::Float;
  return NumKind::None;
}

bool is_number(nb::handle h) noexcept { return num_kind(h) != NumKind::None; }

// ---------------------------------------------------------------------------
// The Python-facing array: a type-erased container over the core Array<T>
// ---------------------------------------------------------------------------
struct DlxArray {
  using Variant = std::variant<dracolix::Array<float>, dracolix::Array<double>,
                               dracolix::Array<int32_t>,
                               dracolix::Array<int64_t>, dracolix::Array<bool>>;
  Variant v;

  DlxArray() = delete;
  explicit DlxArray(Variant value) : v(std::move(value)) {}

  template <typename T>
  explicit DlxArray(dracolix::Array<T> a) : v(std::move(a)) {}

  template <typename T> dracolix::Array<T> *as() {
    return std::get_if<dracolix::Array<T>>(&v);
  }
  template <typename T> const dracolix::Array<T> *as() const {
    return std::get_if<dracolix::Array<T>>(&v);
  }

  dracolix::DType dtype() const {
    return std::visit([](const auto &a) { return a.dtype(); }, v);
  }
  size_t size() const {
    return std::visit([](const auto &a) { return a.size(); }, v);
  }
  size_t ndim() const {
    return std::visit([](const auto &a) { return a.ndim(); }, v);
  }
  const std::vector<size_t> &shape() const {
    return std::visit(
        [](const auto &a) -> const std::vector<size_t> & { return a.shape(); },
        v);
  }
  const std::vector<size_t> &strides() const {
    return std::visit(
        [](const auto &a) -> const std::vector<size_t> & {
          return a.strides();
        },
        v);
  }
  dracolix::Layout layout() const {
    return std::visit([](const auto &a) { return a.layout(); }, v);
  }
  size_t itemsize() const { return dracolix::dtype_size(dtype()); }
  void *data_ptr() {
    return std::visit([](auto &a) -> void * { return a.data(); }, v);
  }
  const void *data_ptr() const {
    return std::visit([](const auto &a) -> const void * { return a.data(); },
                      v);
  }
};

// Extract the element type from dracolix::Array<T> (the core has no value_type;
// std::visit gives us the Array type, not the scalar).
template <typename ArrayT> struct array_scalar;
template <typename T> struct array_scalar<dracolix::Array<T>> {
  using type = T;
};

// Registered Python type objects (filled during NB_MODULE).
PyTypeObject *g_dlx_type = nullptr;
PyTypeObject *g_dtype_type = nullptr;

bool is_array(nb::handle h) noexcept {
  return g_dlx_type && PyObject_TypeCheck(h.ptr(), g_dlx_type);
}
bool is_dtype(nb::handle h) noexcept {
  return g_dtype_type && PyObject_TypeCheck(h.ptr(), g_dtype_type);
}

DlxArray &as_array(nb::handle h) {
  if (!is_array(h))
    throw nb::type_error("expected a dracolix Array");
  return *nb::cast<DlxArray *>(h);
}

// ---------------------------------------------------------------------------
// DType wrapper + singleton registry
// ---------------------------------------------------------------------------
struct DlxDType {
  dracolix::DType dt = dracolix::DType::Void;
  constexpr explicit DlxDType(dracolix::DType d) : dt(d) {}
};

nb::object dtype_singleton(dracolix::DType dt) {
  // The singletons are Python-owned (stored as attributes of this module).
  // Look them up on demand instead of caching them in C++ static storage:
  // a static nb::object would outlive the interpreter and never be released,
  // producing nanobind "leaked instances" warnings at interpreter shutdown.
  auto mod = nb::module_::import_("dracolix._dracolix_nb");
  switch (dt) {
  case dracolix::DType::F32:
    return mod.attr("f32");
  case dracolix::DType::F64:
    return mod.attr("f64");
  case dracolix::DType::I32:
    return mod.attr("i32");
  case dracolix::DType::I64:
    return mod.attr("i64");
  case dracolix::DType::Bool:
    return mod.attr("bool");
  default:
    throw nb::type_error("unsupported dtype");
  }
}

std::optional<dracolix::DType> optional_dtype(nb::handle h) {
  if (h.is_none())
    return std::nullopt;
  if (!is_dtype(h))
    throw nb::type_error("dtype must be one of dlx.f32, dlx.f64, dlx.i32, "
                         "dlx.i64, dlx.bool_");
  return nb::cast<DlxDType *>(h)->dt;
}

// ---------------------------------------------------------------------------
// Scalar conversion
// ---------------------------------------------------------------------------
nb::object py_scalar_at(const DlxArray &a, int64_t offset) {
  return std::visit(
      [offset](const auto &arr) { return nb::cast(arr[(size_t)offset]); }, a.v);
}

DlxArray scalar_slot(nb::handle h) {
  switch (num_kind(h)) {
  case NumKind::Bool: {
    dracolix::Array<int32_t> a(std::vector<size_t>{1});
    a[0] = nb::cast<bool>(h) ? 1 : 0;
    return DlxArray(std::move(a));
  }
  case NumKind::Int: {
    int64_t v = nb::cast<int64_t>(h);
    dracolix::Array<int64_t> a(std::vector<size_t>{1});
    a[0] = v;
    return DlxArray(std::move(a));
  }
  case NumKind::Float: {
    double v = nb::cast<double>(h);
    dracolix::Array<double> a(std::vector<size_t>{1});
    a[0] = v;
    return DlxArray(std::move(a));
  }
  default:
    throw nb::type_error("expected a scalar");
  }
}

// ---------------------------------------------------------------------------
// dtype promotion (rules documented in dracolix.pyi)
// ---------------------------------------------------------------------------
int64_t rank_of(dracolix::DType dt) {
  switch (dt) {
  case dracolix::DType::Bool:
    return 0;
  case dracolix::DType::I32:
    return 1;
  case dracolix::DType::I64:
    return 2;
  case dracolix::DType::F32:
    return 3;
  case dracolix::DType::F64:
    return 4;
  default:
    throw nb::type_error("unsupported dtype");
  }
}

dracolix::DType promote_pair(dracolix::DType a, dracolix::DType b) {
  bool a_f = rank_of(a) >= rank_of(dracolix::DType::F32);
  bool b_f = rank_of(b) >= rank_of(dracolix::DType::F32);
  if (a_f || b_f) {
    if (a == dracolix::DType::F64 || b == dracolix::DType::F64)
      return dracolix::DType::F64;
    if (a == dracolix::DType::I64 || b == dracolix::DType::I64)
      return dracolix::DType::F64; // avoid int64 -> f32 precision loss
    return dracolix::DType::F32;
  }
  if (a == dracolix::DType::I64 || b == dracolix::DType::I64)
    return dracolix::DType::I64;
  return dracolix::DType::I32;
}

// ---------------------------------------------------------------------------
// dtype casting (core Array<T>::astype)
// ---------------------------------------------------------------------------
template <typename T> DlxArray cast_cpp(const DlxArray &a) {
  return std::visit(
      [](const auto &arr) { return DlxArray(arr.template astype<T>()); }, a.v);
}

DlxArray cast_dtype(const DlxArray &a, dracolix::DType dt) {
  switch (dt) {
  case dracolix::DType::F32:
    return cast_cpp<float>(a);
  case dracolix::DType::F64:
    return cast_cpp<double>(a);
  case dracolix::DType::I32:
    return cast_cpp<int32_t>(a);
  case dracolix::DType::I64:
    return cast_cpp<int64_t>(a);
  case dracolix::DType::Bool:
    return cast_cpp<bool>(a);
  default:
    throw nb::type_error("unsupported dtype");
  }
}

// ---------------------------------------------------------------------------
// Broadcast helpers (same semantics as core Array<T>::elementwise)
// ---------------------------------------------------------------------------
std::vector<size_t> bcast_shape(const std::vector<size_t> &a,
                                const std::vector<size_t> &b) {
  size_t na = a.size(), nb = b.size();
  size_t n = std::max(na, nb);
  std::vector<size_t> res(n);
  for (int i = (int)n - 1, ia = (int)na - 1, ib = (int)nb - 1; i >= 0;
       --i, --ia, --ib) {
    size_t da = ia >= 0 ? a[ia] : 1;
    size_t db = ib >= 0 ? b[ib] : 1;
    if (da != db && da != 1 && db != 1)
      throw std::invalid_argument("broadcast: incompatible shapes");
    res[i] = std::max(da, db);
  }
  return res;
}

size_t bcast_offset(const std::vector<size_t> &shape,
                    const std::vector<size_t> &strides,
                    const std::vector<size_t> &ridx, size_t rndim) {
  size_t orig = shape.size();
  size_t off = 0;
  for (size_t i = 0; i < rndim; ++i) {
    int orig_d = (int)i - (int)(rndim - orig);
    if (orig_d < 0)
      continue;
    if (shape[orig_d] == 1)
      continue;
    off += ridx[i] * strides[orig_d];
  }
  return off;
}

template <typename R>
dracolix::Array<bool> compare_bcast(const dracolix::Array<R> &a,
                                    const dracolix::Array<R> &b, bool eq) {
  if (a.shape() == b.shape()) {
    dracolix::Array<bool> out(a.shape());
    for (size_t i = 0; i < a.size(); ++i)
      out[i] = eq ? (a[i] == b[i]) : (a[i] != b[i]);
    return out;
  }
  auto bs = bcast_shape(a.shape(), b.shape());
  dracolix::Array<bool> out(bs);
  size_t bnd = bs.size();
  std::vector<size_t> idx(bnd);
  for (size_t flat = 0; flat < out.size(); ++flat) {
    size_t rem = flat;
    for (int d = (int)bnd - 1; d >= 0; --d) {
      idx[d] = rem % bs[d];
      rem /= bs[d];
    }
    size_t off_a = bcast_offset(a.shape(), a.strides(), idx, bnd);
    size_t off_b = bcast_offset(b.shape(), b.strides(), idx, bnd);
    out[flat] = eq ? (a[off_a] == b[off_b]) : (a[off_a] != b[off_b]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Binary operations (promotion + broadcasting; no bool arithmetic)
// ---------------------------------------------------------------------------
enum class BinOp { Add, Sub, Mul, Div, Eq, Ne };

template <typename R>
DlxArray bin_typed(const DlxArray &A, const DlxArray &B, BinOp op) {
  const dracolix::Array<R> &a = *A.as<R>();
  const dracolix::Array<R> &b = *B.as<R>();
  switch (op) {
  case BinOp::Add:
    return DlxArray(a + b);
  case BinOp::Sub:
    return DlxArray(a - b);
  case BinOp::Mul:
    return DlxArray(a * b);
  case BinOp::Div:
    return DlxArray(a / b);
  case BinOp::Eq:
    return DlxArray(compare_bcast<R>(a, b, true));
  case BinOp::Ne:
    return DlxArray(compare_bcast<R>(a, b, false));
  }
  throw std::logic_error("unreachable");
}

DlxArray bin_switch(dracolix::DType r, const DlxArray &A, const DlxArray &B,
                    BinOp op) {
  switch (r) {
  case dracolix::DType::F32:
    return bin_typed<float>(A, B, op);
  case dracolix::DType::F64:
    return bin_typed<double>(A, B, op);
  case dracolix::DType::I32:
    return bin_typed<int32_t>(A, B, op);
  case dracolix::DType::I64:
    return bin_typed<int64_t>(A, B, op);
  default:
    throw nb::type_error("unsupported dtype");
  }
}

bool coerce_operand(nb::handle h, std::optional<DlxArray> &out) {
  if (is_array(h)) {
    out = *nb::cast<DlxArray *>(h);
    return true;
  }
  if (is_number(h)) {
    out = scalar_slot(h);
    return true;
  }
  return false;
}

nb::object dlx_bin_op(nb::object x, nb::object y, BinOp op) {
  std::optional<DlxArray> A, B;
  if (!coerce_operand(x, A) || !coerce_operand(y, B))
    return nb::not_implemented();
  assert(A && B);

  dracolix::DType ra = A->dtype(), rb = B->dtype();
  dracolix::DType r = promote_pair(ra, rb);
  if (r == dracolix::DType::Bool)
    return nb::not_implemented(); // unreachable: promotions never stay bool
  if (op == BinOp::Div) {
    // __truediv__: NumPy-like semantics (int/int -> f64, f32/f32 -> f32)
    r = (ra == dracolix::DType::F32 && rb == dracolix::DType::F32)
            ? dracolix::DType::F32
            : dracolix::DType::F64;
  }
  DlxArray Ca = cast_dtype(*A, r);
  DlxArray Cb = cast_dtype(*B, r);
  return nb::cast<DlxArray>(bin_switch(r, Ca, Cb, op));
}

nb::object dlx_add(nb::object x, nb::object y) {
  return dlx_bin_op(x, y, BinOp::Add);
}
nb::object dlx_sub(nb::object x, nb::object y) {
  return dlx_bin_op(x, y, BinOp::Sub);
}
nb::object dlx_mul(nb::object x, nb::object y) {
  return dlx_bin_op(x, y, BinOp::Mul);
}
nb::object dlx_div(nb::object x, nb::object y) {
  return dlx_bin_op(x, y, BinOp::Div);
}
nb::object dlx_eq(nb::object x, nb::object y) {
  return dlx_bin_op(x, y, BinOp::Eq);
}
nb::object dlx_ne(nb::object x, nb::object y) {
  return dlx_bin_op(x, y, BinOp::Ne);
}

nb::object dlx_neg(nb::object x) {
  if (!is_array(x))
    return nb::not_implemented();
  DlxArray &a = as_array(x);
  return std::visit(
      [](auto &arr) -> nb::object {
        using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
        if constexpr (std::is_same_v<T, bool>) {
          auto v = arr.template astype<int32_t>();
          return nb::cast<DlxArray>(DlxArray(v * int32_t(-1)));
        } else {
          return nb::cast<DlxArray>(DlxArray(arr * T(-1)));
        }
      },
      a.v);
}

// ---------------------------------------------------------------------------
// matmul (core dracolix::linalg::matmul; 1-D @ 1-D -> scalar dot)
// ---------------------------------------------------------------------------
nb::object dlx_matmul(nb::object x, nb::object y) {
  std::optional<DlxArray> A, B;
  if (!coerce_operand(x, A) || !coerce_operand(y, B))
    return nb::not_implemented();
  assert(A && B);
  dracolix::DType dt = A->dtype();
  if (A->dtype() != B->dtype())
    throw nb::type_error("matmul requires matching dtypes");
  if (dt == dracolix::DType::Bool)
    throw nb::type_error("matmul does not support bool arrays");

  if (A->ndim() == 1 && B->ndim() == 1)
    return std::visit(
        [&](const auto &a) -> nb::object {
          using T = typename array_scalar<std::decay_t<decltype(a)>>::type;
          const auto &b = *B->as<T>();
          return nb::cast(dracolix::linalg::dot(a, b));
        },
        A->v);

  return std::visit(
      [&](const auto &a) -> nb::object {
        using T = typename array_scalar<std::decay_t<decltype(a)>>::type;
        const auto &b = *B->as<T>();
        return nb::cast<DlxArray>(DlxArray(dracolix::linalg::matmul(a, b)));
      },
      A->v);
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------
struct KeySpec {
  bool is_int = false;
  int64_t idx = 0;
  dracolix::Slice slice;
};

// Parse a Python index (int, slice, or tuple of int/slice) into KeySpecs.
// PySlice_Unpack marks an omitted component with PY_SSIZE_T_MAX (start)
// or PY_SSIZE_T_MIN (stop); step is left as given.
void parse_slice_h(nb::handle obj, KeySpec &ks) {
  Py_ssize_t start, stop, step;
  if (PySlice_Unpack(obj.ptr(), &start, &stop, &step) != 0)
    throw nb::python_error();
  if (step == 0)
    throw nb::value_error("slice step cannot be 0");
  ks.is_int = false;
  ks.slice.step = step;
  if (start != PY_SSIZE_T_MAX)
    ks.slice.start = (int64_t)start;
  if (stop != PY_SSIZE_T_MIN)
    ks.slice.stop = (int64_t)stop;
}

std::vector<KeySpec> parse_key(nb::object key, size_t ndim) {
  std::vector<KeySpec> out;
  auto parse_one = [&](nb::handle h, KeySpec &ks) {
    if (PySlice_Check(h.ptr())) {
      parse_slice_h(h, ks);
    } else if (PyLong_Check(h.ptr())) {
      ks.is_int = true;
      ks.idx = nb::cast<int64_t>(h);
    } else {
      throw nb::type_error("indices must be integers or slices");
    }
  };

  if (PyTuple_Check(key.ptr())) {
    Py_ssize_t n = PyTuple_GET_SIZE(key.ptr());
    for (Py_ssize_t i = 0; i < n; ++i) {
      KeySpec ks;
      parse_one(PyTuple_GET_ITEM(key.ptr(), i), ks);
      out.push_back(ks);
    }
  } else {
    KeySpec ks;
    parse_one(key, ks);
    out.push_back(ks);
  }

  if (out.size() > ndim)
    throw nb::index_error("too many indices");
  while (out.size() < ndim)
    out.push_back(KeySpec{}); // slice(None)
  return out;
}

nb::object dlx_scalar_index(const DlxArray &a,
                            const std::vector<int64_t> &idx) {
  const std::vector<size_t> &shape = a.shape();
  const std::vector<size_t> &strides = a.strides();
  size_t off = 0;
  for (size_t d = 0; d < shape.size(); ++d) {
    int64_t i = idx[d];
    if (i < 0)
      i += (int64_t)shape[d];
    if (i < 0 || (size_t)i >= shape[d])
      throw nb::index_error("index out of bounds");
    off += (size_t)i * strides[d];
  }
  return py_scalar_at(a, (int64_t)off);
}

// Materialize a (possibly mixed int/slice) index into a new contiguous array.
template <typename T>
dracolix::Array<T> slice_materialize(const dracolix::Array<T> &arr,
                                     const std::vector<KeySpec> &keys) {
  size_t nd = arr.ndim();
  std::vector<int64_t> fixed_off(nd, 0);
  std::vector<int64_t> starts(nd, 0), steps(nd, 1);
  std::vector<int64_t> lens(nd, 0);
  std::vector<size_t> out_shape;

  for (size_t d = 0; d < nd; ++d) {
    const KeySpec &ks = keys[d];
    if (ks.is_int) {
      int64_t i = ks.idx;
      if (i < 0)
        i += (int64_t)arr.shape()[d];
      if (i < 0 || (size_t)i >= arr.shape()[d])
        throw nb::index_error("index out of bounds");
      fixed_off[d] = i * (int64_t)arr.strides()[d];
      lens[d] = 0;
      continue;
    }
    size_t off0, len;
    int64_t step;
    dracolix::normalize_slice(ks.slice, arr.shape()[d], off0, len, step);
    if (len == 0)
      throw nb::value_error("zero-length slices are not supported yet");
    starts[d] = (int64_t)off0;
    steps[d] = step;
    lens[d] = (int64_t)len;
    out_shape.push_back(len);
  }

  dracolix::Array<T> out(out_shape);
  std::vector<size_t> idx(out_shape.size());
  for (size_t flat = 0; flat < out.size(); ++flat) {
    size_t rem = flat;
    for (int dd = (int)out_shape.size() - 1; dd >= 0; --dd) {
      idx[dd] = rem % out_shape[dd];
      rem /= out_shape[dd];
    }
    int64_t src = 0;
    size_t od = 0;
    for (size_t d = 0; d < nd; ++d) {
      if (lens[d] == 0) {
        src += fixed_off[d];
      } else {
        int64_t src_d = starts[d] + (int64_t)idx[od] * steps[d];
        src += src_d * (int64_t)arr.strides()[d];
        ++od;
      }
    }
    out[flat] = arr[(size_t)src];
  }
  return out;
}

nb::object dlx_getitem(const DlxArray &self, nb::object key) {
  std::vector<KeySpec> keys = parse_key(key, self.ndim());

  bool all_int = true;
  for (auto &k : keys)
    if (!k.is_int) {
      all_int = false;
      break;
    }
  if (all_int) {
    std::vector<int64_t> idx;
    idx.reserve(keys.size());
    for (auto &k : keys)
      idx.push_back(k.idx);
    return dlx_scalar_index(self, idx);
  }

  return std::visit(
      [&](const auto &arr) -> nb::object {
        DlxArray out(slice_materialize(arr, keys));
        return nb::cast<DlxArray>(std::move(out));
      },
      self.v);
}

nb::object dlx_setitem(DlxArray &self, nb::object key, nb::object val) {
  std::vector<KeySpec> keys = parse_key(key, self.ndim());
  bool all_int = true;
  for (auto &ks : keys)
    if (!ks.is_int) {
      all_int = false;
      break;
    }
  if (!all_int)
    throw nb::type_error("slice assignment is not supported yet");
  if (keys.size() != self.ndim())
    throw nb::index_error("index rank mismatch");
  if (!is_number(val))
    throw nb::type_error("assignment value must be a scalar");

  size_t off = 0;
  for (size_t d = 0; d < self.ndim(); ++d) {
    int64_t i = keys[d].idx;
    if (i < 0)
      i += (int64_t)self.shape()[d];
    if (i < 0 || (size_t)i >= self.shape()[d])
      throw nb::index_error("index out of bounds");
    off += (size_t)i * self.strides()[d];
  }
  std::visit(
      [&](auto &arr) {
        using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
        T v;
        if constexpr (std::is_same_v<T, bool>) {
          v = nb::cast<bool>(val);
        } else if constexpr (std::is_integral_v<T>) {
          v = (T)nb::cast<int64_t>(val);
        } else {
          v = (T)nb::cast<double>(val);
        }
        arr[(size_t)off] = v;
      },
      self.v);
  return nb::none();
}

// ---------------------------------------------------------------------------
// Buffer protocol export (PEP 3118).
// nanobind does not ship def_buffer(); we wire bf_getbuffer / bf_releasebuffer
// on the registered type ourselves.
// ---------------------------------------------------------------------------
const char *buffer_format(dracolix::DType dt) {
  switch (dt) {
  case dracolix::DType::F32:
    return "f";
  case dracolix::DType::F64:
    return "d";
  case dracolix::DType::I32:
    return "i";
  case dracolix::DType::I64:
    return "q";
  case dracolix::DType::Bool:
    return "?";
  default:
    return "";
  }
}

int dlx_getbuffer(PyObject *obj, Py_buffer *view, int flags) {
  if (view == nullptr) {
    PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
    return -1;
  }
  if (!g_dlx_type || !PyObject_TypeCheck(obj, g_dlx_type)) {
    PyErr_SetString(PyExc_TypeError, "not a dracolix Array");
    return -1;
  }
  DlxArray *self = nb::cast<DlxArray *>(nb::handle(obj));
  const std::vector<size_t> &shape = self->shape();
  const std::vector<size_t> &strides = self->strides();
  size_t ndim = self->ndim();

  bool is_rowmajor = true;
  bool is_colmajor = true;
  if (ndim > 0) {
    {
      size_t expect = 1;
      for (int i = (int)ndim - 1; i >= 0; --i) {
        if (strides[i] != expect) {
          is_rowmajor = false;
          break;
        }
        expect *= shape[i];
      }
    }
    {
      size_t expect = 1;
      for (size_t i = 0; i < ndim; ++i) {
        if (strides[i] != expect) {
          is_colmajor = false;
          break;
        }
        expect *= shape[i];
      }
    }
  }

  // Contiguity checks must compare the full flag mask: PyBUF_C/F_CONTIGUOUS
  // each include the ND+STRIDES bits, which plain requests always carry.
  if ((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS && !is_rowmajor) {
    PyErr_Format(PyExc_BufferError, "dracolix array is not C-contiguous");
    return -1;
  }
  if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS && !is_colmajor) {
    PyErr_Format(PyExc_BufferError, "dracolix array is not Fortran-contiguous");
    return -1;
  }
  // PyBUF_INDIRECT relaxes the format requirements but never suboffsets;
  // we are suboffset-capable exporter (we simply never produce suboffsets).

  bool need_shape = (flags & PyBUF_ND) != 0;
  bool need_strides = (flags & PyBUF_STRIDES) != 0;

  Py_ssize_t itemsize = (Py_ssize_t)self->itemsize();

  Py_ssize_t *c_shape = nullptr;
  Py_ssize_t *c_strides = nullptr;
  if (need_shape) {
    c_shape = (Py_ssize_t *)PyMem_Malloc(sizeof(Py_ssize_t) *
                                         std::max<size_t>(ndim, 1));
    if (!c_shape) {
      PyErr_NoMemory();
      return -1;
    }
    for (size_t d = 0; d < ndim; ++d)
      c_shape[d] = (Py_ssize_t)shape[d];
  }
  if (need_strides) {
    c_strides = (Py_ssize_t *)PyMem_Malloc(sizeof(Py_ssize_t) *
                                           std::max<size_t>(ndim, 1));
    if (!c_strides) {
      PyMem_Free(c_shape);
      PyErr_NoMemory();
      return -1;
    }
    // PEP 3118 specifies byte strides, not element strides.
    for (size_t d = 0; d < ndim; ++d)
      c_strides[d] = (Py_ssize_t)strides[d] * itemsize;
  }

  // Simple requests with no ND/STRIDES and a 1-D array: hand out a flat
  // contiguous view (shape NULL is fine for simple consumers).
  const char *fmt = buffer_format(self->dtype());
  char *fmt_copy = (char *)PyMem_Malloc(strlen(fmt) + 1);
  if (!fmt_copy) {
    PyMem_Free(c_shape);
    PyMem_Free(c_strides);
    PyErr_NoMemory();
    return -1;
  }
  std::strcpy(fmt_copy, fmt);

  if (!need_shape && !need_strides && ndim == 1) {
    c_shape = (Py_ssize_t *)PyMem_Malloc(sizeof(Py_ssize_t));
    if (!c_shape) {
      PyErr_NoMemory();
      return -1;
    }
    c_shape[0] = (Py_ssize_t)shape[0];
    view->ndim = 1;
    view->shape = c_shape;
    view->strides = nullptr;
    view->suboffsets = nullptr;
    view->internal = nullptr;
    view->obj = obj;
    Py_INCREF(obj);
    view->buf = const_cast<void *>(self->data_ptr());
    view->len = (Py_ssize_t)self->size() * itemsize;
    view->readonly = 0;
    view->itemsize = itemsize;
    view->format = fmt_copy;
    return 0;
  }

  view->obj = obj;
  Py_INCREF(obj);
  view->buf = const_cast<void *>(self->data_ptr());
  view->len = (Py_ssize_t)self->size() * itemsize;
  view->readonly = 0;
  view->format = fmt_copy;
  view->itemsize = itemsize;
  view->ndim = need_shape ? (Py_ssize_t)ndim : 0;
  view->shape = c_shape;
  view->strides = c_strides;
  view->suboffsets = nullptr;
  view->internal = nullptr;
  return 0;
}

void dlx_releasebuffer(PyObject *, Py_buffer *view) {
  if (view->shape)
    PyMem_Free(view->shape);
  if (view->strides)
    PyMem_Free(view->strides);
  if (view->format)
    PyMem_Free(view->format);
  view->shape = nullptr;
  view->strides = nullptr;
  view->format = nullptr;
}

void install_buffer_slots() {
  if (!g_dlx_type)
    return;
  PyBufferProcs *bp = g_dlx_type->tp_as_buffer;
  if (!bp)
    return;
  bp->bf_getbuffer = &dlx_getbuffer;
  bp->bf_releasebuffer = &dlx_releasebuffer;
}

// ---------------------------------------------------------------------------
// Buffer input (numpy / memoryview -> DlxArray)
// ---------------------------------------------------------------------------
dracolix::DType infer_format_dtype(const char *fmt, Py_ssize_t itemsize) {
  if (fmt == nullptr || *fmt == 0) {
    if (itemsize == 1)
      return dracolix::DType::I32;
    return dracolix::DType::Void;
  }
  switch (*fmt) {
  case 'f':
    return dracolix::DType::F32;
  case 'd':
    return dracolix::DType::F64;
  case 'q':
  case 'Q':
  case 'L':
  case 'N':
    return dracolix::DType::I64;
  case 'i':
  case 'l':
    return itemsize >= 8 ? dracolix::DType::I64 : dracolix::DType::I32;
  case 'b':
  case 'h':
  case 'B':
  case 'H':
  case 'I':
  case 'n':
    return dracolix::DType::I32;
  case 'e':
    return dracolix::DType::F32;
  case '?':
    return dracolix::DType::Bool;
  default:
    return dracolix::DType::Void;
  }
}

double read_double_at(const char *p, char tag, Py_ssize_t itemsize) {
  switch (tag) {
  case 'f':
    return (double)(*(const float *)p);
  case 'd':
    return *(const double *)p;
  case 'e': {
    uint16_t h;
    std::memcpy(&h, p, 2);
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    double d;
    if (exp == 0)
      d = mant ? std::ldexp((double)mant, -24) : 0.0;
    else if (exp == 31)
      d = mant ? std::numeric_limits<double>::quiet_NaN()
               : std::numeric_limits<double>::infinity();
    else
      d = std::ldexp((double)(mant + 1024), (int)exp - 25);
    return sign ? -d : d;
  }
  case '?':
    return *(const bool *)p ? 1.0 : 0.0;
  default: {
    if (itemsize == 1)
      return (double)(*(const int8_t *)p);
    if (itemsize == 2)
      return (double)(*(const int16_t *)p);
    if (itemsize == 4)
      return (double)(*(const int32_t *)p);
    return (double)(*(const int64_t *)p);
  }
  }
}

int64_t read_int_at(const char *p, char tag, Py_ssize_t itemsize) {
  bool is_unsigned = (tag == 'B' || tag == 'H' || tag == 'I' || tag == 'L' ||
                      tag == 'N' || tag == 'Q');
  bool is_bool = tag == '?';
  if (is_bool)
    return *(const bool *)p ? 1 : 0;
  if (itemsize == 1)
    return is_unsigned ? (int64_t)(*(const uint8_t *)p)
                       : (int64_t)(*(const int8_t *)p);
  if (itemsize == 2)
    return is_unsigned ? (int64_t)(*(const uint16_t *)p)
                       : (int64_t)(*(const int16_t *)p);
  if (itemsize == 4)
    return is_unsigned ? (int64_t)(*(const uint32_t *)p)
                       : (int64_t)(*(const int32_t *)p);
  return is_unsigned ? (int64_t)(*(const uint64_t *)p) : (*(const int64_t *)p);
}

template <typename T> T from_raw(const char *p, char tag, Py_ssize_t itemsize);

template <>
float from_raw<float>(const char *p, char tag, Py_ssize_t itemsize) {
  if (tag == 'd' || tag == 'e' || tag == '?' || tag == 'f')
    return (float)read_double_at(p, tag, itemsize);
  return (float)read_int_at(p, tag, itemsize);
}
template <>
double from_raw<double>(const char *p, char tag, Py_ssize_t itemsize) {
  return read_double_at(p, tag, itemsize);
}
template <>
int32_t from_raw<int32_t>(const char *p, char tag, Py_ssize_t itemsize) {
  return (int32_t)read_int_at(p, tag, itemsize);
}
template <>
int64_t from_raw<int64_t>(const char *p, char tag, Py_ssize_t itemsize) {
  return read_int_at(p, tag, itemsize);
}
template <> bool from_raw<bool>(const char *p, char tag, Py_ssize_t itemsize) {
  if (tag == '?')
    return *(const bool *)p;
  return read_double_at(p, tag, itemsize) != 0.0;
}

template <typename T>
DlxArray buffer_to_array(const Py_buffer &view, char tag) {
  if (view.ndim == 0)
    throw nb::value_error("0-dimensional buffers are not supported yet");
  std::vector<size_t> shape((size_t)view.ndim);
  Py_ssize_t n = 1;
  for (Py_ssize_t d = 0; d < view.ndim; ++d) {
    shape[(size_t)d] = (size_t)view.shape[d];
    n *= view.shape[d];
  }
  for (auto s : shape)
    if (s == 0)
      throw nb::value_error("empty buffers are not supported yet");

  dracolix::Array<T> out(shape);
  if (n != (Py_ssize_t)out.size())
    throw std::logic_error("buffer size mismatch");

  Py_ssize_t itemsize = view.itemsize;
  const Py_ssize_t *st = view.strides;
  std::vector<Py_ssize_t> fallback_strides;
  if (st == nullptr) {
    fallback_strides.resize((size_t)view.ndim);
    for (Py_ssize_t d = view.ndim - 1, acc = 1; d >= 0; --d) {
      fallback_strides[(size_t)d] = acc;
      acc *= view.shape[d];
    }
    st = fallback_strides.data();
  }

  for (Py_ssize_t flat = 0; flat < n; ++flat) {
    size_t rem = (size_t)flat;
    size_t off = 0;
    for (Py_ssize_t d = view.ndim - 1; d >= 0; --d) {
      size_t idx = rem % (size_t)view.shape[d];
      rem /= (size_t)view.shape[d];
      off += idx * (size_t)st[d];
    }
    const char *p = (const char *)view.buf + off * itemsize;
    out[(size_t)flat] = from_raw<T>(p, tag, itemsize);
  }
  return DlxArray(std::move(out));
}

DlxArray from_buffer(nb::object o, std::optional<dracolix::DType> dt) {
  Py_buffer view;
  if (PyObject_GetBuffer(o.ptr(), &view, PyBUF_CONTIG_RO | PyBUF_FORMAT) != 0)
    throw nb::python_error();
  struct Guard {
    Py_buffer &v;
    ~Guard() { PyBuffer_Release(&v); }
  } guard{view};

  dracolix::DType target =
      dt.value_or(infer_format_dtype(view.format, view.itemsize));
  if (target == dracolix::DType::Void)
    throw nb::type_error("unsupported buffer format");

  char tag = (view.format && *view.format) ? *view.format : 'B';
  switch (target) {
  case dracolix::DType::F32:
    return buffer_to_array<float>(view, tag);
  case dracolix::DType::F64:
    return buffer_to_array<double>(view, tag);
  case dracolix::DType::I32:
    return buffer_to_array<int32_t>(view, tag);
  case dracolix::DType::I64:
    return buffer_to_array<int64_t>(view, tag);
  case dracolix::DType::Bool:
    return buffer_to_array<bool>(view, tag);
  default:
    throw nb::type_error("unsupported dtype");
  }
}

// ---------------------------------------------------------------------------
// list / tuple input
// ---------------------------------------------------------------------------
void walk_sequence(nb::object o, size_t depth, std::vector<size_t> &shape,
                   std::vector<nb::object> &leaves, bool &any_float,
                   bool &any_int, bool &any_bool) {
  if (!PyList_Check(o.ptr()) && !PyTuple_Check(o.ptr()))
    throw nb::type_error("expected list or tuple of numbers");

  Py_ssize_t len = PySequence_Size(o.ptr());
  if (depth >= shape.size()) {
    shape.push_back((size_t)len);
  } else if ((size_t)len != shape[depth]) {
    throw nb::value_error("ragged nested sequence");
  }
  for (Py_ssize_t i = 0; i < len; ++i) {
    nb::object item = nb::steal(PySequence_GetItem(o.ptr(), i));
    if (PyList_Check(item.ptr()) || PyTuple_Check(item.ptr())) {
      walk_sequence(item, depth + 1, shape, leaves, any_float, any_int,
                    any_bool);
    } else if (is_number(item)) {
      leaves.push_back(item);
      switch (num_kind(item)) {
      case NumKind::Float:
        any_float = true;
        break;
      case NumKind::Int:
        any_int = true;
        break;
      default:
        any_bool = true;
        break;
      }
    } else {
      throw nb::type_error("dracolix arrays only support bool, int and "
                           "float elements");
    }
  }
}

dracolix::DType infer_list_dtype(bool any_float, bool any_int, bool any_bool) {
  if (any_float)
    return dracolix::DType::F64;
  if (any_int)
    return dracolix::DType::I64;
  return dracolix::DType::Bool;
}

template <typename T>
void fill_leaf(dracolix::Array<T> &out, size_t i, nb::object leaf);

template <>
void fill_leaf<bool>(dracolix::Array<bool> &out, size_t i, nb::object leaf) {
  out[i] = nb::cast<bool>(leaf);
}
template <>
void fill_leaf<int32_t>(dracolix::Array<int32_t> &out, size_t i,
                        nb::object leaf) {
  if (num_kind(leaf) == NumKind::Float) {
    double d = nb::cast<double>(leaf);
    if (d != std::trunc(d))
      throw nb::type_error(
          "cannot convert non-integral float to integer dtype");
    out[i] = (int32_t)d;
  } else {
    out[i] = (int32_t)nb::cast<int64_t>(leaf);
  }
}
template <>
void fill_leaf<int64_t>(dracolix::Array<int64_t> &out, size_t i,
                        nb::object leaf) {
  if (num_kind(leaf) == NumKind::Float) {
    double d = nb::cast<double>(leaf);
    if (d != std::trunc(d))
      throw nb::type_error(
          "cannot convert non-integral float to integer dtype");
    out[i] = (int64_t)d;
  } else {
    out[i] = nb::cast<int64_t>(leaf);
  }
}
template <>
void fill_leaf<float>(dracolix::Array<float> &out, size_t i, nb::object leaf) {
  out[i] = (float)nb::cast<double>(leaf);
}
template <>
void fill_leaf<double>(dracolix::Array<double> &out, size_t i,
                       nb::object leaf) {
  out[i] = nb::cast<double>(leaf);
}

DlxArray from_list(nb::object o, std::optional<dracolix::DType> dt) {
  std::vector<size_t> shape;
  std::vector<nb::object> leaves;
  bool any_float = false, any_int = false, any_bool = false;
  walk_sequence(o, 0, shape, leaves, any_float, any_int, any_bool);
  dracolix::DType target =
      dt.value_or(infer_list_dtype(any_float, any_int, any_bool));
  if (shape.empty() || leaves.empty())
    throw nb::value_error("empty arrays are not supported yet");

  switch (target) {
  case dracolix::DType::F32: {
    dracolix::Array<float> out(shape);
    for (size_t i = 0; i < leaves.size(); ++i)
      fill_leaf<float>(out, i, leaves[i]);
    return DlxArray(std::move(out));
  }
  case dracolix::DType::F64: {
    dracolix::Array<double> out(shape);
    for (size_t i = 0; i < leaves.size(); ++i)
      fill_leaf<double>(out, i, leaves[i]);
    return DlxArray(std::move(out));
  }
  case dracolix::DType::I32: {
    dracolix::Array<int32_t> out(shape);
    for (size_t i = 0; i < leaves.size(); ++i)
      fill_leaf<int32_t>(out, i, leaves[i]);
    return DlxArray(std::move(out));
  }
  case dracolix::DType::I64: {
    dracolix::Array<int64_t> out(shape);
    for (size_t i = 0; i < leaves.size(); ++i)
      fill_leaf<int64_t>(out, i, leaves[i]);
    return DlxArray(std::move(out));
  }
  case dracolix::DType::Bool: {
    dracolix::Array<bool> out(shape);
    for (size_t i = 0; i < leaves.size(); ++i)
      fill_leaf<bool>(out, i, leaves[i]);
    return DlxArray(std::move(out));
  }
  default:
    throw nb::type_error("unsupported dtype");
  }
}

// ---------------------------------------------------------------------------
// dlx.array / dlx.asarray
// ---------------------------------------------------------------------------
DlxArray dlx_array(nb::object data, nb::object dtype) {
  std::optional<dracolix::DType> dt = optional_dtype(dtype);
  if (is_array(data))
    return dt ? cast_dtype(*nb::cast<DlxArray *>(data), *dt)
              : DlxArray(*nb::cast<DlxArray *>(data));
  if (PyObject_CheckBuffer(data.ptr()))
    return from_buffer(data, dt);
  if (PyList_Check(data.ptr()) || PyTuple_Check(data.ptr()))
    return from_list(data, dt);
  throw nb::type_error("expected a sequence, buffer, or dracolix Array");
}

nb::object dlx_asarray(nb::object data, nb::object dtype) {
  std::optional<dracolix::DType> dt = optional_dtype(dtype);
  if (is_array(data)) {
    if (dt && *dt != nb::cast<DlxArray *>(data)->dtype())
      return nb::cast<DlxArray>(cast_dtype(*nb::cast<DlxArray *>(data), *dt));
    return data; // no copy
  }
  return nb::cast<DlxArray>(dlx_array(data, dtype));
}

// ---------------------------------------------------------------------------
// shape parsing for zeros / ones / full / empty
// ---------------------------------------------------------------------------
std::vector<size_t> parse_shape(nb::object o) {
  std::vector<size_t> shape;
  if (PyLong_Check(o.ptr())) {
    int64_t n = nb::cast<int64_t>(o);
    if (n <= 0)
      throw nb::value_error("dimensions must be positive");
    shape.push_back((size_t)n);
    return shape;
  }
  if (PyList_Check(o.ptr()) || PyTuple_Check(o.ptr())) {
    Py_ssize_t len = PySequence_Size(o.ptr());
    for (Py_ssize_t i = 0; i < len; ++i) {
      nb::object item = nb::steal(PySequence_GetItem(o.ptr(), i));
      int64_t n = nb::cast<int64_t>(item);
      if (n <= 0)
        throw nb::value_error("dimensions must be positive");
      shape.push_back((size_t)n);
    }
    return shape;
  }
  throw nb::type_error("shape must be an int or a sequence of ints");
}

template <typename T> T scalar_to_dtype(nb::handle h);
template <> float scalar_to_dtype<float>(nb::handle h) {
  return (float)nb::cast<double>(h);
}
template <> double scalar_to_dtype<double>(nb::handle h) {
  return nb::cast<double>(h);
}
template <> int32_t scalar_to_dtype<int32_t>(nb::handle h) {
  double d = nb::cast<double>(h);
  if (d != std::trunc(d))
    throw nb::type_error("cannot convert non-integral float to i32");
  return (int32_t)nb::cast<int64_t>(h);
}
template <> int64_t scalar_to_dtype<int64_t>(nb::handle h) {
  double d = nb::cast<double>(h);
  if (d != std::trunc(d))
    throw nb::type_error("cannot convert non-integral float to i64");
  return nb::cast<int64_t>(h);
}
template <> bool scalar_to_dtype<bool>(nb::handle h) {
  return nb::cast<bool>(h);
}

template <typename T>
DlxArray make_filled(const std::vector<size_t> &shape, T v) {
  dracolix::Array<T> a(shape);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = v;
  return DlxArray(std::move(a));
}

DlxArray dlx_zeros(nb::object shape, nb::object dtype) {
  std::vector<size_t> s = parse_shape(shape);
  std::optional<dracolix::DType> dt = optional_dtype(dtype);
  dracolix::DType t = dt.value_or(dracolix::DType::F64);
  switch (t) {
  case dracolix::DType::F32:
    return DlxArray(dracolix::Array<float>::zeros(s));
  case dracolix::DType::F64:
    return DlxArray(dracolix::Array<double>::zeros(s));
  case dracolix::DType::I32:
    return DlxArray(dracolix::Array<int32_t>::zeros(s));
  case dracolix::DType::I64:
    return DlxArray(dracolix::Array<int64_t>::zeros(s));
  case dracolix::DType::Bool:
    return DlxArray(dracolix::Array<bool>::zeros(s));
  default:
    throw nb::type_error("unsupported dtype");
  }
}

DlxArray dlx_ones(nb::object shape, nb::object dtype) {
  std::vector<size_t> s = parse_shape(shape);
  std::optional<dracolix::DType> dt = optional_dtype(dtype);
  dracolix::DType t = dt.value_or(dracolix::DType::F64);
  switch (t) {
  case dracolix::DType::F32:
    return DlxArray(dracolix::Array<float>::ones(s));
  case dracolix::DType::F64:
    return DlxArray(dracolix::Array<double>::ones(s));
  case dracolix::DType::I32:
    return DlxArray(dracolix::Array<int32_t>::ones(s));
  case dracolix::DType::I64:
    return DlxArray(dracolix::Array<int64_t>::ones(s));
  case dracolix::DType::Bool:
    return DlxArray(dracolix::Array<bool>::ones(s));
  default:
    throw nb::type_error("unsupported dtype");
  }
}

DlxArray dlx_empty(nb::object shape, nb::object dtype) {
  // The core always zero-initializes storage; empty() == zeros() for now.
  return dlx_zeros(shape, dtype);
}

DlxArray dlx_full(nb::object shape, nb::object value, nb::object dtype) {
  std::vector<size_t> s = parse_shape(shape);
  if (!is_number(value))
    throw nb::type_error("full(): fill value must be a scalar");
  std::optional<dracolix::DType> dt = optional_dtype(dtype);
  dracolix::DType t =
      dt.value_or(num_kind(value) == NumKind::Float  ? dracolix::DType::F64
                  : num_kind(value) == NumKind::Bool ? dracolix::DType::Bool
                                                     : dracolix::DType::I64);
  switch (t) {
  case dracolix::DType::F32:
    return make_filled<float>(s, scalar_to_dtype<float>(value));
  case dracolix::DType::F64:
    return make_filled<double>(s, scalar_to_dtype<double>(value));
  case dracolix::DType::I32:
    return make_filled<int32_t>(s, scalar_to_dtype<int32_t>(value));
  case dracolix::DType::I64:
    return make_filled<int64_t>(s, scalar_to_dtype<int64_t>(value));
  case dracolix::DType::Bool:
    return make_filled<bool>(s, scalar_to_dtype<bool>(value));
  default:
    throw nb::type_error("unsupported dtype");
  }
}

DlxArray dlx_arange(nb::object start, nb::object stop, nb::object step) {
  bool has_stop = !stop.is_none();
  nb::object a0 = has_stop ? start : nb::object(nb::int_(0));
  nb::object a1 = has_stop ? stop : start;
  nb::object a2 = step.is_none() ? nb::object(nb::int_(1)) : step;

  if (!is_number(a0) || !is_number(a1) || !is_number(a2))
    throw nb::type_error("arange arguments must be numbers");
  if (num_kind(a2) == NumKind::Int && nb::cast<int64_t>(a2) == 0)
    throw nb::value_error("arange step cannot be 0");

  bool any_float = false;
  for (nb::handle h : {nb::handle(a0), nb::handle(a1), nb::handle(a2)})
    if (num_kind(h) == NumKind::Float)
      any_float = true;

  if (!any_float) {
    int64_t s0 = nb::cast<int64_t>(a0), s1 = nb::cast<int64_t>(a1),
            s2 = nb::cast<int64_t>(a2);
    int64_t span = s1 - s0;
    int64_t n64 = 0;
    if (s2 > 0 && span > 0)
      n64 = (span + s2 - 1) / s2;
    if (s2 < 0 && span < 0)
      n64 = (span + s2 + 1) / s2;
    if (n64 < 0)
      n64 = 0;
    dracolix::Array<int64_t> a(std::vector<size_t>{(size_t)n64});
    for (int64_t i = 0; i < n64; ++i)
      a[(size_t)i] = s0 + i * s2;
    return DlxArray(std::move(a));
  }
  double d0 = nb::cast<double>(a0), d1 = nb::cast<double>(a1),
         d2 = nb::cast<double>(a2);
  double span = d1 - d0;
  int64_t n64 = 0;
  if (d2 != 0.0) {
    if (d2 > 0 && span > 0)
      n64 = (int64_t)std::ceil(span / d2);
    if (d2 < 0 && span < 0)
      n64 = (int64_t)std::ceil(span / d2);
  }
  if (n64 < 0)
    n64 = 0;
  dracolix::Array<double> a(std::vector<size_t>{(size_t)n64});
  for (int64_t i = 0; i < n64; ++i)
    a[(size_t)i] = d0 + (double)i * d2;
  return DlxArray(std::move(a));
}

// ---------------------------------------------------------------------------
// Iteration
// ---------------------------------------------------------------------------
struct FlatIter {
  const DlxArray *arr = nullptr;
  int64_t i = 0;
  FlatIter &operator++() {
    ++i;
    return *this;
  }
  FlatIter operator++(int) {
    FlatIter c = *this;
    ++i;
    return c;
  }
  bool operator==(const FlatIter &o) const { return i == o.i && arr == o.arr; }
  bool operator!=(const FlatIter &o) const { return !(*this == o); }
  nb::object operator*() const { return py_scalar_at(*arr, i); }

  using iterator_category = std::forward_iterator_tag;
  using value_type = nb::object;
  using difference_type = std::ptrdiff_t;
  using pointer = nb::object *;
  using reference = nb::object;
};

nb::object make_iter(const DlxArray &self, nb::handle scope) {
  return nb::make_iterator<nb::rv_policy::copy>(
      scope, "dracolix_array_iterator", FlatIter{&self, 0},
      FlatIter{&self, (int64_t)self.size()});
}

// ---------------------------------------------------------------------------
// reductions
// ---------------------------------------------------------------------------
nb::object dlx_sum(const DlxArray &a) {
  return std::visit(
      [](const auto &arr) -> nb::object {
        using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
        if constexpr (std::is_same_v<T, bool>) {
          size_t cnt = 0;
          for (size_t i = 0; i < arr.size(); ++i)
            if (arr[i])
              ++cnt;
          return nb::cast((int64_t)cnt);
        } else {
          return nb::cast(arr.sum());
        }
      },
      a.v);
}

nb::object dlx_min(const DlxArray &a) {
  return std::visit(
      [](const auto &arr) -> nb::object {
        using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
        if constexpr (std::is_same_v<T, bool>) {
          bool any = false;
          for (size_t i = 0; i < arr.size(); ++i)
            if (arr[i]) {
              any = true;
              break;
            }
          return nb::cast(any);
        } else {
          return nb::cast(arr.min());
        }
      },
      a.v);
}

nb::object dlx_max(const DlxArray &a) {
  return std::visit(
      [](const auto &arr) -> nb::object {
        using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
        if constexpr (std::is_same_v<T, bool>) {
          bool any = false;
          for (size_t i = 0; i < arr.size(); ++i)
            if (arr[i]) {
              any = true;
              break;
            }
          return nb::cast(any);
        } else {
          return nb::cast(arr.max());
        }
      },
      a.v);
}

nb::object dlx_mean(const DlxArray &a) {
  return std::visit(
      [](const auto &arr) -> nb::object {
        using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
        if constexpr (std::is_same_v<T, bool>) {
          size_t cnt = 0;
          for (size_t i = 0; i < arr.size(); ++i)
            if (arr[i])
              ++cnt;
          return nb::cast((double)cnt / (double)arr.size());
        } else {
          return nb::cast(arr.mean());
        }
      },
      a.v);
}

int64_t normalize_axis(const DlxArray &a, int64_t axis) {
  if (axis < 0)
    axis += (int64_t)a.ndim();
  if (axis < 0 || (size_t)axis >= a.ndim())
    throw nb::index_error("axis out of bounds");
  return axis;
}

DlxArray dlx_sum_axis(const DlxArray &a, int64_t axis) {
  axis = normalize_axis(a, axis);
  return std::visit(
      [&](const auto &arr) -> DlxArray {
        using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
        if constexpr (std::is_same_v<T, bool>) {
          // logical OR along axis (documented in dracolix.pyi)
          dracolix::Array<bool> r = arr.sum((size_t)axis);
          return DlxArray(std::move(r));
        } else {
          return DlxArray(arr.sum((size_t)axis));
        }
      },
      a.v);
}

DlxArray dlx_min_axis(const DlxArray &a, int64_t axis) {
  axis = normalize_axis(a, axis);
  return std::visit(
      [&](const auto &arr) -> DlxArray {
        return DlxArray(arr.min_axis((size_t)axis));
      },
      a.v);
}

DlxArray dlx_max_axis(const DlxArray &a, int64_t axis) {
  axis = normalize_axis(a, axis);
  return std::visit(
      [&](const auto &arr) -> DlxArray {
        return DlxArray(arr.max_axis((size_t)axis));
      },
      a.v);
}

DlxArray dlx_mean_axis(const DlxArray &a, int64_t axis) {
  axis = normalize_axis(a, axis);
  return std::visit(
      [&](const auto &arr) -> DlxArray {
        return DlxArray(arr.mean_axis((size_t)axis));
      },
      a.v);
}

// ---------------------------------------------------------------------------
// tolist / repr
// ---------------------------------------------------------------------------
template <typename T>
nb::object build_list(const dracolix::Array<T> &a, size_t d, size_t offset,
                      const std::vector<size_t> &rst) {
  nb::list l;
  size_t n = a.shape()[d];
  if (d + 1 == a.ndim()) {
    for (size_t i = 0; i < n; ++i)
      l.append(nb::cast(a[offset + i * rst[d]]));
    return l;
  }
  for (size_t i = 0; i < n; ++i)
    l.append(build_list(a, d + 1, offset + i * rst[d], rst));
  return l;
}

nb::object dlx_tolist(const DlxArray &a) {
  return std::visit(
      [&](const auto &arr) -> nb::object {
        using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
        std::vector<size_t> rst(a.ndim());
        size_t acc = 1;
        for (int d = (int)a.ndim() - 1; d >= 0; --d) {
          rst[(size_t)d] = acc;
          acc *= a.shape()[(size_t)d];
        }
        return build_list<T>(arr, 0, 0, rst);
      },
      a.v);
}

std::string repr_py(nb::object o) {
  nb::object r = nb::steal(PyObject_Repr(o.ptr()));
  return nb::cast<std::string>(r);
}

std::string dlx_repr(const DlxArray &a) {
  std::string out = "array(";
  out += repr_py(dlx_tolist(a));
  out += ", dtype=";
  out += dracolix::dtype_name(a.dtype());
  out += ")";
  return out;
}

// ---------------------------------------------------------------------------
// reshape / transpose / astype / copy
// ---------------------------------------------------------------------------
DlxArray dlx_reshape(const DlxArray &a, nb::object shape) {
  std::vector<size_t> s = parse_shape(shape);
  return std::visit(
      [&](const auto &arr) -> DlxArray {
        dracolix::Array<
            typename array_scalar<std::decay_t<decltype(arr)>>::type>
            c = arr;
        c.reshape(s);
        return DlxArray(std::move(c));
      },
      a.v);
}

DlxArray dlx_transpose(const DlxArray &a) {
  if (a.ndim() != 2)
    throw nb::value_error("transpose() requires 2-D");
  return std::visit([](const auto &arr) { return DlxArray(arr.transpose()); },
                    a.v);
}

DlxArray dlx_copy(const DlxArray &a) {
  return std::visit([](const auto &arr) { return DlxArray(arr.clone()); }, a.v);
}

DlxArray dlx_astype(const DlxArray &a, nb::object dtype) {
  std::optional<dracolix::DType> dt = optional_dtype(dtype);
  if (!dt)
    throw nb::type_error("astype() requires a dtype");
  return cast_dtype(a, *dt);
}

// ---------------------------------------------------------------------------
// Sparse matrices (CSR / CSC wrappers over the core header)
// ---------------------------------------------------------------------------
// Scalar types supported by sparse storage (no bool: bool x bool matmul is
// meaningless and would need dedicated semantics).
using CsrVariant =
    std::variant<dracolix::CsrMatrix<float>, dracolix::CsrMatrix<double>,
                 dracolix::CsrMatrix<int32_t>, dracolix::CsrMatrix<int64_t>>;
using CscVariant =
    std::variant<dracolix::CscMatrix<float>, dracolix::CscMatrix<double>,
                 dracolix::CscMatrix<int32_t>, dracolix::CscMatrix<int64_t>>;

template <typename M> struct matrix_scalar;
template <typename T> struct matrix_scalar<dracolix::CsrMatrix<T>> {
  using type = T;
};
template <typename T> struct matrix_scalar<dracolix::CscMatrix<T>> {
  using type = T;
};

// MSVC refuses the std::variant converting constructor when the alternative
// type is a dependent name inside a template. These helpers narrow T to a
// concrete scalar via if constexpr so every std::variant construction sees a
// concrete, non-dependent matrix type.
template <typename T> CsrVariant make_csr(dracolix::CsrMatrix<T> m) {
    if constexpr (std::is_same_v<T, float>)
        return CsrVariant(std::move(m));
    else if constexpr (std::is_same_v<T, double>)
        return CsrVariant(std::move(m));
    else if constexpr (std::is_same_v<T, int32_t>)
        return CsrVariant(std::move(m));
    else
        return CsrVariant(std::move(m));
}
template <typename T> CscVariant make_csc(dracolix::CscMatrix<T> m) {
    if constexpr (std::is_same_v<T, float>)
        return CscVariant(std::move(m));
    else if constexpr (std::is_same_v<T, double>)
        return CscVariant(std::move(m));
    else if constexpr (std::is_same_v<T, int32_t>)
        return CscVariant(std::move(m));
    else
        return CscVariant(std::move(m));
}

struct DlxCsr {
  CsrVariant v;
  explicit DlxCsr(CsrVariant value) : v(std::move(value)) {}
  template <typename T> explicit DlxCsr(dracolix::CsrMatrix<T> m)
      : v(make_csr(std::move(m))) {}
  template <typename T> dracolix::CsrMatrix<T> *as() {
    return std::get_if<dracolix::CsrMatrix<T>>(&v);
  }
  template <typename T> const dracolix::CsrMatrix<T> *as() const {
    return std::get_if<dracolix::CsrMatrix<T>>(&v);
  }
};

struct DlxCsc {
  CscVariant v;
  explicit DlxCsc(CscVariant value) : v(std::move(value)) {}
  template <typename T> explicit DlxCsc(dracolix::CscMatrix<T> m)
      : v(make_csc(std::move(m))) {}
  template <typename T> dracolix::CscMatrix<T> *as() {
    return std::get_if<dracolix::CscMatrix<T>>(&v);
  }
  template <typename T> const dracolix::CscMatrix<T> *as() const {
    return std::get_if<dracolix::CscMatrix<T>>(&v);
  }
};

template <typename V> dracolix::DType sparse_dtype(const V &m) {
  return std::visit(
      [](const auto &mm) {
        using M = std::decay_t<decltype(mm)>;
        using T = typename matrix_scalar<M>::type;
        if constexpr (std::is_same_v<T, float>)
          return dracolix::DType::F32;
        if constexpr (std::is_same_v<T, double>)
          return dracolix::DType::F64;
        if constexpr (std::is_same_v<T, int32_t>)
          return dracolix::DType::I32;
        return dracolix::DType::I64;
      },
      m.v);
}

PyTypeObject *g_csr_type = nullptr;
PyTypeObject *g_csc_type = nullptr;
bool is_csr(nb::handle h) noexcept {
  return g_csr_type && PyObject_TypeCheck(h.ptr(), g_csr_type);
}
bool is_csc(nb::handle h) noexcept {
  return g_csc_type && PyObject_TypeCheck(h.ptr(), g_csc_type);
}

template <typename T>
dracolix::CsrMatrix<T> dense_to_csr(const dracolix::Array<T> &a) {
  return dracolix::CsrMatrix<T>::from_dense(a);
}
template <typename T>
dracolix::CscMatrix<T> dense_to_csc(const dracolix::Array<T> &a) {
  return dracolix::CscMatrix<T>::from_dense(a);
}

DlxCsc csc_from_dense(const DlxArray &a) {
  if (a.ndim() != 2)
    throw nb::type_error("csc_matrix() requires a 2-D array");
  if (a.dtype() == dracolix::DType::Bool)
    throw nb::type_error("sparse matrices do not support bool entries");
  // Concrete per-dtype branches: MSVC cannot resolve std::variant ctors when
  // the scalar dracolix::Array<T> type comes from a nested alias (see helpers
  // below), so spell each alternative out.
  switch (a.dtype()) {
    case dracolix::DType::F32: {
      const dracolix::Array<float> &src = *a.as<float>();
      return DlxCsc(CscVariant(dense_to_csc<float>(src)));
    }
    case dracolix::DType::F64: {
      const dracolix::Array<double> &src = *a.as<double>();
      return DlxCsc(CscVariant(dense_to_csc<double>(src)));
    }
    case dracolix::DType::I32: {
      const dracolix::Array<int32_t> &src = *a.as<int32_t>();
      return DlxCsc(CscVariant(dense_to_csc<int32_t>(src)));
    }
    case dracolix::DType::I64: {
      const dracolix::Array<int64_t> &src = *a.as<int64_t>();
      return DlxCsc(CscVariant(dense_to_csc<int64_t>(src)));
    }
    default: throw nb::type_error("unsupported sparse dtype");
  }
}

// ---- construction
// ------------------------------------------------------------

DlxCsr csr_from_dense(const DlxArray &a) {
  if (a.ndim() != 2)
    throw nb::type_error("csr_matrix() requires a 2-D array");
  if (a.dtype() == dracolix::DType::Bool)
    throw nb::type_error("sparse matrices do not support bool entries");
  switch (a.dtype()) {
    case dracolix::DType::F32: {
      const dracolix::Array<float> &src = *a.as<float>();
      return DlxCsr(CsrVariant(dense_to_csr<float>(src)));
    }
    case dracolix::DType::F64: {
      const dracolix::Array<double> &src = *a.as<double>();
      return DlxCsr(CsrVariant(dense_to_csr<double>(src)));
    }
    case dracolix::DType::I32: {
      const dracolix::Array<int32_t> &src = *a.as<int32_t>();
      return DlxCsr(CsrVariant(dense_to_csr<int32_t>(src)));
    }
    case dracolix::DType::I64: {
      const dracolix::Array<int64_t> &src = *a.as<int64_t>();
      return DlxCsr(CsrVariant(dense_to_csr<int64_t>(src)));
    }
    default: throw nb::type_error("unsupported sparse dtype");
  }
}

std::vector<size_t> idx_list(nb::object o) {
  if (is_array(o)) {
    DlxArray &a = *nb::cast<DlxArray *>(o);
    if (a.ndim() != 1)
      throw nb::value_error("index arrays must be 1-D");
    std::vector<size_t> out(a.size());
    std::visit(
        [&](const auto &arr) {
          using U = typename array_scalar<std::decay_t<decltype(arr)>>::type;
          for (size_t i = 0; i < a.size(); ++i) {
            int64_t v = (int64_t)arr[i];
            if (v < 0)
              throw nb::value_error("indices must be non-negative");
            out[i] = (size_t)v;
          }
        },
        a.v);
    return out;
  }
  if (PyList_Check(o.ptr()) || PyTuple_Check(o.ptr())) {
    Py_ssize_t len = PySequence_Size(o.ptr());
    std::vector<size_t> out((size_t)len);
    for (Py_ssize_t i = 0; i < len; ++i) {
      nb::object item = nb::steal(PySequence_GetItem(o.ptr(), i));
      int64_t v = nb::cast<int64_t>(item);
      if (v < 0)
        throw nb::value_error("indices must be non-negative");
      out[(size_t)i] = (size_t)v;
    }
    return out;
  }
  throw nb::type_error("indices must be a list/tuple/array of ints");
}

template <typename T> std::vector<T> dlx_values(const DlxArray &a) {
  std::vector<T> out(a.size());
  std::visit(
      [&](const auto &arr) {
        using U = typename array_scalar<std::decay_t<decltype(arr)>>::type;
        for (size_t i = 0; i < a.size(); ++i)
          out[i] = (T)arr[i];
      },
      a.v);
  return out;
}

template <typename T>
DlxCsr coo_to_csr_typed(size_t rows, size_t cols, std::vector<size_t> ri,
                        std::vector<size_t> ci, std::vector<T> vv,
                        bool drop_zeros) {
  return DlxCsr(dracolix::CsrMatrix<T>::from_coo(
      rows, cols, std::move(ri), std::move(ci), std::move(vv), drop_zeros));
}
template <typename T>
DlxCsc coo_to_csc_typed(size_t rows, size_t cols, std::vector<size_t> ri,
                        std::vector<size_t> ci, std::vector<T> vv,
                        bool drop_zeros) {
  return DlxCsc(dracolix::CscMatrix<T>::from_coo(
      rows, cols, std::move(ri), std::move(ci), std::move(vv), drop_zeros));
}

DlxCsr csr_from_coo(size_t rows, size_t cols, nb::object row_ind,
                    nb::object col_ind, nb::object values, bool drop_zeros) {
  DlxArray va = dlx_array(values, nb::none());
  if (va.dtype() == dracolix::DType::Bool)
    throw nb::type_error("sparse matrices do not support bool entries");
  std::vector<size_t> ri = idx_list(row_ind);
  std::vector<size_t> ci = idx_list(col_ind);
  if (ri.size() != ci.size() || ci.size() != va.size())
    throw nb::value_error("COO triplet arrays must have equal length");
  switch (va.dtype()) {
  case dracolix::DType::F32:
    return coo_to_csr_typed<float>(rows, cols, std::move(ri), std::move(ci),
                                   dlx_values<float>(va), drop_zeros);
  case dracolix::DType::F64:
    return coo_to_csr_typed<double>(rows, cols, std::move(ri), std::move(ci),
                                    dlx_values<double>(va), drop_zeros);
  case dracolix::DType::I32:
    return coo_to_csr_typed<int32_t>(rows, cols, std::move(ri), std::move(ci),
                                     dlx_values<int32_t>(va), drop_zeros);
  case dracolix::DType::I64:
    return coo_to_csr_typed<int64_t>(rows, cols, std::move(ri), std::move(ci),
                                     dlx_values<int64_t>(va), drop_zeros);
  default:
    throw nb::type_error("unsupported sparse dtype");
  }
}

DlxCsc csc_from_coo(size_t rows, size_t cols, nb::object row_ind,
                    nb::object col_ind, nb::object values, bool drop_zeros) {
  DlxArray va = dlx_array(values, nb::none());
  if (va.dtype() == dracolix::DType::Bool)
    throw nb::type_error("sparse matrices do not support bool entries");
  std::vector<size_t> ri = idx_list(row_ind);
  std::vector<size_t> ci = idx_list(col_ind);
  if (ri.size() != ci.size() || ci.size() != va.size())
    throw nb::value_error("COO triplet arrays must have equal length");
  switch (va.dtype()) {
  case dracolix::DType::F32:
    return coo_to_csc_typed<float>(rows, cols, std::move(ri), std::move(ci),
                                   dlx_values<float>(va), drop_zeros);
  case dracolix::DType::F64:
    return coo_to_csc_typed<double>(rows, cols, std::move(ri), std::move(ci),
                                    dlx_values<double>(va), drop_zeros);
  case dracolix::DType::I32:
    return coo_to_csc_typed<int32_t>(rows, cols, std::move(ri), std::move(ci),
                                     dlx_values<int32_t>(va), drop_zeros);
  case dracolix::DType::I64:
    return coo_to_csc_typed<int64_t>(rows, cols, std::move(ri), std::move(ci),
                                     dlx_values<int64_t>(va), drop_zeros);
  default:
    throw nb::type_error("unsupported sparse dtype");
  }
}

// ---- accessors
// ----------------------------------------------------------------

nb::tuple int_tuple2(size_t a, size_t b) {
  nb::tuple t = nb::steal<nb::tuple>(PyTuple_New(2));
  PyTuple_SET_ITEM(t.ptr(), 0, nb::int_((int64_t)a).release().ptr());
  PyTuple_SET_ITEM(t.ptr(), 1, nb::int_((int64_t)b).release().ptr());
  return t;
}

DlxArray idx_to_array(const std::vector<size_t> &v) {
  dracolix::Array<int64_t> a(std::vector<size_t>{v.size()});
  for (size_t i = 0; i < v.size(); ++i)
    a[i] = (int64_t)v[i];
  return DlxArray(std::move(a));
}

template <typename V, typename GetIdx>
nb::object sparse_index_array(const V &m, GetIdx get) {
  return std::visit(
      [&](const auto &mm) -> nb::object {
        return nb::cast<DlxArray>(idx_to_array(get(mm)));
      },
      m.v);
}

nb::object csr_values_array(const DlxCsr &m) {
  return std::visit(
      [](const auto &mm) -> nb::object {
        using T = typename matrix_scalar<std::decay_t<decltype(mm)>>::type;
        const auto &vals = mm.values();
        dracolix::Array<T> a(std::vector<size_t>{vals.size()});
        for (size_t i = 0; i < vals.size(); ++i)
          a[i] = vals[i];
        return nb::cast<DlxArray>(DlxArray(std::move(a)));
      },
      m.v);
}
nb::object csc_values_array(const DlxCsc &m) {
  return std::visit(
      [](const auto &mm) -> nb::object {
        using T = typename matrix_scalar<std::decay_t<decltype(mm)>>::type;
        const auto &vals = mm.values();
        dracolix::Array<T> a(std::vector<size_t>{vals.size()});
        for (size_t i = 0; i < vals.size(); ++i)
          a[i] = vals[i];
        return nb::cast<DlxArray>(DlxArray(std::move(a)));
      },
      m.v);
}

nb::object sparse_at_csr(const DlxCsr &m, int64_t i, int64_t j) {
  return std::visit(
      [&](const auto &mm) { return nb::cast(mm.at((size_t)i, (size_t)j)); },
      m.v);
}
nb::object sparse_at_csc(const DlxCsc &m, int64_t i, int64_t j) {
  return std::visit(
      [&](const auto &mm) { return nb::cast(mm.at((size_t)i, (size_t)j)); },
      m.v);
}

nb::object sparse_to_dense_csr(const DlxCsr &m) {
  return std::visit(
      [](const auto &mm) {
        return nb::cast<DlxArray>(DlxArray(mm.to_dense()));
      },
      m.v);
}
nb::object sparse_to_dense_csc(const DlxCsc &m) {
  return std::visit(
      [](const auto &mm) {
        return nb::cast<DlxArray>(DlxArray(mm.to_dense()));
      },
      m.v);
}

nb::object csr_transpose(const DlxCsr &m) {
  return std::visit(
      [](const auto &mm) { return nb::cast<DlxCsc>(DlxCsc(mm.transpose())); },
      m.v);
}
nb::object csc_transpose(const DlxCsc &m) {
  return std::visit(
      [](const auto &mm) { return nb::cast<DlxCsr>(DlxCsr(mm.transpose())); },
      m.v);
}
nb::object csr_to_csc(const DlxCsr &m) {
  return std::visit(
      [](const auto &mm) { return nb::cast<DlxCsc>(DlxCsc(mm.to_csc())); },
      m.v);
}
nb::object csc_to_csr(const DlxCsc &m) {
  return std::visit(
      [](const auto &mm) { return nb::cast<DlxCsr>(DlxCsr(mm.to_csr())); },
      m.v);
}
nb::object csr_to_csr(const DlxCsr &m) {
  return std::visit([](const auto &mm) { return nb::cast<DlxCsr>(DlxCsr(mm)); },
                    m.v);
}
nb::object csc_to_csc(const DlxCsc &m) {
  return std::visit([](const auto &mm) { return nb::cast<DlxCsc>(DlxCsc(mm)); },
                    m.v);
}

// ---- matmul
// ---------------------------------------------------------------------

template <typename T>
nb::object csr_matmul_typed(const dracolix::CsrMatrix<T> &A, nb::object other) {
  if (is_csr(other)) {
    auto &B = *nb::cast<DlxCsr *>(other);
    const auto *Bm = B.as<T>();
    if (!Bm)
      throw nb::type_error("sparse @ sparse requires matching dtypes");
    return nb::cast<DlxCsr>(DlxCsr(A.matmul(*Bm)));
  }
  if (is_csc(other)) {
    auto &C = *nb::cast<DlxCsc *>(other);
    const auto *Cm = C.as<T>();
    if (!Cm)
      throw nb::type_error("sparse @ sparse requires matching dtypes");
    return nb::cast<DlxCsr>(DlxCsr(A.matmul(Cm->to_csr())));
  }
  bool dense_operand =
      is_array(other) || PyObject_CheckBuffer(other.ptr()) ||
      PyList_Check(other.ptr()) || PyTuple_Check(other.ptr());
  if (dense_operand) {
    DlxArray b = dlx_array(other, nb::none());
    if (b.dtype() == dracolix::DType::Bool)
      throw nb::type_error("sparse @ dense does not support bool arrays");
    const auto *bd = b.as<T>();
    if (!bd)
      throw nb::type_error("sparse @ dense requires matching dtypes");
    if (bd->ndim() == 1)
      return nb::cast<DlxArray>(DlxArray(A.matvec(*bd)));
    if (bd->ndim() == 2) {
      auto B = dracolix::CsrMatrix<T>::from_dense(*bd);
      return nb::cast<DlxArray>(DlxArray(A.matmul(B).to_dense()));
    }
    throw nb::type_error("sparse @ dense: dense operand must be 1-D or 2-D");
  }
  return nb::not_implemented();
}

nb::object csr_matmul(const DlxCsr &A, nb::object other) {
  return std::visit(
      [&](const auto &mm) -> nb::object {
        using T = typename matrix_scalar<std::decay_t<decltype(mm)>>::type;
        return csr_matmul_typed<T>(mm, other);
      },
      A.v);
}

nb::object csc_matmul(const DlxCsc &C, nb::object other) {
  // CSC has no matmul kernel of its own: materialize CSR (same matrix) and
  // delegate. Correctness is preserved; the CSR path is the reference impl.
  return std::visit(
      [&](const auto &mm) -> nb::object {
        using T = typename matrix_scalar<std::decay_t<decltype(mm)>>::type;
        dracolix::CsrMatrix<T> csr = mm.to_csr();
        return csr_matmul_typed<T>(csr, other);
      },
      C.v);
}

nb::object sparse_rmatmul(nb::object x, nb::object y) {
  if (!is_array(x))
    return nb::not_implemented();
  DlxArray &a = as_array(x);
  if (a.dtype() == dracolix::DType::Bool)
    throw nb::type_error("dense @ sparse does not support bool arrays");
  if (a.ndim() != 2)
    throw nb::type_error("dense @ sparse requires a 2-D dense array");
  if (is_csr(y)) {
    DlxCsr &s = *nb::cast<DlxCsr *>(y);
    return std::visit(
        [&](const auto &arr) -> nb::object {
          using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
          if constexpr (std::is_same_v<T, bool>) {
            throw nb::type_error("dense @ sparse does not support bool arrays");
          } else {
            const auto *Bm = s.as<T>();
            if (!Bm)
              throw nb::type_error("dense @ sparse requires matching dtypes");
            auto A = dracolix::CsrMatrix<T>::from_dense(arr);
            return nb::cast<DlxArray>(DlxArray(A.matmul(*Bm).to_dense()));
          }
        },
        a.v);
  }
  if (is_csc(y)) {
    DlxCsc &s = *nb::cast<DlxCsc *>(y);
    return std::visit(
        [&](const auto &arr) -> nb::object {
          using T = typename array_scalar<std::decay_t<decltype(arr)>>::type;
          if constexpr (std::is_same_v<T, bool>) {
            throw nb::type_error("dense @ sparse does not support bool arrays");
          } else {
            const auto *Cm = s.as<T>();
            if (!Cm)
              throw nb::type_error("dense @ sparse requires matching dtypes");
            dracolix::CsrMatrix<T> B = Cm->to_csr();
            auto A = dracolix::CsrMatrix<T>::from_dense(arr);
            return nb::cast<DlxArray>(DlxArray(A.matmul(B).to_dense()));
          }
        },
        a.v);
  }
  return nb::not_implemented();
}

// ---------------------------------------------------------------------------
// module registration
// ---------------------------------------------------------------------------
} // namespace

NB_MODULE(_dracolix_nb, m) {
  m.doc() = "DracoLIX - a scientific computing runtime with a zero-NumPy "
            "Python API (interoperates via the PEP 3118 buffer protocol).";

  m.attr("__version__") = "0.2.0";

  nb::class_<DlxDType>(m, "DType")
      .def("__eq__",
           [](const DlxDType &a, const nb::object &other) -> nb::object {
             if (!is_dtype(other))
               return nb::not_implemented();
             return nb::cast(a.dt == nb::cast<DlxDType *>(other)->dt);
           })
      .def("__ne__",
           [](const DlxDType &a, const nb::object &other) -> nb::object {
             if (!is_dtype(other))
               return nb::not_implemented();
             return nb::cast(a.dt != nb::cast<DlxDType *>(other)->dt);
           })
      .def("__hash__", [](const DlxDType &a) { return (size_t)a.dt; })
      .def("__repr__",
           [](const DlxDType &a) {
             return "dtype('" + dracolix::dtype_name(a.dt) + "')";
           })
      .def_prop_ro(
          "name", [](const DlxDType &a) { return dracolix::dtype_name(a.dt); });

  nb::class_<DlxArray> arr_cls(m, "Array");
  arr_cls
      .def_prop_ro("dtype",
                   [](const DlxArray &a) { return dtype_singleton(a.dtype()); })
      .def_prop_ro(
          "shape",
          [](const DlxArray &a) {
            nb::tuple t =
                nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t)a.ndim()));
            for (size_t d = 0; d < a.ndim(); ++d)
              PyTuple_SET_ITEM(t.ptr(), (Py_ssize_t)d,
                               nb::int_((int64_t)a.shape()[d]).release().ptr());
            return t;
          })
      .def_prop_ro("strides",
                   [](const DlxArray &a) {
                     nb::tuple t = nb::steal<nb::tuple>(
                         PyTuple_New((Py_ssize_t)a.ndim()));
                     for (size_t d = 0; d < a.ndim(); ++d)
                       PyTuple_SET_ITEM(
                           t.ptr(), (Py_ssize_t)d,
                           nb::int_((int64_t)a.strides()[d]).release().ptr());
                     return t;
                   })
      .def_prop_ro("ndim",
                   [](const DlxArray &a) { return (Py_ssize_t)a.ndim(); })
      .def_prop_ro("size",
                   [](const DlxArray &a) { return (Py_ssize_t)a.size(); })
      .def_prop_ro("itemsize",
                   [](const DlxArray &a) { return (Py_ssize_t)a.itemsize(); })
      .def_prop_ro("layout",
                   [](const DlxArray &a) {
                     return a.layout() == dracolix::Layout::RowMajor
                                ? "row-major"
                                : "col-major";
                   })
      .def_prop_ro("T",
                   [](const DlxArray &a) -> DlxArray {
                     if (a.ndim() != 2)
                       throw nb::value_error(".T requires 2-D");
                     return dlx_transpose(a);
                   })

      .def("__len__",
           [](const DlxArray &a) {
             return (Py_ssize_t)(a.ndim() ? a.shape()[0] : 1);
           })
      .def("__bool__",
           [](const DlxArray &a) -> bool {
             if (a.size() != 1)
               throw nb::type_error("the truth value of an array with more "
                                    "than one element is ambiguous");
             return std::visit(
                 [](const auto &arr) -> bool { return (bool)arr[0]; }, a.v);
           })
      .def(
          "__iter__",
          [scope = nb::handle(m)](const DlxArray &self) {
            return make_iter(self, scope);
          },
          nb::keep_alive<0, 1>())
      .def("__getitem__", &dlx_getitem)
      .def("__setitem__", &dlx_setitem)
      .def("__repr__", &dlx_repr)
      .def("__str__", &dlx_repr)

      .def("copy", &dlx_copy)
      .def("astype", &dlx_astype, nb::arg("dtype"))
      .def("reshape", &dlx_reshape, nb::arg("shape"))
      .def("transpose", &dlx_transpose)
      .def("tolist", &dlx_tolist)

      .def("sum", [](const DlxArray &a) { return dlx_sum(a); })
      .def(
          "sum",
          [](const DlxArray &a, int64_t axis) { return dlx_sum_axis(a, axis); },
          nb::arg("axis"))
      .def("min", [](const DlxArray &a) { return dlx_min(a); })
      .def(
          "min",
          [](const DlxArray &a, int64_t axis) { return dlx_min_axis(a, axis); },
          nb::arg("axis"))
      .def("max", [](const DlxArray &a) { return dlx_max(a); })
      .def(
          "max",
          [](const DlxArray &a, int64_t axis) { return dlx_max_axis(a, axis); },
          nb::arg("axis"))
      .def("mean", [](const DlxArray &a) { return dlx_mean(a); })
      .def(
          "mean",
          [](const DlxArray &a, int64_t axis) {
            return dlx_mean_axis(a, axis);
          },
          nb::arg("axis"))

      .def("__add__", &dlx_add)
      .def("__radd__", &dlx_add)
      .def("__sub__", &dlx_sub)
      .def("__rsub__", [](nb::object x, nb::object y) { return dlx_sub(y, x); })
      .def("__mul__", &dlx_mul)
      .def("__rmul__", &dlx_mul)
      .def("__truediv__", &dlx_div)
      .def("__rtruediv__",
           [](nb::object x, nb::object y) { return dlx_div(y, x); })
      .def("__neg__", &dlx_neg)
      .def("__eq__", &dlx_eq)
      .def("__ne__", &dlx_ne)
      .def("__matmul__", &dlx_matmul)
      .def("__rmatmul__",
           [](nb::object x, nb::object y) { return dlx_matmul(y, x); });

  // ---- sparse matrices ----------------------------------------------------
  nb::class_<DlxCsr> csr_cls(m, "CsrMatrix");
  csr_cls
      .def("__init__",
           [](DlxCsr *self, nb::object data, nb::object dtype) {
               DlxArray a = dlx_array(data, dtype);
               new (self) DlxCsr(csr_from_dense(a));
           },
           nb::arg("data"), nb::arg("dtype") = nb::none())
      .def_static("from_coo", &csr_from_coo, nb::arg("rows"), nb::arg("cols"),
                  nb::arg("row_ind"), nb::arg("col_ind"), nb::arg("values"),
                  nb::arg("drop_zeros") = true)
      .def_prop_ro("dtype",
                   [](const DlxCsr &m) { return dtype_singleton(sparse_dtype(m)); })
      .def_prop_ro("shape", [](const DlxCsr &m) {
          return std::visit(
              [](const auto &mm) { return int_tuple2(mm.rows(), mm.cols()); },
              m.v);
      })
      .def_prop_ro("rows", [](const DlxCsr &m) {
          return std::visit([](const auto &mm) { return (Py_ssize_t)mm.rows(); }, m.v);
      })
      .def_prop_ro("cols", [](const DlxCsr &m) {
          return std::visit([](const auto &mm) { return (Py_ssize_t)mm.cols(); }, m.v);
      })
      .def_prop_ro("nnz", [](const DlxCsr &m) {
          return std::visit([](const auto &mm) { return (Py_ssize_t)mm.nnz(); }, m.v);
      })
      .def_prop_ro("row_ptr", [](const DlxCsr &m) {
          return std::visit(
              [](const auto &mm) { return nb::cast<DlxArray>(idx_to_array(mm.row_ptr())); },
              m.v);
      })
      .def_prop_ro("col_ind", [](const DlxCsr &m) {
          return std::visit(
              [](const auto &mm) { return nb::cast<DlxArray>(idx_to_array(mm.col_ind())); },
              m.v);
      })
      .def_prop_ro("values", [](const DlxCsr &m) { return csr_values_array(m); })
      .def_prop_ro("T", [](const DlxCsr &m) { return csr_transpose(m); })
      .def("to_dense", [](const DlxCsr &m) { return sparse_to_dense_csr(m); })
      .def("to_csr", [](const DlxCsr &m) { return csr_to_csr(m); })
      .def("to_csc", [](const DlxCsr &m) { return csr_to_csc(m); })
      .def("at", [](const DlxCsr &m, int64_t i, int64_t j) {
          return sparse_at_csr(m, i, j);
      })
      .def("__matmul__", [](const DlxCsr &a, nb::object other) {
          return csr_matmul(a, other);
      })
      .def("__rmatmul__", [](nb::object x, nb::object y) -> nb::object {
          if (is_csr(x) || is_csc(x)) return sparse_rmatmul(y, x);
          if (is_csr(y) || is_csc(y)) return sparse_rmatmul(x, y);
          return nb::not_implemented();
      })
      .def_prop_ro_static("__array_ufunc__", [](nb::handle) { return nb::none(); })
      .def("__repr__", [](const DlxCsr &m) {
          return std::visit(
              [](const auto &mm) {
                  return "csr_matrix(" + std::to_string(mm.rows()) + " x " +
                         std::to_string(mm.cols()) + ", nnz=" +
                         std::to_string(mm.nnz()) + ")";
              },
              m.v);
      });

  nb::class_<DlxCsc> csc_cls(m, "CscMatrix");
  csc_cls
      .def("__init__",
           [](DlxCsc *self, nb::object data, nb::object dtype) {
               DlxArray a = dlx_array(data, dtype);
               new (self) DlxCsc(csc_from_dense(a));
           },
           nb::arg("data"), nb::arg("dtype") = nb::none())
      .def_static("from_coo", &csc_from_coo, nb::arg("rows"), nb::arg("cols"),
                  nb::arg("row_ind"), nb::arg("col_ind"), nb::arg("values"),
                  nb::arg("drop_zeros") = true)
      .def_prop_ro("dtype",
                   [](const DlxCsc &m) { return dtype_singleton(sparse_dtype(m)); })
      .def_prop_ro("shape", [](const DlxCsc &m) {
          return std::visit(
              [](const auto &mm) { return int_tuple2(mm.rows(), mm.cols()); },
              m.v);
      })
      .def_prop_ro("rows", [](const DlxCsc &m) {
          return std::visit([](const auto &mm) { return (Py_ssize_t)mm.rows(); }, m.v);
      })
      .def_prop_ro("cols", [](const DlxCsc &m) {
          return std::visit([](const auto &mm) { return (Py_ssize_t)mm.cols(); }, m.v);
      })
      .def_prop_ro("nnz", [](const DlxCsc &m) {
          return std::visit([](const auto &mm) { return (Py_ssize_t)mm.nnz(); }, m.v);
      })
      .def_prop_ro("col_ptr", [](const DlxCsc &m) {
          return std::visit(
              [](const auto &mm) { return nb::cast<DlxArray>(idx_to_array(mm.col_ptr())); },
              m.v);
      })
      .def_prop_ro("row_ind", [](const DlxCsc &m) {
          return std::visit(
              [](const auto &mm) { return nb::cast<DlxArray>(idx_to_array(mm.row_ind())); },
              m.v);
      })
      .def_prop_ro("values", [](const DlxCsc &m) { return csc_values_array(m); })
      .def_prop_ro("T", [](const DlxCsc &m) { return csc_transpose(m); })
      .def("to_dense", [](const DlxCsc &m) { return sparse_to_dense_csc(m); })
      .def("to_csr", [](const DlxCsc &m) { return csc_to_csr(m); })
      .def("to_csc", [](const DlxCsc &m) { return csc_to_csc(m); })
      .def("at", [](const DlxCsc &m, int64_t i, int64_t j) {
          return sparse_at_csc(m, i, j);
      })
      .def("__matmul__", [](const DlxCsc &a, nb::object other) {
          return csc_matmul(a, other);
      })
      .def("__rmatmul__", [](nb::object x, nb::object y) -> nb::object {
          if (is_csr(x) || is_csc(x)) return sparse_rmatmul(y, x);
          if (is_csr(y) || is_csc(y)) return sparse_rmatmul(x, y);
          return nb::not_implemented();
      })
      .def_prop_ro_static("__array_ufunc__", [](nb::handle) { return nb::none(); })
      .def("__repr__", [](const DlxCsc &m) {
          return std::visit(
              [](const auto &mm) {
                  return "csc_matrix(" + std::to_string(mm.rows()) + " x " +
                         std::to_string(mm.cols()) + ", nnz=" +
                         std::to_string(mm.nnz()) + ")";
              },
              m.v);
      });

  // singletons: Python-owned module attributes (no C++ static refs - see
  // dtype_singleton()). Keep the objects stable, so "dlx.f32 is dlx.f32".
  m.attr("f32") = nb::cast<DlxDType>(DlxDType(dracolix::DType::F32));
  m.attr("f64") = nb::cast<DlxDType>(DlxDType(dracolix::DType::F64));
  m.attr("i32") = nb::cast<DlxDType>(DlxDType(dracolix::DType::I32));
  m.attr("i64") = nb::cast<DlxDType>(DlxDType(dracolix::DType::I64));
  m.attr("bool") = nb::cast<DlxDType>(DlxDType(dracolix::DType::Bool));

  // factories
  m.def("array", &dlx_array, nb::arg("data"), nb::arg("dtype") = nb::none());
  m.def("asarray", &dlx_asarray, nb::arg("data"),
        nb::arg("dtype") = nb::none());
  m.def("zeros", &dlx_zeros, nb::arg("shape"), nb::arg("dtype") = nb::none());
  m.def("ones", &dlx_ones, nb::arg("shape"), nb::arg("dtype") = nb::none());
  m.def("empty", &dlx_empty, nb::arg("shape"), nb::arg("dtype") = nb::none());
  m.def("full", &dlx_full, nb::arg("shape"), nb::arg("fill_value"),
        nb::arg("dtype") = nb::none());
  m.def("arange", &dlx_arange, nb::arg("start"), nb::arg("stop") = nb::none(),
        nb::arg("step") = nb::none());

  // wire PEP 3118 export slots after the class exists
  g_dlx_type = reinterpret_cast<PyTypeObject *>(nb::type<DlxArray>().ptr());
  g_dtype_type = reinterpret_cast<PyTypeObject *>(nb::type<DlxDType>().ptr());
  install_buffer_slots();

  g_csr_type = reinterpret_cast<PyTypeObject *>(nb::type<DlxCsr>().ptr());
  g_csc_type = reinterpret_cast<PyTypeObject *>(nb::type<DlxCsc>().ptr());
}
