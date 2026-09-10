#include "dracolix/array.hpp"
// Explicit instantiations for implemented dtypes (keeps compile times + ABI stable)
namespace dracolix {
template class Array<float>;
template class Array<double>;
template class Array<int32_t>;
template class Array<int64_t>;
template class Array<bool>;
}
