#pragma once
// Umbrella header for the core. Outgoing-only: no language bindings included
// here. Licensed under GPL-3.0-only
#include "alloc.hpp"
#include "array.hpp"
#include "array_view.hpp"
#include "cpu.hpp"
#include "decomp.hpp"
#include "dtype.hpp"
#include "layout.hpp"
#include "linalg.hpp"
#include "sparse.hpp"
#include "eigen.hpp"
#include "svd.hpp"
#include "thread_pool.hpp"
#include "mem.hpp"
#include "numerics.hpp"
#include "ode.hpp"

namespace dracolix {
constexpr const char *version = "0.1.0-core";
}
