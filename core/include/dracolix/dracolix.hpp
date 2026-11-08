#pragma once

// Umbrella header for the core.
// Outgoing-only: no language bindings included here.
// Licensed under GPL-3.0+

#include "alloc.hpp"
#include "array.hpp"
#include "array_view.hpp"
#include "complex_ops.hpp"
#include "constants.hpp"
#include "cpu.hpp"
#include "decomp.hpp"
#include "distributions.hpp"
#include "dtype.hpp"
#include "eigen.hpp"
#include "layout.hpp"
#include "least_squares.hpp"
#include "linalg.hpp"
#include "mem.hpp"
#include "numerics.hpp"
#include "ode.hpp"
#include "optimize.hpp"
#include "pde.hpp"
#include "random.hpp"
#include "signal.hpp"
#include "sparse.hpp"
#include "svd.hpp"
#include "thread_pool.hpp"
#include "timeseries.hpp"
#include "units.hpp"

namespace dracolix {
constexpr const char *version = "0.3.4";
}
