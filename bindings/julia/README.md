# DracoLIX Julia Bindings — Phase 5 prototype

Simple `ccall` interop to the C++ core via `core/include/dracolix/c_api.h`.

```julia
using DracoLIX
DracoLIX.version() |> println

A = [1.0 2.0; 3.0 4.0]
B = [5.0 6.0; 7.0 8.0]
C = DracoLIX.matmul(A, B) # 2×2
```

Build the shared lib first:

```bash
cmake -S . -B build -DDRACOLIX_BUILD_JULIA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
julia --project=bindings/julia -e 'using Pkg; Pkg.test()'
```

Covers `f64` only for the prototype; other dtypes follow the same `c_api.h` pattern.
