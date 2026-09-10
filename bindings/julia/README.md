# Julia bindings - outgoing-only

Depends on `dracolix_core` via CxxWrap.jl. Core never includes Julia headers.

Planned layout:

```
bindings/julia/
  Project.toml
  src/DracoLIX.jl  -> ccall to libdracolix_core.so
  test/
```

Core exposes `extern "C"` API from `core/include/dracolix/c_api.h` (to be added).
