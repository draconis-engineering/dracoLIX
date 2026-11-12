# DracoLIX Version Roadmap

Design decisions and scope are recorded here **per version** so that future
development has a single, authoritative place to look before touching code.
The phase-level roadmap lives in [`ROADMAP.md`](ROADMAP.md); this file tracks
release-level scope, semantic decisions, and deferrals.

## Conventions

- Newest version first. Append a new section when a version is started.
- Every section lists: goals, in-scope, out-of-scope, design decisions (with
  rationale), testing plan, and release checklist.
- Decisions are binding until changed explicitly in a later section; when a
  decision is reversed, note it under the newer version and link back.
- Reference files/functions with `path:line` where it helps.

---

## v0.3.5 (In progress)

### Goals

Add the NumPy-style elementwise selection helpers that were deferred from the
0.3.4 ufunc surface.

### In scope

- Python module-level `maximum(x, y)` / `minimum(x, y)` with broadcasting.
- Python module-level `where(condition, x, y)` with three-way broadcasting.
- Array, sequence and scalar operands for all arguments; scalar/scalar input
  returns a Python scalar.
- Exports, canonical stub, tests, CHANGELOG.

### Out of scope

- `fmax`/`fmin` (NaN-ignoring variants) and the one-argument
  `where(condition)` index form.
- `nonzero`/`count_nonzero`.
- Empty arrays (see v0.3.6) and N-D shape work (see v0.3.7).

### Design decisions

- **D1** - `maximum`/`minimum` are module-level only (no `Array` methods),
  matching NumPy. They promote through the usual pair rules, so mixed real
  dtypes widen as in the binary operators.
- **D2** - NaN propagates in `maximum`/`minimum` (unlike `fmax`/`fmin`).
  Complex values are ordered lexicographically by `(real, imag)`; NaN in
  either component propagates that operand.
- **D3** - Scalars participate anywhere. A Python bool stays `bool` (so
  `maximum(bool_array, True) -> bool`); otherwise scalars use the existing
  weak promotion (Python `int` behaves like `i64`). Two scalar operands return
  a Python scalar because dracolix has no 0-d arrays.
- **D4** - `where` promotes only `x` and `y`; the condition may have any dtype
  and uses Python truthiness (complex is true when either component is
  non-zero). Three scalar operands return a Python scalar.

### Testing plan

- `maximum`/`minimum`: dtype matrix, broadcasting, NaN propagation, complex
  ordering, scalar operands, `numpy` cross-checks.
- `where`: bool/real/complex truthiness, three-way broadcasting, scalar
  returns, dtype promotion, mismatched-shape errors.

### Release checklist

- [x] Bindings + exports + canonical stub
- [x] Python tests (numpy cross-checks where available)
- [x] CHANGELOG + version bump
- [x] pytest + ctest green
- [ ] Tag and publish v0.3.5

---

## v0.3.7 (Planned)

### Goals

NumPy-inspired shape manipulation that is currently missing or subtly wrong.

### In scope

- Fix `Array::reshape`: it currently pads or truncates when the element count
  changes (`element copy + resize`); NumPy requires the same total size and
  supports a single `-1` placeholder. Add a `reshaped()` (copy returning)
  helper; keep in-place `reshape` for C++ callers.
- `ravel()` / `flatten()` (copy; `order="C"` only).
- `transpose(axes=None)` for N-D (general permutation), `swapaxes(a, b)`,
  `moveaxis(src, dst)` (current `transpose()` is a 2-D copy only).
- `squeeze(axis=None)`, `expand_dims(axis)`, `atleast_1d/2d`.
- `broadcast_to(shape)` (materialized copy) and `broadcast_arrays` if cheap.
- Stacking: `concatenate`, `stack`, `vstack`, `hstack`.
- Preserve dtype; reject incompatible sizes with `ValueError`.

### Out of scope

- Fancy/boolean-mask indexing.
- Public stride/view API, `order="F"`, structured dtypes.
- Zero-cost views (materialized copies are acceptable first).

### Design decisions

- **D1** - Shape arguments accept `int` or sequences; a single `-1` is
  inferred from the total size. More than one `-1`, or a non-divisible size,
  raises `ValueError`.
- **D2** - `reshape` returns a new array (never aliases) until a view model
  exists; document the copy cost.
- **D3** - Axis arguments use the shared normalizer and accept negatives.
- **D4** - `transpose()` with no arguments reverses all axes (NumPy default),
  replacing the current 2-D-only restriction.

### Testing plan

- `reshape` with `-1`, invalid sizes, 1-D <-> 2-D <-> 3-D round-trips.
- N-D transpose/swapaxes/moveaxis against `numpy`.
- squeeze/expand_dims edge cases (size-1 axes, out-of-range).
- concatenate/stack/ravel/flatten dtype and shape checks.

### Release checklist

- [ ] Core shape helpers implemented
- [ ] Bindings + canonical stub
- [ ] Python tests (numpy cross-checks where available)
- [ ] CHANGELOG + version bump
- [ ] ctest + pytest green

---

## v0.3.6 (In progress)

### Goals

Support zero-size (empty) arrays end-to-end. Today the core builds them, but
the Python layer rejects them at construction/indexing, which blocks
masking/filtering use cases and NumPy empty identities. Also ship the
Windows/MSVC GEMM fix found while benchmarking 0.3.5.

### In scope

- Remove the binding guards (`dracolix_nb.cpp` ~991, 1646, 1766, 1776) and
  audit methods that assume `size_ > 0`.
- Construction: `array([])`, `zeros(0)`, `zeros((2, 0))`, `empty/full` with a
  zero dim, `arange(0)`, `linspace(..., num=0)`.
- Slicing: zero-length slices (`a[0:0]`), empty views, `tolist() == []`.
- Buffer export with zero length (PEP 3118).
- Reductions on empty arrays follow NumPy identities:
  `sum() -> 0`, `prod() -> 1`, `any() -> False`, `all() -> True`;
  `min`/`max`/`argmin`/`argmax` raise `ValueError`; `mean`/`std`/`var` raise
  `ValueError` (we keep explicit errors instead of NumPy's NaN + warning).
- Axis reductions over a zero-length reduced axis: `sum -> zeros`,
  `prod -> ones`, `any -> False`, `all -> True`; stats raise.
- If the *output* is empty because an outer dimension is zero (for example
  `zeros((2, 0)).min(axis=0)`), return an empty result instead of raising;
  only a zero-length *reduced* axis raises (matching NumPy).
- Broadcasting with a zero dimension follows NumPy (`(0, 3) + (1, 3) ->
  (0, 3)`); `broadcast_shape` must not widen `0` to `1`.
- GEMM: build the AVX2/AVX-512 microkernels on MSVC via per-ISA translation
  units (`/arch:AVX2`, `/arch:AVX512`), require FMA in dispatch, and cover the
  blocked/tile edge shapes from Python so the wheel build exercises them.

### Out of scope

- 0-d/scalar arrays (keep `ndim >= 1`).
- `resize`/`repeat`/`pad`.

### Design decisions

- **D1** - A zero dimension is legal; the total element count is the product
  of dimensions and may be `0`.
- **D2** - `array([])` infers `f64` (same as an empty Python float list would
  in NumPy) and produces shape `(0,)`.
- **D3** - Empty reductions use the identities above. `min`/`max`/`argmin`/
  `argmax` and statistical reductions raise `ValueError` only when the reduced
  axis itself has length 0; zero-size outer dimensions yield empty outputs.
- **D4** - MSVC has no per-function `target` attributes, so `gemm_micro_avx2.cpp`
  and `gemm_micro_avx512.cpp` are compiled once per ISA and left in the same
  DLL. The runtime CPUID check in `dispatch.hpp` still gates every call, so the
  wheel stays portable while no longer silently falling back to scalar.

### Testing plan

- Construct/extend/tolist/buffer round-trips for shapes `(0,)`, `(2, 0)`,
  `(0, 3)`.
- Every reduction on empty arrays and zero-length axes, including empty
  outputs from a zero-size outer dimension.
- Slicing to empty and back, plus zero-dimension broadcasting.
- `numpy` cross-checks for dtypes and identities.
- `test_matmul_blocked_edge_shapes` crosses the 256 block and microkernel tile
  boundaries for `f64`/`f32`; cibuildwheel runs it against the MSVC wheel.

### Release checklist

- [x] Binding guards removed, invariants audited
- [x] Core/binding empty handling
- [x] Tests added and green
- [x] MSVC AVX2/AVX-512 GEMM fix + edge-shape coverage
- [ ] CHANGELOG + version bump

---

## v0.3.4 (Released 2026-11-08)

### Goals

Correct and expose `std`/`var`, complete the arithmetic operator set, and add
the missing elementwise math (ufunc) surface.

### In scope

- Fix `Array::std(axis)` / `Array::var(axis)` (currently `s - s²/n²`, which is
  not variance) with a stable two-pass `mean(|x − mean|²)` computation
  (`core/include/dracolix/array.hpp`).
- Add `ddof` support (default `0`, population) to global and axis forms.
- Complex `std`/`var`: compute from the modulus and return real `f64`.
- Bind `std`/`var` (global + axis + `ddof`) to Python.
- Arithmetic completeness:
  - `//` and `%` (both directions), with Python/NumPy floor/remainder signs.
  - unary `+`, fix unary `-` for bool (raise, like NumPy).
  - `~`, `&`, `|`, `^` (bool + integers only).
  - `<<` and `>>` (integers only).
  - `pow`/`rpow` already exist; keep semantics, add tests.
- Elementwise math (module-level, NumPy-style): `sqrt`, `square`, `exp`,
  `expm1`, `log`, `log1p`, `log2`, `log10`, `sin`, `cos`, `tan`, `asin`,
  `acos`, `atan`, `sinh`, `cosh`, `tanh`, `floor`, `ceil`, `trunc`,
  `isfinite`, `isnan`, `isinf`.
- Retire the dead `bindings/python/dracolix/dracolix.pyi` and keep
  `__init__.pyi` as the single shipped stub (see v0.3.3 decision 8).

### Out of scope

- Module-level reduction functions (`dracolix.sum(a)`, ...).
- In-place operators (`+=`, ...), `divmod`, `round`.
- Integer reduction promotion/overflow changes; power overflow checks.
- Fancy indexing, reshape work (see v0.3.6/v0.3.7).

### Design decisions

- **D1** - `std`/`var` return `f64` (real) for all real/integer/bool inputs;
  complex inputs return `f64` computed from `mean(|x − mean|²)`, matching
  NumPy. Two-pass algorithms are used to avoid catastrophic cancellation.
- **D2** - `ddof` must satisfy `0 <= ddof < n`; otherwise raise `ValueError`
  (NumPy returns NaN/inf with warnings; we prefer an explicit error).
- **D3** - Axis stats are exposed as `std_axis(axis, ddof=0)` /
  `var_axis(axis, ddof=0)` in C++, with global `std(ddof=0)` /
  `var(ddof=0)`. Renaming avoids the `arr.var(0)` ambiguity between "axis 0"
  and "ddof 0"; the old (broken, unused) axis overloads are removed. The
  Python surface follows NumPy's argument order: `arr.std(axis=None, ddof=0)`,
  so positional `arr.std(0)` reduces axis 0 and the scalar form uses the
  keyword (`arr.std(ddof=1)`).
- **D4** - Axis reductions over a zero-length reduced dimension raise
  `ValueError` (consistent with argmin/argmax), rather than returning NaN.
- **D5** - Bool arithmetic remains rejected (`TypeError`): NumPy's bool
  arithmetic has surprising promotions (`bool+bool -> bool`,
  `bool**bool -> int8`), and we keep the explicit-error policy already shipped
  in 0.3.3. Unary `-`/`+` and `<<`/`>>` on bool also raise; `~bool` is
  logical NOT and is supported, and bitwise `&`/`|`/`^` between bool arrays
  stays bool.
- **D6** - `//`/`%` follow Python/NumPy sign semantics (floor towards `-inf`,
  remainder has the divisor's sign) for all real dtypes. Integer division or
  remainder by zero raises `ZeroDivisionError`; float follows IEEE
  (`inf`/`nan`, no warnings); complex raises `TypeError`.
- **D7** - Bitwise `& | ^ ~` work for bool and integer dtypes (promoted via
  the existing pair rules); float/complex raise `TypeError`. Shifts require
  integer operands (bool is rejected, see D5); the shift count must be
  `0 <= count < bit width`,
  otherwise `ValueError` (avoids C++ UB). Left-shift wraps through the
  unsigned type; scalars keep the existing weak-promotion behavior (a Python
  `int` operand promotes like `i64`).
- **D8** - Ufunc dtype rules: `f32 -> f32`; `f64 -> f64`; integers/bool ->
  `f64`; `c64 -> c64` and `c128 -> c128` for complex-capable functions.
  `floor`/`ceil`/`trunc` reject complex. IEEE domain results are returned
  unsuppressed (`sqrt(-1) -> nan`, `log(0) -> -inf`); no warnings.
- **D9** - Ufuncs are **module-level only** (`dracolix.sqrt(a)`), matching
  NumPy; no `Array.sqrt()` methods. In 0.3.4 they accept `Array` inputs only
  (list/scalar coercion is not wired into the ufunc entry points yet).
  `square` preserves the input dtype (including bool); `abs`/`absolute` keep
  returning real magnitudes for complex.
- **D10** - `isfinite`/`isnan`/`isinf` return bool arrays for every dtype.
  For complex input a value is finite when **both** components are finite,
  and NaN/inf when **either** component is NaN/inf (NumPy parity).
- **D11** - `pow`/`rpow` stay as shipped: integer negative exponents raise
  `ValueError`, integer results wrap (no overflow checks), bool pow raises.
- **D12** - Retire `dracolix.pyi`; `__init__.pyi` remains the only stub.

### Testing plan

- `var`/`std`: population vs sample (`ddof=1`), 1-D/2-D, negative axis,
  bool, complex, invalid `ddof`, complex raise-free parity with `numpy.var`.
- Arithmetic: `//`, `%`, mixed int/float, negative operands, division by
  zero, complex rejection, reflected forms.
- Unary/bitwise/shift operators: bool, signed/unsigned, dtype preservation,
  invalid operands, invalid shift counts.
- Ufuncs: dtype matrix (f32/f64/int/bool/c64/c128), domain/edge values, shape
  preservation, complex restrictions, `numpy` cross-checks.
- Canonical stub has no references to `dracolix.pyi`.

### Release checklist

- [x] Core std/var fixed and covered by C++ tests
- [x] Bindings + canonical stub
- [x] Arithmetic/ufuncs bound and exported from `__init__.py`
- [x] Python tests added
- [x] CHANGELOG + version bump
- [x] ctest + pytest green
- [x] Tag and publish v0.3.4

---

## v0.3.3 (Released 2026-11-08)

### Goals

Finish the reduction surface that 0.3.1 only stubbed, and close the last
comparison gap, so comparison masks and basic statistics are usable from
Python.

### In scope

- Fix the broken `Array::prod()` and `Array::prod(axis)` core reductions.
- Add `Array::argmin(axis)` / `Array::argmax(axis)` returning **i64 indices**
  (the existing implementations return values of the element dtype).
- Add global/axis `any()` and `all()`.
- Bind `prod`, `argmin`, `argmax`, `any`, `all` (global + axis) to Python.
- Fix bool `min` (currently returns logical OR; should be AND/all).
- Accept list/tuple operands in binary operators (comparisons **and**
  arithmetic) by coercing them through the existing `array()` constructor.
- Make `bindings/python/dracolix/__init__.pyi` the authoritative stub.

### Out of scope (deferred)

- `std`/`var` fixes and binding → **v0.3.4** (decided 0.3.3 kickoff).
- Complex `std`/`var` (needs the v0.3.4 modulus definition).
- Module-level reductions (`dracolix.sum(a)`, ...).
- `cumsum`/`cumprod`, `sort`/`argsort`, `where`, `nonzero`, `count_nonzero`.
- Integer `sum`/`prod` dtype promotion and overflow handling.
- Aligning `sum`/`min`/`max`/`mean` empty-array behavior with the new NumPy
  identities (they still raise) → **v0.3.6**, together with empty-array
  construction support.

### Design decisions

1. **Reduction identities follow NumPy for the new functions.**
   `prod()` of an empty array is `1`; `prod(axis)` over a zero-length axis is
   `ones` in the output shape; `any()` of empty is `False`; `all()` of empty is
   `True`. Existing `sum`/`min`/`max`/`mean` keep raising on empty (deferred
   alignment, see out-of-scope). Note: the Python layer still rejects
   construction of empty/zero-length arrays (`ValueError: empty arrays are not
   supported yet`), so these identities are covered by the C++ tests until
   empty-array support lands.
   *Rationale:* new surface should not inherit the old inconsistency.
2. **`prod` boolean semantics:** `prod` on `bool` returns `i64` `0`/`1`
   (logical AND) for the scalar form and an `i64` array along an axis, matching
   `numpy.prod`. `min` on `bool` returns `bool` logical AND (`all`); `max`
   returns `bool` logical OR (`any`), matching `numpy.min`/`numpy.max`.
   *Rationale:* NumPy parity; also fixes the current `min` bug.
3. **`argmin`/`argmax`:** scalar forms return a Python `int` (flat C-order
   index); axis forms return `i64` arrays with the reduced axis removed.
   Empty arrays / zero-length reduced axes raise `ValueError`; complex arrays
   raise `TypeError` (no ordering). Negative axes are normalized.
   *Rationale:* matches NumPy dtypes and the library's existing complex policy.
4. **`any`/`all`:** scalar forms return a Python `bool`; axis forms return
   `bool` arrays. Truthiness is "non-zero" for all dtypes; NaN is truthy and
   complex is truthy when either part is non-zero (NumPy). Negative axes
   normalized; out-of-range raises `IndexError`.
5. **Sequence operands:** binary operators accept `list`/`tuple` operands by
   converting them with the same path as `array()`; `str` is not accepted as a
   numeric sequence. Buffer objects (`bytes`, `memoryview`, `numpy.ndarray`)
   continue to go through the existing buffer path.
   *Rationale:* `arr < [1, 2, 3]` and `arr + [1, 2, 3]` should behave like
   NumPy; one coercion path avoids divergent semantics.
6. **Axis normalization** is shared: `-ndim <= axis < ndim`; anything else is
   `IndexError`.
7. **`std`/`var` are not bound in 0.3.3.** The current axis implementations are
   mathematically wrong, so exposing them would ship incorrect results. They
   are removed from the canonical stub until v0.3.4.
8. **Canonical stub:** `bindings/python/dracolix/__init__.pyi` is the stub type
   checkers use for `import dracolix`;
   `bindings/python/dracolix/dracolix.pyi` is dead weight (nothing imports
   `dracolix.dracolix`). Keep the latter temporarily, remove it in v0.3.4.
9. **No module-level reductions** in 0.3.3; methods only.
10. **Integer overflow behavior is unchanged** (reductions keep the input
    dtype); documenting NumPy-style promotion is deferred.

### Implementation notes

- Core: `core/include/dracolix/array.hpp`
  - `prod()` must seed from the first element (or identity for empty), not
    `T{}` (`array.hpp:472`).
  - `prod(axis)` must pass multiply-identity `T{1}`, not `T{}`
    (`array.hpp:560`).
  - New `argmin_axis` / `argmax_axis` return `Array<int64_t>`; do not reuse
    `reduce_axis_minmax` (it returns values).
  - New `any`/`all` global + `any_axis`/`all_axis`.
- Bindings: `bindings/python/src/dracolix_nb.cpp`
  - New `dlx_prod`, `dlx_prod_axis`, `dlx_argmin`, `dlx_argmax`,
    `dlx_argmin_axis`, `dlx_argmax_axis`, `dlx_any`, `dlx_all`,
    `dlx_any_axis`, `dlx_all_axis`; register on `Array`.
  - Fix bool branch of `dlx_min` (`dracolix_nb.cpp:2081`).
  - Extend `coerce_operand` (`dracolix_nb.cpp:753`) for sequences.
- Stubs: `bindings/python/dracolix/__init__.pyi` (authoritative).

### Testing plan

- `prod`: global/axis on 1-D/2-D, f64/i64/bool, negative axis; empty
  identities covered in C++ (Python cannot construct empty arrays yet).
- `argmin`/`argmax`: 1-D/2-D, ties (first index), negative axis, bool, complex
  raises; empty raises covered in C++.
- `any`/`all`: global/axis, bool + numeric + complex; empty identities covered
  in C++.
- `min`/`max` on bool (AND/OR), existing complex rejection.
- Comparisons and arithmetic with list/tuple operands and broadcasting.

### Release checklist

- [x] Core reductions fixed/added
- [x] Bindings registered
- [x] Canonical stub updated (no `std`/`var`)
- [x] Tests added and green (`pytest`, `ctest`)
- [x] CHANGELOG + version bump to 0.3.3
- [ ] Tag/publish (only when explicitly requested)

---

## v0.3.2 (Released 2026-11-07)

Metadata-only release: version bump plus formatting in `core/src/array.cpp`
and `pyproject.toml`. No API changes.

---

## v0.3.1 (Released 2026-11-06)

Added integer dtypes, `linspace`, `item`/`__index__`, comparison operators, and
overflow-checked construction. Reductions were stubbed in `dracolix.pyi` only
and were **not** bound; that gap is closed in v0.3.3.
