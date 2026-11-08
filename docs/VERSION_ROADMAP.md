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

## v0.3.4 (Planned)

### Goals

- Correct and expose `std`/`var`.

### In scope

- Fix `Array::std(axis)` / `Array::var(axis)` (currently `s - s²/n²`, which is
  not variance) to use `E[x²] − E[x]²` with a numerically stable two-pass
  fallback where needed (`core/include/dracolix/array.hpp`).
- Add `ddof` support (default `0`, population) to global and axis forms.
- Decide and implement complex `std`/`var` (NumPy returns a **real** array
  computed from `mean(|x − mean|²)`; the current code path throws via
  `mean()`).
- Bind `std`/`var` to Python (methods) once the core math is correct.
- Retire the dead `bindings/python/dracolix/dracolix.pyi` and keep
  `__init__.pyi` as the single shipped stub (see v0.3.3 decision 8).

### Out of scope

- Module-level reduction functions.
- Integer reduction promotion/overflow changes.

### Design decisions

- **D1** — `std`/`var` return `f64` (real) for all real/integer/bool inputs;
  complex inputs return `f64` variance/std from the modulus, matching NumPy.
- **D2** — `ddof` must satisfy `0 <= ddof < n`; otherwise raise `ValueError`
  (NumPy returns NaN/inf with warnings; we prefer an explicit error).
- **D3** — Axis reductions over a zero-length reduced dimension raise
  `ValueError` (consistent with argmin/argmax), rather than returning NaN.

### Testing plan

- Population vs sample (`ddof=1`) against hand-computed values.
- Axis forms on 1-D/2-D, negative axis, zero-length dimension.
- Complex variance against `numpy.var`.
- Bool arrays against `numpy.var`/`numpy.std`.

### Release checklist

- [ ] Core math fixed and covered by C++ tests
- [ ] Bindings + canonical stub
- [ ] Python tests added
- [ ] CHANGELOG + version bump
- [ ] ctest + pytest green

---

## v0.3.3 (In progress)

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
  identities (they still raise).

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

- [ ] Core reductions fixed/added
- [ ] Bindings registered
- [ ] Canonical stub updated (no `std`/`var`)
- [ ] Tests added and green (`pytest`, `ctest`)
- [ ] CHANGELOG + version bump to 0.3.3
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
