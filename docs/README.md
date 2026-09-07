# DracoLIX

### Draconis Scientific Computing & High-Performance Numerical Computing

> **A native, multi-language scientific computing stack built from scratch around performance, reliability and numerical control.**

DracoLIX is an open-source scientific computing project developed by **Draconis Engineering**.

The goal is to build a high-performance numerical computing stack spanning **linear algebra, numerical methods, scientific simulation and HPC**, with a particular focus on giving the developer direct control over data types, memory layout and computation.

DracoLIX is designed to be **independent of NumPy and similar numerical runtimes**. Instead of wrapping an existing numerical backend, DracoLIX aims to provide its own array types, dtype system, memory model and computational kernels.

The project is currently in an early stage. At present, the implementation is focused primarily on matrix operations, but the architecture is being developed with a much larger scientific computing ecosystem in mind.

---

## Vision

Scientific computing is built on numerical primitives.

Matrices, vectors, tensors, numerical solvers and differential equations form the foundation of everything from engineering simulations to data analysis and computational physics.

DracoLIX aims to provide that foundation for the **Draconis Engineering ecosystem**.

The long-term vision is a platform capable of powering:

* Linear algebra
* Numerical analysis
* Scientific simulation
* Optimization
* Differential equations
* CFD
* Engineering computation
* High-performance data analysis
* HPC workloads

The core principle is simple:

> **Build the numerical infrastructure instead of depending on it.**

---

# Architecture

DracoLIX is designed as a multi-language system where each language has a specific role.

```text
                         DracoLIX
                            │
                 ┌──────────┴──────────┐
                 │                     │
            Public APIs            Core Runtime
                 │                     │
        ┌────────┼────────┐            │
        │        │        │            │
     Python    Julia     Rust          │
        │        │        │            │
        └────────┴────────┴────────────┘
                            │
                     DracoLIX Core
                            │
              ┌─────────────┼─────────────┐
              │             │             │
           Arrays        DTypes        Memory
              │             │             │
              └─────────────┼─────────────┘
                            │
                     Compute Runtime
                            │
              ┌─────────────┴─────────────┐
              │                           │
        Rust Kernels                Fortran Kernels
              │                           │
              └─────────────┬─────────────┘
                            │
                       CPU / SIMD
```

### Rust

Rust forms the foundation of DracoLIX.

It is responsible for:

* Core data structures
* Memory management
* Array implementations
* DType system
* Runtime and dispatch
* Parallel execution
* SIMD-oriented computation
* Language bindings
* Performance-critical kernels

Rust provides the systems-level control required to make DracoLIX independent of external numerical runtimes.

### Fortran

Fortran provides a second numerical backend, particularly for traditional HPC and numerical computing workloads.

It is intended for:

* Numerical kernels
* Scientific algorithms
* Linear algebra
* Numerical solvers
* Future HPC functionality

Where appropriate, Fortran and Rust will communicate through well-defined native interfaces.

### Python

Python is one of the primary high-level interfaces to DracoLIX.

The Python API is intended to make high-performance numerical computation accessible without requiring users to write Rust or Fortran.

Python bindings are implemented directly against the DracoLIX core rather than relying on NumPy as an underlying execution engine.

```python
import dracolix as dlx

A = dlx.matrix(
    [[1.0, 2.0],
     [3.0, 4.0]],
    dtype=dlx.f64
)

B = dlx.matrix(
    [[5.0, 6.0],
     [7.0, 8.0]],
    dtype=dlx.f64
)

C = A @ B
```

### Julia

Julia is planned as a first-class scientific computing interface.

Julia's strengths in numerical programming make it particularly suitable for:

* Scientific research
* Mathematical experimentation
* Numerical methods
* Differential equations
* Optimization
* Simulation

The Julia interface should communicate directly with the DracoLIX core rather than passing through Python.

---

# Native DTypes

One of the core goals of DracoLIX is to provide its **own numerical type system**.

DracoLIX should not require NumPy's `dtype` system to represent numerical data.

Instead:

```python
dlx.f32
dlx.f64
dlx.i32
dlx.i64
dlx.c64
dlx.c128
```

would represent native DracoLIX types.

Conceptually:

```text
                  DType
                    │
        ┌───────────┼───────────┐
        │           │           │
     Integer      Float      Complex
        │           │           │
     i8/i16/...   f32/f64    c64/c128
```

The DType system is intended to become part of the DracoLIX runtime itself.

This allows the runtime to make decisions based on:

* Element size
* Alignment
* Numerical representation
* SIMD compatibility
* Kernel availability
* Memory requirements
* Future accelerator support

The goal is to build a **numerical type system designed around DracoLIX's own execution model**, not just copy NumPy.

---

# Native Arrays

DracoLIX will provide its own numerical data structures.

```python
dlx.Array
dlx.Vector
dlx.Matrix
dlx.Tensor
```

These structures will own and manage their underlying memory through the DracoLIX runtime.

A conceptual array representation might contain:

```text
Array<T>
 ├── data
 ├── shape
 ├── strides
 ├── dtype
 └── layout
```

This gives DracoLIX control over important performance characteristics such as:

* Contiguous memory
* Strided views
* Row-major and column-major layouts
* Memory alignment
* Cache efficiency
* SIMD compatibility
* Zero-copy views
* Slicing
* Parallel access

This is a deliberate design choice.

DracoLIX should not simply become:

```text
Python
  ↓
NumPy
  ↓
C
  ↓
Rust
```

Instead:

```text
Python
  ↓
PyO3
  ↓
DracoLIX Core
  ↓
Native Kernels
  ↓
CPU / SIMD / HPC
```

---

# Computational Model

The long-term architecture separates **what is being calculated** from **how it is calculated**.

For example, a high-level operation might request:

```python
C = A @ B
```

The runtime can then determine the appropriate implementation based on:

* DType
* Matrix dimensions
* Memory layout
* Hardware capabilities
* Available kernels
* Parallel execution options

Conceptually:

```text
                 Matrix Multiplication
                          │
                          ▼
                    Kernel Dispatch
                          │
              ┌───────────┼───────────┐
              │           │           │
            Scalar      SIMD      Parallel
              │           │           │
              └───────────┼───────────┘
                          ▼
                     Native Code
```

This dispatch architecture is intended to become increasingly important as DracoLIX grows beyond basic linear algebra.

---

# Planned Scientific Computing Stack

DracoLIX is intended to grow incrementally.

### 1. Core

* Native DTypes
* Arrays
* Vectors
* Matrices
* Tensors
* Memory management
* Views and slicing
* Basic arithmetic

### 2. Linear Algebra

* Matrix multiplication
* Vector operations
* Decompositions
* Linear system solvers
* Eigenvalue problems
* Sparse matrices
* Advanced matrix algorithms

### 3. Numerical Computing

* Numerical integration
* Interpolation
* Optimization
* Root finding
* ODE solvers
* PDE solvers
* Numerical differentiation

### 4. Scientific Computing

* Scientific simulation
* Computational physics
* Engineering workloads
* CFD
* Large-scale numerical analysis

### 5. HPC

Future research areas may include:

* Advanced SIMD
* Multithreading
* GPU acceleration
* Distributed computation
* MPI
* Domain decomposition
* Large-scale simulation

These features are **long-term goals**, not current capabilities.

---

# Draconis Ecosystem

DracoLIX is intended to become the numerical computation backbone of Draconis Engineering.

```text
                     Draconis
                         │
              ┌──────────┼──────────┐
              │          │          │
           Olympus     DuraPy     ICARUS
              │          │          │
              └──────────┼──────────┘
                         │
                     DracoLIX
                         │
              Numerical Computation
```

### Olympus

Olympus Analytics Engine will be one of the first major consumers of DracoLIX.

As Olympus processes increasingly large quantities of training and physiological data, DracoLIX can provide the numerical foundation for:

* Time-series analysis
* Statistical calculations
* Signal processing
* Rolling computations
* Training-load calculations
* Numerical models
* Large-scale historical analysis

Olympus understands the **domain**.

DracoLIX handles the **mathematics**.

### DuraPy

DuraPy can use DracoLIX as its numerical backend for high-performance scientific and endurance-sports computation.

### ICARUS

ICARUS can use DracoLIX when numerical computation becomes part of an agent workflow, allowing the agent to delegate computationally intensive operations to a native numerical engine.

---

# Design Principles

### From the ground-up

DracoLIX should own its core numerical representation rather than simply wrapping another numerical library.

### Speed

Performance matters.

Memory layout, cache behavior, SIMD, parallelism and algorithmic complexity should be considered fundamental parts of the architecture.

### Reliability

Numerical software must be predictable.

Correctness, testing, numerical stability and deterministic behavior are core concerns.

### Multi-language

Different languages are good at different things.

DracoLIX embraces Python, Julia, Rust and Fortran instead of forcing everything into one language.

### Open Source

DracoLIX is fully open source.

The goal is to build a transparent numerical computing stack that can be studied, modified and extended by others.

### Modular Architecture

DracoLIX should consist of smaller components rather than becoming one enormous monolithic library.

```text
dracolix-core
dracolix-array
dracolix-linalg
dracolix-solver
dracolix-runtime
dracolix-python
dracolix-julia
...
```

The exact crate/package structure may evolve as the project develops.

---

# Current Status

DracoLIX is **experimental and heavily under development**.

Current functionality is limited compared to the long-term vision.

At the moment, the project primarily contains early implementations of numerical operations such as matrix multiplication.

The architecture and roadmap described above represent the **direction of the project**, not a claim that all of these capabilities currently exist.

---

# Roadmap

- [x] Project foundation
- [x] Initial Rust numerical core
- [x] Initial Python interface
- [x] Matrix multiplication prototype
- [ ] Native DType system
- [ ] Native Array / Matrix types
- [ ] Memory layout system
- [ ] Better benchmarking infrastructure
- [ ] SIMD kernels
- [ ] Parallel execution
- [ ] Expanded linear algebra
- [ ] Sparse matrices
- [ ] Numerical solvers
- [ ] Julia interface
- [ ] ODE/PDE tooling
- [ ] Optimization
- [ ] Scientific simulation
- [ ] CFD
- [ ] GPU acceleration
- [ ] Distributed HPC


The roadmap is intentionally ambitious.

Not every feature is guaranteed to be implemented, and priorities may change as the project evolves.

---

# Philosophy

DracoLIX started as a simple linear algebra library.

The long-term goal is considerably larger.

Instead of building another abstraction layer around existing numerical software, DracoLIX aims to explore what a **modern, native, multi-language scientific computing stack** could look like when the numerical runtime itself is designed from the ground up.

> **Own the data. Own the types. Own the computation.**

---

# License

DracoLIX is open source and intended to remain freely available to the community.

See the repository license for details.

---

**Draconis Engineering** - *Semper Ultra*
