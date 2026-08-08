# DracoLIX | Draconis Linear Algebra Xllerated

## What is DracoLIX?

DracoLIX is a linear algebra library written in Python, Rust and Fortran, which focuses on performance, reliability and ease of use. DracoLIX is totally open source, and contributions are welcome. Currently, DracoLIX is only designed for linear algebra, but we are planning to make it a general purpose HPC library in the future, with support for CFD computations and other numerical simulations. DracoLIX is the numerical computation backbone of Draconis, especially the DuraPy library and the ICARUS Agent.

## Architecture

DracoLIX constists of a series of smaller crates for each component of the library, such as matrix operations, linear solvers, and tensor operations. Python is the default interface, but usage of DracoLIX directly through Rust is possible. Rust and Fortran as the numerical computation backend. DracoLIX is written from scratch in Rust/Fortran whereever possible.

## Modules 

## Notes
