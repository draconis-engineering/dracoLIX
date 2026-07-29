# DracoLIX | Draconis Linear Algebra Xllerated

## What is DracoLIX?

DracoLIX is a linear algebra library written in Rust, which focuses on performance and ease of use, and also as a way to educate engineers new to rust or to linear algebra. DracoLIX is totally open source, and contributions are welcome. Currently, DracoLIX is only designed for linear algebra, but we are planning to make it a general purpose HPC library for engineers. DracoLIX is the numerical computation backbone of Draconis.

## Architecture

DracoLIX constists of a series of smaller crates for each component of the library, such as matrix operations, linear solvers, and tensor operations. Python acts as the interface (or "glue" as some would say) and Rust and Fortran as the numerical computation backend. DracoLIX is written from scratch in Rust whereever possible.
