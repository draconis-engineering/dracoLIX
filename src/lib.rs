// src/lib.rs

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{Bound, PyResult, Python, exceptions::PyValueError};

unsafe extern "C" {
    fn matmatmul_c(a: *const f64, b: *const f64, res: *mut f64, m: i32, n: i32, p: i32);
    fn matvecmul_c(a: *const f64, b: *const f64, res: *mut f64, n: i32, m: i32);
}

#[pyo3::pyfunction]
fn matmatmul<'py>(
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray2<f64>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let m = i32::try_from(a.dims()[0]).unwrap();
    let n1 = i32::try_from(a.dims()[1]).unwrap();
    let n2 = i32::try_from(b.dims()[0]).unwrap();
    let p = i32::try_from(b.dims()[1]).unwrap();

    if n1 != n2 {
        return Err(PyValueError::new_err(
            "Inner matrix dimensions must match (columns of A must equal rows of B).",
        ));
    }

    let a_readonly = a.as_array();
    let b_readonly = b.as_array();

    let a_slice = a_readonly
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Array 'a' must be contiguous."))?;
    let b_slice = b_readonly
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Array 'b' must be contiguous."))?;

    // PERFORMANCE FIX: Allocate raw uninitialized memory instead of clearing it with zeros.
    // This removes the costly O(N^2) memory-writing pre-pass.
    let res_array = unsafe { PyArray2::<f64>::new(py, [m as usize, p as usize], false) };

    let a_addr = a_slice.as_ptr() as usize;
    let b_addr = b_slice.as_ptr() as usize;
    let res_addr = res_array.data() as usize;

    py.detach(move || unsafe {
        matmatmul_c(
            b_addr as *const f64,
            a_addr as *const f64,
            res_addr as *mut f64,
            p,
            n1,
            m,
        );
    });

    Ok(res_array)
}

#[pyo3::pyfunction]
fn matvecmul<'py>(
    a: PyReadonlyArray2<f64>,
    b: PyReadonlyArray1<f64>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n = i32::try_from(a.dims()[0]).unwrap();
    let m1 = i32::try_from(a.dims()[1]).unwrap();
    let m2 = i32::try_from(b.dims()[0]).unwrap();

    if m1 != m2 {
        return Err(PyValueError::new_err(
            "Matrix columns must match the length of the vector.",
        ));
    }

    let a_readonly = a.as_array();
    let b_readonly = b.as_array();

    let a_slice = a_readonly
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Matrix 'a' must be contiguous."))?;
    let b_slice = b_readonly
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Vector 'b' must be contiguous."))?;

    // Allocate an uninitialized 1D output vector
    let res_array = unsafe { PyArray1::<f64>::new(py, [n as usize], false) };

    let a_addr = a_slice.as_ptr() as usize;
    let b_addr = b_slice.as_ptr() as usize;
    let res_addr = res_array.data() as usize;

    py.detach(move || unsafe {
        matvecmul_c(
            a_addr as *const f64,
            b_addr as *const f64,
            res_addr as *mut f64,
            n,
            m1,
        );
    });

    Ok(res_array)
}

#[pyo3::pymodule]
fn dracolix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(matmatmul, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(matvecmul, m)?)?;
    //m.add_function(pyo3::wrap_pyfunction!(vecmatmul, m)?)?;
    //m.add_function(pyo3::wrap_pyfunction!(dot, m)?)?;
    //m.add_function(pyo3::wrap_pyfunction!(outer, m)?)?;
    Ok(())
}
