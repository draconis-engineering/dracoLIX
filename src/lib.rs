// lib.rs

use numpy::{PyArray2, PyArrayMethods};
use pyo3::{exceptions::PyValueError, Bound, PyResult};

unsafe extern "C" {
    fn matmatmul_c(a: *const f64, b: *const f64, res: *mut f64, m: i32, n: i32, p: i32);
}

#[pyo3::pyfunction]
fn matmatmul<'py>(
    a: numpy::PyReadonlyArray2<f64>,
    b: numpy::PyReadonlyArray2<f64>,
    py: pyo3::Python<'py>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let m = i32::try_from(a.dims()[0]).unwrap();
    let n1 = i32::try_from(a.dims()[1]).unwrap();
    let n2 = i32::try_from(b.dims()[0]).unwrap();
    let p = i32::try_from(b.dims()[1]).unwrap();
    if n1 != n2 {
        let err = PyValueError::new_err("a and b must have the same number of columns");
        return Err(err.into());
    }
    todo!()
}
