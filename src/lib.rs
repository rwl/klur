use itertools::interleave;
use std::alloc::{alloc, Layout};
use std::iter::zip;
use suitesparse_sys::{
    klu_analyze, klu_common, klu_defaults, klu_factor, klu_free_numeric, klu_free_symbolic,
    klu_numeric, klu_solve, klu_symbolic, klu_tsolve, klu_z_factor, klu_z_free_numeric,
    klu_z_solve, klu_z_tsolve,
};

use numpy::{Complex64, PyReadonlyArray1, PyReadwriteArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Performs LU factorization of a sparse matrix in compressed column format
/// and solves for one or more right-hand-side vectors.
#[pyfunction(n, rowind, colptr, nz, b, trans = false)]
#[pyo3(text_signature = "(n, rowind, colptr, nz, b, /, trans=False)")]
fn factor_solve(
    n: i32,
    rowind: PyReadonlyArray1<i32>,
    colptr: PyReadonlyArray1<i32>,
    nz: PyReadonlyArray1<f64>,
    mut b: PyReadwriteArrayDyn<f64>,
    trans: bool,
) -> PyResult<()> {
    let b = b
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(format!("b: {}", err)))?;

    let a_i = rowind
        .as_slice()
        .map_err(|err| PyValueError::new_err(format!("rowind: {}", err)))?;
    let a_p = colptr
        .as_slice()
        .map_err(|err| PyValueError::new_err(format!("colptr: {}", err)))?;
    let a_x = nz
        .as_slice()
        .map_err(|err| PyValueError::new_err(format!("nz: {}", err)))?;

    let common = unsafe { alloc(Layout::new::<klu_common>()) as *mut klu_common };
    if common.is_null() {
        return Err(PyValueError::new_err("error allocating common".to_string()));
    }
    if unsafe { klu_defaults(common) } != 1 {
        return Err(PyValueError::new_err("error calling klu_defaults"));
    }

    let mut symbolic = unsafe { klu_analyze(n, a_p.as_ptr(), a_i.as_ptr(), common) };
    if symbolic.is_null() {
        return Err(PyValueError::new_err("error calling klu_analyze"));
    }

    let mut numeric =
        unsafe { klu_factor(a_p.as_ptr(), a_i.as_ptr(), a_x.as_ptr(), symbolic, common) };
    if numeric.is_null() {
        unsafe {
            klu_free_symbolic(&mut symbolic as *mut *mut klu_symbolic, common);
        }
        return Err(PyValueError::new_err("error calling klu_factor"));
    }

    let nrhs = b.len() as i32 / n;
    let rv = if trans {
        unsafe { klu_tsolve(symbolic, numeric, n, nrhs, b.as_mut_ptr(), common) }
    } else {
        unsafe { klu_solve(symbolic, numeric, n, nrhs, b.as_mut_ptr(), common) }
    };
    unsafe {
        klu_free_numeric(&mut numeric as *mut *mut klu_numeric, common);
        klu_free_symbolic(&mut symbolic as *mut *mut klu_symbolic, common);
    }
    if rv != 1 {
        return Err(PyValueError::new_err("error calling klu_solve"));
    }

    Ok(())
}

/// Performs LU factorization of a sparse complex matrix in compressed column format
/// and solves for one or more complex right-hand-side vectors.
#[pyfunction(n, rowind, colptr, nz, b, trans = false)]
#[pyo3(text_signature = "(n, rowind, colptr, nz, b, /, trans=False)")]
fn z_factor_solve(
    n: i32,
    rowind: PyReadonlyArray1<i32>,
    colptr: PyReadonlyArray1<i32>,
    nz: PyReadonlyArray1<Complex64>,
    mut b: PyReadwriteArrayDyn<Complex64>,
    trans: bool,
) -> PyResult<()> {
    let b0 = b
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(format!("b: {}", err)))?;
    let mut b: Vec<f64> = interleave(b0.iter().map(|v| v.re), b0.iter().map(|v| v.im)).collect();

    let a_i = rowind
        .as_slice()
        .map_err(|err| PyValueError::new_err(format!("rowind: {}", err)))?;
    let a_p = colptr
        .as_slice()
        .map_err(|err| PyValueError::new_err(format!("colptr: {}", err)))?;
    let a_x = nz
        .as_slice()
        .map_err(|err| PyValueError::new_err(format!("nz: {}", err)))?;
    let a_x: Vec<f64> = interleave(a_x.iter().map(|v| v.re), a_x.iter().map(|v| v.im)).collect();

    let common = unsafe { alloc(Layout::new::<klu_common>()) as *mut klu_common };
    if common.is_null() {
        return Err(PyValueError::new_err("error allocating common".to_string()));
    }
    if unsafe { klu_defaults(common) } != 1 {
        return Err(PyValueError::new_err("error calling klu_defaults"));
    }

    let mut symbolic = unsafe { klu_analyze(n, a_p.as_ptr(), a_i.as_ptr(), common) };
    if symbolic.is_null() {
        return Err(PyValueError::new_err("error calling klu_analyze"));
    }

    let mut numeric =
        unsafe { klu_z_factor(a_p.as_ptr(), a_i.as_ptr(), a_x.as_ptr(), symbolic, common) };
    if numeric.is_null() {
        unsafe {
            klu_free_symbolic(&mut symbolic as *mut *mut klu_symbolic, common);
        }
        return Err(PyValueError::new_err("error calling klu_z_factor"));
    }

    let nrhs = b.len() as i32 / n;
    let rv = if trans {
        unsafe { klu_z_tsolve(symbolic, numeric, n, nrhs, b.as_mut_ptr(), 0, common) }
    } else {
        unsafe { klu_z_solve(symbolic, numeric, n, nrhs, b.as_mut_ptr(), common) }
    };
    unsafe {
        klu_z_free_numeric(&mut numeric as *mut *mut klu_numeric, common);
        klu_free_symbolic(&mut symbolic as *mut *mut klu_symbolic, common);
    }
    if rv != 1 {
        return Err(PyValueError::new_err("error calling klu_z_solve"));
    }

    zip(b.chunks_exact(2), b0).for_each(|(v, z)| {
        z.re = v[0];
        z.im = v[1];
    });

    Ok(())
}

/// Provides sparse LU factorization with partial pivoting.
#[pymodule]
fn klur(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(factor_solve, m)?)?;
    m.add_function(wrap_pyfunction!(z_factor_solve, m)?)?;
    Ok(())
}
