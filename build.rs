// build.rs

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // 1. Tell Cargo to rerun this script if the Fortran file changes
    let fortran_src = PathBuf::from("fortran").join("linalg.f90");
    println!("cargo:rerun-if-changed={}", fortran_src.display());

    // 2. Set up working output directory scopes
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let liblinalg = out_dir.join("liblinalg.a");
    let linalg_o = out_dir.join("linalg.o");

    // 3. Compile Fortran file into a clean static object file (-fPIC is required for PyO3)
    let gfstatus = Command::new("gfortran")
        .arg("-O3")
        .arg("-march=native")
        .arg("-ffast-math")
        .arg("-funroll-loops")
        .arg("-fexternal-blas")
        .arg("-fPIC") // Keep -fPIC so the code can be included in your Python extension
        .arg("-c") // Compile only, do not link (Dropped the conflicting '-shared' flag)
        .arg(&fortran_src)
        .arg("-o")
        .arg(&linalg_o)
        .status()
        .expect("Failed to execute gfortran compiler pipeline");

    assert!(gfstatus.success(), "gfortran compilation failed!");

    // 4. Package the static object file into a standard static library archive (.a)
    let arstatus = Command::new("ar")
        .arg("rcs")
        .arg(&liblinalg)
        .arg(&linalg_o)
        .status()
        .expect("Failed to execute ar archiving command");

    assert!(arstatus.success(), "ar archiving into liblinalg.a failed!");

    // 5. Instruct Cargo on how to bind the resulting library to your Rust binary
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=linalg");

    // 6. CRITICAL FOR FORTRAN: Link the system Fortran runtime library
    // This provides necessary internal system routines (like matmul allocations)
    println!("cargo:rustc-link-lib=gfortran");
    println!("cargo:rustc-link-lib=openblas"); // <-- Link the multi-threaded backend
}
