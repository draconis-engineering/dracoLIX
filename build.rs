// build.rs

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Set up paths
    let fortran_dir = PathBuf::from("fortran").join("linalg.f90");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let liblinalg = out_dir.join("liblinalg.a");
    let linalg_o = out_dir.join("linalg.o");

    // Set up commands
    let mut gfcmd = Command::new("gfortran");
    let mut arcmd = Command::new("ar");

    // Add arguments to commands
    gfcmd.arg("-c").arg(fortran_dir).arg("-o").arg(&linalg_o);
    arcmd.arg("rcs").arg(&liblinalg).arg(&linalg_o);

    // Run commands
    let gfstatus = gfcmd.status().unwrap();
    let arstatus = arcmd.status().unwrap();

    // Check that gfortran command succeeded
    assert!(gfstatus.success());

    // Check that ar command succeeded
    assert!(arstatus.success());

    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=linalg");
}
