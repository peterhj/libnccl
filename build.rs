extern crate walkdir;

use walkdir::{WalkDir};

use std::env;
use std::path::{PathBuf};
use std::process::{Command};

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  for entry in WalkDir::new("nccl") {
    let entry = entry.unwrap();
    println!("cargo:rerun-if-changed={}", entry.path().display());
  }

  let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
  let out_dir = env::var("OUT_DIR").unwrap();

  let cc = env::var("CC").unwrap_or(format!("gcc"));
  let cxx = env::var("CXX").unwrap_or(format!("g++"));
  let fc = env::var("FC").unwrap_or(format!("gfortran"));
  let cuda_home = env::var("CUDA_HOME").unwrap_or(format!("/usr/local/cuda"));

  let mut nccl_lib_dst_path = PathBuf::from(&out_dir);
  nccl_lib_dst_path.push("libnccl_static.a");

  {
    let mut nccl_src_path = PathBuf::from(&manifest_dir);
    nccl_src_path.push("nccl");
    let mut nccl_build_path = PathBuf::from(&out_dir);
    nccl_build_path.push("nccl-build");
    let mut nccl_lib_path = PathBuf::from(&nccl_build_path);
    nccl_lib_path.push("build");
    nccl_lib_path.push("lib");
    nccl_lib_path.push("libnccl_static.a");

    Command::new("cp")
      .current_dir(&out_dir)
      .arg("-r")
      .arg(nccl_src_path.to_str().unwrap())
      .arg(nccl_build_path.to_str().unwrap())
      .status().unwrap();

    Command::new("make")
      .current_dir(&nccl_build_path)
      .env("CC",  &cc)
      .env("CXX", &cxx)
      .env("FC",  &fc)
      .arg("-j8")
      .arg(&format!("CUDA_HOME={}", cuda_home))
      .arg("NVCC_GENCODE=-gencode=arch=compute_52,code=sm_52")
      .arg("staticlib")
      .status().unwrap();

    Command::new("cp")
      .current_dir(&out_dir)
      .arg(nccl_lib_path.to_str().unwrap())
      .arg(nccl_lib_dst_path.to_str().unwrap())
      .status().unwrap();
  }

  println!("cargo:rustc-link-search=native={}", out_dir);
}
