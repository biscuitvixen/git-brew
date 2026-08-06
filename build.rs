//! Builds the CUDA static library via CMake and tells rustc how to link it.
//!
//! The native profile is pinned to Release regardless of the cargo profile:
//! the library is the hashing hot path, and a debug build of it would make
//! `cargo build` produce a tool that is quietly hundreds of times slower.

use std::env;

fn main() {
    let arch = env::var("GIT_BREW_CUDA_ARCH").unwrap_or_else(|_| "121".to_string());
    let dst = cmake::Config::new("native")
        .generator("Ninja")
        .profile("Release")
        .define("BUILD_TESTING", "OFF")
        .define("CMAKE_CUDA_ARCHITECTURES", &arch)
        .build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());

    let cuda_home = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:rustc-link-search=native={cuda_home}/lib64");

    println!("cargo:rustc-link-lib=static=brew_native");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Watched paths name the source subdirectories individually: watching
    // native/ itself would also watch native/build, the standalone ctest
    // build tree, and trigger rebuild loops.
    println!("cargo:rerun-if-changed=native/CMakeLists.txt");
    println!("cargo:rerun-if-changed=native/core");
    println!("cargo:rerun-if-changed=native/backend");
    println!("cargo:rerun-if-changed=native/include");
    println!("cargo:rerun-if-env-changed=GIT_BREW_CUDA_ARCH");
}
