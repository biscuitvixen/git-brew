//! Builds the CUDA static library via CMake and tells rustc how to link it.
//!
//! The native profile is pinned to Release regardless of the cargo profile:
//! the library is the hashing hot path, and a debug build of it would make
//! `cargo build` produce a tool that is quietly hundreds of times slower.

use std::env;

fn main() {
    // An explicit arch list wins, then CMake's own CUDAARCHS convention.
    // Without either, CMake picks the shipped fatbin list for the toolkit
    // it found. The shipped-list switch is passed both ways because CMake
    // caches it in OUT_DIR, and a stale ON would mask a later override.
    let arch = env::var("GIT_BREW_CUDA_ARCH")
        .or_else(|_| env::var("CUDAARCHS"))
        .ok();
    let mut config = cmake::Config::new("native");
    config
        .generator("Ninja")
        .profile("Release")
        .define("BUILD_TESTING", "OFF")
        .define(
            "GIT_BREW_SHIPPED_ARCHS",
            if arch.is_some() { "OFF" } else { "ON" },
        );
    if let Some(arch) = &arch {
        config.define("CMAKE_CUDA_ARCHITECTURES", arch);
    }
    let dst = config.build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());

    let cuda_home = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:rustc-link-search=native={cuda_home}/lib64");

    // Left-to-right resolution: brew_native's runtime-API references
    // resolve from cudart_static, whose dlopen/shm_open/pthread references
    // resolve from the shared libraries after it. dl/rt/pthread are empty
    // compat shims on glibc >= 2.34 but load-bearing on older systems.
    println!("cargo:rustc-link-lib=static=brew_native");
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=pthread");

    // Watched paths name the source subdirectories individually: watching
    // native/ itself would also watch native/build, the standalone ctest
    // build tree, and trigger rebuild loops.
    println!("cargo:rerun-if-changed=native/CMakeLists.txt");
    println!("cargo:rerun-if-changed=native/core");
    println!("cargo:rerun-if-changed=native/backend");
    println!("cargo:rerun-if-changed=native/include");
    println!("cargo:rerun-if-env-changed=GIT_BREW_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDAARCHS");
}
