//! Builds the CUDA static library via CMake and tells rustc how to link it.
//!
//! The native profile is pinned to Release regardless of the cargo profile:
//! the library is the hashing hot path, and a debug build of it would make
//! `cargo build` produce a tool that is quietly hundreds of times slower.

use std::env;
use std::fs;
use std::path::Path;

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

    // The runtime comes from the toolkit whose nvcc compiled the kernel, as
    // located by CMake. That covers conda's targets/<triple>/lib layout and
    // module-loaded toolkits without guessing at an install prefix.
    let report = dst.join("share/cudart_static_path");
    let cudart = fs::read_to_string(&report).unwrap_or_else(|e| {
        panic!(
            "native build left no cudart location at {}: {e}",
            report.display()
        )
    });
    let cudart_dir = Path::new(cudart.trim())
        .parent()
        .expect("cudart_static path has no parent directory");
    println!("cargo:rustc-link-search=native={}", cudart_dir.display());

    // Left-to-right resolution: brew_native's runtime-API references
    // resolve from cudart_static, whose dlopen/shm_open/pthread references
    // resolve from the shared libraries after it. dl/rt/pthread are empty
    // compat shims on glibc >= 2.34 but load-bearing on older systems.
    // MSVC pulls in its own C++ runtime and has no counterparts to these,
    // so a Windows target links the two static libraries alone. The target
    // is read from cargo, since cfg! here describes the build host.
    println!("cargo:rustc-link-lib=static=brew_native");
    println!("cargo:rustc-link-lib=static=cudart_static");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=rt");
        println!("cargo:rustc-link-lib=dylib=pthread");
    }

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
