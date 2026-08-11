//! Builds the CUDA static library via CMake and tells rustc how to link it.
//!
//! The native profile is pinned to Release regardless of the cargo profile:
//! the library is the hashing hot path, and a debug build of it would make
//! `cargo build` produce a tool that is quietly hundreds of times slower.

use std::env;

/// Real SASS for every architecture CUDA 13 can target, plus sm_120 PTX for
/// whatever comes after Blackwell. Plain -real, no a/f suffix: the kernel
/// needs nothing newer than lop3 (sm_50). Minor-forward cubin compatibility
/// fills the gaps: 80-real runs on sm_87 (Orin), 100-real on sm_103,
/// 120-real on sm_121 (GB10). 110-real is listed because Thor is covered by
/// neither a neighbouring cubin nor the sm_120 PTX.
const DEFAULT_CUDA_ARCHS: &str =
    "75-real;80-real;86-real;89-real;90-real;100-real;110-real;120-real;120-virtual";

fn main() {
    // The define is passed unconditionally: CMake caches it in OUT_DIR, so
    // passing it only when the env var is set would leave a stale arch list
    // behind after an override build.
    let arch = env::var("GIT_BREW_CUDA_ARCH").unwrap_or_else(|_| DEFAULT_CUDA_ARCHS.to_string());
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
}
