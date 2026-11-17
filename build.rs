//! Builds the CUDA kernel image via CMake and tells rustc where it is.
//!
//! The image is a fatbin that src/gpu.rs embeds with `include_bytes!` and
//! hands to the driver at run time, so nothing native is linked into the
//! executable and no link directives are emitted here.
//!
//! The native profile is pinned to Release regardless of the cargo profile:
//! the kernel is the hashing hot path, and a debug build of it would make
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

    // rustc-env makes the path visible to env!() in the crate, which is
    // how include_bytes! reaches a file under OUT_DIR.
    let fatbin = dst.join("share/kernel.fatbin");
    assert!(
        fatbin.is_file(),
        "native build left no kernel image at {}",
        fatbin.display()
    );
    println!(
        "cargo:rustc-env=GIT_BREW_KERNEL_FATBIN={}",
        fatbin.display()
    );

    // Watched paths name the source subdirectories individually: watching
    // native/ itself would also watch native/build, the standalone ctest
    // build tree, and trigger rebuild loops.
    println!("cargo:rerun-if-changed=native/CMakeLists.txt");
    println!("cargo:rerun-if-changed=native/core");
    println!("cargo:rerun-if-changed=native/backend");
    println!("cargo:rerun-if-env-changed=GIT_BREW_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDAARCHS");
}
