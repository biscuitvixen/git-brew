//! The GPU search: one kernel launch over one candidate range.
//!
//! The kernel is compiled ahead of time to a fatbin, embedded in the
//! executable, and handed to the driver when a Searcher is made. A
//! Searcher owns everything that has to outlive a single launch: a
//! reference on the device's primary context, the loaded module, and the
//! addresses of the kernel and its two result globals. The primary
//! context is the one the CUDA runtime itself uses, so scheduling and
//! synchronisation behave as they would under the runtime API.
//!
//! Every launch writes its hit through the same two device globals, so
//! two launches from one module must not overlap. `search_range` takes
//! `&mut self`, which makes an overlap a borrow error instead of a race.
//! A Searcher also holds raw driver handles, which keeps it off other
//! threads; that matters because the context is current only on the
//! thread that made it.
//!
//! Argument checking and mask building are shared with the other
//! backends and live in `crate::backend`.

mod driver;

pub use self::driver::CudaError;

use std::ffi::{c_int, c_void, CStr};
use std::mem::{offset_of, size_of};
use std::ops::Range;
use std::ptr;

use self::driver::{
    CUcontext, CUdevice, CUdeviceptr, CUfunction, CUmodule, Driver,
    DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
};
use crate::backend::{Backend, SearchArgs, SearchError};

/// Kernel launch shape, scaled off the SM count at launch. On GB10 the
/// kernel is bound by the SHA-1 round dependency chain, not occupancy or
/// memory: a warm sweep of threads in {128..512} x blocks-per-SM in
/// {16..128} holds flat at 11.1-11.3 GH/s, so shape is not a throughput
/// lever here. 256 threads is a clean warp multiple; the blocks-per-SM is
/// set high so the adaptive driver's smaller tail chunks still cover all
/// SMs rather than to raise steady-state rate.
const KERNEL_THREADS: u32 = 256;
const KERNEL_BLOCKS_PER_SM: u32 = 32;

/// First range the search loop asks for. 2^28 is small enough that even
/// a slow GPU answers in well under a second.
const START_CHUNK: u64 = 1 << 28;

/// Value of d_result while no thread has claimed a hit.
const NO_RESULT: u64 = u64::MAX;

/// Names the kernel image exports; C linkage in kernel.cu keeps them
/// unmangled.
const KERNEL_NAME: &CStr = c"vanity_kernel";
const RESULT_NAME: &CStr = c"d_result";
const FOUND_NAME: &CStr = c"d_found";

/// Wrapper that raises the alignment of the embedded image. A bare
/// `include_bytes!` is aligned to one byte, and the driver reads the
/// image's 64-bit header fields in place.
#[repr(C, align(16))]
struct Aligned<T: ?Sized>(T);

/// The fatbin built by CMakeLists.txt beside this file; build.rs supplies the path.
static KERNEL_IMAGE: &Aligned<[u8]> = &Aligned(*include_bytes!(env!("GIT_BREW_KERNEL_FATBIN")));

/// Mirror of SearchParams in kernel.cu, passed to the
/// kernel by value. The assertions below and the static_asserts beside
/// the C++ definition pin the same layout from both sides.
#[repr(C)]
#[derive(Clone, Copy)]
struct SearchParams {
    /// SHA-1 midstate h0..h4.
    ms: [u32; 5],
    /// Commit object length in bytes; total_len % 64 == 48.
    total_len: u64,
    /// Digest word0 target, pre-masked.
    target0: u32,
    /// High-bit mask covering min(n, 8) nibbles.
    mask0: u32,
    /// Digest word1 target, pre-masked; 0 for n <= 8.
    target1: u32,
    /// High-bit mask covering nibbles 9..n; 0 for n <= 8.
    mask1: u32,
    /// Inclusive.
    start: u64,
    /// Exclusive.
    end: u64,
}

const _: () = assert!(size_of::<SearchParams>() == 64);
const _: () = assert!(offset_of!(SearchParams, total_len) == 24);
const _: () = assert!(offset_of!(SearchParams, target0) == 32);
const _: () = assert!(offset_of!(SearchParams, start) == 48);

/// A device ready to run searches. See the module docs for what it owns.
pub struct Searcher {
    driver: Driver,
    device: CUdevice,
    /// Null until the image is loaded, so Drop can tell how far `new` got.
    module: CUmodule,
    kernel: CUfunction,
    result_address: CUdeviceptr,
    found_address: CUdeviceptr,
    /// Grid size of a full launch: SM count times KERNEL_BLOCKS_PER_SM.
    full_grid_blocks: u32,
}

impl Searcher {
    /// Open the driver, take device 0 and load the kernel image.
    ///
    /// Device 0 is the first device the driver enumerates, which
    /// CUDA_VISIBLE_DEVICES can reorder or restrict.
    pub fn new() -> Result<Self, SearchError> {
        let driver = Driver::load()?;
        let mut device: CUdevice = 0;
        let mut sm_count: c_int = 0;
        let mut context: CUcontext = ptr::null_mut();
        // SAFETY: every out-pointer borrows a live local of the type the
        // driver writes.
        unsafe {
            driver.check("cuInit", (driver.cu_init)(0))?;
            driver.check("cuDeviceGet", (driver.cu_device_get)(&mut device, 0))?;
            driver.check(
                "cuDeviceGetAttribute",
                (driver.cu_device_get_attribute)(
                    &mut sm_count,
                    DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    device,
                ),
            )?;
            driver.check(
                "cuDevicePrimaryCtxRetain",
                (driver.cu_device_primary_ctx_retain)(&mut context, device),
            )?;
        }

        // From here the context reference is held, and dropping the
        // Searcher on an early return releases it.
        let mut searcher = Searcher {
            driver,
            device,
            module: ptr::null_mut(),
            kernel: ptr::null_mut(),
            result_address: 0,
            found_address: 0,
            full_grid_blocks: sm_count.max(1) as u32 * KERNEL_BLOCKS_PER_SM,
        };
        let d = &searcher.driver;
        let mut result_bytes = 0usize;
        let mut found_bytes = 0usize;
        // SAFETY: the image is a complete fatbin that outlives the call;
        // the names are NUL-terminated; out-pointers borrow live fields
        // and locals.
        unsafe {
            d.check("cuCtxSetCurrent", (d.cu_ctx_set_current)(context))?;
            d.check(
                "cuModuleLoadData",
                (d.cu_module_load_data)(&mut searcher.module, KERNEL_IMAGE.0.as_ptr().cast()),
            )?;
            d.check(
                "cuModuleGetFunction",
                (d.cu_module_get_function)(
                    &mut searcher.kernel,
                    searcher.module,
                    KERNEL_NAME.as_ptr(),
                ),
            )?;
            d.check(
                "cuModuleGetGlobal",
                (d.cu_module_get_global)(
                    &mut searcher.result_address,
                    &mut result_bytes,
                    searcher.module,
                    RESULT_NAME.as_ptr(),
                ),
            )?;
            d.check(
                "cuModuleGetGlobal",
                (d.cu_module_get_global)(
                    &mut searcher.found_address,
                    &mut found_bytes,
                    searcher.module,
                    FOUND_NAME.as_ptr(),
                ),
            )?;
        }
        // The copies in search_range assume these widths.
        assert_eq!(result_bytes, size_of::<u64>(), "d_result is not 64 bits");
        assert_eq!(found_bytes, size_of::<c_int>(), "d_found is not an int");
        Ok(searcher)
    }
}

impl Backend for Searcher {
    fn search_range(
        &mut self,
        midstate: &[u32; 5],
        tail: &[u8; 64],
        prefix_nibbles: &[u8],
        range: Range<u64>,
    ) -> Result<Option<u64>, SearchError> {
        let args = SearchArgs::check(tail, prefix_nibbles, &range)?;
        let mut params = SearchParams {
            ms: *midstate,
            total_len: args.total_len,
            target0: args.target.target0,
            mask0: args.target.mask0,
            target1: args.target.target1,
            mask1: args.target.mask1,
            start: range.start,
            end: range.end,
        };
        if range.is_empty() {
            return Ok(None);
        }
        let d = &self.driver;

        // Clear the previous launch's hit.
        let no_result: u64 = NO_RESULT;
        let no_found: c_int = 0;
        // SAFETY: both copies read a live local of exactly the size given.
        unsafe {
            d.check(
                "cuMemcpyHtoD",
                (d.cu_memcpy_htod)(
                    self.result_address,
                    (&raw const no_result).cast(),
                    size_of::<u64>(),
                ),
            )?;
            d.check(
                "cuMemcpyHtoD",
                (d.cu_memcpy_htod)(
                    self.found_address,
                    (&raw const no_found).cast(),
                    size_of::<c_int>(),
                ),
            )?;
        }

        // A range smaller than a full grid gets only the blocks it can
        // fill, so no thread starts past the end.
        let blocks_needed = (range.end - range.start).div_ceil(u64::from(KERNEL_THREADS));
        let blocks = blocks_needed.min(u64::from(self.full_grid_blocks)) as u32;

        // The driver takes one pointer per kernel parameter and copies
        // each by the size recorded in the image, so the struct is read
        // during the call and need not outlive it.
        let mut kernel_params = [(&raw mut params).cast::<c_void>()];
        // SAFETY: the kernel takes a single SearchParams by value, which
        // kernel_params supplies; the launch runs on the default stream
        // and the synchronise below waits for it.
        unsafe {
            d.check(
                "cuLaunchKernel",
                (d.cu_launch_kernel)(
                    self.kernel,
                    blocks,
                    1,
                    1,
                    KERNEL_THREADS,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    kernel_params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
            )?;
            d.check("cuCtxSynchronize", (d.cu_ctx_synchronize)())?;
        }

        // Read the hit back.
        let mut result: u64 = 0;
        let mut found: c_int = 0;
        // SAFETY: both copies write a live local of exactly the size given.
        unsafe {
            d.check(
                "cuMemcpyDtoH",
                (d.cu_memcpy_dtoh)(
                    (&raw mut result).cast(),
                    self.result_address,
                    size_of::<u64>(),
                ),
            )?;
            d.check(
                "cuMemcpyDtoH",
                (d.cu_memcpy_dtoh)(
                    (&raw mut found).cast(),
                    self.found_address,
                    size_of::<c_int>(),
                ),
            )?;
        }
        Ok((found != 0 && result != NO_RESULT).then_some(result))
    }

    fn start_chunk(&self) -> u64 {
        START_CHUNK
    }
}

impl Drop for Searcher {
    fn drop(&mut self) {
        // Failures are ignored: nothing useful can be done with them
        // while tearing down, and the process is usually about to exit.
        // SAFETY: the module handle is either null or one this Searcher
        // loaded, and the context reference is the one `new` retained.
        unsafe {
            if !self.module.is_null() {
                (self.driver.cu_module_unload)(self.module);
            }
            (self.driver.cu_device_primary_ctx_release)(self.device);
        }
    }
}

/// GPU-backed tests early-return when GIT_BREW_SKIP_GPU_TESTS is set, so
/// the suite can run on machines without a device (CI build runners). The
/// default, with the variable unset, runs everything.
#[cfg(test)]
pub(crate) fn skip_gpu_tests() -> bool {
    if std::env::var_os("GIT_BREW_SKIP_GPU_TESTS").is_some() {
        eprintln!("skipping: GIT_BREW_SKIP_GPU_TESTS is set");
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::testing::{fixture, host_nibbles, next_candidate};
    use crate::host::HostSearcher;

    /// A structurally valid padded final block for a 240-byte object.
    fn smoke_tail() -> [u8; 64] {
        let mut tail = [0u8; 64];
        tail[..48].fill(b' ');
        tail[48] = 0x80;
        let bit_len: u64 = 240 * 8;
        tail[56..].copy_from_slice(&bit_len.to_be_bytes());
        tail
    }

    // The smallest end-to-end launch: image load, symbol lookup, parameter
    // passing and readback all have to work for any hit to come back.
    #[test]
    fn one_nibble_smoke_search_finds_a_candidate_on_the_gpu() {
        if skip_gpu_tests() {
            return;
        }
        // Any midstate exercises the search path; correctness against real
        // digests is pinned by the parity test.
        let midstate = [
            0x67452301u32,
            0xEFCDAB89,
            0x98BADCFE,
            0x10325476,
            0xC3D2E1F0,
        ];
        let mut searcher = Searcher::new().expect("device must initialise");
        let hit = searcher
            .search_range(&midstate, &smoke_tail(), &[0xC], 0..1 << 20)
            .expect("smoke search must not error");
        assert!(
            hit.is_some(),
            "1-nibble search over 2^20 candidates found nothing"
        );
    }

    // Host-vs-device parity through the real search path. For each sampled
    // candidate the host hashes the full object, takes its first 12 hex
    // digits as the target, and requires the GPU to recover exactly that
    // candidate from a narrow bracketing range. A 12-nibble target makes a
    // different-candidate collision within the bracket a 2^-48 event, so
    // any mismatch is a real divergence between the host SHA-1 and object
    // encoding and the device core.
    #[test]
    fn the_gpu_recovers_candidates_hashed_on_the_host() {
        if skip_gpu_tests() {
            return;
        }
        let f = fixture();
        let mut searcher = Searcher::new().expect("device must initialise");
        let mut rng = 20260806u64;
        for trial in 0..1000 {
            let candidate = next_candidate(&mut rng);
            let target = host_nibbles(&f, candidate);
            let range = candidate.saturating_sub(500)..candidate + 500;
            let hit = searcher
                .search_range(&f.midstate, &f.tail, &target, range)
                .expect("search near a known candidate must not error");
            // One divergence means the core differs; the assert stops the
            // run at the first.
            assert_eq!(
                hit,
                Some(candidate),
                "trial {trial}: GPU did not recover the host-hashed candidate"
            );
        }
    }

    // The no-hit path: a full sweep with nothing to find, and a range with
    // nothing in it, must both come back empty and not report a stale hit.
    #[test]
    fn a_target_outside_the_range_and_an_empty_range_report_no_hit() {
        if skip_gpu_tests() {
            return;
        }
        let f = fixture();
        let mut searcher = Searcher::new().expect("device must initialise");
        // A 12-nibble target from a candidate outside the searched range:
        // the odds any of the 999 range members collide are ~999 * 2^-48.
        let mut rng = 424242u64;
        let candidate = next_candidate(&mut rng);
        let target = host_nibbles(&f, candidate);
        let outside = searcher
            .search_range(
                &f.midstate,
                &f.tail,
                &target,
                candidate + 1..candidate + 1000,
            )
            .expect("search must not error");
        assert_eq!(outside, None, "target outside the range must not be found");
        let empty = searcher
            .search_range(&f.midstate, &f.tail, &target, candidate..candidate)
            .expect("empty search must not error");
        assert_eq!(empty, None, "empty range must not report a hit");
    }

    // The word0 mask at its narrowest, for every digit value: a 1-nibble
    // search must find some candidate fast, and the host must agree.
    #[test]
    fn every_one_nibble_target_is_found_and_verifies_on_the_host() {
        if skip_gpu_tests() {
            return;
        }
        let f = fixture();
        let mut searcher = Searcher::new().expect("device must initialise");
        for nibble in 0..16u8 {
            let hit = searcher
                .search_range(&f.midstate, &f.tail, &[nibble], 0..1 << 20)
                .expect("1-nibble search must not error")
                .unwrap_or_else(|| panic!("no hit for nibble {nibble:#x} in 2^20 candidates"));
            assert_eq!(
                host_nibbles(&f, hit)[0],
                nibble,
                "GPU hit {hit:#x} does not verify on the host"
            );
        }
    }

    // The two backends against each other on ranges where a hit is
    // neither certain nor absent: a 4-digit target over 2^16 candidates
    // has one match on average and none about a third of the time. The
    // GPU may return any match and the host returns the lowest, so the
    // comparison is whether each found one, and whether the host accepts
    // the GPU's candidate when searching that candidate alone.
    #[test]
    fn the_gpu_and_the_host_agree_on_which_ranges_hold_a_match() {
        if skip_gpu_tests() {
            return;
        }
        let f = fixture();
        let mut gpu = Searcher::new().expect("device must initialise");
        let mut host = HostSearcher::new();
        let mut rng = 20261117u64;
        let (mut with_match, mut without) = (0, 0);
        for trial in 0..64 {
            let start = next_candidate(&mut rng) >> 1;
            let range = start..start + (1 << 16);
            let target: [u8; 4] = std::array::from_fn(|_| (next_candidate(&mut rng) & 0xF) as u8);
            let gpu_hit = gpu
                .search_range(&f.midstate, &f.tail, &target, range.clone())
                .expect("GPU search must not error");
            let host_hit = host
                .search_range(&f.midstate, &f.tail, &target, range)
                .expect("host search must not error");
            assert_eq!(
                gpu_hit.is_some(),
                host_hit.is_some(),
                "trial {trial}: GPU found {gpu_hit:?}, host found {host_hit:?}"
            );
            if let Some(candidate) = gpu_hit {
                let confirmed = host
                    .search_range(&f.midstate, &f.tail, &target, candidate..candidate + 1)
                    .expect("host search must not error");
                assert_eq!(
                    confirmed,
                    Some(candidate),
                    "trial {trial}: the host rejects the GPU's candidate"
                );
                with_match += 1;
            } else {
                without += 1;
            }
        }
        // Both outcomes must have been exercised for the test to mean
        // anything; the seed is fixed, so this cannot flake.
        assert!(
            with_match > 0 && without > 0,
            "{with_match} ranges with a match and {without} without"
        );
    }
}
