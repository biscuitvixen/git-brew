//! The seam to the CUDA backend: one extern function, one safe wrapper.
//!
//! Everything crossing the boundary is plain old data; the native side owns
//! all CUDA state. The wrapper's only jobs are pointer/length marshalling
//! and turning the return code into a Result, so the rest of the crate
//! never sees a raw pointer.

use std::fmt;
use std::ops::Range;

/// Prefix length limit shared with the native side (12 hex digits = 48 bits,
/// the whole pad search space).
pub const MAX_PREFIX_NIBBLES: usize = 12;

#[derive(Debug, PartialEq, Eq)]
pub enum NativeError {
    /// The CUDA runtime failed: no device, driver mismatch, launch failure.
    Cuda,
    /// The wrapper passed arguments the native side rejected; a bug here,
    /// not a user error.
    BadArgs,
    /// The native side returned a code this build does not know.
    Unknown(i32),
}

impl fmt::Display for NativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NativeError::Cuda => write!(f, "CUDA runtime failure (is a GPU visible?)"),
            NativeError::BadArgs => write!(f, "internal error: bad search arguments"),
            NativeError::Unknown(rc) => write!(f, "unknown native return code {rc}"),
        }
    }
}

impl std::error::Error for NativeError {}

// Return codes and prototype from native/include/git_brew.h.
const GV_FOUND: i32 = 0;
const GV_EXHAUSTED: i32 = 1;
const GV_ERR_CUDA: i32 = -1;
const GV_ERR_BAD_ARGS: i32 = -2;

extern "C" {
    fn vanity_search(
        midstate: *const u32,
        tail: *const u8,
        tail_len: u32,
        target_nibbles: *const u8,
        num_nibbles: u32,
        start_candidate: u64,
        end_candidate: u64,
        found_candidate: *mut u64,
    ) -> i32;
}

/// Search `range` for a candidate whose digest starts with `prefix_nibbles`.
///
/// `midstate` is the SHA-1 state after every block before the final one;
/// `tail` is the fully padded final block whose first 48 bytes the search
/// overwrites. Returns `Ok(Some(candidate))` on a hit, `Ok(None)` when the
/// range is exhausted.
pub fn search_range(
    midstate: &[u32; 5],
    tail: &[u8; 64],
    prefix_nibbles: &[u8],
    range: Range<u64>,
) -> Result<Option<u64>, NativeError> {
    assert!(
        !prefix_nibbles.is_empty() && prefix_nibbles.len() <= MAX_PREFIX_NIBBLES,
        "prefix length {} outside 1..={MAX_PREFIX_NIBBLES}",
        prefix_nibbles.len()
    );
    let mut found: u64 = 0;
    // SAFETY: all pointers borrow live slices for the duration of the call;
    // lengths are passed alongside; the native side writes only *found.
    let rc = unsafe {
        vanity_search(
            midstate.as_ptr(),
            tail.as_ptr(),
            tail.len() as u32,
            prefix_nibbles.as_ptr(),
            prefix_nibbles.len() as u32,
            range.start,
            range.end,
            &mut found,
        )
    };
    match rc {
        GV_FOUND => Ok(Some(found)),
        GV_EXHAUSTED => Ok(None),
        GV_ERR_CUDA => Err(NativeError::Cuda),
        GV_ERR_BAD_ARGS => Err(NativeError::BadArgs),
        other => Err(NativeError::Unknown(other)),
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

    /// A structurally valid padded final block for a 240-byte object.
    fn smoke_tail() -> [u8; 64] {
        let mut tail = [0u8; 64];
        tail[..48].fill(b' ');
        tail[48] = 0x80;
        let bit_len: u64 = 240 * 8;
        tail[56..].copy_from_slice(&bit_len.to_be_bytes());
        tail
    }

    #[test]
    fn one_nibble_smoke_search_finds_a_candidate_on_the_gpu() {
        if skip_gpu_tests() {
            return;
        }
        // Any midstate exercises the search path; correctness against real
        // digests is pinned by the native gpu_parity test.
        let midstate = [
            0x67452301u32,
            0xEFCDAB89,
            0x98BADCFE,
            0x10325476,
            0xC3D2E1F0,
        ];
        let hit = search_range(&midstate, &smoke_tail(), &[0xC], 0..1 << 20)
            .expect("smoke search must not error");
        assert!(
            hit.is_some(),
            "1-nibble search over 2^20 candidates found nothing"
        );
    }

    #[test]
    fn malformed_tail_is_reported_as_bad_args() {
        let midstate = [0u32; 5];
        let mut tail = smoke_tail();
        tail[48] = 0; // No 0x80 marker.
        let err = search_range(&midstate, &tail, &[0], 0..10)
            .expect_err("malformed tail must be rejected");
        assert_eq!(err, NativeError::BadArgs, "expected BadArgs, got {err:?}");
    }
}
