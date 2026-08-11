//! Chunked search driver: sweeps the candidate space one FFI call at a time.
//!
//! One kernel launch per chunk keeps the GPU saturated while bounding how
//! long a Ctrl-C or a progress update waits on a running launch. The chunk
//! size adapts to the measured launch time instead of assuming a
//! throughput, so the loop lands near the target cadence on any GPU
//! without configuration.
//!
//! Cancellation is a caller-owned AtomicBool checked between chunks; this
//! module never installs signal handlers itself, which keeps it callable
//! from tests and from whatever binary embeds it.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::ffi;

/// Bits in the candidate space; mirrors the 48 pad characters.
pub const SPACE_BITS: u32 = 48;

/// First chunk size. 2^28 is small enough that even a slow GPU answers in
/// well under a second, so the adaptive loop starts responsive and grows.
const START_CHUNK: u64 = 1 << 28;

/// Chunk growth stops here: past 2^34 a single call could hold the GPU for
/// many seconds on a mid-range card, defeating cancellation.
const MAX_CHUNK: u64 = 1 << 34;

/// Chunk duration the loop steers toward.
const TARGET_CHUNK_TIME: Duration = Duration::from_millis(400);

/// Snapshot handed to the progress callback after every chunk.
pub struct Progress {
    /// Candidates swept so far, including the chunk just finished.
    pub done: u64,
    /// Size of the requested range.
    pub total: u64,
    /// Hashes per second over the chunk just finished.
    pub rate: f64,
    /// Wall time since the search started.
    pub elapsed: Duration,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// A candidate whose digest starts with the prefix.
    Found(u64),
    /// The whole range was swept without a hit.
    Exhausted,
    /// The interrupt flag was raised between chunks.
    Interrupted,
}

/// Sweep `range` for a candidate matching `prefix_nibbles`.
///
/// `interrupt` is checked between chunks; `progress` runs after every chunk.
pub fn run(
    midstate: &[u32; 5],
    tail: &[u8; 64],
    prefix_nibbles: &[u8],
    range: Range<u64>,
    interrupt: &AtomicBool,
    progress: &mut dyn FnMut(&Progress),
) -> Result<Outcome, ffi::NativeError> {
    let total = range.end - range.start;
    let started = Instant::now();
    let mut next = range.start;
    let mut chunk = START_CHUNK;
    while next < range.end {
        if interrupt.load(Ordering::Relaxed) {
            return Ok(Outcome::Interrupted);
        }
        let end = range.end.min(next + chunk);
        let chunk_started = Instant::now();
        if let Some(hit) = ffi::search_range(midstate, tail, prefix_nibbles, next..end)? {
            return Ok(Outcome::Found(hit));
        }
        let took = chunk_started.elapsed();
        let chunk_len = end - next;
        next = end;
        progress(&Progress {
            done: next - range.start,
            total,
            rate: chunk_len as f64 / took.as_secs_f64().max(1e-9),
            elapsed: started.elapsed(),
        });
        // Steer the next chunk toward the target cadence: double while
        // fast, halve when a chunk overshoots badly.
        if took < TARGET_CHUNK_TIME && chunk < MAX_CHUNK {
            chunk *= 2;
        } else if took > 2 * TARGET_CHUNK_TIME && chunk > START_CHUNK {
            chunk /= 2;
        }
    }
    Ok(Outcome::Exhausted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit;
    use crate::sha1;

    struct Fixture {
        midstate: [u32; 5],
        tail: [u8; 64],
        padded: commit::PaddedCommit,
    }

    fn fixture() -> Fixture {
        let content = b"tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904\n\
            author Test Author <author@example.com> 1700000000 +0000\n\
            committer Test Committer <committer@example.com> 1700000000 +0000\n\
            \n\
            Search driver fixture\n";
        let padded = commit::prepare(content).expect("fixture must prepare");
        let mut midstate = sha1::INIT_STATE;
        sha1::update_blocks(&mut midstate, padded.midstate_prefix());
        Fixture {
            midstate,
            tail: padded.tail(),
            padded,
        }
    }

    fn digest_nibbles(digest: &[u8; 20], n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    digest[i / 2] >> 4
                } else {
                    digest[i / 2] & 0xF
                }
            })
            .collect()
    }

    #[test]
    fn a_short_prefix_search_finds_a_verified_hit() {
        if crate::ffi::skip_gpu_tests() {
            return;
        }
        let f = fixture();
        let interrupt = AtomicBool::new(false);
        let mut updates = 0u32;
        let outcome = run(
            &f.midstate,
            &f.tail,
            &[0xB],
            0..1 << SPACE_BITS,
            &interrupt,
            &mut |_| updates += 1,
        )
        .expect("search must not error");
        let Outcome::Found(candidate) = outcome else {
            panic!("1-nibble search must find a hit, got {outcome:?}");
        };
        let digest = sha1::digest(&f.padded.object_for(candidate));
        assert_eq!(digest[0] >> 4, 0xB, "hit must verify on the host SHA-1");
    }

    #[test]
    fn an_impossible_target_over_a_small_range_reports_exhausted() {
        if crate::ffi::skip_gpu_tests() {
            return;
        }
        let f = fixture();
        // A 12-nibble target taken from a candidate outside the range: a
        // collision inside it is a ~2^-38 event.
        let outside = 1_000_000u64;
        let digest = sha1::digest(&f.padded.object_for(outside));
        let target = digest_nibbles(&digest, 12);
        let interrupt = AtomicBool::new(false);
        let mut last_done = 0u64;
        let outcome = run(
            &f.midstate,
            &f.tail,
            &target,
            0..1024,
            &interrupt,
            &mut |p| {
                assert!(p.done > last_done, "progress must advance monotonically");
                last_done = p.done;
            },
        )
        .expect("search must not error");
        assert_eq!(outcome, Outcome::Exhausted, "range without the target");
        assert_eq!(last_done, 1024, "progress must end at the range size");
    }

    #[test]
    fn a_raised_interrupt_flag_stops_before_any_chunk_runs() {
        let f = fixture();
        let interrupt = AtomicBool::new(true);
        let outcome = run(
            &f.midstate,
            &f.tail,
            &[0x0],
            0..1 << SPACE_BITS,
            &interrupt,
            &mut |_| panic!("no chunk may complete after an interrupt"),
        )
        .expect("search must not error");
        assert_eq!(
            outcome,
            Outcome::Interrupted,
            "pre-raised flag must stop the run"
        );
    }
}
