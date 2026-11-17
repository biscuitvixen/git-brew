//! Chunked search driver: sweeps the candidate space one chunk at a time.
//!
//! One backend call per chunk keeps the hardware saturated while bounding
//! how long a Ctrl-C or a progress update waits on a running call. The
//! chunk size starts where the backend asks and adapts to the measured
//! call time instead of assuming a throughput, so the loop lands near the
//! target cadence on any hardware without configuration.
//!
//! The loop reaches the hardware only through the Backend trait, so it
//! holds no device state and its own logic is testable on a machine with
//! no GPU or driver.
//!
//! Cancellation is a caller-owned AtomicBool checked between chunks; this
//! module never installs signal handlers itself, which keeps it callable
//! from tests and from whatever binary embeds it.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::backend::{Backend, SearchError};

/// Bits in the candidate space; mirrors the 48 pad characters.
pub const SPACE_BITS: u32 = 48;

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
    backend: &mut dyn Backend,
    midstate: &[u32; 5],
    tail: &[u8; 64],
    prefix_nibbles: &[u8],
    range: Range<u64>,
    interrupt: &AtomicBool,
    progress: &mut dyn FnMut(&Progress),
) -> Result<Outcome, SearchError> {
    let total = range.end - range.start;
    let started = Instant::now();
    let mut next = range.start;
    let start_chunk = backend.start_chunk();
    let mut chunk = start_chunk;
    while next < range.end {
        if interrupt.load(Ordering::Relaxed) {
            return Ok(Outcome::Interrupted);
        }
        let end = range.end.min(next + chunk);
        let chunk_started = Instant::now();
        if let Some(hit) = backend.search_range(midstate, tail, prefix_nibbles, next..end)? {
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
        } else if took > 2 * TARGET_CHUNK_TIME && chunk > start_chunk {
            chunk /= 2;
        }
    }
    Ok(Outcome::Exhausted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::testing::{fixture, host_nibbles};
    use crate::host::HostSearcher;

    /// A backend for tests in which no range may be searched at all.
    struct NeverCalled;

    impl Backend for NeverCalled {
        fn search_range(
            &mut self,
            _midstate: &[u32; 5],
            _tail: &[u8; 64],
            _prefix_nibbles: &[u8],
            range: Range<u64>,
        ) -> Result<Option<u64>, SearchError> {
            panic!("backend was asked to search {range:?}");
        }

        fn start_chunk(&self) -> u64 {
            1 << 20
        }
    }

    /// A backend that finds nothing and records every range it is given.
    struct Recording {
        ranges: Vec<Range<u64>>,
    }

    impl Backend for Recording {
        fn search_range(
            &mut self,
            _midstate: &[u32; 5],
            _tail: &[u8; 64],
            _prefix_nibbles: &[u8],
            range: Range<u64>,
        ) -> Result<Option<u64>, SearchError> {
            self.ranges.push(range);
            Ok(None)
        }

        fn start_chunk(&self) -> u64 {
            16
        }
    }

    // The ordinary successful run: a hit in the first chunk ends the
    // search, and the candidate handed back is one the verifying SHA-1
    // accepts.
    #[test]
    fn a_short_prefix_search_finds_a_verified_hit() {
        let f = fixture();
        let mut searcher = HostSearcher::new();
        let interrupt = AtomicBool::new(false);
        let mut updates = 0u32;
        let outcome = run(
            &mut searcher,
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
        assert_eq!(
            host_nibbles(&f, candidate)[0],
            0xB,
            "hit must verify on the host SHA-1"
        );
    }

    // A range with nothing to find must end as Exhausted, with the last
    // progress report covering the whole range.
    #[test]
    fn an_impossible_target_over_a_small_range_reports_exhausted() {
        let f = fixture();
        // A 12-nibble target taken from a candidate outside the range: a
        // collision inside it is a ~2^-38 event.
        let outside = 1_000_000u64;
        let target = host_nibbles(&f, outside);
        let interrupt = AtomicBool::new(false);
        let mut last_done = 0u64;
        let mut searcher = HostSearcher::new();
        let outcome = run(
            &mut searcher,
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

    // The chunks of one run must tile the range: a gap is a candidate
    // never searched, an overlap is wasted work. A backend that answers
    // instantly also makes every chunk double, starting from the size
    // the backend asked for, until the range runs out.
    #[test]
    fn chunks_tile_the_range_and_double_from_the_backends_start_size() {
        let f = fixture();
        let mut backend = Recording { ranges: Vec::new() };
        let interrupt = AtomicBool::new(false);
        let outcome = run(
            &mut backend,
            &f.midstate,
            &f.tail,
            &[0x0],
            100..1100,
            &interrupt,
            &mut |_| {},
        )
        .expect("search must not error");
        assert_eq!(outcome, Outcome::Exhausted, "nothing was there to find");
        // 16 + 32 + 64 + 128 + 256 = 496, then 504 of the next 512.
        let expected = [100..116, 116..148, 148..212, 212..340, 340..596, 596..1100];
        assert_eq!(backend.ranges, expected, "ranges handed to the backend");
    }

    // Ctrl-C before the first chunk: the backend must not be asked to
    // search anything.
    #[test]
    fn a_raised_interrupt_flag_stops_before_any_chunk_runs() {
        let f = fixture();
        let interrupt = AtomicBool::new(true);
        let outcome = run(
            &mut NeverCalled,
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
