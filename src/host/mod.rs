//! The host search: candidates hashed on the CPU.
//!
//! This backend needs no driver and no device, so it is the one that
//! runs where the GPU cannot, and the one the device-free tests search
//! with. It is far slower than the kernel and is meant for short
//! prefixes.
//!
//! The hashing itself is in `scalar`; this module checks the arguments
//! and spreads the range over the cores. The range is cut into blocks of
//! BLOCK candidates, and each thread claims the next unclaimed block
//! from a shared counter until none are left, which keeps every core
//! busy however unevenly they run. The lowest hit so far is shared too.
//! Blocks are claimed in ascending order, so when a hit is recorded
//! every block below it is already claimed; those are searched to the
//! end, and no thread starts a block above it. The result is therefore
//! the lowest matching candidate in the range whatever the thread count
//! or timing, the same answer a single thread walking the range gives.
//!
//! Threads are started for each call and joined before it returns. A
//! call lasts a few hundred milliseconds once the search loop has found
//! its pace, against which starting them is not measurable.

mod scalar;

use std::num::NonZero;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use crate::backend::{Backend, SearchArgs, SearchError};

/// Candidates a thread hashes per claim. Small enough that the work
/// done past a hit is a few blocks, large enough that the shared
/// counter is touched a few thousand times a second per thread.
const BLOCK: u64 = 1 << 12;

/// First range the search loop asks for: about a tenth of a second on
/// a single slow core.
const START_CHUNK: u64 = 1 << 20;

/// Value of the shared lowest hit while no thread has found one. No
/// candidate reaches it, as candidates are below 2^48.
const NO_HIT: u64 = u64::MAX;

/// Searches on a fixed number of threads.
pub struct HostSearcher {
    threads: usize,
}

impl HostSearcher {
    /// A searcher using every core the process may run on.
    pub fn new() -> Self {
        Self::with_threads(thread::available_parallelism().map_or(1, NonZero::get))
    }

    /// A searcher using `threads` threads, at least one.
    pub fn with_threads(threads: usize) -> Self {
        HostSearcher {
            threads: threads.max(1),
        }
    }
}

impl Backend for HostSearcher {
    fn search_range(
        &mut self,
        midstate: &[u32; 5],
        tail: &[u8; 64],
        prefix_nibbles: &[u8],
        range: Range<u64>,
    ) -> Result<Option<u64>, SearchError> {
        let args = SearchArgs::check(tail, prefix_nibbles, &range)?;
        let blocks = (range.end - range.start).div_ceil(BLOCK);
        let next_block = AtomicU64::new(0);
        let lowest_hit = AtomicU64::new(NO_HIT);

        // What one thread does: claim blocks in turn until the range or
        // the need for it runs out.
        let work = || loop {
            let block = next_block.fetch_add(1, Ordering::Relaxed);
            if block >= blocks {
                break;
            }
            let start = range.start + block * BLOCK;
            // A block that starts above a recorded hit cannot hold a
            // lower one, and neither can any block claimed after it.
            if start > lowest_hit.load(Ordering::Relaxed) {
                break;
            }
            let end = range.end.min(start + BLOCK);
            if let Some(hit) = scalar::scan(midstate, &args, start..end) {
                lowest_hit.fetch_min(hit, Ordering::Relaxed);
                break;
            }
        };

        // No more threads than blocks; the calling thread is one of
        // them, so a range of a single block starts none.
        let extra_threads = (self.threads as u64).min(blocks).saturating_sub(1);
        thread::scope(|scope| {
            for _ in 0..extra_threads {
                scope.spawn(work);
            }
            work();
        });

        let hit = lowest_hit.into_inner();
        Ok((hit != NO_HIT).then_some(hit))
    }

    fn start_chunk(&self) -> u64 {
        START_CHUNK
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::testing::{fixture, host_nibbles, next_candidate};
    use crate::sha1;

    /// Candidates that stress the encoder: no tabs, all tabs, one tab at
    /// each end, and alternating patterns that put a different bit in
    /// every byte position of a word.
    const EDGE_CANDIDATES: [u64; 6] = [
        0,
        (1 << 48) - 1,
        1,
        1 << 47,
        0x5555_5555_5555,
        0x1234_5678_9ABC,
    ];

    // The block the search hashes must be the block the commit surgery
    // would write for the same candidate, or a hit names an object that
    // does not have the digest it was found with.
    #[test]
    fn the_final_block_matches_the_object_the_surgery_builds() {
        let f = fixture();
        let args = SearchArgs::check(&f.tail, &[0], &(0..1)).expect("fixture tail is valid");
        for candidate in EDGE_CANDIDATES {
            // The surgery's own final block: the object's last 48 bytes,
            // then the padding and length from the prepared tail.
            let object = f.padded.object_for(candidate);
            let mut block = f.tail;
            block[..48].copy_from_slice(&object[object.len() - 48..]);
            let expected: [u32; 16] = std::array::from_fn(|i| {
                u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().expect("4-byte word"))
            });
            assert_eq!(
                scalar::final_block(candidate, args.total_len),
                expected,
                "schedule words for candidate {candidate:#x}"
            );
        }
    }

    // The rounds are a second SHA-1 implementation in this crate; they
    // must agree with the one that verifies hits, on the two words the
    // search compares.
    #[test]
    fn digest_words_match_the_verifying_sha1() {
        let f = fixture();
        let args = SearchArgs::check(&f.tail, &[0], &(0..1)).expect("fixture tail is valid");
        let mut rng = 20261117u64;
        let sampled = std::iter::repeat_with(|| next_candidate(&mut rng)).take(1000);
        for candidate in EDGE_CANDIDATES.into_iter().chain(sampled) {
            let digest = sha1::digest(&f.padded.object_for(candidate));
            let expected = (
                u32::from_be_bytes(digest[0..4].try_into().expect("4-byte word")),
                u32::from_be_bytes(digest[4..8].try_into().expect("4-byte word")),
            );
            let block = scalar::final_block(candidate, args.total_len);
            assert_eq!(
                scalar::digest_head(&f.midstate, block),
                expected,
                "digest words 0 and 1 for candidate {candidate:#x}"
            );
        }
    }

    // The same check the GPU passes, through the Backend interface: the
    // verifying side hashes a candidate, its first 12 hex digits become
    // the target, and the search must recover exactly that candidate
    // from a narrow bracket. This covers the masks and the range bounds
    // as well as the hashing.
    #[test]
    fn the_search_recovers_candidates_hashed_by_the_verifying_sha1() {
        let f = fixture();
        let mut searcher = HostSearcher::new();
        let mut rng = 20260806u64;
        for trial in 0..1000 {
            let candidate = next_candidate(&mut rng);
            let target = host_nibbles(&f, candidate);
            let range = candidate.saturating_sub(500)..candidate + 500;
            let hit = searcher
                .search_range(&f.midstate, &f.tail, &target, range)
                .expect("search near a known candidate must not error");
            assert_eq!(
                hit,
                Some(candidate),
                "trial {trial}: the search did not recover the candidate"
            );
        }
    }

    // The range is half-open: its last member must be searched and its
    // end must not be, or adjacent chunks would skip or repeat a
    // candidate. An empty range holds nothing to find.
    #[test]
    fn the_range_end_is_excluded_and_an_empty_range_reports_no_hit() {
        let f = fixture();
        let mut searcher = HostSearcher::new();
        let mut rng = 424242u64;
        let candidate = next_candidate(&mut rng);
        let target = host_nibbles(&f, candidate);
        let mut search = |range: Range<u64>| {
            searcher
                .search_range(&f.midstate, &f.tail, &target, range)
                .expect("search must not error")
        };
        assert_eq!(
            search(candidate - 999..candidate + 1),
            Some(candidate),
            "the last candidate of a range must be searched"
        );
        assert_eq!(
            search(candidate - 999..candidate),
            None,
            "the end of a range must not be searched"
        );
        assert_eq!(
            search(candidate..candidate),
            None,
            "an empty range must not report a hit"
        );
    }

    // With many matches in the range, the hit is the lowest one. The
    // expected value comes from hashing each candidate in turn with the
    // verifying SHA-1, for every value of a one-digit prefix.
    #[test]
    fn a_one_digit_search_returns_the_lowest_match_for_every_digit() {
        let f = fixture();
        let mut searcher = HostSearcher::new();
        for nibble in 0..16u8 {
            let lowest = (0..1u64 << 12).find(|&c| host_nibbles(&f, c)[0] == nibble);
            let hit = searcher
                .search_range(&f.midstate, &f.tail, &[nibble], 0..1 << 12)
                .expect("search must not error");
            assert_eq!(hit, lowest, "lowest match for prefix digit {nibble:x}");
        }
    }

    // A malformed argument is a caller bug and must come back as an
    // error, not as a search over a block built from it.
    #[test]
    fn malformed_arguments_are_refused() {
        let f = fixture();
        let outcome = HostSearcher::new().search_range(&f.midstate, &f.tail, &[], 0..10);
        assert!(
            matches!(outcome, Err(SearchError::BadArgs(_))),
            "an empty prefix must be refused, got {outcome:?}"
        );
    }

    // The thread count must not change the answer. A 3-digit target has
    // a match every 4096 candidates on average, so a range of 2^18 holds
    // dozens spread over many blocks, and the lowest is rarely in the
    // first block claimed. The range starts and ends off a block
    // boundary so the short last block is searched too.
    #[test]
    fn every_thread_count_returns_the_single_threaded_answer() {
        let f = fixture();
        let mut rng = 7u64;
        for trial in 0..16 {
            let start = next_candidate(&mut rng) >> 1;
            let range = start..start + (1 << 18) + 777;
            let target: [u8; 3] = std::array::from_fn(|_| (next_candidate(&mut rng) & 0xF) as u8);
            let search = |threads: usize| {
                HostSearcher::with_threads(threads)
                    .search_range(&f.midstate, &f.tail, &target, range.clone())
                    .expect("search must not error")
            };
            let single = search(1);
            assert!(
                single.is_some(),
                "trial {trial}: 2^18 candidates held no match"
            );
            for threads in [2, 3, 8, 64] {
                assert_eq!(
                    search(threads),
                    single,
                    "trial {trial}: {threads} threads against one"
                );
            }
        }
    }

    // A match in the very last candidate of a range that is not a whole
    // number of blocks: the short final block must be claimed and
    // searched to its end.
    #[test]
    fn a_match_at_the_end_of_a_ragged_range_is_found() {
        let f = fixture();
        let mut rng = 99u64;
        let candidate = next_candidate(&mut rng) | (1 << 40);
        let target = host_nibbles(&f, candidate);
        let range = candidate - 5 * BLOCK - 123..candidate + 1;
        let hit = HostSearcher::with_threads(4)
            .search_range(&f.midstate, &f.tail, &target, range)
            .expect("search must not error");
        assert_eq!(hit, Some(candidate), "match in the last candidate");
    }
}
