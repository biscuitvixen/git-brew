//! The host search: candidates hashed on the CPU.
//!
//! This backend needs no driver and no device, so it is the one that
//! runs where the GPU cannot, and the one the device-free tests search
//! with. It is far slower than the kernel and is meant for short
//! prefixes.
//!
//! The hashing itself is in `scalar`; this module checks the arguments
//! and hands the range over. The range is swept in ascending order on
//! the calling thread, so a hit is always the lowest matching candidate
//! in the range.

mod scalar;

use std::ops::Range;

use crate::backend::{Backend, SearchArgs, SearchError};

/// Searches on the calling thread.
pub struct HostSearcher;

impl HostSearcher {
    pub fn new() -> Self {
        HostSearcher
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
        Ok(scalar::scan(midstate, &args, range))
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
}
