//! What every search backend shares: the trait the search loop drives,
//! the error type, and the argument checks.
//!
//! A backend searches one candidate range per call and reports at most
//! one hit. The arguments arrive in the form the commit surgery produces
//! them (a midstate, a padded final block, a prefix as nibbles), and
//! `SearchArgs::check` turns them into what a hashing loop needs: the
//! object length that fixes the last two schedule words, and the masked
//! digest words a hit must equal. Every backend calls it first, so a
//! malformed argument is refused the same way whichever device would
//! have run the search.

use std::fmt;
use std::ops::Range;

use crate::commit::NUM_PAD_CHARS;
use crate::cuda::CudaError;

/// Prefix length limit (12 hex digits = 48 bits, the whole pad search
/// space).
pub const MAX_PREFIX_NIBBLES: usize = 12;

/// Something that can search one candidate range in a single call.
pub trait Backend {
    /// Search `range` for a candidate whose digest starts with
    /// `prefix_nibbles`.
    ///
    /// `midstate` is the SHA-1 state after every block before the final
    /// one; `tail` is the fully padded final block whose first 48 bytes
    /// the search overwrites. Returns `Ok(Some(candidate))` on a hit,
    /// `Ok(None)` when the range is exhausted. Which hit is returned when
    /// several candidates in the range match is unspecified.
    fn search_range(
        &mut self,
        midstate: &[u32; 5],
        tail: &[u8; 64],
        prefix_nibbles: &[u8],
        range: Range<u64>,
    ) -> Result<Option<u64>, SearchError>;
}

#[derive(Debug)]
pub enum SearchError {
    /// The driver is missing or one of its calls failed: no device,
    /// driver too old for the kernel image, launch failure.
    Cuda(CudaError),
    /// The caller passed arguments the search cannot run on; a bug in
    /// this crate, not a user error.
    BadArgs(&'static str),
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchError::Cuda(e) => write!(f, "{e}"),
            SearchError::BadArgs(why) => write!(f, "internal error: bad search arguments: {why}"),
        }
    }
}

impl std::error::Error for SearchError {}

impl From<CudaError> for SearchError {
    fn from(e: CudaError) -> Self {
        SearchError::Cuda(e)
    }
}

/// The digest bits a hit must match. Nibbles 0-7 of the prefix land in
/// digest word 0 and nibbles 8-11 in word 1, most significant first.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    /// Digest word 0 target, pre-masked.
    pub target0: u32,
    /// High-bit mask covering min(n, 8) nibbles.
    pub mask0: u32,
    /// Digest word 1 target, pre-masked; 0 for n <= 8.
    pub target1: u32,
    /// High-bit mask covering nibbles 9..n; 0 for n <= 8.
    pub mask1: u32,
}

/// The checked form of one `search_range` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchArgs {
    /// Commit object length in bytes; total_len % 64 == 48.
    pub total_len: u64,
    pub target: Target,
}

impl SearchArgs {
    /// Check the arguments of a `search_range` call and derive the
    /// object length and the target from them.
    ///
    /// `tail` must be a well-formed padded final block whose pad region
    /// starts at byte 0: byte 48 is 0x80, bytes 49-55 are zero, and bytes
    /// 56-63 hold the big-endian total bit length, which defines schedule
    /// words 14 and 15.
    pub fn check(
        tail: &[u8; 64],
        prefix_nibbles: &[u8],
        range: &Range<u64>,
    ) -> Result<Self, SearchError> {
        if prefix_nibbles.is_empty() || prefix_nibbles.len() > MAX_PREFIX_NIBBLES {
            return Err(SearchError::BadArgs("prefix length outside 1..=12"));
        }
        if range.start > range.end || range.end > 1 << 48 {
            return Err(SearchError::BadArgs("candidate range outside 48 bits"));
        }
        if tail[NUM_PAD_CHARS] != 0x80 {
            return Err(SearchError::BadArgs("tail has no 0x80 padding marker"));
        }
        if tail[NUM_PAD_CHARS + 1..56].iter().any(|&b| b != 0) {
            return Err(SearchError::BadArgs("tail padding is not zero"));
        }
        let bit_len = u64::from_be_bytes(tail[56..].try_into().expect("8-byte length field"));
        if bit_len % 8 != 0 || (bit_len / 8) % 64 != NUM_PAD_CHARS as u64 {
            return Err(SearchError::BadArgs(
                "tail length field does not end on the pad block",
            ));
        }

        let mut target = Target::default();
        for (i, &nibble) in prefix_nibbles.iter().enumerate() {
            if nibble > 15 {
                return Err(SearchError::BadArgs("prefix digit above 15"));
            }
            let shift = 28 - 4 * (i as u32 % 8);
            if i < 8 {
                target.target0 |= u32::from(nibble) << shift;
                target.mask0 |= 0xF << shift;
            } else {
                target.target1 |= u32::from(nibble) << shift;
                target.mask1 |= 0xF << shift;
            }
        }
        Ok(SearchArgs {
            total_len: bit_len / 8,
            target,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A structurally valid padded final block for a 240-byte object.
    fn valid_tail() -> [u8; 64] {
        let mut tail = [0u8; 64];
        tail[..48].fill(b' ');
        tail[48] = 0x80;
        let bit_len: u64 = 240 * 8;
        tail[56..].copy_from_slice(&bit_len.to_be_bytes());
        tail
    }

    // Each case is a caller bug that would make a backend hash garbage
    // or run off the candidate space, so none may reach a search.
    #[test]
    fn each_malformed_argument_is_rejected() {
        let good_tail = valid_tail();
        let mut no_marker = good_tail;
        no_marker[48] = 0;
        let mut dirty_padding = good_tail;
        dirty_padding[50] = 1;
        let mut misaligned_length = good_tail;
        misaligned_length[56..].copy_from_slice(&(241u64 * 8).to_be_bytes());

        // The range is two numbers so a reversed one can be written down.
        struct Case<'a> {
            name: &'static str,
            tail: &'a [u8; 64],
            nibbles: &'a [u8],
            start: u64,
            end: u64,
        }
        let cases = [
            Case {
                name: "zero nibbles",
                tail: &good_tail,
                nibbles: &[],
                start: 0,
                end: 10,
            },
            Case {
                name: "13 nibbles",
                tail: &good_tail,
                nibbles: &[0; 13],
                start: 0,
                end: 10,
            },
            Case {
                name: "nibble value 16",
                tail: &good_tail,
                nibbles: &[16],
                start: 0,
                end: 10,
            },
            Case {
                name: "reversed range",
                tail: &good_tail,
                nibbles: &[0],
                start: 10,
                end: 5,
            },
            Case {
                name: "range past 2^48",
                tail: &good_tail,
                nibbles: &[0],
                start: 0,
                end: (1 << 48) + 1,
            },
            Case {
                name: "tail without 0x80 marker",
                tail: &no_marker,
                nibbles: &[0],
                start: 0,
                end: 10,
            },
            Case {
                name: "tail with non-zero padding",
                tail: &dirty_padding,
                nibbles: &[0],
                start: 0,
                end: 10,
            },
            Case {
                name: "tail length off the pad block",
                tail: &misaligned_length,
                nibbles: &[0],
                start: 0,
                end: 10,
            },
        ];
        for case in cases {
            let outcome = SearchArgs::check(case.tail, case.nibbles, &(case.start..case.end));
            assert!(
                matches!(outcome, Err(SearchError::BadArgs(_))),
                "{}: expected BadArgs, got {outcome:?}",
                case.name
            );
        }
    }

    // The masks decide which digest bits a backend compares; a 9-nibble
    // prefix is the shortest that spills into word1.
    #[test]
    fn a_nine_nibble_prefix_fills_word0_and_the_top_of_word1() {
        let nibbles = [0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9];
        let args = SearchArgs::check(&valid_tail(), &nibbles, &(0..10))
            .expect("well-formed arguments must be accepted");
        assert_eq!(args.target.target0, 0x1234_5678, "word0 target");
        assert_eq!(args.target.mask0, 0xFFFF_FFFF, "word0 mask");
        assert_eq!(args.target.target1, 0x9000_0000, "word1 target");
        assert_eq!(args.target.mask1, 0xF000_0000, "word1 mask");
        assert_eq!(args.total_len, 240, "object length from the tail");
    }
}
