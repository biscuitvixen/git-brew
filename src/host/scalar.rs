//! The search loop in portable Rust: one candidate, one compression.
//!
//! Every candidate differs from its neighbours only in the 48 pad
//! characters, which fill schedule words 0 to 11 of the final block;
//! words 12 to 15 are the SHA-1 padding and the object's bit length, and
//! are the same for every candidate of one commit. A candidate therefore
//! costs twelve table lookups to build its block and eighty rounds to
//! hash it. A prefix of at most twelve hex digits lies wholly inside the
//! first two digest words, which are the midstate's first two words plus
//! the working variables a and b after the last round, so the other
//! three state additions and the digest bytes are never formed.
//!
//! The rounds are written out here and not shared with `crate::sha1`,
//! which re-verifies every hit before a ref moves: a mistake in one of
//! the two would then have to be repeated in the other to go unnoticed.

use std::ops::Range;

use crate::backend::SearchArgs;

/// Schedule word for four consecutive candidate bits. Bit 0 of the index
/// selects the word's most significant byte, as the pad characters are
/// laid out in block order and the words are big-endian; a clear bit is
/// a space (0x20) and a set bit a tab (0x09).
const PAD_WORDS: [u32; 16] = pad_words();

const fn pad_words() -> [u32; 16] {
    let mut words = [0u32; 16];
    let mut bits = 0;
    while bits < 16 {
        let mut byte = 0;
        while byte < 4 {
            let pad_char = if (bits >> byte) & 1 == 1 { 0x09 } else { 0x20 };
            words[bits] |= pad_char << (24 - 8 * byte);
            byte += 1;
        }
        bits += 1;
    }
    words
}

/// The 16 schedule words of the final block for `candidate`.
#[inline(always)]
pub(super) fn final_block(candidate: u64, total_len: u64) -> [u32; 16] {
    let mut w = [0u32; 16];
    // Words 0 to 11: four pad characters each, lowest candidate bits
    // first.
    for (j, word) in w.iter_mut().take(12).enumerate() {
        *word = PAD_WORDS[(candidate >> (4 * j)) as usize & 0xF];
    }
    // Word 12 opens the SHA-1 padding with the 0x80 marker, word 13 is
    // zero, and words 14 and 15 are the object length in bits.
    let bit_len = total_len * 8;
    w[12] = 0x8000_0000;
    w[14] = (bit_len >> 32) as u32;
    w[15] = bit_len as u32;
    w
}

/// One round: the working variables shift along by one and `a` takes
/// the new value (FIPS 180-4 section 6.1.2 step 3).
#[inline(always)]
fn round(v: &mut [u32; 5], f: u32, k: u32, wt: u32) {
    let [a, b, c, d, e] = *v;
    let next = a
        .rotate_left(5)
        .wrapping_add(f)
        .wrapping_add(e)
        .wrapping_add(k)
        .wrapping_add(wt);
    *v = [next, a, b.rotate_left(30), c, d];
}

/// Schedule word for round `t >= 16`, kept in a rolling window of 16
/// (FIPS 180-4 section 6.1.3).
#[inline(always)]
fn expand(w: &mut [u32; 16], t: usize) -> u32 {
    let wt = (w[(t + 13) & 15] ^ w[(t + 8) & 15] ^ w[(t + 2) & 15] ^ w[t & 15]).rotate_left(1);
    w[t & 15] = wt;
    wt
}

/// Digest words 0 and 1 of the object whose final block is `w`.
#[inline(always)]
pub(super) fn digest_head(midstate: &[u32; 5], mut w: [u32; 16]) -> (u32, u32) {
    let mut v = *midstate;
    // The four stages of twenty rounds, each with its own function of
    // b, c and d and its own constant (FIPS 180-4 sections 4.1.1 and
    // 4.2.1). Only the first sixteen rounds read the block directly.
    for t in 0..20 {
        let wt = if t < 16 { w[t] } else { expand(&mut w, t) };
        let [_, b, c, d, _] = v;
        round(&mut v, (b & c) | (!b & d), 0x5A82_7999, wt);
    }
    for t in 20..40 {
        let wt = expand(&mut w, t);
        let [_, b, c, d, _] = v;
        round(&mut v, b ^ c ^ d, 0x6ED9_EBA1, wt);
    }
    for t in 40..60 {
        let wt = expand(&mut w, t);
        let [_, b, c, d, _] = v;
        round(&mut v, (b & c) ^ (b & d) ^ (c & d), 0x8F1B_BCDC, wt);
    }
    for t in 60..80 {
        let wt = expand(&mut w, t);
        let [_, b, c, d, _] = v;
        round(&mut v, b ^ c ^ d, 0xCA62_C1D6, wt);
    }
    (
        midstate[0].wrapping_add(v[0]),
        midstate[1].wrapping_add(v[1]),
    )
}

/// Hash every candidate in `range` in ascending order and return the
/// first whose digest matches the target.
pub(super) fn scan(midstate: &[u32; 5], args: &SearchArgs, range: Range<u64>) -> Option<u64> {
    let target = &args.target;
    range.into_iter().find(|&candidate| {
        let (word0, word1) = digest_head(midstate, final_block(candidate, args.total_len));
        // mask1 is zero for prefixes of 8 digits or fewer, which makes
        // the second comparison always true.
        word0 & target.mask0 == target.target0 && word1 & target.mask1 == target.target1
    })
}
