//! Commit object surgery: pure byte manipulation, no subprocesses.
//!
//! A commit object is `commit <len>\0` followed by header lines, a blank
//! line, and the message. The surgery appends `k` constant spaces and 48
//! mutable whitespace characters after the message, choosing `k` so the
//! mutable run starts exactly on a 64-byte SHA-1 block boundary. Everything
//! before the run is then a whole number of blocks (the midstate input) and
//! the run plus the SHA-1 padding is the single block the GPU recomputes
//! per candidate.
//!
//! The pad-character mapping (candidate bit i: 0 = space, 1 = tab at byte
//! i) mirrors native/core/encode.hpp. The two sides are locked together by
//! the end-to-end check that re-hashes the reconstructed winner before any
//! ref is updated.

use std::fmt;

/// Width of the mutable whitespace run; mirrors NUM_PAD_CHARS in
/// native/core/encode.hpp.
pub const NUM_PAD_CHARS: usize = 48;

/// Upper bound (exclusive) on the constant filler run. A 128-wide window is
/// the smallest power of two guaranteed to contain an alignment solution;
/// see [`solve_filler`].
const MAX_FILLER: usize = 128;

#[derive(Debug, PartialEq, Eq)]
pub enum SurgeryError {
    /// No blank line between headers and message: not a commit object.
    NoHeaderBoundary,
    /// No filler count aligns the pad run; cannot happen for well-formed
    /// commits (see [`solve_filler`]) but reported rather than asserted.
    NoAlignment,
}

impl fmt::Display for SurgeryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SurgeryError::NoHeaderBoundary => {
                write!(f, "commit object has no header/message boundary")
            }
            SurgeryError::NoAlignment => {
                write!(f, "no filler length aligns the pad run to a block boundary")
            }
        }
    }
}

impl std::error::Error for SurgeryError {}

/// Split commit content (wrapper already stripped) at the first blank line
/// into (headers, message). The `\n\n` separator belongs to neither part.
pub fn split(content: &[u8]) -> Result<(&[u8], &[u8]), SurgeryError> {
    let pos = content
        .windows(2)
        .position(|w| w == b"\n\n")
        .ok_or(SurgeryError::NoHeaderBoundary)?;
    Ok((&content[..pos], &content[pos + 2..]))
}

/// Whether the headers carry a signature. A signed commit's signature
/// covers the whole object except the gpgsig header itself, so appending
/// padding to the message would break verification; callers must refuse.
/// Continuation lines start with a space and never match, so only real
/// header names are tested.
pub fn has_gpgsig(headers: &[u8]) -> bool {
    headers
        .split(|&b| b == b'\n')
        .any(|line| line.starts_with(b"gpgsig ") || line.starts_with(b"gpgsig-sha256 "))
}

/// Strip the trailing run of spaces and tabs from a message, so re-running
/// the tool replaces its own padding instead of growing the object. A
/// message that legitimately ends in spaces or tabs loses them; newlines
/// are never touched.
pub fn strip_trailing_padding(message: &[u8]) -> &[u8] {
    let end = message
        .iter()
        .rposition(|&b| b != b' ' && b != b'\t')
        .map_or(0, |i| i + 1);
    &message[..end]
}

/// Pad character for bit `i` of the candidate; mirrors gv::pad_char.
pub fn pad_char(candidate: u64, i: usize) -> u8 {
    if (candidate >> i) & 1 == 0 {
        b' '
    } else {
        b'\t'
    }
}

/// A commit prepared for the search: everything up to the mutable run, plus
/// the object length the wrapper already encodes.
pub struct PaddedCommit {
    /// `commit <len>\0` + headers + `\n\n` + message + filler spaces.
    /// Always a whole number of 64-byte blocks: the midstate input.
    prefix: Vec<u8>,
    /// Full object length: `prefix.len() + NUM_PAD_CHARS`.
    total_len: u64,
}

/// Choose the filler count k in [0, MAX_FILLER) that puts the pad run on a
/// block boundary, i.e. total object length L ≡ 48 (mod 64).
///
/// k is circular: it changes the content length, which can change the digit
/// count of `<len>` in the wrapper, which changes L again. The loop just
/// tries every k. Existence: L(k+1) - L(k) is 1, or 2 when the digit count
/// grows, so over a 128-wide window L sweeps a range of width >= 128
/// skipping at most one value per digit boundary; the window contains two
/// residue targets 64 apart and a realistic commit (content > 100 bytes)
/// meets at most one digit boundary, which can only skip one of them.
fn solve_filler(headers: &[u8], message: &[u8]) -> Result<(usize, u64), SurgeryError> {
    for k in 0..MAX_FILLER {
        let content_len = headers.len() + 2 + message.len() + k + NUM_PAD_CHARS;
        let wrapper_len = "commit ".len() + content_len.to_string().len() + 1;
        let total_len = wrapper_len + content_len;
        if total_len % 64 == NUM_PAD_CHARS {
            return Ok((k, total_len as u64));
        }
    }
    Err(SurgeryError::NoAlignment)
}

/// Prepare commit content (wrapper already stripped) for the search:
/// strip old padding, solve the filler, rebuild the wrapped prefix.
pub fn prepare(content: &[u8]) -> Result<PaddedCommit, SurgeryError> {
    let (headers, message) = split(content)?;
    let message = strip_trailing_padding(message);
    let (filler, total_len) = solve_filler(headers, message)?;

    let content_len = headers.len() + 2 + message.len() + filler + NUM_PAD_CHARS;
    let mut prefix = format!("commit {content_len}\0").into_bytes();
    prefix.extend_from_slice(headers);
    prefix.extend_from_slice(b"\n\n");
    prefix.extend_from_slice(message);
    prefix.resize(prefix.len() + filler, b' ');
    debug_assert_eq!(prefix.len() % 64, 0, "prefix must be whole blocks");
    debug_assert_eq!(prefix.len() as u64 + NUM_PAD_CHARS as u64, total_len);
    Ok(PaddedCommit { prefix, total_len })
}

impl PaddedCommit {
    /// The midstate input: every block before the mutable final block.
    pub fn midstate_prefix(&self) -> &[u8] {
        &self.prefix
    }

    /// Full object length in bytes, wrapper included.
    pub fn total_len(&self) -> u64 {
        self.total_len
    }

    /// The SHA-1-padded final block the FFI expects: placeholder pad
    /// characters, then 0x80, zeros, and the big-endian bit length.
    pub fn tail(&self) -> [u8; 64] {
        let mut tail = [0u8; 64];
        tail[..NUM_PAD_CHARS].fill(b' ');
        tail[NUM_PAD_CHARS] = 0x80;
        tail[56..].copy_from_slice(&(self.total_len * 8).to_be_bytes());
        tail
    }

    /// The complete wrapped object for one candidate, ready to hash or to
    /// hand to `git hash-object`.
    pub fn object_for(&self, candidate: u64) -> Vec<u8> {
        let mut object = Vec::with_capacity(self.total_len as usize);
        object.extend_from_slice(&self.prefix);
        object.extend((0..NUM_PAD_CHARS).map(|i| pad_char(candidate, i)));
        object
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha1;

    fn sample_content() -> Vec<u8> {
        b"tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904\n\
          author Test Author <author@example.com> 1700000000 +0000\n\
          committer Test Committer <committer@example.com> 1700000000 +0000\n\
          \n\
          Fixture commit for git-brew tests\n"
            .to_vec()
    }

    #[test]
    fn split_separates_headers_from_message() {
        let content = sample_content();
        let (headers, message) = split(&content).expect("sample must split");
        assert!(
            headers.starts_with(b"tree ") && headers.ends_with(b"+0000"),
            "headers are the lines before the blank line"
        );
        assert_eq!(
            message, b"Fixture commit for git-brew tests\n",
            "message is everything after the blank line"
        );
        assert_eq!(
            split(b"no blank line here"),
            Err(SurgeryError::NoHeaderBoundary),
            "content without a blank line must not split"
        );
    }

    #[test]
    fn gpgsig_detection_matches_header_names_only() {
        assert!(
            has_gpgsig(b"tree x\ngpgsig -----BEGIN PGP SIGNATURE-----\n rest"),
            "gpgsig header must be detected"
        );
        assert!(
            has_gpgsig(b"tree x\ngpgsig-sha256 -----BEGIN\n rest"),
            "gpgsig-sha256 header must be detected"
        );
        assert!(
            !has_gpgsig(b"tree x\nauthor gpgsig <g@x> 0 +0000"),
            "'gpgsig' inside another header must not match"
        );
        assert!(
            !has_gpgsig(b"tree x\n gpgsig fake continuation"),
            "continuation lines must not match"
        );
    }

    #[test]
    fn strip_removes_exactly_the_trailing_whitespace_run() {
        assert_eq!(strip_trailing_padding(b"msg\n \t  \t"), b"msg\n");
        assert_eq!(strip_trailing_padding(b"msg\n"), b"msg\n");
        assert_eq!(strip_trailing_padding(b"msg \t inner"), b"msg \t inner");
        assert_eq!(strip_trailing_padding(b"   \t"), b"");
        assert_eq!(
            strip_trailing_padding(b"ends with newline \n"),
            b"ends with newline \n",
            "a newline terminates the strippable run"
        );
    }

    #[test]
    fn pad_char_mirrors_the_native_encoder() {
        for i in 0..NUM_PAD_CHARS {
            assert_eq!(pad_char(0, i), b' ', "candidate 0 is all spaces");
            assert_eq!(
                pad_char(u64::MAX >> 16, i),
                b'\t',
                "candidate 2^48-1 is all tabs"
            );
        }
        assert_eq!(pad_char(0b100, 2), b'\t', "bit i selects byte i");
        assert_eq!(pad_char(0b100, 1), b' ', "unset bits stay spaces");
    }

    #[test]
    fn prepared_prefix_is_whole_blocks_and_length_is_aligned() {
        let padded = prepare(&sample_content()).expect("sample must prepare");
        assert_eq!(
            padded.midstate_prefix().len() % 64,
            0,
            "midstate prefix must be whole blocks"
        );
        assert_eq!(
            padded.total_len() % 64,
            NUM_PAD_CHARS as u64,
            "object length must put the pad run on a block boundary"
        );
        let object = padded.object_for(0);
        assert_eq!(
            object.len() as u64,
            padded.total_len(),
            "object length mismatch"
        );
        let nul = object.iter().position(|&b| b == 0).expect("wrapper NUL");
        let claimed: usize = std::str::from_utf8(&object[7..nul])
            .expect("wrapper length is ASCII")
            .parse()
            .expect("wrapper length parses");
        assert_eq!(
            claimed,
            object.len() - nul - 1,
            "wrapper <len> must match the content length"
        );
    }

    // The invariants must hold across wrapper digit-count boundaries. The
    // sweep covers content lengths whose padded totals cross 999 -> 1000,
    // where <len> gains a digit and the alignment solve goes circular.
    #[test]
    fn every_content_length_from_100_to_1200_prepares_cleanly() {
        for n in 100..1200usize {
            let mut content = b"a\n\n".to_vec();
            content.resize(n, b'x');
            let padded = prepare(&content)
                .unwrap_or_else(|e| panic!("content length {n} failed to prepare: {e}"));
            assert_eq!(
                padded.midstate_prefix().len() % 64,
                0,
                "unaligned prefix at content length {n}"
            );
            assert_eq!(
                padded.total_len() % 64,
                NUM_PAD_CHARS as u64,
                "unaligned total at content length {n}"
            );
        }
    }

    // Re-running the surgery on an already-padded object must reproduce it
    // byte for byte: the tool's own padding never accumulates.
    #[test]
    fn surgery_is_idempotent_on_its_own_output() {
        let first = prepare(&sample_content()).expect("first prepare");
        let object = first.object_for(0x0000_1234_5678_9ABC);
        let nul = object.iter().position(|&b| b == 0).expect("wrapper NUL");
        let second = prepare(&object[nul + 1..]).expect("second prepare");
        assert_eq!(
            first.object_for(7).as_slice(),
            second.object_for(7).as_slice(),
            "re-padding a padded object must be byte-identical"
        );
    }

    // The midstate + tail handed to the FFI must describe exactly the bytes
    // object_for() produces: hashing prefix blocks then the candidate-
    // encoded tail equals one-shot hashing the whole object.
    #[test]
    fn midstate_plus_encoded_tail_equals_one_shot_object_digest() {
        let padded = prepare(&sample_content()).expect("sample must prepare");
        for candidate in [0u64, 1, (1 << 48) - 1, 0x0000_5A5A_5A5A_5A5A] {
            let mut state = sha1::INIT_STATE;
            let consumed = sha1::update_blocks(&mut state, padded.midstate_prefix());
            assert_eq!(
                consumed,
                padded.midstate_prefix().len(),
                "midstate prefix must consume completely"
            );
            let mut final_block = padded.tail();
            for (i, byte) in final_block[..NUM_PAD_CHARS].iter_mut().enumerate() {
                *byte = pad_char(candidate, i);
            }
            sha1::compress(&mut state, &final_block);
            let mut streamed = [0u8; 20];
            for (i, word) in state.iter().enumerate() {
                streamed[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
            }
            assert_eq!(
                streamed,
                sha1::digest(&padded.object_for(candidate)),
                "streamed digest diverges for candidate {candidate:#x}"
            );
        }
    }
}
