//! Host-side SHA-1: midstate computation and independent verification.
//!
//! This deliberately duplicates the C++ core rather than calling it. The
//! native side and git each compute the digest their own way; a winning
//! object must satisfy this implementation, the prefix, and `git
//! hash-object` before any ref moves, so a silent divergence in any one of
//! the three implementations cannot corrupt a repository. Only whole-buffer
//! and midstate hashing exist here; throughput hashing is the GPU's job.

/// SHA-1 initial hash state H0..H4 (FIPS 180-4 section 5.3.1).
pub const INIT_STATE: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];

/// One SHA-1 compression: folds a 64-byte block into the running state.
pub fn compress(state: &mut [u32; 5], block: &[u8; 64]) {
    let mut w = [0u32; 16];
    for (i, word) in w.iter_mut().enumerate() {
        *word = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
    }
    let [mut a, mut b, mut c, mut d, mut e] = *state;
    for t in 0..80 {
        let wt = if t < 16 {
            w[t]
        } else {
            let x =
                (w[(t + 13) & 15] ^ w[(t + 8) & 15] ^ w[(t + 2) & 15] ^ w[t & 15]).rotate_left(1);
            w[t & 15] = x;
            x
        };
        let (f, k) = match t {
            0..=19 => (d ^ (b & (c ^ d)), 0x5A827999u32),
            20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
            40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
            _ => (b ^ c ^ d, 0xCA62C1D6),
        };
        let tmp = a
            .rotate_left(5)
            .wrapping_add(f)
            .wrapping_add(e)
            .wrapping_add(k)
            .wrapping_add(wt);
        e = d;
        d = c;
        c = b.rotate_left(30);
        b = a;
        a = tmp;
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
}

/// Compress every full 64-byte block of `data` into `state` and return the
/// number of bytes consumed. This is the midstate primitive: the caller
/// keeps the unconsumed tail for the search or for `finalize`.
pub fn update_blocks(state: &mut [u32; 5], data: &[u8]) -> usize {
    let (blocks, _tail) = data.as_chunks::<64>();
    for block in blocks {
        compress(state, block);
    }
    blocks.len() * 64
}

/// Apply FIPS 180-4 padding to the trailing partial block (`tail.len() < 64`)
/// and produce the digest. `total_len` is the whole message length in bytes.
pub fn finalize(mut state: [u32; 5], tail: &[u8], total_len: u64) -> [u8; 20] {
    assert!(
        tail.len() < 64,
        "finalize takes a partial block, got {} bytes",
        tail.len()
    );
    let mut block = [0u8; 64];
    block[..tail.len()].copy_from_slice(tail);
    block[tail.len()] = 0x80;
    if tail.len() >= 56 {
        // No room for the 8-byte length after 0x80: one extra block.
        compress(&mut state, &block);
        block = [0u8; 64];
    }
    block[56..].copy_from_slice(&(total_len * 8).to_be_bytes());
    compress(&mut state, &block);
    let mut out = [0u8; 20];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// One-shot SHA-1 of a whole buffer.
pub fn digest(data: &[u8]) -> [u8; 20] {
    let mut state = INIT_STATE;
    let consumed = update_blocks(&mut state, data);
    finalize(state, &data[consumed..], data.len() as u64)
}

/// Lowercase hex of a digest, the form git prints.
pub fn to_hex(digest: &[u8; 20]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex_digest(data: &[u8]) -> String {
        to_hex(&digest(data))
    }

    // NIST FIPS 180-4 / CAVP SHA-1 vectors, as listed in RFC 3174 section
    // 7.3; the same fixtures as native/tests/test_sha1.cpp so the two
    // implementations are pinned to identical references.
    #[test]
    fn nist_vectors_match() {
        for (msg, expected) in [
            (&b""[..], "da39a3ee5e6b4b0d3255bfef95601890afd80709"),
            (&b"abc"[..], "a9993e364706816aba3e25717850c26c9cd0d89d"),
            (
                &b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"[..],
                "84983e441c3bd26ebaae4aa1f95129e5e54670f1",
            ),
        ] {
            assert_eq!(hex_digest(msg), expected, "digest mismatch for {msg:?}");
        }
        assert_eq!(
            hex_digest(&vec![b'a'; 1_000_000]),
            "34aa973cd4c4daa4f61eeb2bdbad27316534016f",
            "digest mismatch for one million 'a'"
        );
    }

    // Exact lengths around the padding boundary. Golden digests from GNU
    // coreutils sha1sum 9.4: printf 'a%.0s' {1..N} | sha1sum
    #[test]
    fn padding_boundaries_match_sha1sum() {
        for (n, expected) in [
            (55, "c1c8bbdc22796e28c0e15163d20899b65621d65a"),
            (56, "c2db330f6083854c99d4b5bfb6e8f29f201be699"),
            (63, "03f09f5b158a7a8cdad920bddc29b81c18a551f5"),
            (64, "0098ba824b5c16427bd7a1122a5a442a25ec644d"),
            (65, "11655326c708d70319be2610e8a57d9a5b959d3b"),
        ] {
            assert_eq!(
                hex_digest(&vec![b'a'; n]),
                expected,
                "digest mismatch for {n} bytes of 'a'"
            );
        }
    }

    // The same commit fixture as the C++ tests; provenance is recorded in
    // native/tests/test_sha1.cpp.
    #[test]
    fn git_commit_oid_matches_wrapper_plus_content() {
        let content = b"tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904\n\
            author Test Author <author@example.com> 1700000000 +0000\n\
            committer Test Committer <committer@example.com> 1700000000 +0000\n\
            \n\
            Fixture commit for git-brew tests\n";
        let mut object = format!("commit {}\0", content.len()).into_bytes();
        object.extend_from_slice(content);
        assert_eq!(
            hex_digest(&object),
            "34405e61dc9daaf8c8b30c66f0fd25939b8c46e0",
            "commit oid mismatch"
        );
    }

    #[test]
    fn midstate_path_agrees_with_one_shot_digest() {
        let msg: Vec<u8> = (0..200u8).collect();
        let mut state = INIT_STATE;
        let consumed = update_blocks(&mut state, &msg);
        assert_eq!(consumed, 192, "expected three full blocks consumed");
        assert_eq!(
            finalize(state, &msg[consumed..], msg.len() as u64),
            digest(&msg),
            "midstate digest diverges from one-shot digest"
        );
    }
}
