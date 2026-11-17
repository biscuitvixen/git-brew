/**
 * @file sha1.hpp
 * @brief Portable SHA-1 core shared between the CPU baseline and the GPU kernel.
 *
 * Implements the FIPS PUB 180-4 SHA-1 compression function over a plain
 * uint32_t[16] big-endian word view of a 64-byte block. Every function is
 * usable from both host and device code behind the GV_HD shim, so the CPU
 * reference path and the CUDA kernel share one implementation by
 * construction. The three-input round functions route through a single
 * lop3.b32 on device (truth-table immediates 0xCA, 0xE8, 0x96) and degrade
 * to plain bitwise expressions everywhere else.
 *
 * SHA-1 is cryptographically broken and used here purely as git's content
 * address. Only single-threaded, whole-buffer hashing is provided on the
 * host side; throughput hashing lives in the CUDA backend.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define GV_HD __host__ __device__
#define GV_UNROLL _Pragma("unroll")
#else
#define GV_HD
#define GV_UNROLL
#endif

namespace gv {

/// SHA-1 initial hash state H0..H4 (FIPS 180-4 §5.3.1).
inline constexpr uint32_t kSha1InitState[5] = {0x67452301u,
                                               0xEFCDAB89u,
                                               0x98BADCFEu,
                                               0x10325476u,
                                               0xC3D2E1F0u};

/// Rotate x left by n bits; n must be in [1, 31] so neither shift is by 32.
GV_HD inline uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

/// Ch(x,y,z) = (x & y) | (~x & z); one lop3.b32 with immediate 0xCA on device.
GV_HD inline uint32_t sha1_ch(uint32_t x, uint32_t y, uint32_t z) {
#if defined(__CUDA_ARCH__)
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(r) : "r"(x), "r"(y), "r"(z));
    return r;
#else
    return z ^ (x & (y ^ z));
#endif
}

/// Maj(x,y,z) = (x & y) | (x & z) | (y & z); lop3.b32 immediate 0xE8 on device.
GV_HD inline uint32_t sha1_maj(uint32_t x, uint32_t y, uint32_t z) {
#if defined(__CUDA_ARCH__)
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(r) : "r"(x), "r"(y), "r"(z));
    return r;
#else
    return (x & y) | (x & z) | (y & z);
#endif
}

/// Parity(x,y,z) = x ^ y ^ z; lop3.b32 immediate 0x96 on device.
GV_HD inline uint32_t sha1_parity(uint32_t x, uint32_t y, uint32_t z) {
#if defined(__CUDA_ARCH__)
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(r) : "r"(x), "r"(y), "r"(z));
    return r;
#else
    return x ^ y ^ z;
#endif
}

/**
 * @brief The 80 SHA-1 rounds, without the final state addition.
 *
 * Exposed separately so the search kernel can run the rounds and test the
 * digest's first words directly: word0 of the digest is h0 + a and word1 is
 * h1 + b, so a prefix check needs neither the remaining three additions nor
 * any digest assembly.
 *
 * @param w Block schedule words W0-W15; clobbered by the rolling schedule.
 * @param a Working variable, seeded with h0; holds a_80 on return.
 * @param b Working variable, seeded with h1; holds a_79 on return.
 * @param c Working variable, seeded with h2.
 * @param d Working variable, seeded with h3.
 * @param e Working variable, seeded with h4.
 */
GV_HD inline void sha1_rounds(uint32_t w[16],
                              uint32_t &a,
                              uint32_t &b,
                              uint32_t &c,
                              uint32_t &d,
                              uint32_t &e) {
    // Rolling 16-word schedule: W[t] for t >= 16 overwrites slot t & 15,
    // trading the 80-word array for 16 registers (FIPS 180-4 §6.1.3 step 1).
    GV_UNROLL
    for (int t = 0; t < 80; ++t) {
        uint32_t wt;
        if (t < 16) {
            wt = w[t];
        } else {
            wt = rotl32(
              w[(t + 13) & 15] ^ w[(t + 8) & 15] ^ w[(t + 2) & 15] ^ w[t & 15], 1);
            w[t & 15] = wt;
        }
        // Round constants K per FIPS 180-4 §4.2.1; kept literal in the
        // branches so device code needs no reference to host-side storage.
        uint32_t f;
        uint32_t k;
        if (t < 20) {
            f = sha1_ch(b, c, d);
            k = 0x5A827999u;
        } else if (t < 40) {
            f = sha1_parity(b, c, d);
            k = 0x6ED9EBA1u;
        } else if (t < 60) {
            f = sha1_maj(b, c, d);
            k = 0x8F1BBCDCu;
        } else {
            f = sha1_parity(b, c, d);
            k = 0xCA62C1D6u;
        }
        uint32_t tmp = rotl32(a, 5) + f + e + k + wt;
        e = d;
        d = c;
        c = rotl32(b, 30);
        b = a;
        a = tmp;
    }
}

/**
 * @brief One SHA-1 compression: folds a 64-byte block into the running state.
 *
 * @param state Running hash state h0..h4, updated in place.
 * @param w_in  The block as 16 big-endian 32-bit words (FIPS 180-4 §6.1.2).
 */
GV_HD inline void sha1_compress(uint32_t state[5], const uint32_t w_in[16]) {
    uint32_t w[16];
    for (int i = 0; i < 16; ++i) {
        w[i] = w_in[i];
    }
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    sha1_rounds(w, a, b, c, d, e);
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
}

/// Load a big-endian 32-bit word (FIPS 180-4 §3.1 byte ordering).
GV_HD inline uint32_t load_be32(const uint8_t *p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8)
           | uint32_t(p[3]);
}

/// Store a 32-bit word as four big-endian bytes.
GV_HD inline void store_be32(uint8_t *p, uint32_t x) {
    p[0] = uint8_t(x >> 24);
    p[1] = uint8_t(x >> 16);
    p[2] = uint8_t(x >> 8);
    p[3] = uint8_t(x);
}

/**
 * @brief Compress all full 64-byte blocks of a buffer into the state.
 *
 * Leaves any trailing partial block untouched; the caller finishes with
 * sha1_final. This is the midstate primitive: hashing everything up to the
 * final block once, so a search only recomputes the last compression.
 *
 * @param state Running hash state, updated in place.
 * @param data  Input buffer.
 * @param len   Buffer length in bytes.
 * @return Number of bytes consumed (largest multiple of 64 <= len).
 */
inline size_t sha1_update_blocks(uint32_t state[5], const uint8_t *data, size_t len) {
    size_t consumed = 0;
    uint32_t w[16];
    while (len - consumed >= 64) {
        for (int i = 0; i < 16; ++i) {
            w[i] = load_be32(data + consumed + size_t(i) * 4);
        }
        sha1_compress(state, w);
        consumed += 64;
    }
    return consumed;
}

/**
 * @brief Apply FIPS 180-4 §5.1.1 padding to the trailing partial block and
 *        finish the digest.
 *
 * @param state     Running hash state after all full blocks; consumed.
 * @param tail      Remaining input bytes (fewer than 64).
 * @param tail_len  Length of tail in bytes.
 * @param total_len Total message length in bytes (for the 64-bit bit count).
 * @param out       Receives the 20-byte big-endian digest.
 */
inline void sha1_final(uint32_t state[5],
                       const uint8_t *tail,
                       size_t tail_len,
                       uint64_t total_len,
                       uint8_t out[20]) {
    uint8_t block[64] = {0};
    for (size_t i = 0; i < tail_len; ++i) {
        block[i] = tail[i];
    }
    block[tail_len] = 0x80;
    uint32_t w[16];
    if (tail_len >= 56) {
        // No room for the 8-byte length after 0x80: one extra block.
        for (int i = 0; i < 16; ++i) {
            w[i] = load_be32(block + size_t(i) * 4);
        }
        sha1_compress(state, w);
        for (int i = 0; i < 64; ++i) {
            block[i] = 0;
        }
    }
    uint64_t bit_len = total_len * 8;
    store_be32(block + 56, uint32_t(bit_len >> 32));
    store_be32(block + 60, uint32_t(bit_len));
    for (int i = 0; i < 16; ++i) {
        w[i] = load_be32(block + size_t(i) * 4);
    }
    sha1_compress(state, w);
    for (int i = 0; i < 5; ++i) {
        store_be32(out + size_t(i) * 4, state[i]);
    }
}

/**
 * @brief One-shot SHA-1 of a whole buffer.
 *
 * @param data Input buffer.
 * @param len  Length in bytes.
 * @param out  Receives the 20-byte big-endian digest.
 */
inline void sha1_hash(const uint8_t *data, size_t len, uint8_t out[20]) {
    uint32_t state[5];
    for (int i = 0; i < 5; ++i) {
        state[i] = kSha1InitState[i];
    }
    size_t consumed = sha1_update_blocks(state, data, len);
    sha1_final(state, data + consumed, len - consumed, len, out);
}

}  // namespace gv
