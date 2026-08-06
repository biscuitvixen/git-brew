/**
 * @file encode.hpp
 * @brief Maps a 48-bit search candidate onto the final SHA-1 block.
 *
 * The host lays out the commit object so that its last 48 bytes are a run
 * of whitespace starting exactly on a 64-byte block boundary: bytes 0-47
 * of the final block are the mutable pad characters, byte 48 is the 0x80
 * padding marker, and bytes 56-63 hold the big-endian bit length. Under
 * that layout every candidate differs only in schedule words W0-W11 and
 * one compression finishes the digest; W12-W15 are constants for a given
 * object length.
 *
 * Candidate bit i selects the character at block byte i: 0 is a space
 * (0x20), 1 is a tab (0x09). The Rust host mirrors this mapping when it
 * reconstructs the winning object; the two are locked together by the
 * hash-prediction check before any ref is updated.
 */
#pragma once

#include <cstdint>

#include "core/sha1.hpp"

namespace gv {

/// Width of the mutable whitespace run, in bytes == candidate bits.
constexpr int NUM_PAD_CHARS = 48;

/// Pad character for bit position i of the candidate. Branchless select:
/// 0x20 ^ 0x29 == 0x09, so masking the XOR with the bit picks space or tab.
GV_HD inline uint8_t pad_char(uint64_t candidate, int i) {
    uint32_t bit = uint32_t(candidate >> i) & 1u;
    return uint8_t(0x20u ^ (0x29u & (0u - bit)));
}

/**
 * @brief Fill the 16 schedule words of the final block for one candidate.
 *
 * @param candidate Search candidate; bits [0,48) select the pad characters.
 * @param total_len Total commit object length in bytes, including the
 *                  "commit <len>\0" wrapper. Must satisfy total_len % 64 == 48
 *                  so the pad run occupies bytes 0-47 of the final block.
 * @param w         Receives W0-W15 as big-endian words: W0-W11 from the pad
 *                  characters, W12 = 0x80000000, W13 = 0, W14-W15 = bit length.
 */
GV_HD inline void encode_final_block(uint64_t candidate,
                                     uint64_t total_len,
                                     uint32_t w[16]) {
    for (int j = 0; j < NUM_PAD_CHARS / 4; ++j) {
        w[j] = (uint32_t(pad_char(candidate, 4 * j)) << 24)
               | (uint32_t(pad_char(candidate, 4 * j + 1)) << 16)
               | (uint32_t(pad_char(candidate, 4 * j + 2)) << 8)
               | uint32_t(pad_char(candidate, 4 * j + 3));
    }
    w[12] = 0x80000000u;
    w[13] = 0;
    uint64_t bit_len = total_len * 8;
    w[14] = uint32_t(bit_len >> 32);
    w[15] = uint32_t(bit_len);
}

}  // namespace gv
