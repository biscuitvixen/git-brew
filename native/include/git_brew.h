/**
 * @file git_brew.h
 * @brief The single FFI entry point into the native search backend.
 *
 * This prototype is the whole seam between the Rust host and the CUDA
 * backend: plain C types, no CUDA types, no git knowledge. The host
 * computes the SHA-1 midstate and the padded final block; the backend
 * sweeps a candidate range and reports the first hit.
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes for vanity_search. */
#define GV_FOUND 0
#define GV_EXHAUSTED 1
#define GV_ERR_CUDA (-1)
#define GV_ERR_BAD_ARGS (-2)

/**
 * @brief Search [start_candidate, end_candidate) for a candidate whose
 *        digest starts with the target prefix.
 *
 * @param midstate h0..h4 after compressing every block before the final one.
 * @param tail The final block, fully SHA-1 padded: bytes [0,48) are the
 *             mutable pad region (overwritten per candidate), byte 48 is
 *             0x80, bytes 49-55 are zero, bytes 56-63 hold the big-endian
 *             total bit length.
 * @param tail_len Must be 64 in v1; the parameter exists so a two-block
 *                 tail can be added without changing the seam.
 * @param target_nibbles One hex digit per byte, values 0-15, most
 *                       significant digit first.
 * @param num_nibbles Prefix length, 1 to 12.
 * @param start_candidate Inclusive; candidates use bits [0,48) only.
 * @param end_candidate Exclusive; at most 1 << 48.
 * @param found_candidate Output: receives the winning candidate iff the
 *                        return is GV_FOUND.
 * @return GV_FOUND, GV_EXHAUSTED, or a negative error code. Which hit is
 *         returned when several candidates in the range match is
 *         unspecified.
 */
int32_t vanity_search(const uint32_t midstate[5],
                      const uint8_t *tail,
                      uint32_t tail_len,
                      const uint8_t *target_nibbles,
                      uint32_t num_nibbles,
                      uint64_t start_candidate,
                      uint64_t end_candidate,
                      uint64_t *found_candidate);

#ifdef __cplusplus
}
#endif
