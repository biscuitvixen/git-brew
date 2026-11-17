/**
 * @file kernel.cu
 * @brief CUDA backend: grid-stride sweep of the 48-bit pad space.
 *
 * One kernel launch covers one candidate range. Each thread encodes its
 * candidate into the final block's schedule words, runs the shared 80-round
 * core, and tests digest word0 (and word1 for prefixes past 8 hex digits)
 * against the target. The midstate makes every candidate cost exactly one
 * compression; the word0/word1 masked compares make a rejected candidate
 * cost nothing beyond the rounds themselves - no state additions, no
 * digest assembly, no byte swaps.
 *
 * The whole workload is integer ALU on registers. There is deliberately no
 * shared memory, no cudaMalloc, and no device-side state beyond the two
 * result globals: host-device traffic per call is the parameter struct in
 * and 12 bytes out.
 *
 * This file holds device code only and is compiled to a fatbin rather
 * than an object. The Rust host loads that image through the driver API
 * and looks the kernel and the two globals up by name, so all three have
 * C linkage; a C++ name would carry a per-file hash on Linux and MSVC
 * decoration on Windows. SearchParams crosses the same boundary by value
 * and is mirrored by a repr(C) struct in src/gpu.rs, with the layout
 * pinned here by static_assert.
 */
#include <cstddef>
#include <cstdint>

#include "core/encode.hpp"
#include "core/sha1.hpp"

namespace {

/// How many candidates a thread hashes between polls of the found flag.
/// Bounds post-hit wasted work without measurable polling overhead.
constexpr uint32_t kPollInterval = 256;

}  // namespace

struct SearchParams {
    uint32_t ms[5];      ///< SHA-1 midstate h0..h4.
    uint64_t total_len;  ///< Commit object length in bytes; total_len % 64 == 48.
    uint32_t target0;    ///< Digest word0 target, pre-masked.
    uint32_t mask0;      ///< High-bit mask covering min(n, 8) nibbles.
    uint32_t target1;    ///< Digest word1 target, pre-masked; 0 for n <= 8.
    uint32_t mask1;      ///< High-bit mask covering nibbles 9..n; 0 for n <= 8.
    uint64_t start;      ///< Inclusive.
    uint64_t end;        ///< Exclusive.
};

static_assert(sizeof(SearchParams) == 64, "SearchParams size differs from src/gpu.rs");
static_assert(offsetof(SearchParams, total_len) == 24,
              "SearchParams::total_len offset differs from src/gpu.rs");
static_assert(offsetof(SearchParams, target0) == 32,
              "SearchParams::target0 offset differs from src/gpu.rs");
static_assert(offsetof(SearchParams, start) == 48,
              "SearchParams::start offset differs from src/gpu.rs");

extern "C" {

/// Winning candidate; the host resets it to ~0 before each launch.
__device__ unsigned long long d_result;
/// Set once a thread has claimed d_result; the host resets it to 0.
__device__ int d_found;

__global__ void vanity_kernel(SearchParams p) {
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    uint32_t poll = 0;
    for (uint64_t cand = p.start + uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         cand < p.end;
         cand += stride) {
        if ((++poll & (kPollInterval - 1)) == 0 && *(volatile int *)&d_found) {
            return;
        }
        uint32_t w[16];
        gv::encode_final_block(cand, p.total_len, w);
        uint32_t a = p.ms[0];
        uint32_t b = p.ms[1];
        uint32_t c = p.ms[2];
        uint32_t d = p.ms[3];
        uint32_t e = p.ms[4];
        gv::sha1_rounds(w, a, b, c, d, e);
        // Digest word0 = ms[0] + a, word1 = ms[1] + b. mask1 == 0 makes the
        // second compare vacuous for prefixes of 8 nibbles or fewer.
        if (((p.ms[0] + a) & p.mask0) == p.target0
            && ((p.ms[1] + b) & p.mask1) == p.target1) {
            atomicCAS(&d_result, ~0ULL, (unsigned long long)cand);
            __threadfence();
            d_found = 1;
            return;
        }
    }
}

}  // extern "C"
