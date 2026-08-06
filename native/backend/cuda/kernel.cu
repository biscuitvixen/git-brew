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
 */
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>

#include "core/encode.hpp"
#include "core/sha1.hpp"
#include "git_brew.h"

namespace {

/// Kernel launch shape, scaled off the SM count at launch. On GB10 the
/// kernel is bound by the SHA-1 round dependency chain, not occupancy or
/// memory: a warm sweep of threads in {128..512} x blocks-per-SM in
/// {16..128} holds flat at 11.1-11.3 GH/s, so shape is not a throughput
/// lever here. 256 threads is a clean warp multiple; the blocks-per-SM is
/// set high so the adaptive driver's smaller tail chunks still cover all
/// SMs rather than to raise steady-state rate.
constexpr uint32_t KERNEL_THREADS = 256;
constexpr uint32_t KERNEL_BLOCKS_PER_SM = 32;

/// How many candidates a thread hashes between polls of the found flag.
/// Bounds post-hit wasted work without measurable polling overhead.
constexpr uint32_t kPollInterval = 256;

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

__device__ unsigned long long d_result;
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

/// SM count of the current device, queried once. -1 when CUDA is unusable.
int device_sm_count() {
    static int sm_count = [] {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            return -1;
        }
        int sms = 0;
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device)
            != cudaSuccess) {
            return -1;
        }
        return sms;
    }();
    return sm_count;
}

}  // namespace

extern "C" int32_t vanity_search(const uint32_t midstate[5],
                                 const uint8_t *tail,
                                 uint32_t tail_len,
                                 const uint8_t *target_nibbles,
                                 uint32_t num_nibbles,
                                 uint64_t start_candidate,
                                 uint64_t end_candidate,
                                 uint64_t *found_candidate) {
    if (midstate == nullptr || tail == nullptr || target_nibbles == nullptr
        || found_candidate == nullptr) {
        return GV_ERR_BAD_ARGS;
    }
    if (num_nibbles < 1 || num_nibbles > 12) {
        return GV_ERR_BAD_ARGS;
    }
    if (tail_len != 64) {
        return GV_ERR_BAD_ARGS;
    }
    if (start_candidate > end_candidate || end_candidate > (uint64_t(1) << 48)) {
        return GV_ERR_BAD_ARGS;
    }
    // The tail must be a well-formed padded final block whose pad region
    // starts at byte 0; its length field defines the W14/W15 constants.
    if (tail[gv::NUM_PAD_CHARS] != 0x80) {
        return GV_ERR_BAD_ARGS;
    }
    for (int i = gv::NUM_PAD_CHARS + 1; i < 56; ++i) {
        if (tail[i] != 0) {
            return GV_ERR_BAD_ARGS;
        }
    }
    uint64_t bit_len =
      (uint64_t(gv::load_be32(tail + 56)) << 32) | gv::load_be32(tail + 60);
    if (bit_len % 8 != 0 || (bit_len / 8) % 64 != uint64_t(gv::NUM_PAD_CHARS)) {
        return GV_ERR_BAD_ARGS;
    }

    SearchParams p = {};
    for (int i = 0; i < 5; ++i) {
        p.ms[i] = midstate[i];
    }
    p.total_len = bit_len / 8;
    for (uint32_t i = 0; i < num_nibbles; ++i) {
        if (target_nibbles[i] > 15) {
            return GV_ERR_BAD_ARGS;
        }
        uint32_t shift = 28 - 4 * (i % 8);
        if (i < 8) {
            p.target0 |= uint32_t(target_nibbles[i]) << shift;
            p.mask0 |= 0xFu << shift;
        } else {
            p.target1 |= uint32_t(target_nibbles[i]) << shift;
            p.mask1 |= 0xFu << shift;
        }
    }
    p.start = start_candidate;
    p.end = end_candidate;
    if (p.start == p.end) {
        return GV_EXHAUSTED;
    }

    int sms = device_sm_count();
    if (sms <= 0) {
        return GV_ERR_CUDA;
    }
    // d_result and d_found are process-global device state, so concurrent
    // calls would read each other's hits; one search runs at a time.
    static std::mutex search_mutex;
    std::lock_guard<std::mutex> guard(search_mutex);
    const unsigned long long sentinel = ~0ULL;
    const int zero = 0;
    if (cudaMemcpyToSymbol(d_result, &sentinel, sizeof sentinel) != cudaSuccess
        || cudaMemcpyToSymbol(d_found, &zero, sizeof zero) != cudaSuccess) {
        return GV_ERR_CUDA;
    }

    uint64_t range = p.end - p.start;
    uint64_t blocks_needed = (range + KERNEL_THREADS - 1) / KERNEL_THREADS;
    uint32_t blocks = uint32_t(sms) * KERNEL_BLOCKS_PER_SM;
    if (blocks_needed < blocks) {
        blocks = uint32_t(blocks_needed);
    }
    vanity_kernel<<<blocks, KERNEL_THREADS>>>(p);
    if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        return GV_ERR_CUDA;
    }

    unsigned long long result = 0;
    int found = 0;
    if (cudaMemcpyFromSymbol(&result, d_result, sizeof result) != cudaSuccess
        || cudaMemcpyFromSymbol(&found, d_found, sizeof found) != cudaSuccess) {
        return GV_ERR_CUDA;
    }
    if (found != 0 && result != sentinel) {
        *found_candidate = uint64_t(result);
        return GV_FOUND;
    }
    return GV_EXHAUSTED;
}
