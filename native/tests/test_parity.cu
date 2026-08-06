/**
 * @file test_parity.cu
 * @brief CPU-vs-GPU parity through the real vanity_search entry point.
 *
 * For each sampled candidate the CPU computes the digest via the shared
 * core, takes its first 12 hex digits as the target, and requires the GPU
 * to recover exactly that candidate from a narrow bracketing range. A
 * 12-nibble target makes a different-candidate collision within the
 * bracket a 2^-48 event, so any mismatch is a real divergence between the
 * host and device paths.
 */
#include <cstring>
#include <string>

#include "core/encode.hpp"
#include "core/sha1.hpp"
#include "git_brew.h"
#include "tests/check.hpp"

namespace {

/// Deterministic 48-bit candidate stream (Knuth MMIX LCG, high bits).
uint64_t next_candidate(uint64_t &state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (state >> 16) & ((uint64_t(1) << 48) - 1);
}

struct Fixture {
    uint32_t midstate[5];
    uint8_t tail[64];
    uint64_t total_len;
};

/// Commit-shaped 192-byte prefix + 48-byte pad block: total_len 240.
Fixture make_fixture() {
    std::string prefix = "commit 229";
    prefix.push_back('\0');
    prefix +=
      "tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904\n"
      "author A <a@example.com> 1700000000 +0000\n"
      "committer C <c@example.com> 1700000000 +0000\n"
      "\n"
      "GPU parity fixture";
    prefix.resize(192, '.');

    Fixture f = {};
    f.total_len = prefix.size() + gv::NUM_PAD_CHARS;
    for (int i = 0; i < 5; ++i) {
        f.midstate[i] = gv::kSha1InitState[i];
    }
    gv::sha1_update_blocks(
      f.midstate, reinterpret_cast<const uint8_t *>(prefix.data()), prefix.size());
    for (int i = 0; i < gv::NUM_PAD_CHARS; ++i) {
        f.tail[i] = 0x20;
    }
    f.tail[gv::NUM_PAD_CHARS] = 0x80;
    gv::store_be32(f.tail + 56, uint32_t((f.total_len * 8) >> 32));
    gv::store_be32(f.tail + 60, uint32_t(f.total_len * 8));
    return f;
}

void cpu_digest(const Fixture &f, uint64_t candidate, uint8_t out[20]) {
    uint32_t state[5];
    std::memcpy(state, f.midstate, sizeof state);
    uint32_t w[16];
    gv::encode_final_block(candidate, f.total_len, w);
    gv::sha1_compress(state, w);
    for (int i = 0; i < 5; ++i) {
        gv::store_be32(out + size_t(i) * 4, state[i]);
    }
}

void digest_nibbles(const uint8_t digest[20], uint8_t out[12]) {
    for (int i = 0; i < 12; ++i) {
        out[i] = (i % 2 == 0) ? (digest[i / 2] >> 4) : (digest[i / 2] & 0xF);
    }
}

void test_gpu_recovers_cpu_candidates(const Fixture &f) {
    uint64_t rng = 20260806;
    for (int trial = 0; trial < 1000; ++trial) {
        uint64_t cand = next_candidate(rng);
        uint8_t digest[20];
        cpu_digest(f, cand, digest);
        uint8_t nibbles[12];
        digest_nibbles(digest, nibbles);

        uint64_t lo = cand < 500 ? 0 : cand - 500;
        uint64_t hi = cand + 500;
        uint64_t found = 0;
        int32_t rc = vanity_search(f.midstate, f.tail, 64, nibbles, 12, lo, hi, &found);
        CHECK_EQ_U64(GV_FOUND, rc, "search near a known candidate returns FOUND");
        CHECK_EQ_U64(cand, found, "GPU recovers the CPU-hashed candidate");
        if (gv_test_failures() > 0) {
            return;  // One divergence means the core differs; stop the spam.
        }
    }
}

void test_range_exhaustion(const Fixture &f) {
    // A 12-nibble target from a candidate outside the searched range: the
    // odds any of the 999 range members collide are ~999 * 2^-48.
    uint64_t rng = 424242;
    uint64_t cand = next_candidate(rng);
    uint8_t digest[20];
    cpu_digest(f, cand, digest);
    uint8_t nibbles[12];
    digest_nibbles(digest, nibbles);
    uint64_t found = 0;
    int32_t rc =
      vanity_search(f.midstate, f.tail, 64, nibbles, 12, cand + 1, cand + 1000, &found);
    CHECK_EQ_U64(GV_EXHAUSTED, rc, "target outside range reports EXHAUSTED");
    rc = vanity_search(f.midstate, f.tail, 64, nibbles, 12, cand, cand, &found);
    CHECK_EQ_U64(GV_EXHAUSTED, rc, "empty range reports EXHAUSTED");
}

void test_short_prefix_search(const Fixture &f) {
    // A 1-nibble search must find some candidate fast; verify it on the CPU.
    for (uint8_t nibble = 0; nibble < 16; ++nibble) {
        uint64_t found = 0;
        int32_t rc = vanity_search(
          f.midstate, f.tail, 64, &nibble, 1, 0, uint64_t(1) << 20, &found);
        CHECK_EQ_U64(GV_FOUND, rc, "1-nibble search over 2^20 finds a hit");
        if (rc != GV_FOUND) {
            continue;
        }
        uint8_t digest[20];
        cpu_digest(f, found, digest);
        CHECK_EQ_U64(nibble, digest[0] >> 4, "GPU hit verifies on the CPU");
    }
}

void test_argument_validation(const Fixture &f) {
    uint8_t nibbles[12] = {0};
    uint64_t found = 0;
    CHECK_EQ_U64(GV_ERR_BAD_ARGS,
                 vanity_search(nullptr, f.tail, 64, nibbles, 1, 0, 10, &found),
                 "null midstate rejected");
    CHECK_EQ_U64(GV_ERR_BAD_ARGS,
                 vanity_search(f.midstate, f.tail, 64, nibbles, 0, 0, 10, &found),
                 "zero nibbles rejected");
    CHECK_EQ_U64(GV_ERR_BAD_ARGS,
                 vanity_search(f.midstate, f.tail, 64, nibbles, 13, 0, 10, &found),
                 "13 nibbles rejected");
    CHECK_EQ_U64(GV_ERR_BAD_ARGS,
                 vanity_search(f.midstate, f.tail, 63, nibbles, 1, 0, 10, &found),
                 "tail_len != 64 rejected");
    CHECK_EQ_U64(GV_ERR_BAD_ARGS,
                 vanity_search(f.midstate, f.tail, 64, nibbles, 1, 10, 5, &found),
                 "reversed range rejected");
    CHECK_EQ_U64(
      GV_ERR_BAD_ARGS,
      vanity_search(
        f.midstate, f.tail, 64, nibbles, 1, 0, (uint64_t(1) << 48) + 1, &found),
      "range past 2^48 rejected");
    uint8_t bad_nibble = 16;
    CHECK_EQ_U64(GV_ERR_BAD_ARGS,
                 vanity_search(f.midstate, f.tail, 64, &bad_nibble, 1, 0, 10, &found),
                 "nibble value 16 rejected");
    uint8_t bad_tail[64];
    std::memcpy(bad_tail, f.tail, 64);
    bad_tail[gv::NUM_PAD_CHARS] = 0;
    CHECK_EQ_U64(GV_ERR_BAD_ARGS,
                 vanity_search(f.midstate, bad_tail, 64, nibbles, 1, 0, 10, &found),
                 "tail without 0x80 marker rejected");
}

}  // namespace

int main() {
    Fixture f = make_fixture();
    test_gpu_recovers_cpu_candidates(f);
    test_range_exhaustion(f);
    test_short_prefix_search(f);
    test_argument_validation(f);
    if (gv_test_failures() == 0) {
        std::printf("test_parity: all checks passed\n");
    }
    return gv_test_failures();
}
