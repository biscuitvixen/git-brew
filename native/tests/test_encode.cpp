/**
 * @file test_encode.cpp
 * @brief Fixtures for the candidate encoder and the final-block layout.
 *
 * The last test runs the exact computation the GPU performs per candidate
 * (midstate + one encoded compression) on the CPU and compares it against
 * the one-shot digest of the equivalent byte buffer, pinning the whole
 * midstate pipeline before CUDA enters the picture.
 */
#include <cstring>
#include <string>
#include <vector>

#include "core/encode.hpp"
#include "core/sha1.hpp"
#include "tests/check.hpp"

namespace {

void test_pad_char_mapping() {
    for (int i = 0; i < gv::NUM_PAD_CHARS; ++i) {
        CHECK_EQ_U64(0x20, gv::pad_char(0, i), "candidate 0 is all spaces");
        CHECK_EQ_U64(0x09,
                     gv::pad_char((uint64_t(1) << 48) - 1, i),
                     "candidate 2^48-1 is all tabs");
        uint64_t one_bit = uint64_t(1) << i;
        for (int j = 0; j < gv::NUM_PAD_CHARS; ++j) {
            CHECK_EQ_U64(j == i ? 0x09 : 0x20,
                         gv::pad_char(one_bit, j),
                         "single-bit candidate sets exactly one tab");
        }
    }
}

void test_constant_words() {
    // 240 % 64 == 48, so 240 is a valid object length for the layout.
    uint32_t w[16];
    gv::encode_final_block(0, 240, w);
    for (int j = 0; j < 12; ++j) {
        CHECK_EQ_U64(0x20202020u, w[j], "all-spaces candidate packs 0x20202020");
    }
    CHECK_EQ_U64(0x80000000u, w[12], "W12 is the padding marker");
    CHECK_EQ_U64(0, w[13], "W13 is zero");
    CHECK_EQ_U64(0, w[14], "W14 high bit-length word");
    CHECK_EQ_U64(240 * 8, w[15], "W15 low bit-length word");

    // A length whose bit count crosses 32 bits exercises W14. 2^30 blocks
    // won't happen in practice; the encoder must still be exact.
    gv::encode_final_block(0, (uint64_t(1) << 32) + 48, w);
    CHECK_EQ_U64(8, w[14], "W14 for a > 512 MiB object");
    CHECK_EQ_U64(48 * 8, w[15], "W15 for a > 512 MiB object");
}

// Byte-level equivalence: encode_final_block on a candidate must produce
// the same digest as one-shot hashing the buffer it stands for.
void test_encoded_compression_matches_one_shot() {
    const uint64_t candidates[] = {0,
                                   1,
                                   (uint64_t(1) << 48) - 1,
                                   0x0000AAAAAAAAAAAA,
                                   0x0000123456789ABC,
                                   0x00005A5A5A5A5A5A};
    // Two full blocks of commit-shaped prefix + one block of pad:
    // L = 192 + 48 = 240. The NUL is appended explicitly because a string
    // literal would truncate there; the wrapper length is not meant to be
    // self-consistent, only the byte layout matters here.
    std::string prefix = "commit 229";
    prefix.push_back('\0');
    prefix +=
      "tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904\n"
      "author A <a@example.com> 1700000000 +0000\n"
      "committer C <c@example.com> 1700000000 +0000\n"
      "\n"
      "Encoder equivalence fixture";
    prefix.resize(192, '.');
    for (uint64_t candidate : candidates) {
        std::string object = prefix;
        for (int i = 0; i < gv::NUM_PAD_CHARS; ++i) {
            object.push_back(char(gv::pad_char(candidate, i)));
        }
        CHECK_EQ_U64(240, object.size(), "fixture object length");

        uint8_t expected[20];
        gv::sha1_hash(
          reinterpret_cast<const uint8_t *>(object.data()), object.size(), expected);

        // The GPU path: midstate over the full blocks, one encoded compression.
        uint32_t state[5];
        for (int i = 0; i < 5; ++i) {
            state[i] = gv::kSha1InitState[i];
        }
        size_t consumed = gv::sha1_update_blocks(
          state, reinterpret_cast<const uint8_t *>(prefix.data()), prefix.size());
        CHECK_EQ_U64(192, consumed, "midstate consumed both prefix blocks");
        uint32_t w[16];
        gv::encode_final_block(candidate, object.size(), w);
        gv::sha1_compress(state, w);
        uint8_t actual[20];
        for (int i = 0; i < 5; ++i) {
            gv::store_be32(actual + size_t(i) * 4, state[i]);
        }
        CHECK(std::memcmp(expected, actual, 20) == 0,
              "encoded compression == one-shot digest");
    }
}

}  // namespace

int main() {
    test_pad_char_mapping();
    test_constant_words();
    test_encoded_compression_matches_one_shot();
    if (gv_test_failures() == 0) {
        std::printf("test_encode: all checks passed\n");
    }
    return gv_test_failures();
}
