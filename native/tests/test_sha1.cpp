/**
 * @file test_sha1.cpp
 * @brief SHA-1 fixture tests: NIST vectors plus a real git commit object.
 *
 * Runs entirely on the CPU through the same header the CUDA kernel
 * compiles, so a pass here pins the shared core's correctness before any
 * GPU involvement.
 */
#include <cstring>
#include <string>
#include <vector>

#include "core/sha1.hpp"
#include "tests/check.hpp"

namespace {

void check_sha1(const std::string &msg, const char *expected_hex, const char *label) {
    uint8_t digest[20];
    gv::sha1_hash(reinterpret_cast<const uint8_t *>(msg.data()), msg.size(), digest);
    CHECK_EQ_DIGEST(expected_hex, digest, label);
}

// NIST FIPS 180-4 / CAVP SHA-1 short and long message vectors, as listed in
// RFC 3174 section 7.3 and the NIST example values document
// (https://csrc.nist.gov/projects/cryptographic-standards-and-guidelines/example-values).
void test_nist_vectors() {
    check_sha1("", "da39a3ee5e6b4b0d3255bfef95601890afd80709", "empty message");
    check_sha1("abc", "a9993e364706816aba3e25717850c26c9cd0d89d", "'abc'");
    check_sha1("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
               "84983e441c3bd26ebaae4aa1f95129e5e54670f1",
               "two-block 56-byte message");
    check_sha1(
      "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnop"
      "jklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
      "a49b2446a02c645bf419f995b67091253a04a259",
      "four-block 112-byte message");
    check_sha1(std::string(1000000, 'a'),
               "34aa973cd4c4daa4f61eeb2bdbad27316534016f",
               "one million 'a'");
}

// Exact byte lengths around the padding boundary: 55 is the longest message
// whose padding fits one block, 56 forces the extra block, 64 is an exact
// block. Golden digests from coreutils sha1sum 9.4:
//   printf 'a%.0s' {1..N} | sha1sum
void test_padding_boundaries() {
    check_sha1(std::string(55, 'a'),
               "c1c8bbdc22796e28c0e15163d20899b65621d65a",
               "55-byte message (padding fits)");
    check_sha1(std::string(56, 'a'),
               "c2db330f6083854c99d4b5bfb6e8f29f201be699",
               "56-byte message (padding overflows)");
    check_sha1(std::string(63, 'a'),
               "03f09f5b158a7a8cdad920bddc29b81c18a551f5",
               "63-byte message");
    check_sha1(std::string(64, 'a'),
               "0098ba824b5c16427bd7a1122a5a442a25ec644d",
               "64-byte exact block");
    check_sha1(std::string(65, 'a'),
               "11655326c708d70319be2610e8a57d9a5b959d3b",
               "65-byte message");
}

// A real commit object, regenerable with git 2.43:
//   export GIT_AUTHOR_NAME="Test Author" GIT_AUTHOR_EMAIL="author@example.com" \
//          GIT_AUTHOR_DATE="1700000000 +0000" GIT_COMMITTER_NAME="Test Committer" \
//          GIT_COMMITTER_EMAIL="committer@example.com" \
//          GIT_COMMITTER_DATE="1700000000 +0000"
//   git init && git commit --allow-empty -m "Fixture commit for git-brew tests"
//   git rev-parse HEAD   -> 34405e61dc9daaf8c8b30c66f0fd25939b8c46e0
//   git cat-file commit HEAD   -> the content below (204 bytes)
void test_git_commit_object() {
    const std::string content =
      "tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904\n"
      "author Test Author <author@example.com> 1700000000 +0000\n"
      "committer Test Committer <committer@example.com> 1700000000 +0000\n"
      "\n"
      "Fixture commit for git-brew tests\n";
    CHECK_EQ_U64(204, content.size(), "fixture content length");
    const std::string object =
      "commit " + std::to_string(content.size()) + '\0' + content;
    uint8_t digest[20];
    gv::sha1_hash(
      reinterpret_cast<const uint8_t *>(object.data()), object.size(), digest);
    CHECK_EQ_DIGEST("34405e61dc9daaf8c8b30c66f0fd25939b8c46e0",
                    digest,
                    "git commit oid via wrapper + content");
}

// The midstate path (update full blocks, finish the tail separately) must
// agree with the one-shot digest for every split point of a two-block
// message.
void test_midstate_matches_one_shot() {
    const std::string msg =
      "The quick brown fox jumps over the lazy dog, twice over, to make this "
      "message span multiple SHA-1 blocks for the midstate test.";
    uint8_t expected[20];
    gv::sha1_hash(reinterpret_cast<const uint8_t *>(msg.data()), msg.size(), expected);

    const uint8_t *data = reinterpret_cast<const uint8_t *>(msg.data());
    uint32_t state[5];
    for (int i = 0; i < 5; ++i) {
        state[i] = gv::kSha1InitState[i];
    }
    size_t consumed = gv::sha1_update_blocks(state, data, msg.size());
    CHECK_EQ_U64((msg.size() / 64) * 64, consumed, "blocks consumed");
    uint8_t digest[20];
    gv::sha1_final(state, data + consumed, msg.size() - consumed, msg.size(), digest);
    CHECK(std::memcmp(expected, digest, 20) == 0, "midstate digest == one-shot");
}

}  // namespace

int main() {
    test_nist_vectors();
    test_padding_boundaries();
    test_git_commit_object();
    test_midstate_matches_one_shot();
    if (gv_test_failures() == 0) {
        std::printf("test_sha1: all checks passed\n");
    }
    return gv_test_failures();
}
