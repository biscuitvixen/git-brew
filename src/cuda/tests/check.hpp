/**
 * @file check.hpp
 * @brief Minimal test checks: each failure prints a labelled line, main()
 *        returns the failure count so ctest sees a nonzero exit status.
 *
 * Deliberately not a framework. The tests here are table-driven fixture
 * comparisons; a labelled check macro and a failure counter cover them
 * without a configure-time dependency fetch.
 */
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>

inline int &gv_test_failures() {
    static int failures = 0;
    return failures;
}

/// Format n bytes as lowercase hex into out (needs 2n + 1 chars).
inline void gv_hex(const uint8_t *bytes, size_t n, char *out) {
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < n; ++i) {
        out[2 * i] = digits[bytes[i] >> 4];
        out[2 * i + 1] = digits[bytes[i] & 0xF];
    }
    out[2 * n] = '\0';
}

#define CHECK(cond, label)                                              \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, label); \
            ++gv_test_failures();                                       \
        }                                                               \
    } while (0)

#define CHECK_EQ_U64(expected, actual, label)                     \
    do {                                                          \
        uint64_t gv_e = uint64_t(expected);                       \
        uint64_t gv_a = uint64_t(actual);                         \
        if (gv_e != gv_a) {                                       \
            std::printf(                                          \
              "FAIL %s:%d: %s: expected %llu (0x%llx), got %llu " \
              "(0x%llx)\n",                                       \
              __FILE__,                                           \
              __LINE__,                                           \
              label,                                              \
              (unsigned long long)gv_e,                           \
              (unsigned long long)gv_e,                           \
              (unsigned long long)gv_a,                           \
              (unsigned long long)gv_a);                          \
            ++gv_test_failures();                                 \
        }                                                         \
    } while (0)

/// Compare a 20-byte digest against an expected lowercase hex string.
#define CHECK_EQ_DIGEST(expected_hex, digest20, label)           \
    do {                                                         \
        char gv_hexbuf[41];                                      \
        gv_hex(digest20, 20, gv_hexbuf);                         \
        if (std::strcmp(expected_hex, gv_hexbuf) != 0) {         \
            std::printf("FAIL %s:%d: %s: expected %s, got %s\n", \
                        __FILE__,                                \
                        __LINE__,                                \
                        label,                                   \
                        expected_hex,                            \
                        gv_hexbuf);                              \
            ++gv_test_failures();                                \
        }                                                        \
    } while (0)
