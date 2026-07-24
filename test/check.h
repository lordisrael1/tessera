#pragma once
// Dependency-free test harness (project rule: no gtest, no catch2).
// Usage: CHECK(cond); CHECK_NEAR(a, b, tol); then return test_summary() from main.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

inline int& g_fail_count() { static int n = 0; return n; }
inline int& g_check_count() { static int n = 0; return n; }

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_check_count();                                                   \
        if (!(cond)) {                                                       \
            ++g_fail_count();                                                \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond); \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                        \
    do {                                                                             \
        ++g_check_count();                                                           \
        double _da = (a), _db = (b), _t = (tol);                                     \
        if (std::fabs(_da - _db) > _t) {                                             \
            ++g_fail_count();                                                        \
            std::printf("  FAIL %s:%d  |%g - %g| = %g > %g\n", __FILE__, __LINE__,   \
                        _da, _db, std::fabs(_da - _db), _t);                         \
        }                                                                            \
    } while (0)

inline int test_summary() {
    std::printf("%d checks, %d failures\n", g_check_count(), g_fail_count());
    return g_fail_count() == 0 ? 0 : 1;
}
