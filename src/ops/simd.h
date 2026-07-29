#pragma once

// SIMD capability detection and per-function target attributes.
//
// WHY THE ATTRIBUTES: the Release build compiles with -march=native, but the
// ASan/Debug/CI trees do not, and CI deliberately targets x86-64-v3 on a
// non-Zen machine. Sprinkling <immintrin.h> through files compiled without
// -mavx2 is a build break waiting to happen. Marking each kernel
// __attribute__((target("avx2,fma"))) instead means:
//   * every tree compiles the AVX2 code, at any -march,
//   * the compiler refuses to hoist AVX2 into non-AVX2 callers,
//   * and dispatch stays a runtime decision (bible §4.3), which is exactly what
//     lets CI exercise the scalar fallback on the same binary.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define TESSERA_X86 1
#else
#  define TESSERA_X86 0
#endif

#if TESSERA_X86
#  include <immintrin.h>
#endif

#if TESSERA_X86 && defined(__GNUC__)
#  define TESSERA_TARGET_AVX2 __attribute__((target("avx2,fma")))
#else
#  define TESSERA_TARGET_AVX2
#endif

namespace ops {

// Cached cpuid probe. Returns false on non-x86 so the scalar path is taken.
bool has_avx2();

// Test hook: force the scalar path even on an AVX2 machine, so the fallback
// that CI exercises can also be exercised locally.
void set_force_scalar(bool on);

}  // namespace ops
