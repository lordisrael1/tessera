#include "simd.h"

#if TESSERA_X86 && defined(_MSC_VER)
#  include <intrin.h>
#endif

namespace ops {
namespace {
bool g_force_scalar = false;

bool probe_avx2() {
#if !TESSERA_X86
    return false;
#elif defined(__GNUC__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#elif defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuidex(r, 7, 0);
    bool avx2 = (r[1] & (1 << 5)) != 0;
    __cpuid(r, 1);
    bool fma = (r[2] & (1 << 12)) != 0;
    return avx2 && fma;
#else
    return false;
#endif
}
}  // namespace

bool has_avx2() {
    static const bool cached = probe_avx2();
    return cached && !g_force_scalar;
}

void set_force_scalar(bool on) { g_force_scalar = on; }

}  // namespace ops
