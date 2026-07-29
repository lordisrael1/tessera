// ============================================================================
// peak_bench — the COMPUTE ROOF of the roofline chart (bible §4.4).
// ============================================================================
//
// Dependency-free instruction chains, run long enough to be real, on 1 core and
// on all cores, reporting BOTH a 10-second burst and a sustained window. On a
// 15-25 W mobile part the gap between those two IS the story.
//
// Four ceilings are measured, and the differences between them are the point:
//
//   fp32-fma    _mm256_fmadd_ps          8 lanes x 2 flop  -> the classic number
//   i8-vnni-ish maddubs + madd + add     32 MAC in 3 ops   -> what INT8 could be
//                                                             if the unsigned
//                                                             operand were free
//   i8-actual   sign + maddubs+madd+add  32 MAC in 4 ops   -> what our kernel
//                                                             really executes
//   i8-1op      (hypothetical vpdpbusd)  32 MAC in 1 op    -> printed as an
//                                                             extrapolation, NOT
//                                                             measured; Zen 3
//                                                             has no AVX512-VNNI
//
// THE MISSING-VNNI TAX. On a VNNI machine one `vpdpbusd` does a 32-way int8
// dot-accumulate. Zen 3 has neither AVX512-VNNI nor AVX-VNNI, so the same work
// takes maddubs (multiply + pairwise add to int16) then madd (pairwise add to
// int32) then an accumulate — and because we keep the unsigned operand in
// [0,127] to prove saturation cannot happen, one more `vpsignb`. Four
// instructions where one would do. Measuring i8-actual against i8-1op quantifies
// exactly how much of the INT8 story on this laptop is an instruction-set
// accident rather than a memory-system fact. That distinction is the whole
// reason fixed-function inference silicon exists.
//
// Usage: peak_bench [--secs 10] [--sustain 0] [--csv out.csv]
// ============================================================================
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "core/threadpool.h"
#include "ops/simd.h"

namespace {

using Clock = std::chrono::steady_clock;
double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

constexpr int kChains = 12;   // >= 8 to cover 2 FMA pipes x 4-cycle latency
constexpr int64_t kInner = 256;

#if TESSERA_X86

// Returns flop executed.
TESSERA_TARGET_AVX2
double fp32_fma_burst(double secs, double* out_seconds) {
    __m256 acc[kChains];
    for (int j = 0; j < kChains; ++j) acc[j] = _mm256_set1_ps(1e-6f * static_cast<float>(j + 1));
    const __m256 a = _mm256_set1_ps(1.0000001f);
    const __m256 b = _mm256_set1_ps(0.9999999f);

    double flop = 0.0;
    double t0 = now_s(), t = t0;
    while ((t = now_s()) - t0 < secs) {
        for (int64_t it = 0; it < kInner; ++it)
            for (int j = 0; j < kChains; ++j) acc[j] = _mm256_fmadd_ps(a, b, acc[j]);
        flop += static_cast<double>(kInner) * kChains * 8 * 2;
    }
    *out_seconds = t - t0;

    // Consume the accumulators so nothing is dead-code eliminated.
    __m256 s = acc[0];
    for (int j = 1; j < kChains; ++j) s = _mm256_add_ps(s, acc[j]);
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, s);
    volatile float sink = tmp[0] + tmp[7];
    (void)sink;
    return flop;
}

// `with_sign` selects between what our kernel really runs and the cheaper
// sequence you would get if the unsigned operand needed no fixing up.
// Returns int8 MACs executed.
template <bool with_sign>
TESSERA_TARGET_AVX2 double i8_burst(double secs, double* out_seconds) {
    __m256i acc[kChains];
    for (int j = 0; j < kChains; ++j) acc[j] = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i av = _mm256_set1_epi8(37);
    __m256i wv = _mm256_set1_epi8(-11);
    __m256i a_abs = _mm256_sign_epi8(av, av);

    double macs = 0.0;
    double t0 = now_s(), t = t0;
    while ((t = now_s()) - t0 < secs) {
        for (int64_t it = 0; it < kInner; ++it)
            for (int j = 0; j < kChains; ++j) {
                __m256i w = with_sign ? _mm256_sign_epi8(wv, av) : wv;
                __m256i p16 = _mm256_maddubs_epi16(a_abs, w);
                acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(p16, ones));
            }
        macs += static_cast<double>(kInner) * kChains * 32;
    }
    *out_seconds = t - t0;

    __m256i s = acc[0];
    for (int j = 1; j < kChains; ++j) s = _mm256_add_epi32(s, acc[j]);
    alignas(32) int32_t tmp[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), s);
    volatile int32_t sink = tmp[0] + tmp[7];
    (void)sink;
    return macs;
}
#endif  // TESSERA_X86

struct Kernel {
    const char* name;
    // ops-per-second reported as GOP/s where one MAC == 2 ops, so INT8 and fp32
    // land on the same y-axis of the roofline.
    double (*run)(double, double*);
    const char* note;
};

double run_fp32(double s, double* d) {
#if TESSERA_X86
    return fp32_fma_burst(s, d);
#else
    (void)s; *d = 1; return 0;
#endif
}
double run_i8_sign(double s, double* d) {
#if TESSERA_X86
    return i8_burst<true>(s, d) * 2.0;  // 1 MAC == 2 ops
#else
    (void)s; *d = 1; return 0;
#endif
}
double run_i8_nosign(double s, double* d) {
#if TESSERA_X86
    return i8_burst<false>(s, d) * 2.0;
#else
    (void)s; *d = 1; return 0;
#endif
}

// Runs `k` on `nt` threads simultaneously and returns aggregate GOP/s.
double parallel_gops(const Kernel& k, int nt, double secs) {
    std::vector<std::thread> ts;
    std::vector<double> ops(static_cast<size_t>(nt), 0.0);
    std::vector<double> el(static_cast<size_t>(nt), 1.0);
    std::atomic<int> ready{0};
    for (int i = 0; i < nt; ++i)
        ts.emplace_back([&, i] {
            ready.fetch_add(1);
            while (ready.load() < nt) { /* start together, so the whole burst overlaps */ }
            ops[static_cast<size_t>(i)] = k.run(secs, &el[static_cast<size_t>(i)]);
        });
    for (auto& t : ts) t.join();
    double total = 0.0, maxel = 0.0;
    for (int i = 0; i < nt; ++i) {
        total += ops[static_cast<size_t>(i)];
        maxel = std::max(maxel, el[static_cast<size_t>(i)]);
    }
    return total / maxel / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    double secs = 10.0, sustain = 0.0;
    std::string csv;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--secs") secs = std::atof(next());
        else if (a == "--sustain") sustain = std::atof(next());
        else if (a == "--csv") csv = next();
    }

    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned phys = ThreadPool::default_threads();
    std::printf("peak_bench: hardware_concurrency=%u, assuming %u physical cores\n", hw, phys);
    std::printf("  AVX2+FMA available: %s\n\n", ops::has_avx2() ? "yes" : "NO (scalar fallback)");

    const Kernel kernels[] = {
        {"fp32-fma  ", run_fp32, "vfmadd231ps, 8 lanes x 2 flop"},
        {"i8-actual ", run_i8_sign, "vpsignb+vpmaddubsw+vpmaddwd+vpaddd, 32 MAC / 4 ops"},
        {"i8-nosign ", run_i8_nosign, "vpmaddubsw+vpmaddwd+vpaddd,       32 MAC / 3 ops"},
    };

    std::vector<std::string> rows{"kernel,threads,gops,window_s"};
    double core1[3] = {0, 0, 0}, coreN[3] = {0, 0, 0};

    for (int ki = 0; ki < 3; ++ki) {
        const Kernel& k = kernels[ki];
        double g1 = parallel_gops(k, 1, secs);
        double gp = parallel_gops(k, static_cast<int>(phys), secs);
        double gh = parallel_gops(k, static_cast<int>(hw ? hw : phys), secs);
        core1[ki] = g1;
        coreN[ki] = gp;
        std::printf("%s  1 thread %8.1f GOP/s | %2u threads %8.1f | %2u threads (SMT) %8.1f"
                    "   scaling %.2fx\n",
                    k.name, g1, phys, gp, hw, gh, gp / (g1 > 0 ? g1 : 1));
        std::printf("             %s\n", k.note);
        rows.push_back(std::string(k.name) + ",1," + std::to_string(g1) + "," +
                       std::to_string(secs));
        rows.push_back(std::string(k.name) + "," + std::to_string(phys) + "," +
                       std::to_string(gp) + "," + std::to_string(secs));
        rows.push_back(std::string(k.name) + "," + std::to_string(hw) + "," +
                       std::to_string(gh) + "," + std::to_string(secs));
    }

    std::printf("\n=== COMPUTE ROOF (%.0f s burst) ===\n", secs);
    std::printf("  fp32 FMA        : %8.1f GFLOP/s all-core (%.1f per core)\n", coreN[0], core1[0]);
    std::printf("  INT8 as we run  : %8.1f GOP/s   all-core (%.1f per core)\n", coreN[1], core1[1]);
    std::printf("  INT8 w/o vpsignb: %8.1f GOP/s   -> the sign fixup costs %.0f%%\n",
                coreN[2], 100.0 * (coreN[2] - coreN[1]) / (coreN[2] > 0 ? coreN[2] : 1));
    std::printf("  [extrapolated] a single-uop vpdpbusd would give ~%.1f GOP/s\n",
                coreN[2] * 3.0);
    std::printf("  NOT MEASURED — Zen 3 has no VNNI. Printed to size the gap that a\n"
                "  fixed-function INT8 datapath closes for free.\n");

    if (sustain > 0.0) {
        std::printf("\n=== SUSTAINED (%.0f s, all cores, fp32 FMA) ===\n", sustain);
        // Sample in 5-second windows so thermal roll-off is visible rather than
        // averaged away. On a 15 W mobile part the first window is the boost
        // clock and the last is the steady state; publishing only the first is
        // how laptop benchmarks lie.
        int windows = static_cast<int>(sustain / 5.0);
        if (windows < 1) windows = 1;
        double first = 0, last = 0;
        for (int wnd = 0; wnd < windows; ++wnd) {
            double g = parallel_gops(kernels[0], static_cast<int>(phys), 5.0);
            std::printf("  t=%3ds  %8.1f GFLOP/s\n", wnd * 5, g);
            rows.push_back(std::string("fp32-sustain,") + std::to_string(phys) + "," +
                           std::to_string(g) + ",5");
            if (wnd == 0) first = g;
            last = g;
        }
        std::printf("  throttle: %.1f%% lost from first window to last\n",
                    100.0 * (first - last) / (first > 0 ? first : 1));
    }

    if (!csv.empty()) {
        FILE* f = std::fopen(csv.c_str(), "w");
        if (f) {
            for (const auto& r : rows) std::fprintf(f, "%s\n", r.c_str());
            std::fclose(f);
            std::printf("\nwrote %s\n", csv.c_str());
        }
    }
    return 0;
}
