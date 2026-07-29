// ============================================================================
// stream_bench — the MEMORY ROOF of the roofline chart (bible §4.4).
// ============================================================================
//
// Measures achievable DRAM bandwidth with the four classic STREAM kernels over
// arrays far larger than the 16 MB L3, so nothing is served from cache:
//
//   copy   a[i] = b[i]
//   scale  a[i] = s*b[i]
//   add    a[i] = b[i] + c[i]
//   triad  a[i] = b[i] + s*c[i]      <- the number quoted as "the memory roof"
//
// THREE THINGS THIS BENCHMARK GETS RIGHT THAT A NAIVE ONE DOES NOT
//
// 1. IT IS THREADED, AND SWEEPS THREAD COUNT.
//    One Zen 3 core cannot saturate dual-channel DDR4-3200. A single core has a
//    bounded number of outstanding L1/L2 miss buffers, and Little's law caps it
//    at roughly (miss buffers x line size) / latency — typically 60-70% of the
//    channel pair. Reporting a single-threaded triad as "the memory roof" is the
//    single most common way to publish a roofline that is wrong by 40%.
//
// 2. IT REPORTS BOTH "STREAM-COUNTED" AND "TRUE" TRAFFIC.
//    Classic STREAM counts triad as 3 arrays (read b, read c, write a). But an
//    ordinary store to a line that is not in cache first triggers a
//    READ-FOR-OWNERSHIP: the CPU fetches the line it is about to overwrite
//    entirely. So the DRAM actually moves 4 arrays' worth. Both numbers are
//    printed, because they answer different questions: STREAM-counted is what
//    you compare against other STREAM results, TRUE is what you must put on a
//    roofline if you want the diagonal to mean anything.
//
// 3. IT HAS A NON-TEMPORAL VARIANT.
//    _mm256_stream_pd writes straight to memory and skips the RFO, collapsing
//    the gap in (2). The delta between the two triads is a direct measurement of
//    what write-allocate costs — and it is the CPU-side rehearsal of exactly the
//    argument for M3's software-managed scratchpad: an explicit DMA_STORE never
//    pays for a cache line it is about to overwrite, because there are no caches
//    to be coherent with.
//
// Usage: stream_bench [--mib 192] [--reps 12] [--threads N] [--csv out.csv]
// ============================================================================
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/align.h"
#include "core/threadpool.h"
#include "ops/simd.h"

namespace {

using Clock = std::chrono::steady_clock;
double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

struct Result {
    double gbs_stream = 0;  // classic STREAM accounting
    double gbs_true = 0;    // including read-for-ownership on the stores
};

// One kernel's worth of description: how many arrays STREAM counts, and how many
// the memory controller actually moves.
struct KernelSpec {
    const char* name;
    int arrays_stream;
    int arrays_true;  // +1 vs stream for every written array, unless NT stores
};

void touch_parallel(ThreadPool& pool, double* p, int64_t n, double v) {
    // First-touch matters: pages faulted in by the main thread all land on
    // whatever the allocator gives it. Touching from the same threads that will
    // later read them is the honest setup.
    pool.parallel_for(n, [&](int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; ++i) p[i] = v;
    });
}

#if TESSERA_X86
TESSERA_TARGET_AVX2
void triad_nt(double* a, const double* b, const double* c, double s, int64_t lo, int64_t hi) {
    const __m256d sv = _mm256_set1_pd(s);
    int64_t i = lo;
    // Non-temporal stores need 32-byte alignment; step up to it scalar-wise.
    while (i < hi && (reinterpret_cast<uintptr_t>(a + i) & 31u) != 0) {
        a[i] = b[i] + s * c[i];
        ++i;
    }
    for (; i + 4 <= hi; i += 4) {
        __m256d bv = _mm256_loadu_pd(b + i);
        __m256d cv = _mm256_loadu_pd(c + i);
        _mm256_stream_pd(a + i, _mm256_fmadd_pd(sv, cv, bv));
    }
    for (; i < hi; ++i) a[i] = b[i] + s * c[i];
}
#endif

}  // namespace

int main(int argc, char** argv) {
    int64_t mib = 192;
    int reps = 12;
    int fixed_threads = 0;
    std::string csv;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--mib") mib = std::atoll(next());
        else if (a == "--reps") reps = std::atoi(next());
        else if (a == "--threads") fixed_threads = std::atoi(next());
        else if (a == "--csv") csv = next();
    }

    const int64_t N = mib * 1024 * 1024 / static_cast<int64_t>(sizeof(double));
    const double bytes_per_array = static_cast<double>(N) * sizeof(double);
    std::printf("stream_bench: %lld MiB per array (%lld doubles), %d reps\n",
                static_cast<long long>(mib), static_cast<long long>(N), reps);
    std::printf("  L3 is 16 MiB on the target machine, so %lldx L3 -> pure DRAM.\n\n",
                static_cast<long long>(mib / 16));

    auto* A = static_cast<double*>(aligned_alloc64(static_cast<size_t>(N) * sizeof(double)));
    auto* B = static_cast<double*>(aligned_alloc64(static_cast<size_t>(N) * sizeof(double)));
    auto* C = static_cast<double*>(aligned_alloc64(static_cast<size_t>(N) * sizeof(double)));

    const KernelSpec specs[] = {
        {"copy ", 2, 3},   // read b, write a (+RFO on a)
        {"scale", 2, 3},
        {"add  ", 3, 4},
        {"triad", 3, 4},
        {"triad-nt", 3, 3},  // NT store skips the RFO: true == stream
    };

    std::vector<int> thread_counts;
    if (fixed_threads > 0) {
        thread_counts.push_back(fixed_threads);
    } else {
        unsigned hw = std::thread::hardware_concurrency();
        for (int t : {1, 2, 3, 4, 6, 8, 12})
            if (static_cast<unsigned>(t) <= (hw ? hw : 12u)) thread_counts.push_back(t);
    }

    std::vector<std::string> csv_rows;
    csv_rows.push_back("kernel,threads,gbs_stream_counted,gbs_true_traffic");

    double best_triad_true = 0.0, best_triad_stream = 0.0;
    int best_triad_threads = 0;

    for (int nt : thread_counts) {
        ThreadPool pool(static_cast<unsigned>(nt));
        pool.pin_to_cores();
        touch_parallel(pool, A, N, 0.0);
        touch_parallel(pool, B, N, 1.0);
        touch_parallel(pool, C, N, 2.0);
        const double s = 3.0;

        std::printf("threads=%-2d  ", nt);
        for (const KernelSpec& k : specs) {
            Result best{};
            for (int r = 0; r < reps; ++r) {
                double t0 = now_s();
                pool.parallel_for(N, [&](int64_t lo, int64_t hi) {
                    if (std::strcmp(k.name, "copy ") == 0) {
                        for (int64_t i = lo; i < hi; ++i) A[i] = B[i];
                    } else if (std::strcmp(k.name, "scale") == 0) {
                        for (int64_t i = lo; i < hi; ++i) A[i] = s * B[i];
                    } else if (std::strcmp(k.name, "add  ") == 0) {
                        for (int64_t i = lo; i < hi; ++i) A[i] = B[i] + C[i];
                    } else if (std::strcmp(k.name, "triad") == 0) {
                        for (int64_t i = lo; i < hi; ++i) A[i] = B[i] + s * C[i];
                    } else {
#if TESSERA_X86
                        if (ops::has_avx2()) { triad_nt(A, B, C, s, lo, hi); return; }
#endif
                        for (int64_t i = lo; i < hi; ++i) A[i] = B[i] + s * C[i];
                    }
                });
                double dt = now_s() - t0;
                Result cur;
                cur.gbs_stream = k.arrays_stream * bytes_per_array / dt / 1e9;
                cur.gbs_true = k.arrays_true * bytes_per_array / dt / 1e9;
                if (cur.gbs_stream > best.gbs_stream) best = cur;
            }
            std::printf("%s %5.1f/%5.1f  ", k.name, best.gbs_stream, best.gbs_true);
            csv_rows.push_back(std::string(k.name) + "," + std::to_string(nt) + "," +
                               std::to_string(best.gbs_stream) + "," +
                               std::to_string(best.gbs_true));
            if (std::strcmp(k.name, "triad") == 0 && best.gbs_true > best_triad_true) {
                best_triad_true = best.gbs_true;
                best_triad_stream = best.gbs_stream;
                best_triad_threads = nt;
            }
        }
        std::printf("  (GB/s stream-counted / true-traffic)\n");
    }

    // Guard against the whole thing being optimised away.
    volatile double sink = A[0] + A[N / 2] + A[N - 1];
    (void)sink;

    std::printf("\n=== MEMORY ROOF ===\n");
    std::printf("  triad, best over thread counts: %.1f GB/s stream-counted, "
                "%.1f GB/s true traffic (at %d threads)\n",
                best_triad_stream, best_triad_true, best_triad_threads);
    std::printf("  DDR4-3200 dual channel theoretical = 51.2 GB/s -> %.0f%% of peak\n",
                100.0 * best_triad_true / 51.2);
    std::printf("  Use the TRUE number as the roofline's memory ceiling.\n");

    if (!csv.empty()) {
        FILE* f = std::fopen(csv.c_str(), "w");
        if (f) {
            for (const auto& r : csv_rows) std::fprintf(f, "%s\n", r.c_str());
            std::fclose(f);
            std::printf("\nwrote %s\n", csv.c_str());
        }
    }

    aligned_free64(A);
    aligned_free64(B);
    aligned_free64(C);
    return 0;
}
