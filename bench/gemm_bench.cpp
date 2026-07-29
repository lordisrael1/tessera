// ============================================================================
// gemm_bench — the POINTS on the roofline chart, plus the optimization ladder.
// ============================================================================
//
// Runs the four matrix shapes that Qwen2.5-0.5B actually contains, at the M
// values that prefill and decode actually use, through every kernel generation,
// and writes a CSV with (arithmetic intensity, GOP/s) so the roofline plots
// itself.
//
// WHY ARITHMETIC INTENSITY IS COMPUTED, NOT ASSUMED. For C[M,N] = A[M,K] @ W^T:
//     ops   = 2 * M * K * N
//     bytes = N*K            (int8 weights)
//           + N*(K/QK)*4     (fp32 block scales)
//           + M*K*4          (fp32 activations in)
//           + M*N*4          (fp32 results out)
// At M=1 the weights dominate utterly and AI ~ 1.9 ops/byte, which is far below
// the machine balance (compute roof / memory roof ~ 20 ops/byte on this laptop).
// Decode therefore CANNOT be compute-bound, no matter how good the kernel is.
// At M=512 the same weights are amortized 512 ways and AI ~ 900 ops/byte, which
// is deep in the compute-bound region. The identical kernel moves from one side
// of the roofline ridge to the other purely because of M. That single fact is
// the thesis of this whole project, and this benchmark is where it is measured
// rather than asserted.
//
// Usage: gemm_bench [--secs 1.0] [--sweep] [--csv out.csv]
// ============================================================================
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/threadpool.h"
#include "ops/gemm_avx2.h"
#include "ops/ops.h"
#include "ops/quant.h"

namespace {

using Clock = std::chrono::steady_clock;
double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

struct Shape {
    const char* name;
    int64_t in, out;
};

// The real Qwen2.5-0.5B projection shapes. hidden=896, intermediate=4864,
// vocab=151936, and the attention output dim is 14 heads x 64 = 896.
const Shape kShapes[] = {
    {"qkv/o  [896->896]",    896, 896},
    {"gate/up[896->4864]",   896, 4864},
    {"down   [4864->896]",  4864, 896},
    {"lm_head[896->151936]", 896, 151936},
};

std::vector<float> rand_mat(int64_t rows, int64_t cols, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, 0.02f);  // realistic weight scale
    std::vector<float> m(static_cast<size_t>(rows * cols));
    for (auto& x : m) x = d(rng);
    return m;
}

constexpr double d(int64_t v) { return static_cast<double>(v); }

double int8_bytes(int64_t M, int64_t IN, int64_t OUT) {
    return d(OUT) * d(IN)                          // int8 weights
           + d(OUT) * d(IN / quant::QK) * 4.0      // fp32 block scales
           + d(M) * d(IN) * 4.0                    // fp32 activations in
           + d(M) * d(OUT) * 4.0;                  // fp32 results out
}

// Times `fn` for at least `secs`, returns GOP/s counting 2 ops per MAC.
double time_gops(int64_t M, int64_t IN, int64_t OUT, double secs, const std::function<void()>& fn) {
    fn();  // warm caches / first-touch
    int iters = 0;
    double t0 = now_s();
    while (now_s() - t0 < secs) {
        fn();
        ++iters;
    }
    double dt = now_s() - t0;
    double ops = 2.0 * d(M) * d(IN) * d(OUT) * iters;
    return ops / dt / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    double secs = 1.0;
    bool sweep = false;
    std::string csv;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--secs") secs = std::atof(next());
        else if (a == "--sweep") sweep = true;
        else if (a == "--csv") csv = next();
    }

    const int64_t Ms[] = {1, 8, 32, 128, 512};
    std::vector<std::string> rows{
        "shape,in,out,M,kernel,threads,arith_intensity_ops_per_byte,gops"};

    std::printf("gemm_bench: INT8 GEMM, %u-thread pool\n", global_pool().size());
    std::printf("%-22s %6s %10s %12s %12s %12s %12s\n", "shape", "M", "AI(op/B)",
                "naive-1T", "tuned-1T", "tuned-NT", "fp32-NT");
    std::printf("%s\n", std::string(94, '-').c_str());

    for (const Shape& sh : kShapes) {
        auto Wf = rand_mat(sh.out, sh.in, 1234);
        auto qw = quant::quantize_weight(Wf.data(), sh.out, sh.in);
        auto qp = quant::pack_weight(qw);

        for (int64_t M : Ms) {
            // lm_head at M=512 is a 300 MB fp32 output; skip the silly ones.
            if (d(M) * d(sh.out) > 40e6) continue;

            auto A = rand_mat(M, sh.in, 99);
            std::vector<float> C(static_cast<size_t>(M * sh.out));

            double ai = 2.0 * d(M) * d(sh.in) * d(sh.out) /
                        int8_bytes(M, sh.in, sh.out);

            double g_naive = time_gops(M, sh.in, sh.out, secs, [&] {
                ops::gemm_i8_naive(A.data(), M, qw, nullptr, C.data());
            });
            double g_t1 = time_gops(M, sh.in, sh.out, secs, [&] {
                ops::gemm_i8(A.data(), M, qp, nullptr, C.data(), 1);
            });
            double g_tn = time_gops(M, sh.in, sh.out, secs, [&] {
                ops::gemm_i8(A.data(), M, qp, nullptr, C.data(), -1);
            });
            // fp32 for scale: same shape, same threads, 4x the weight bytes.
            double g_f32 = time_gops(M, sh.in, sh.out, secs, [&] {
                ops::linear_fast(A.data(), Wf.data(), nullptr, C.data(), M, sh.in, sh.out);
            });

            std::printf("%-22s %6lld %10.1f %12.1f %12.1f %12.1f %12.1f\n", sh.name,
                        static_cast<long long>(M), ai, g_naive, g_t1, g_tn, g_f32);

            auto row = [&](const char* k, int th, double g) {
                rows.push_back(std::string(sh.name) + "," + std::to_string(sh.in) + "," +
                               std::to_string(sh.out) + "," + std::to_string(M) + "," + k + "," +
                               std::to_string(th) + "," + std::to_string(ai) + "," +
                               std::to_string(g));
            };
            row("i8-naive", 1, g_naive);
            row("i8-tuned", 1, g_t1);
            row("i8-tuned", static_cast<int>(global_pool().size()), g_tn);
            row("fp32", static_cast<int>(global_pool().size()), g_f32);
        }
        std::printf("\n");
    }

    if (sweep) {
        // The tuning sweep the bible asks to publish rather than argue about
        // (§4.3): register-tile height MR, L2 panel-block width, prefetch on/off.
        std::printf("=== TUNING SWEEP (gate/up [896->4864], 1 thread) ===\n");
        const Shape& sh = kShapes[1];
        auto Wf = rand_mat(sh.out, sh.in, 7);
        auto qp = quant::quantize_and_pack(Wf.data(), sh.out, sh.in);
        std::printf("%6s %5s %8s %6s %10s\n", "M", "MR", "panels", "pref", "GOP/s");
        for (int64_t M : {int64_t{1}, int64_t{8}, int64_t{128}}) {
            auto A = rand_mat(M, sh.in, 5);
            std::vector<float> C(static_cast<size_t>(M * sh.out));
            for (int mr : {1, 2, 4})
                for (int64_t nc : {int64_t{8}, int64_t{32}, int64_t{54}, int64_t{256}})
                    for (bool pf : {false, true}) {
                        ops::GemmTuning t;
                        t.mr = mr;
                        t.nc = nc;
                        t.prefetch = pf;
                        double g = time_gops(M, sh.in, sh.out, 0.3, [&] {
                            ops::gemm_i8_tuned(A.data(), M, qp, nullptr, C.data(), 1, t);
                        });
                        std::printf("%6lld %5d %8lld %6s %10.1f\n", static_cast<long long>(M),
                                    mr, static_cast<long long>(nc), pf ? "on" : "off", g);
                        rows.push_back(std::string("sweep,") + std::to_string(sh.in) + "," +
                                       std::to_string(sh.out) + "," + std::to_string(M) +
                                       ",mr" + std::to_string(mr) + "_nc" + std::to_string(nc) +
                                       (pf ? "_pf" : "_nopf") + ",1,0," + std::to_string(g));
                    }
        }
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
