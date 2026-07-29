#include "gemm_avx2.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "../core/threadpool.h"

// ============================================================================
// ARITHMETIC CONTRACT
// ============================================================================
// Every kernel in this file computes each output element in exactly this order,
// which is why they agree BIT-FOR-BIT and why test/test_gemm.cpp can assert ==
// rather than a tolerance:
//
//   For output (m, n):
//     float lane[8] = {0}
//     for each block b of QK reduction elements (QK=32 today; the code is
//     written for any multiple of 32, one 32-byte chunk per step):
//         int32 s[j] = sum over c in {0, 32, .. QK-32}, d in {0,1,2,3} of
//                      a[b*QK + c + 4j + d] * w[b*QK + c + 4j + d]     (j = 0..7)
//         float mul   = a_scale[m][b] * w_scale[n][b]
//         lane[j]     = fma(float(s[j]), mul, lane[j])                 (j = 0..7)
//     h[j] = lane[j] + lane[j+4]                                       (j = 0..3)
//     y    = (h[0] + h[2]) + (h[1] + h[3])
//     C[m,n] = y + bias[n]
//
// The peculiar lane mapping is not a choice; it is what the hardware does.
//   _mm256_maddubs_epi16 pairs adjacent BYTES  -> word k = a[2k]w[2k]+a[2k+1]w[2k+1]
//   _mm256_madd_epi16    pairs adjacent WORDS  -> dword j = word[2j] + word[2j+1]
// so one 32-byte chunk lands 4 consecutive products in int32 lane j. Writing the
// scalar reference to match the hardware's grouping (instead of summing left to
// right and calling the difference "rounding") is what turns "the AVX2 kernel
// looks about right" into a == assertion.
//
// The float accumulator (rather than draining to double every block) is the one
// place we trade exactness for speed: it keeps the per-block dequantize as a
// single vector FMA. lane[] holds 8 partial sums, so the error behaves like a
// tree reduction of depth log2(blocks/8), not a serial chain.
// ============================================================================

namespace ops {

using quant::NRP;
using quant::QK;
using quant::QWeight;
using quant::QWeightPacked;

namespace {

// ---------------------------------------------------------------------------
// Scalar reference
// ---------------------------------------------------------------------------

// int32 lanes for one QK block, exactly as the hardware groups them.
inline void block_lanes_scalar(const int8_t* a, const int8_t* w, int32_t s[8]) {
    for (int j = 0; j < 8; ++j) s[j] = 0;
    for (int c = 0; c < QK; c += 32)
        for (int j = 0; j < 8; ++j)
            for (int d = 0; d < 4; ++d) {
                int idx = c + 4 * j + d;
                s[j] += static_cast<int32_t>(a[idx]) * w[idx];
            }
}

inline float hsum8_fixed(const float lane[8]) {
    float h0 = lane[0] + lane[4];
    float h1 = lane[1] + lane[5];
    float h2 = lane[2] + lane[6];
    float h3 = lane[3] + lane[7];
    return (h0 + h2) + (h1 + h3);
}

// ---------------------------------------------------------------------------
// AVX2 helpers
// ---------------------------------------------------------------------------
#if TESSERA_X86

TESSERA_TARGET_AVX2
inline __m256i chunk_dot32(__m256i a_abs, __m256i a_raw, __m256i w, __m256i ones) {
    // |a| is the UNSIGNED operand (0..127) and w*sign(a) the signed one, so the
    // product is a*w and each int16 partial is bounded by 2*127^2 = 32258.
    // maddubs therefore cannot saturate — see docs/02-int8-saturation.md.
    __m256i w_sgn = _mm256_sign_epi8(w, a_raw);
    __m256i p16 = _mm256_maddubs_epi16(a_abs, w_sgn);
    return _mm256_madd_epi16(p16, ones);
}

TESSERA_TARGET_AVX2
inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);                  // h[j] = v[j] + v[j+4]
    __m128 sh = _mm_movehl_ps(lo, lo);        // [h2, h3, ..]
    lo = _mm_add_ps(lo, sh);                  // [h0+h2, h1+h3, ..]
    sh = _mm_shuffle_ps(lo, lo, 0x1);         // [h1+h3, ..]
    lo = _mm_add_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}

// ---------------------------------------------------------------------------
// The micro-kernel: MR activation rows x NRP(=4) packed output rows.
//
// MR is a template parameter so the MR loops fully unroll and the accumulators
// live in registers. Budget on AVX2 (16 ymm):
//   MR=1: 4 accf + 4 acc32 + 2 a-regs + w + ones     = 12   -> no spill
//   MR=2: 8 accf + 8 acc32 + 4 a-regs + w + ones     = 22   -> accf spills, but
//         accf is touched once per 64-element block (8 vector ops of work), so
//         the spill traffic is ~10% and the doubled weight reuse wins for M>1.
//   MR=4: spills hard. Measured, kept as a knob, not the default. bench/ shows it.
// ---------------------------------------------------------------------------
template <int MR>
TESSERA_TARGET_AVX2 void micro_kernel(const int8_t* const a_rows[MR],
                                      const float* const a_scales[MR], const int8_t* wp,
                                      const float* ws, int64_t blocks, float out[MR][NRP],
                                      bool prefetch) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 accf[MR][NRP];
    for (int m = 0; m < MR; ++m)
        for (int r = 0; r < NRP; ++r) accf[m][r] = _mm256_setzero_ps();

    for (int64_t b = 0; b < blocks; ++b) {
        const int8_t* wb = wp + b * NRP * QK;
        if (prefetch)
            // One block ahead: 256 B = 4 cache lines. Zen 3's L2 streamer usually
            // has this covered; bench/gemm_bench.cpp reports the delta either way.
            _mm_prefetch(reinterpret_cast<const char*>(wb + NRP * QK), _MM_HINT_T0);

        __m256i acc32[MR][NRP];
        for (int m = 0; m < MR; ++m)
            for (int r = 0; r < NRP; ++r) acc32[m][r] = _mm256_setzero_si256();

        for (int c = 0; c < QK; c += 32) {
            __m256i a_raw[MR], a_abs[MR];
            for (int m = 0; m < MR; ++m) {
                a_raw[m] = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(a_rows[m] + b * QK + c));
                a_abs[m] = _mm256_sign_epi8(a_raw[m], a_raw[m]);
            }
            for (int r = 0; r < NRP; ++r) {
                __m256i wv =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wb + r * QK + c));
                for (int m = 0; m < MR; ++m)
                    acc32[m][r] = _mm256_add_epi32(
                        acc32[m][r], chunk_dot32(a_abs[m], a_raw[m], wv, ones));
            }
        }

        const float* wsb = ws + b * NRP;
        for (int m = 0; m < MR; ++m)
            for (int r = 0; r < NRP; ++r) {
                __m256 mul = _mm256_set1_ps(a_scales[m][b] * wsb[r]);
                accf[m][r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc32[m][r]), mul, accf[m][r]);
            }
    }

    for (int m = 0; m < MR; ++m)
        for (int r = 0; r < NRP; ++r) out[m][r] = hsum256_ps(accf[m][r]);
}

// ---------------------------------------------------------------------------
// Naive AVX2 baseline: row-major weights, one output element at a time.
// This is the "before" in the before/after. Same arithmetic contract.
// ---------------------------------------------------------------------------
TESSERA_TARGET_AVX2
float dot_rowmajor_avx2(const int8_t* a, const int8_t* w, const float* wscale,
                        const float* a_scale, int64_t blocks) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 accf = _mm256_setzero_ps();
    for (int64_t b = 0; b < blocks; ++b) {
        __m256i acc32 = _mm256_setzero_si256();
        for (int c = 0; c < QK; c += 32) {
            __m256i av = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(a + b * QK + c));
            __m256i wv = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(w + b * QK + c));
            __m256i abs_a = _mm256_sign_epi8(av, av);
            acc32 = _mm256_add_epi32(acc32, chunk_dot32(abs_a, av, wv, ones));
        }
        __m256 mul = _mm256_set1_ps(a_scale[b] * wscale[b]);
        accf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc32), mul, accf);
    }
    return hsum256_ps(accf);
}
#endif  // TESSERA_X86

// Reusable per-thread activation scratch. Growing a thread_local vector means
// the allocation happens once per thread for the whole process lifetime, so the
// per-token path stays allocation-free (bible §2.4) without threading a scratch
// buffer through every signature.
std::vector<int8_t>& act_scratch() {
    static thread_local std::vector<int8_t> buf;
    return buf;
}
std::vector<float>& act_scales() {
    static thread_local std::vector<float> buf;
    return buf;
}

// Activations get per-QK-block scales, mirroring the weights — see the note on
// outlier features in quant.h. The scale array is therefore [M, IN/QK].
void quantize_activations(const float* A, int64_t M, int64_t IN) {
    const int64_t B = IN / quant::QK;
    act_scratch().resize(static_cast<size_t>(M * IN));
    act_scales().resize(static_cast<size_t>(M * B));
    for (int64_t m = 0; m < M; ++m)
        quant::quantize_activation_row(A + m * IN, IN, act_scratch().data() + m * IN,
                                       act_scales().data() + m * B);
}

}  // namespace

// ---------------------------------------------------------------------------
// gemm_i8_ref — the definition.
// ---------------------------------------------------------------------------
void gemm_i8_ref(const float* A, int64_t M, const QWeight& W, const float* bias, float* C) {
    const int64_t IN = W.in, N = W.out, B = W.blocks;
    quantize_activations(A, M, IN);
    const int8_t* Aq = act_scratch().data();

    for (int64_t m = 0; m < M; ++m) {
        const float* a_scale = act_scales().data() + m * B;
        for (int64_t n = 0; n < N; ++n) {
            const int8_t* wrow = W.row(n);
            float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int64_t b = 0; b < B; ++b) {
                int32_t s[8];
                block_lanes_scalar(Aq + m * IN + b * QK, wrow + b * QK, s);
                float mul = a_scale[b] * W.scale_at(n, b);
                for (int j = 0; j < 8; ++j)
                    lane[j] = std::fma(static_cast<float>(s[j]), mul, lane[j]);
            }
            C[m * N + n] = hsum8_fixed(lane) + (bias ? bias[n] : 0.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// gemm_i8_naive — the "before" measurement.
// ---------------------------------------------------------------------------
void gemm_i8_naive(const float* A, int64_t M, const QWeight& W, const float* bias, float* C) {
#if TESSERA_X86
    if (!has_avx2()) { gemm_i8_ref(A, M, W, bias, C); return; }
    const int64_t IN = W.in, N = W.out, B = W.blocks;
    quantize_activations(A, M, IN);
    const int8_t* Aq = act_scratch().data();
    for (int64_t m = 0; m < M; ++m) {
        const float* a_scale = act_scales().data() + m * B;
        for (int64_t n = 0; n < N; ++n)
            C[m * N + n] = dot_rowmajor_avx2(Aq + m * IN, W.row(n),
                                             W.scale.data() + n * B, a_scale, B) +
                           (bias ? bias[n] : 0.0f);
    }
#else
    gemm_i8_ref(A, M, W, bias, C);
#endif
}

// ---------------------------------------------------------------------------
// gemm_i8 — packed + register-tiled + L2-blocked + threaded.
// ---------------------------------------------------------------------------
namespace {

#if TESSERA_X86
// Runs the panel range [p_lo, p_hi) for all M rows.
template <int MR>
TESSERA_TARGET_AVX2 void panel_range(const int8_t* Aq, const float* a_scale, int64_t M,
                                     const QWeightPacked& W, const float* bias, float* C,
                                     int64_t p_lo, int64_t p_hi, int64_t panel_block,
                                     bool prefetch) {
    const int64_t IN = W.in, N = W.out, B = W.blocks;

    // L2 BLOCKING. For each block of `panel_block` panels we sweep all M
    // activation rows, so that panel's weights (panel_block * NRP * IN bytes)
    // are pulled from DRAM once and then re-read from L2 for every m-tile. With
    // one panel at a time and large M, the same weights would be evicted and
    // re-fetched M/MR times; that is the difference the doc reports.
    for (int64_t p0 = p_lo; p0 < p_hi; p0 += panel_block) {
        const int64_t p1 = std::min(p0 + panel_block, p_hi);
        for (int64_t m0 = 0; m0 < M; m0 += MR) {
            const int64_t mr = std::min<int64_t>(MR, M - m0);
            const int8_t* a_rows[MR];
            const float* a_sc[MR];
            for (int i = 0; i < MR; ++i) {
                // Short final tile: replay row 0. Its outputs are computed and
                // then discarded, which costs a little work but keeps the
                // kernel branch-free and the loads in bounds.
                int64_t src = (i < mr) ? m0 + i : m0;
                a_rows[i] = Aq + src * IN;
                a_sc[i] = a_scale + src * B;
            }
            for (int64_t p = p0; p < p1; ++p) {
                float out[MR][NRP];
                micro_kernel<MR>(a_rows, a_sc, W.panel(p), W.panel_scale(p), B, out, prefetch);
                for (int64_t i = 0; i < mr; ++i)
                    for (int64_t r = 0; r < NRP; ++r) {
                        int64_t n = p * NRP + r;
                        if (n < N)
                            C[(m0 + i) * N + n] = out[i][r] + (bias ? bias[n] : 0.0f);
                    }
            }
        }
    }
}
#endif

void gemm_i8_range_scalar(const int8_t* Aq, const float* a_scale, int64_t M,
                          const QWeightPacked& W, const float* bias, float* C, int64_t p_lo,
                          int64_t p_hi) {
    const int64_t IN = W.in, N = W.out, B = W.blocks;
    for (int64_t p = p_lo; p < p_hi; ++p) {
        const int8_t* wp = W.panel(p);
        const float* ws = W.panel_scale(p);
        for (int64_t r = 0; r < NRP; ++r) {
            int64_t n = p * NRP + r;
            if (n >= N) break;
            for (int64_t m = 0; m < M; ++m) {
                float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                for (int64_t b = 0; b < B; ++b) {
                    int32_t s[8];
                    block_lanes_scalar(Aq + m * IN + b * QK, wp + (b * NRP + r) * QK, s);
                    float mul = a_scale[m * B + b] * ws[b * NRP + r];
                    for (int j = 0; j < 8; ++j)
                        lane[j] = std::fma(static_cast<float>(s[j]), mul, lane[j]);
                }
                C[m * N + n] = hsum8_fixed(lane) + (bias ? bias[n] : 0.0f);
            }
        }
    }
}

}  // namespace

void gemm_i8_tuned(const float* A, int64_t M, const QWeightPacked& W, const float* bias,
                   float* C, int threads, const GemmTuning& t) {
    const int64_t IN = W.in;
    quantize_activations(A, M, IN);
    // Snapshot the scratch pointers HERE, on the calling thread. act_scratch()
    // is thread_local, so a pool worker calling it would get its own (empty)
    // buffer; the workers must read the caller's. Capturing raw pointers keeps
    // gemm_i8 reentrant across independent caller threads while the workers do
    // read-only access to whichever caller's buffer they were handed.
    const int8_t* Aq = act_scratch().data();
    const float* Asc = act_scales().data();

    // Panel block sized so one block of packed weights is ~192 KB, comfortably
    // inside the 512 KB private L2 alongside the A tile and the C tile.
    int64_t panel_block = t.nc;
    if (panel_block <= 0) {
        constexpr int64_t kL2Budget = 192 * 1024;
        panel_block = std::max<int64_t>(1, kL2Budget / (NRP * IN));
    }
    int mr = t.mr;
    if (mr <= 0) mr = (M >= 4) ? 2 : 1;  // measured default; see docs/01-roofline.md

    auto run = [&](int64_t p_lo, int64_t p_hi) {
#if TESSERA_X86
        if (has_avx2()) {
            switch (mr) {
                case 1: panel_range<1>(Aq, Asc, M, W, bias, C, p_lo, p_hi, panel_block,
                                       t.prefetch); return;
                case 4: panel_range<4>(Aq, Asc, M, W, bias, C, p_lo, p_hi, panel_block,
                                       t.prefetch); return;
                default: panel_range<2>(Aq, Asc, M, W, bias, C, p_lo, p_hi, panel_block,
                                        t.prefetch); return;
            }
        }
#endif
        gemm_i8_range_scalar(Aq, Asc, M, W, bias, C, p_lo, p_hi);
    };

    // Threading splits PANELS, so every output element's whole reduction stays
    // on one thread and the result is bit-identical at any thread count.
    //
    // THE THRESHOLD IS NOT DECORATION. A condition-variable fan-out/join costs
    // tens of microseconds. A decode-shaped [896 -> 896] projection is 0.8 MMAC,
    // about 30 us of work, so handing it to six threads MEASURED 6x SLOWER than
    // running it on one (3.9 vs 24.7 GOP/s in bench/results/gemm.csv). Threads
    // only pay off once the work per call clears the join cost, which for this
    // pool lands around a few MMAC. The lm_head at M=1 is 136 MMAC and does
    // benefit (22 -> 50 GOP/s), which is why the rule is work-based rather than
    // "decode never threads".
    constexpr int64_t kThreadThresholdMacs = 4 << 20;
    if (threads == 1 || W.panels < 8 || M * IN * W.out < kThreadThresholdMacs) {
        run(0, W.panels);
        return;
    }
    global_pool().parallel_for(W.panels, run);
}

void gemm_i8(const float* A, int64_t M, const QWeightPacked& W, const float* bias, float* C,
             int threads) {
    gemm_i8_tuned(A, M, W, bias, C, threads, GemmTuning{});
}

}  // namespace ops
