// AVX2 + threaded fp32 linear layer.
//
// This is the fp32 path the model actually runs. `ops::linear` (scalar, double
// accumulation) stays the oracle; this file must agree with it to ~1e-5, which
// test/test_ops.cpp asserts.
//
// DESIGN NOTE — why accumulate in double when the whole point is speed:
// a Qwen2.5-0.5B decode step reads ~2 GB of fp32 weights (494M params) and does
// ~1 GFLOP. On this laptop that is ~57 ms of DRAM traffic against ~9 ms of
// arithmetic — a 6:1 memory:compute ratio. Widening f32->f64 roughly halves
// arithmetic throughput and costs literally nothing wall-clock, because
// arithmetic is not the constraint. This is the M2 thesis showing up inside the
// fp32 path before we have even written the INT8 kernel: decode is
// bandwidth-starved, so accuracy here is free. See docs/01-roofline.md.
#include "../core/threadpool.h"
#include "ops.h"
#include "simd.h"

namespace ops {
namespace {

// Fixed reduction order (4 accumulators, combined (a0+a1)+(a2+a3), then lanes
// 0..3 in order) so the result does not depend on thread count or -O level.
#if TESSERA_X86
TESSERA_TARGET_AVX2
double dot_f64_avx2(const float* x, const float* w, int64_t n) {
    __m256d a0 = _mm256_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 w0 = _mm256_loadu_ps(w + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 w1 = _mm256_loadu_ps(w + i + 8);
        a0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(x0)),
                             _mm256_cvtps_pd(_mm256_castps256_ps128(w0)), a0);
        a1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(x0, 1)),
                             _mm256_cvtps_pd(_mm256_extractf128_ps(w0, 1)), a1);
        a2 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(x1)),
                             _mm256_cvtps_pd(_mm256_castps256_ps128(w1)), a2);
        a3 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(x1, 1)),
                             _mm256_cvtps_pd(_mm256_extractf128_ps(w1, 1)), a3);
    }
    __m256d s = _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3));
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, s);
    double acc = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]));
    for (; i < n; ++i) acc += static_cast<double>(x[i]) * w[i];
    return acc;
}

// 4 output rows at a time against one activation row: the four weight streams
// are independent, which lets the out-of-order engine keep 16 FMAs in flight,
// and the activation loads are shared 4 ways.
TESSERA_TARGET_AVX2
void dot4_f64_avx2(const float* x, const float* w, int64_t stride, int64_t n, double* out) {
    __m256d acc[4][2];
    for (int r = 0; r < 4; ++r) { acc[r][0] = _mm256_setzero_pd(); acc[r][1] = acc[r][0]; }
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256d xlo = _mm256_cvtps_pd(_mm256_castps256_ps128(xv));
        __m256d xhi = _mm256_cvtps_pd(_mm256_extractf128_ps(xv, 1));
        for (int r = 0; r < 4; ++r) {
            __m256 wv = _mm256_loadu_ps(w + r * stride + i);
            acc[r][0] = _mm256_fmadd_pd(xlo, _mm256_cvtps_pd(_mm256_castps256_ps128(wv)),
                                        acc[r][0]);
            acc[r][1] = _mm256_fmadd_pd(xhi, _mm256_cvtps_pd(_mm256_extractf128_ps(wv, 1)),
                                        acc[r][1]);
        }
    }
    for (int r = 0; r < 4; ++r) {
        alignas(32) double lanes[8];
        _mm256_store_pd(lanes, acc[r][0]);
        _mm256_store_pd(lanes + 4, acc[r][1]);
        double s = ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) +
                   ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]));
        for (int64_t k = i; k < n; ++k) s += static_cast<double>(x[k]) * w[r * stride + k];
        out[r] = s;
    }
}
#endif  // TESSERA_X86

void linear_range_scalar(const float* X, const float* W, const float* bias, float* Y,
                         int64_t M, int64_t IN, int64_t OUT, int64_t o_lo, int64_t o_hi) {
    for (int64_t o = o_lo; o < o_hi; ++o) {
        const float* wrow = W + o * IN;
        double b = bias ? static_cast<double>(bias[o]) : 0.0;
        for (int64_t m = 0; m < M; ++m) {
            double acc = 0.0;
            const float* xrow = X + m * IN;
            for (int64_t k = 0; k < IN; ++k) acc += static_cast<double>(xrow[k]) * wrow[k];
            Y[m * OUT + o] = static_cast<float>(acc + b);
        }
    }
}

#if TESSERA_X86
TESSERA_TARGET_AVX2
void linear_range_avx2(const float* X, const float* W, const float* bias, float* Y,
                       int64_t M, int64_t IN, int64_t OUT, int64_t o_lo, int64_t o_hi) {
    int64_t o = o_lo;
    // Weight-stationary: the o loop is outermost so each weight row is streamed
    // from memory once and reused across all M activation rows. For M=1 (decode)
    // there is no reuse to be had and this degenerates to pure streaming — which
    // is precisely why decode sits on the bandwidth diagonal of the roofline.
    for (; o + 4 <= o_hi; o += 4) {
        const float* wrow = W + o * IN;
        for (int64_t m = 0; m < M; ++m) {
            double out[4];
            dot4_f64_avx2(X + m * IN, wrow, IN, IN, out);
            for (int r = 0; r < 4; ++r) {
                double b = bias ? static_cast<double>(bias[o + r]) : 0.0;
                Y[m * OUT + o + r] = static_cast<float>(out[r] + b);
            }
        }
    }
    for (; o < o_hi; ++o) {
        const float* wrow = W + o * IN;
        double b = bias ? static_cast<double>(bias[o]) : 0.0;
        for (int64_t m = 0; m < M; ++m)
            Y[m * OUT + o] = static_cast<float>(dot_f64_avx2(X + m * IN, wrow, IN) + b);
    }
}
#endif

// Below this many multiply-adds, thread hand-off costs more than it saves.
constexpr int64_t kThreadThreshold = 1 << 16;

}  // namespace

void linear_fast(const float* X, const float* W, const float* bias, float* Y,
                 int64_t M, int64_t IN, int64_t OUT) {
    auto run = [&](int64_t lo, int64_t hi) {
#if TESSERA_X86
        if (has_avx2()) {
            linear_range_avx2(X, W, bias, Y, M, IN, OUT, lo, hi);
            return;
        }
#endif
        linear_range_scalar(X, W, bias, Y, M, IN, OUT, lo, hi);
    };

    if (M * IN * OUT < kThreadThreshold) {
        run(0, OUT);
        return;
    }
    // Splitting on OUT keeps every output element's entire reduction on one
    // thread, so the result is bit-identical for any thread count. That is the
    // property that lets the M1 oracle stay green with threading enabled.
    global_pool().parallel_for(OUT, run);
}

}  // namespace ops
