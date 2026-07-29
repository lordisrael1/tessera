#pragma once
#include <cstdint>

#include "quant.h"
#include "simd.h"

// INT8 GEMM:  C[M,N] = A[M,IN] (fp32) @ W^T + bias,  where W has W.in == IN and
// W.out == N. C is row-major [M, N].
//
// There are three implementations on purpose. They are all bit-for-bit
// identical (see the ARITHMETIC CONTRACT in gemm_avx2.cpp), so the fast ones can
// be validated against the readable one, and the readable one can be validated
// against fp32. The naive kernel is kept as the published "before" number for
// the packing/tiling/blocking speedup in docs/01-roofline.md — a before/after
// you cannot re-run is a claim, not a measurement.
//
//   gemm_i8_ref    scalar. No intrinsics. Runs anywhere. The definition.
//   gemm_i8_naive  AVX2, row-major weights, one output at a time. The baseline.
//   gemm_i8        AVX2, packed weights + register tiling + L2 blocking +
//                  threads. The one the model uses.
namespace ops {

void gemm_i8_ref(const float* A, int64_t M, const quant::QWeight& W,
                 const float* bias, float* C);

void gemm_i8_naive(const float* A, int64_t M, const quant::QWeight& W,
                   const float* bias, float* C);

// Optimized. `threads` <= 0 means "use the global pool"; 1 means single-thread
// (what bench/ uses to report per-core numbers).
void gemm_i8(const float* A, int64_t M, const quant::QWeightPacked& W,
             const float* bias, float* C, int threads = -1);

// Tuning knobs, exposed so bench/gemm_bench.cpp can sweep them instead of
// hardcoding the answer. mr must be 1, 2 or 4.
struct GemmTuning {
    int mr = 0;        // activation rows per micro-kernel tile; 0 = auto by M
    int64_t nc = 0;    // output rows per L2 panel block; 0 = auto by IN
    bool prefetch = true;
};
void gemm_i8_tuned(const float* A, int64_t M, const quant::QWeightPacked& W,
                   const float* bias, float* C, int threads, const GemmTuning& t);

}  // namespace ops
