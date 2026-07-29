#pragma once
#include <cstdint>
#include <vector>

// INT8 quantization for the M2 GEMM (W8A8: both operands are int8).
//
// Weights: symmetric INT8, per-block along the reduction (IN) dimension.
//   Per block: scale chosen to minimise squared reconstruction error near
//   max(|w|)/127, q = round(w/scale) clamped to [-127,127].
//
// Activations: quantized on the fly to symmetric SIGNED int8, ALSO per block.
//   The GEMM multiplies int8 x int8 on AVX2 via the sign-epi8 trick: |a|
//   (0..127) is the unsigned operand to maddubs, w*sign(a) the signed one.
//   Keeping the unsigned operand <= 127 bounds each int16 partial to
//   2*127^2 = 32258 < 32767, so the accumulator PROVABLY never saturates.
//   (The naive +128 zero-point shift makes it 0..255 and 255*127*2 = 64770,
//   which saturates hard — see docs/02-int8-saturation.md. This is a documented
//   deviation from bible §4.1; the deviation table in BIBLE.md §13 records it.)
namespace quant {

// Reduction-block size. The bible starts at 64 and says to drop to 32 if the
// numbers demand it; they did. Measured end-to-end on Qwen2.5-0.5B:
//
//   QK   logit rel-L2   teacher-forced top-1 vs HF   scale storage
//   64      4.33%                94.8%                 1/64 of weights
//   32      3.28%                98.4%                 1/32 of weights
//
// Must be a multiple of 32: the AVX2 micro-kernel consumes the block in 32-byte
// chunks, so a smaller block would leave the vector registers half empty.
constexpr int QK = 32;
constexpr int NRP = 4;   // output rows per packed panel (see QWeightPacked)

// A weight matrix [OUT, IN] quantized to int8, blocked along IN.
// Row-major reference layout — readable, and what the scalar path walks.
struct QWeight {
    int64_t out = 0, in = 0;
    int64_t blocks = 0;               // in / QK
    std::vector<int8_t> q;            // [out * in]
    std::vector<float> scale;         // [out * blocks]

    const int8_t* row(int64_t o) const { return q.data() + o * in; }
    float scale_at(int64_t o, int64_t b) const {
        return scale[static_cast<size_t>(o * blocks + b)];
    }
};

// The same weights, re-laid-out for the AVX2 micro-kernel.
//
// WHY PACKING MATTERS: the row-major layout makes a micro-kernel that computes
// NRP output rows at once read NRP streams that are `in` bytes apart. Zen 3's
// prefetchers cope, but every block boundary is a fresh page-walk-ish jump and
// the loads are not adjacent, so the load/store unit cannot coalesce anything.
// Packing interleaves them: all NRP rows' bytes for block b sit in one
// contiguous NRP*QK run, so the inner loop is a single sequential stream.
// Pack cost is paid once per weight matrix and amortizes over every token.
//
// Layout:  q[((p * blocks) + b) * NRP * QK + r * QK + k]
//          scale[((p * blocks) + b) * NRP + r]
// where p = panel index, r = row within panel (output row = p*NRP + r).
// Rows past `out` in the final panel are zero-filled with scale 0, so they
// contribute nothing and no tail special-casing is needed in the kernel.
struct QWeightPacked {
    int64_t out = 0, in = 0, blocks = 0, panels = 0;
    std::vector<int8_t> q;
    std::vector<float> scale;

    const int8_t* panel(int64_t p) const {
        return q.data() + p * blocks * NRP * QK;
    }
    const float* panel_scale(int64_t p) const {
        return scale.data() + p * blocks * NRP;
    }
};

// Quantize a row-major [out, in] fp32 weight matrix. in must be a multiple of QK.
QWeight quantize_weight(const float* W, int64_t out, int64_t in);

// Re-lay-out an already-quantized matrix for the micro-kernel. Lossless.
QWeightPacked pack_weight(const QWeight& w);

// Convenience: quantize and pack in one step.
QWeightPacked quantize_and_pack(const float* W, int64_t out, int64_t in);

// Quantize one activation row of length `in` into SIGNED int8 in [-127,127],
// with ONE SCALE PER QK-BLOCK. a_s8 holds `in` bytes; scales holds in/QK floats.
//
// WHY PER-BLOCK AND NOT PER-ROW. A single scale for the whole row is the obvious
// implementation and it is badly wrong for transformers. LLM activations have
// "outlier features": a handful of channels whose magnitude is one to two orders
// of magnitude above the rest, in every layer, consistently. One row-wide scale
// is set by those outliers, so every ordinary channel collapses into a couple of
// int8 levels. Measured on Qwen2.5-0.5B, per-row activation scales gave 13.5%
// relative L2 error on logits and 14.6% top-1 agreement with HuggingFace — the
// model was, in effect, broken. Per-QK-block scales confine each outlier's
// damage to its own block. The numbers are in docs/02-int8-saturation.md.
//
// This costs nothing in the kernel: the per-block dequantize multiplier was
// already a[scale] * w[scale][b], so it simply becomes a[scale][b] * w[scale][b].
void quantize_activation_row(const float* x, int64_t in, int8_t* a_s8, float* scales);

// Largest |int16 partial| the maddubs idiom would produce for this block
// pairing (under the sign-epi8 scheme). Bounded by 32258; a value reaching
// 32767 would mean saturation. Used by the saturation study.
int32_t max_int16_partial(const int8_t* a_s8, const int8_t* w_s8, int64_t n);

// Same, but for the naive +128 unsigned-activation scheme the bible originally
// proposed. Returns the worst pairwise sum BEFORE int16 clamping, so a value
// above 32767 is a proof of saturation rather than a suspicion of it. This is
// what docs/02-int8-saturation.md measures on the real model weights.
int32_t max_int16_partial_zp128(const int8_t* a_s8, const int8_t* w_s8, int64_t n);

}  // namespace quant
