#include "quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace quant {
namespace {

// Quantize one QK-block, choosing the scale that minimises squared error.
//
// The obvious scale is amax/127 — it is what llama.cpp's Q8_0 uses and it has
// the nice property that nothing ever clips. But it is chosen by the single
// largest magnitude in the block, so one outlier stretches the grid and every
// other value in the block rounds onto a coarser lattice than it needed to. A
// slightly SMALLER scale clips the outlier a little and buys back resolution for
// the other 31 values, and for weight distributions (roughly Gaussian with the
// occasional spike) that trade is usually worth taking.
//
// So: try a handful of shrink factors, keep whichever minimises the true
// reconstruction error sum (w - q*scale)^2. Nine candidates, load-time only,
// and no change whatsoever to the kernel — the output is still a plain int8
// block plus one float. Measured contribution is in docs/02-int8-saturation.md.
float quantize_block(const float* w, int8_t* q) {
    float amax = 0.0f;
    for (int k = 0; k < QK; ++k) amax = std::max(amax, std::fabs(w[k]));
    if (amax == 0.0f) {
        for (int k = 0; k < QK; ++k) q[k] = 0;
        return 0.0f;
    }

    float best_scale = amax / 127.0f;
    double best_err = -1.0;
    int8_t cand[QK];
    for (int step = 0; step <= 8; ++step) {
        // 1.00 down to 0.80 of the amax scale.
        float scale = (amax / 127.0f) * (1.0f - 0.025f * static_cast<float>(step));
        float inv = 1.0f / scale;
        double err = 0.0;
        for (int k = 0; k < QK; ++k) {
            int v = static_cast<int>(std::lround(w[k] * inv));
            v = std::max(-127, std::min(127, v));
            cand[k] = static_cast<int8_t>(v);
            double d = static_cast<double>(w[k]) - static_cast<double>(v) * scale;
            err += d * d;
        }
        if (best_err < 0.0 || err < best_err) {
            best_err = err;
            best_scale = scale;
            std::memcpy(q, cand, QK);
        }
    }
    return best_scale;
}

}  // namespace

QWeight quantize_weight(const float* W, int64_t out, int64_t in) {
    if (in % QK != 0)
        throw std::runtime_error("quantize_weight: in must be a multiple of QK");
    QWeight qw;
    qw.out = out;
    qw.in = in;
    qw.blocks = in / QK;
    qw.q.resize(static_cast<size_t>(out * in));
    qw.scale.resize(static_cast<size_t>(out * qw.blocks));

    for (int64_t o = 0; o < out; ++o) {
        for (int64_t b = 0; b < qw.blocks; ++b) {
            const float* wblk = W + o * in + b * QK;
            int8_t* qblk = qw.q.data() + o * in + b * QK;
            qw.scale[static_cast<size_t>(o * qw.blocks + b)] = quantize_block(wblk, qblk);
        }
    }
    return qw;
}

QWeightPacked pack_weight(const QWeight& w) {
    QWeightPacked p;
    p.out = w.out;
    p.in = w.in;
    p.blocks = w.blocks;
    p.panels = (w.out + NRP - 1) / NRP;
    // Zero-fill covers the padding rows of the last panel: their weights are 0
    // and their scales are 0, so they produce exact zeros rather than garbage.
    p.q.assign(static_cast<size_t>(p.panels * p.blocks * NRP * QK), 0);
    p.scale.assign(static_cast<size_t>(p.panels * p.blocks * NRP), 0.0f);

    for (int64_t o = 0; o < w.out; ++o) {
        int64_t pi = o / NRP, r = o % NRP;
        for (int64_t b = 0; b < w.blocks; ++b) {
            std::memcpy(p.q.data() + ((pi * w.blocks + b) * NRP + r) * QK,
                        w.row(o) + b * QK, QK);
            p.scale[static_cast<size_t>((pi * w.blocks + b) * NRP + r)] = w.scale_at(o, b);
        }
    }
    return p;
}

QWeightPacked quantize_and_pack(const float* W, int64_t out, int64_t in) {
    return pack_weight(quantize_weight(W, out, in));
}

void quantize_activation_row(const float* x, int64_t in, int8_t* a_s8, float* scales) {
    const int64_t nblocks = in / QK;
    for (int64_t b = 0; b < nblocks; ++b) {
        const float* xb = x + b * QK;
        float amax = 0.0f;
        for (int k = 0; k < QK; ++k) amax = std::max(amax, std::fabs(xb[k]));
        float scale = amax / 127.0f;
        float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
        int8_t* qb = a_s8 + b * QK;
        for (int k = 0; k < QK; ++k) {
            int v = static_cast<int>(std::lround(xb[k] * inv));
            v = std::max(-127, std::min(127, v));  // signed int8, no +128 shift
            qb[k] = static_cast<int8_t>(v);
        }
        scales[b] = scale;
    }
    // Any tail beyond the last whole block would need its own scale; every
    // reduction dimension in this model is a multiple of QK, so assert-by-
    // construction (quantize_weight throws otherwise) rather than pad here.
}

int32_t max_int16_partial(const int8_t* a_s8, const int8_t* w_s8, int64_t n) {
    int32_t worst = 0;
    for (int64_t i = 0; i + 1 < n; i += 2) {
        // The kernel's unsigned operand is |a| and its signed operand is
        // w*sign(a); the product of the pair is therefore |a|*w*sign(a) == a*w.
        int32_t p = static_cast<int32_t>(a_s8[i]) * w_s8[i] +
                    static_cast<int32_t>(a_s8[i + 1]) * w_s8[i + 1];
        worst = std::max(worst, std::abs(p));
    }
    return worst;
}

int32_t max_int16_partial_zp128(const int8_t* a_s8, const int8_t* w_s8, int64_t n) {
    int32_t worst = 0;
    for (int64_t i = 0; i + 1 < n; i += 2) {
        // The scheme the bible proposed: shift activations to unsigned by +128,
        // so the unsigned operand ranges over [1, 255] instead of [0, 127].
        int32_t a0 = static_cast<int32_t>(a_s8[i]) + 128;
        int32_t a1 = static_cast<int32_t>(a_s8[i + 1]) + 128;
        int32_t p = a0 * w_s8[i] + a1 * w_s8[i + 1];
        worst = std::max(worst, std::abs(p));
    }
    return worst;
}

}  // namespace quant
