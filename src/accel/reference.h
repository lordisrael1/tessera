#pragma once
#include <cmath>
#include <vector>

#include "../ops/ops.h"
#include "lower.h"

// ============================================================================
// The CPU fp32 reference for one decode step, written against an HbmLayout.
// ============================================================================
// This is the thing the T1 path must equal BIT FOR BIT. It reads the SAME HBM
// image the simulator does, so there is exactly one copy of the weights and no
// chance of the two paths being fed different numbers.
//
// It is written directly against ops:: in the order the compiler emits, and it
// is deliberately NOT built out of any compiler machinery — a reference that
// shares code with the thing it checks proves nothing. It lives in a header
// shared by test/compiler_parity.cpp (tiny synthetic model, every commit) and
// tools/run_sim.cpp (the real 0.5B, on demand) so that the two cannot drift
// apart and quietly test different semantics.
//
// Precondition: hbm[layout.act_in .. +hidden] holds the token embedding, and
// kcache/vcache hold positions [0, pos).
// ============================================================================

namespace accel {

inline void cpu_reference_decode(const HbmLayout& L, const float* hbm, int64_t pos,
                                 float* kcache, float* vcache, float* logits_out) {
    const ModelConfig& cfg = L.cfg;
    const int64_t H = cfg.hidden_size, HD = cfg.head_dim();
    const int64_t NH = cfg.num_attention_heads, NKV = cfg.num_key_value_heads;
    const int64_t QDIM = NH * HD, KVDIM = NKV * HD, INTER = cfg.intermediate_size;
    const int64_t V = cfg.vocab_size, qpk = cfg.q_per_kv();

    std::vector<float> h(hbm + L.act_in, hbm + L.act_in + H);
    std::vector<float> normed(static_cast<size_t>(H)), q(static_cast<size_t>(QDIM));
    std::vector<float> kcur(static_cast<size_t>(KVDIM)), vcur(static_cast<size_t>(KVDIM));
    std::vector<float> attn(static_cast<size_t>(QDIM)), o(static_cast<size_t>(H));
    std::vector<float> scores(static_cast<size_t>(pos + 1));
    std::vector<float> gate(static_cast<size_t>(INTER)), up(static_cast<size_t>(INTER));
    std::vector<float> ffn(static_cast<size_t>(INTER)), down(static_cast<size_t>(H));

    for (int64_t li = 0; li < cfg.num_hidden_layers; ++li) {
        const auto& la = L.layers[static_cast<size_t>(li)];
        ops::rmsnorm(h.data(), hbm + la.in_ln, normed.data(), H, cfg.rms_norm_eps);
        ops::linear(normed.data(), hbm + la.q_w, hbm + la.q_b, q.data(), 1, H, QDIM);
        ops::linear(normed.data(), hbm + la.k_w, hbm + la.k_b, kcur.data(), 1, H, KVDIM);
        ops::linear(normed.data(), hbm + la.v_w, hbm + la.v_b, vcur.data(), 1, H, KVDIM);

        for (int64_t hh = 0; hh < NH; ++hh) ops::rope(q.data() + hh * HD, HD, pos, cfg.rope_theta);
        for (int64_t hh = 0; hh < NKV; ++hh)
            ops::rope(kcur.data() + hh * HD, HD, pos, cfg.rope_theta);

        float* kslot = kcache + (li * L.max_seq + pos) * KVDIM;
        float* vslot = vcache + (li * L.max_seq + pos) * KVDIM;
        for (int64_t i = 0; i < KVDIM; ++i) {
            kslot[i] = kcur[static_cast<size_t>(i)];
            vslot[i] = vcur[static_cast<size_t>(i)];
        }

        for (int64_t qh = 0; qh < NH; ++qh) {
            const int64_t kvh = qh / qpk;
            for (int64_t t = 0; t <= pos; ++t) {
                const float* kt = kcache + (li * L.max_seq + t) * KVDIM + kvh * HD;
                double s = 0.0;
                for (int64_t d = 0; d < HD; ++d)
                    s += static_cast<double>(q[static_cast<size_t>(qh * HD + d)]) * kt[d];
                scores[static_cast<size_t>(t)] = static_cast<float>(s);
            }
            // The compiler emits QK^T then a SEPARATE V_SCALE, so the reference
            // must scale after the dot too — folding 1/sqrt(d) into the dot
            // would change the rounding and break bit-exactness.
            const float sc = 1.0f / std::sqrt(static_cast<float>(HD));
            for (int64_t t = 0; t <= pos; ++t) scores[static_cast<size_t>(t)] *= sc;
            ops::softmax(scores.data(), pos + 1);

            for (int64_t d = 0; d < HD; ++d) {
                double s = 0.0;
                for (int64_t t = 0; t <= pos; ++t) {
                    const float* vt = vcache + (li * L.max_seq + t) * KVDIM + kvh * HD;
                    s += static_cast<double>(scores[static_cast<size_t>(t)]) * vt[d];
                }
                attn[static_cast<size_t>(qh * HD + d)] = static_cast<float>(s);
            }
        }

        ops::linear(attn.data(), hbm + la.o_w, nullptr, o.data(), 1, QDIM, H);
        for (int64_t i = 0; i < H; ++i)
            h[static_cast<size_t>(i)] += o[static_cast<size_t>(i)];

        ops::rmsnorm(h.data(), hbm + la.post_ln, normed.data(), H, cfg.rms_norm_eps);
        ops::linear(normed.data(), hbm + la.gate_w, nullptr, gate.data(), 1, H, INTER);
        ops::linear(normed.data(), hbm + la.up_w, nullptr, up.data(), 1, H, INTER);
        ops::silu_mul(gate.data(), up.data(), ffn.data(), INTER);
        ops::linear(ffn.data(), hbm + la.down_w, nullptr, down.data(), 1, INTER, H);
        for (int64_t i = 0; i < H; ++i)
            h[static_cast<size_t>(i)] += down[static_cast<size_t>(i)];
    }

    ops::rmsnorm(h.data(), hbm + L.final_ln, normed.data(), H, cfg.rms_norm_eps);
    ops::linear(normed.data(), hbm + L.embed, nullptr, logits_out, 1, H, V);
}

}  // namespace accel
