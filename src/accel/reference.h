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

// ---------------------------------------------------------------------------
// The same thing for a PREFILL CHUNK: `ntokens` tokens starting at absolute
// position `base`, whose embeddings the host has written to
// hbm[act_in .. + ntokens*hidden].
//
// LAYER-MAJOR, like the compiled program: every token goes through layer 0,
// then every token through layer 1. That is not the order the M1 CPU model uses
// (it runs one token through all layers), and it does not need to be — causal
// masking means token r never reads a key past r, so the two orders are
// numerically identical. What DOES have to match is the order of operations
// inside each dot product, and that is what this function exists to pin down.
//
// Only the LAST token's logits are produced. Prefill exists to fill the KV
// cache; the other M-1 logit rows would be 545 MB of weight traffic apiece for
// values nobody reads.
// ---------------------------------------------------------------------------
inline void cpu_reference_prefill(const HbmLayout& L, const float* hbm, int64_t base,
                                  int64_t ntokens, float* kcache, float* vcache,
                                  float* logits_out) {
    const ModelConfig& cfg = L.cfg;
    const int64_t H = cfg.hidden_size, HD = cfg.head_dim();
    const int64_t NH = cfg.num_attention_heads, NKV = cfg.num_key_value_heads;
    const int64_t QDIM = NH * HD, KVDIM = NKV * HD, INTER = cfg.intermediate_size;
    const int64_t V = cfg.vocab_size, qpk = cfg.q_per_kv();
    const int64_t M = ntokens, KEYS = base + ntokens;

    // This chunk's embeddings start at row `base`, not at row 0 — the same
    // offset the compiled DMA descriptor uses.
    std::vector<float> h(hbm + L.act_in + base * H, hbm + L.act_in + (base + M) * H);
    std::vector<float> normed(static_cast<size_t>(M * H)), q(static_cast<size_t>(M * QDIM));
    std::vector<float> kcur(static_cast<size_t>(M * KVDIM)), vcur(static_cast<size_t>(M * KVDIM));
    std::vector<float> attn(static_cast<size_t>(M * QDIM)), o(static_cast<size_t>(M * H));
    std::vector<float> scores(static_cast<size_t>(KEYS));
    std::vector<float> gate(static_cast<size_t>(M * INTER)), up(static_cast<size_t>(M * INTER));
    std::vector<float> ffn(static_cast<size_t>(M * INTER)), down(static_cast<size_t>(M * H));

    for (int64_t li = 0; li < cfg.num_hidden_layers; ++li) {
        const auto& la = L.layers[static_cast<size_t>(li)];
        for (int64_t r = 0; r < M; ++r)
            ops::rmsnorm(h.data() + r * H, hbm + la.in_ln, normed.data() + r * H, H,
                         cfg.rms_norm_eps);
        ops::linear(normed.data(), hbm + la.q_w, hbm + la.q_b, q.data(), M, H, QDIM);
        ops::linear(normed.data(), hbm + la.k_w, hbm + la.k_b, kcur.data(), M, H, KVDIM);
        ops::linear(normed.data(), hbm + la.v_w, hbm + la.v_b, vcur.data(), M, H, KVDIM);

        for (int64_t r = 0; r < M; ++r) {
            for (int64_t hh = 0; hh < NH; ++hh)
                ops::rope(q.data() + r * QDIM + hh * HD, HD, base + r, cfg.rope_theta);
            for (int64_t hh = 0; hh < NKV; ++hh)
                ops::rope(kcur.data() + r * KVDIM + hh * HD, HD, base + r, cfg.rope_theta);
        }

        for (int64_t r = 0; r < M; ++r) {
            float* kslot = kcache + (li * L.max_seq + base + r) * KVDIM;
            float* vslot = vcache + (li * L.max_seq + base + r) * KVDIM;
            for (int64_t i = 0; i < KVDIM; ++i) {
                kslot[i] = kcur[static_cast<size_t>(r * KVDIM + i)];
                vslot[i] = vcur[static_cast<size_t>(r * KVDIM + i)];
            }
        }

        for (int64_t qh = 0; qh < NH; ++qh) {
            const int64_t kvh = qh / qpk;
            for (int64_t r = 0; r < M; ++r) {
                const int64_t n = base + r + 1;  // causal: token r sees keys [0, n)
                for (int64_t t = 0; t < n; ++t) {
                    const float* kt = kcache + (li * L.max_seq + t) * KVDIM + kvh * HD;
                    double s = 0.0;
                    for (int64_t d = 0; d < HD; ++d)
                        s += static_cast<double>(q[static_cast<size_t>(r * QDIM + qh * HD + d)]) *
                             kt[d];
                    scores[static_cast<size_t>(t)] = static_cast<float>(s);
                }
                const float sc = 1.0f / std::sqrt(static_cast<float>(HD));
                for (int64_t t = 0; t < n; ++t) scores[static_cast<size_t>(t)] *= sc;
                ops::softmax(scores.data(), n);
                // The machine sums over all KEYS, but the masked entries are
                // exactly 0.0 and adding 0.0 to a double accumulator changes
                // nothing, so summing the valid prefix is bit-identical.
                for (int64_t d = 0; d < HD; ++d) {
                    double s = 0.0;
                    for (int64_t t = 0; t < n; ++t) {
                        const float* vt = vcache + (li * L.max_seq + t) * KVDIM + kvh * HD;
                        s += static_cast<double>(scores[static_cast<size_t>(t)]) * vt[d];
                    }
                    attn[static_cast<size_t>(r * QDIM + qh * HD + d)] = static_cast<float>(s);
                }
            }
        }

        ops::linear(attn.data(), hbm + la.o_w, nullptr, o.data(), M, QDIM, H);
        for (int64_t i = 0; i < M * H; ++i)
            h[static_cast<size_t>(i)] += o[static_cast<size_t>(i)];

        for (int64_t r = 0; r < M; ++r)
            ops::rmsnorm(h.data() + r * H, hbm + la.post_ln, normed.data() + r * H, H,
                         cfg.rms_norm_eps);
        ops::linear(normed.data(), hbm + la.gate_w, nullptr, gate.data(), M, H, INTER);
        ops::linear(normed.data(), hbm + la.up_w, nullptr, up.data(), M, H, INTER);
        ops::silu_mul(gate.data(), up.data(), ffn.data(), M * INTER);
        ops::linear(ffn.data(), hbm + la.down_w, nullptr, down.data(), M, INTER, H);
        for (int64_t i = 0; i < M * H; ++i)
            h[static_cast<size_t>(i)] += down[static_cast<size_t>(i)];
    }

    ops::rmsnorm(h.data() + (M - 1) * H, hbm + L.final_ln, normed.data(), H, cfg.rms_norm_eps);
    ops::linear(normed.data(), hbm + L.embed, nullptr, logits_out, 1, H, V);
}

}  // namespace accel
