#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "../core/arena.h"
#include "../core/safetensors.h"
#include "../ops/quant.h"
#include "config.h"
#include "kv_cache.h"

// The Qwen2.5-0.5B forward pass, in two precisions off one implementation.
//
//   Precision::F32 — the M1 reference. Every projection is ops::linear_fast,
//                    which accumulates in double and is the path logit_parity
//                    holds to <1e-3 against HuggingFace, forever.
//   Precision::I8  — the M2 path. Every projection weight is quantized to
//                    per-QK-block symmetric INT8 (QK=32; see quant.h) and
//                    packed for the AVX2 micro-kernel at load time. Same
//                    control flow, same KV
//                    cache, same everything else; the only difference is which
//                    kernel a WMat dispatches to. That is deliberate: a second
//                    forward pass written specially for INT8 would be able to
//                    drift from the reference in ways no test could see.
//
// Two entry points from day one because M4 depends on the split existing:
//   prefill()     — process N prompt tokens, fill KV, return last-position logits
//   decode_step() — one token in, one logits row out

enum class Precision { F32, I8 };

// The BISECTION LADDER. When logits disagree with the oracle, the question is
// never "are they wrong" but "where did they first go wrong". These three taps
// are the same three points tools/dump_logits.py records on the HuggingFace
// side, so the first one that diverges names the culprit:
//   h_embed  differs -> embedding lookup / tokenizer ids
//   h_layer0 differs -> attention or FFN (rope convention, GQA head mapping...)
//   h_final  differs -> a later layer, or the final norm
//   only logits differ -> the tied-embedding output projection
struct ForwardTaps {
    std::vector<float> h_embed;   // residual stream straight out of the embedding
    std::vector<float> h_layer0;  // residual stream after layer 0
    std::vector<float> h_final;   // after the final RMSNorm, i.e. lm_head input
};

// One projection matrix, in whichever precision the model was built with.
// Holding both representations behind one apply() is what keeps the forward
// pass free of `if (int8)` branches.
class WMat {
public:
    void build(std::vector<float>&& w, int64_t out, int64_t in, Precision p);
    void apply(const float* x, const float* bias, float* y, int64_t M) const;
    int64_t in() const { return in_; }
    int64_t out() const { return out_; }
    // Resident bytes, for the memory-traffic accounting in docs/01-roofline.md.
    std::size_t bytes() const;

private:
    std::vector<float> f32_;
    quant::QWeightPacked q_;
    int64_t in_ = 0, out_ = 0;
    bool quantized_ = false;
};

class Qwen2 {
public:
    Qwen2(const ModelConfig& cfg, const std::string& safetensors_path,
          Precision precision = Precision::F32);

    void prefill(const int32_t* ids, int64_t n, KVCache& kv, float* logits_out);
    void decode_step(int32_t id, KVCache& kv, float* logits_out);

    // Records the ladder for every token forwarded (so after prefill it holds
    // the last prompt position). Pass nullptr to disable; disabled by default so
    // the per-token path costs one never-taken branch.
    void set_taps(ForwardTaps* t) { taps_ = t; }

    const ModelConfig& config() const { return cfg_; }
    Precision precision() const { return precision_; }
    int64_t vocab_size() const { return cfg_.vocab_size; }
    std::size_t activation_high_water() const { return arena_.high_water(); }
    // Total weight bytes touched by one forward pass. The numerator of the
    // decode-is-memory-bound argument.
    std::size_t weight_bytes() const;

private:
    // Runs the network for one token at absolute position `pos`, appends its
    // K/V into the cache, and writes vocab logits. prefill == decode looped.
    void forward_one(int32_t id, int64_t pos, KVCache& kv, float* logits_out);

    // Load a [OUT,IN] (or [N]) tensor as f32, asserting the element count.
    std::vector<float> load(const std::string& name, int64_t expected);
    // Load only if the tensor exists (optional biases / untied lm_head).
    std::vector<float> load_optional(const std::string& name, int64_t expected);
    // Load a projection and immediately build it in the model's precision.
    void load_mat(const std::string& name, WMat& dst, int64_t out, int64_t in);

    ModelConfig cfg_;
    Precision precision_;
    SafeTensors st_;

    struct Layer {
        std::vector<float> in_ln;         // [hidden]
        WMat q_w, k_w, v_w, o_w;
        std::vector<float> q_b, k_b, v_b; // Qwen2 has QKV biases; Llama does not
        std::vector<float> post_ln;       // [hidden]
        WMat gate_w, up_w, down_w;
    };

    std::vector<float> embed_;     // [vocab, hidden] fp32; the embedding LOOKUP
                                   // is a gather, never a matmul, so it stays
                                   // fp32 in both precisions.
    WMat lm_head_;                 // tied to embed_ or a separate matrix
    std::vector<Layer> layers_;
    std::vector<float> final_ln_;  // [hidden]

    ForwardTaps* taps_ = nullptr;

    Arena arena_;  // per-token activation scratch; reset at top of forward_one
};
