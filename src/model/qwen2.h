#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "../core/arena.h"
#include "../core/safetensors.h"
#include "config.h"
#include "kv_cache.h"

// The reference fp32 forward pass for Qwen2.5-0.5B. Scalar ops only (the M2 AVX2
// kernel and M3 simulator must match THIS bit-for-bit given the same op order).
//
// Two entry points from day one because M4 depends on the split existing:
//   prefill()     — process N prompt tokens, fill KV, return last-position logits
//   decode_step() — one token in, one logits row out
class Qwen2 {
public:
    Qwen2(const ModelConfig& cfg, const std::string& safetensors_path,
          int64_t max_seq = 4096);

    void prefill(const int32_t* ids, int64_t n, KVCache& kv, float* logits_out);
    void decode_step(int32_t id, KVCache& kv, float* logits_out);

    const ModelConfig& config() const { return cfg_; }
    int64_t vocab_size() const { return cfg_.vocab_size; }
    std::size_t activation_high_water() const { return arena_.high_water(); }

private:
    // Runs the network for one token at absolute position `pos`, appends its
    // K/V into the cache, and writes vocab logits. prefill == decode looped.
    void forward_one(int32_t id, int64_t pos, KVCache& kv, float* logits_out);

    // Load a [OUT,IN] (or [N]) tensor as f32 into dst, asserting the element count.
    void load(const std::string& name, std::vector<float>& dst, int64_t expected);
    // Load only if the tensor exists (optional biases / untied lm_head).
    // Returns true if loaded; leaves dst empty and returns false otherwise.
    bool load_optional(const std::string& name, std::vector<float>& dst, int64_t expected);

    ModelConfig cfg_;
    SafeTensors st_;

    struct Layer {
        std::vector<float> in_ln;                  // [hidden]
        std::vector<float> q_w, q_b;               // [n_heads*hd, hidden], [n_heads*hd]
        std::vector<float> k_w, k_b;               // [n_kv*hd, hidden], [n_kv*hd]
        std::vector<float> v_w, v_b;
        std::vector<float> o_w;                    // [hidden, n_heads*hd] (no bias)
        std::vector<float> post_ln;                // [hidden]
        std::vector<float> gate_w, up_w;           // [inter, hidden]
        std::vector<float> down_w;                 // [hidden, inter]
    };

    std::vector<float> embed_;     // [vocab, hidden]; also the tied lm_head
    std::vector<float> lm_head_;   // [vocab, hidden]; only when NOT tied
    std::vector<Layer> layers_;
    std::vector<float> final_ln_;  // [hidden]

    // The output projection weight: tied to embeddings, or a separate lm_head.
    const float* lm_head_weight() const {
        return cfg_.tie_word_embeddings ? embed_.data() : lm_head_.data();
    }

    Arena arena_;  // per-token activation scratch; reset at top of forward_one
};
