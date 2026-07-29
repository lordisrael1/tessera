#pragma once
#include <cstdint>
#include <memory>
#include "config.h"

// One up-front slab for the whole KV cache (project rule: no per-token alloc).
//
// GQA trap: this is sized by num_key_value_heads (2 on Qwen2.5-0.5B), NOT
// num_attention_heads (14). Each KV head is shared by q_per_kv() query heads,
// so the cache is 7x smaller than a naive per-query-head layout.
//
// Layout per tensor, contiguous:  [layer][position][kv_head * head_dim]
class KVCache {
public:
    KVCache(const ModelConfig& cfg, int64_t max_seq);

    int64_t max_seq() const { return max_seq_; }
    int64_t len() const { return len_; }
    void set_len(int64_t n) { len_ = n; }
    void reset() { len_ = 0; }

    // Pointer to the K (or V) vector for (layer, pos): kv_heads*head_dim floats.
    float* k_at(int64_t layer, int64_t pos) {
        return k_.get() + (layer * max_seq_ + pos) * kv_dim_;
    }
    float* v_at(int64_t layer, int64_t pos) {
        return v_.get() + (layer * max_seq_ + pos) * kv_dim_;
    }
    const float* k_at(int64_t layer, int64_t pos) const {
        return k_.get() + (layer * max_seq_ + pos) * kv_dim_;
    }
    const float* v_at(int64_t layer, int64_t pos) const {
        return v_.get() + (layer * max_seq_ + pos) * kv_dim_;
    }

    int64_t kv_dim() const { return kv_dim_; }         // kv_heads * head_dim
    int64_t bytes() const { return 2 * n_layers_ * max_seq_ * kv_dim_ * (int64_t)sizeof(float); }

    // Throws if position `pos` would not fit. Called ONCE per token by the model
    // (not per k_at()) so the hot path stays branch-free while an over-long
    // prompt still fails loudly instead of scribbling past the slab.
    void require_pos(int64_t pos) const;

private:
    int64_t n_layers_, kv_dim_, max_seq_, len_ = 0;
    std::unique_ptr<float[]> k_, v_;
};
