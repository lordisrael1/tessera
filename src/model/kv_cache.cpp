#include "kv_cache.h"

#include <stdexcept>
#include <string>

KVCache::KVCache(const ModelConfig& cfg, int64_t max_seq)
    : n_layers_(cfg.num_hidden_layers),
      kv_dim_(cfg.num_key_value_heads * cfg.head_dim()),
      max_seq_(max_seq) {
    if (n_layers_ <= 0 || kv_dim_ <= 0 || max_seq_ <= 0)
        throw std::runtime_error("kv_cache: non-positive geometry");
    int64_t elems = n_layers_ * max_seq_ * kv_dim_;
    // make_unique<float[]> value-initializes, so the slab starts zeroed. That is
    // deliberate: reading an unwritten position then yields 0 rather than
    // whatever the allocator left behind, which keeps bugs reproducible.
    k_ = std::make_unique<float[]>(static_cast<size_t>(elems));
    v_ = std::make_unique<float[]>(static_cast<size_t>(elems));
}

void KVCache::require_pos(int64_t pos) const {
    if (pos < 0 || pos >= max_seq_)
        throw std::runtime_error("kv_cache: position " + std::to_string(pos) +
                                 " out of range [0," + std::to_string(max_seq_) +
                                 ") — grow max_seq or shorten the prompt");
}
