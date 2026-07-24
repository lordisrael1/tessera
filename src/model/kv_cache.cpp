#include "kv_cache.h"

KVCache::KVCache(const ModelConfig& cfg, int64_t max_seq)
    : n_layers_(cfg.num_hidden_layers),
      kv_dim_(cfg.num_key_value_heads * cfg.head_dim()),
      max_seq_(max_seq) {
    int64_t elems = n_layers_ * max_seq_ * kv_dim_;
    k_ = std::make_unique<float[]>(static_cast<size_t>(elems));
    v_ = std::make_unique<float[]>(static_cast<size_t>(elems));
}
