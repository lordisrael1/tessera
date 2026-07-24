#pragma once
#include <cstdint>
#include <string>

// Everything about the model geometry is DERIVED from config.json at startup,
// never hardcoded. The two traps this struct exists to make explicit:
//   * GQA: num_key_value_heads (2) << num_attention_heads (14) on Qwen2.5-0.5B.
//     KV cache is sized by KV heads, 7x smaller than the naive layout.
//   * Tied embeddings: tie_word_embeddings == true means there is NO
//     lm_head.weight tensor; logits = hidden @ embed_tokens.weight^T.
struct ModelConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t vocab_size = 0;
    int64_t max_position_embeddings = 0;
    float   rms_norm_eps = 1e-6f;
    float   rope_theta = 1000000.0f;
    bool    tie_word_embeddings = true;

    // Derived.
    int64_t head_dim() const { return hidden_size / num_attention_heads; }
    // Each KV head serves this many query heads.
    int64_t q_per_kv() const { return num_attention_heads / num_key_value_heads; }

    static ModelConfig from_json_file(const std::string& path);
    std::string summary() const;
};
