#include "qwen2.h"
#include "../ops/ops.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {
// Sizing the activation arena: the largest transient buffers are the two FFN
// projections (intermediate = 4864) plus assorted hidden-sized scratch. A few
// hundred KB total per token; 8 MB is comfortable headroom and its high_water()
// will report the true working set for M3's scratchpad sizing.
constexpr std::size_t kArenaBytes = 8u * 1024u * 1024u;

float dot(const float* a, const float* b, int64_t n) {
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) acc += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(acc);
}
}  // namespace

void Qwen2::load(const std::string& name, std::vector<float>& dst, int64_t expected) {
    const TensorInfo& t = st_.get(name);
    if (t.numel() != expected)
        throw std::runtime_error("qwen2: tensor '" + name + "' numel " +
                                 std::to_string(t.numel()) + " != expected " +
                                 std::to_string(expected));
    dst.resize(static_cast<size_t>(expected));
    st_.load_as_f32(name, dst.data());
}

bool Qwen2::load_optional(const std::string& name, std::vector<float>& dst, int64_t expected) {
    if (!st_.has(name)) {
        dst.clear();
        return false;
    }
    load(name, dst, expected);
    return true;
}

Qwen2::Qwen2(const ModelConfig& cfg, const std::string& safetensors_path, int64_t max_seq)
    : cfg_(cfg), st_(safetensors_path), arena_(kArenaBytes) {
    (void)max_seq;
    const int64_t H = cfg_.hidden_size;
    const int64_t HD = cfg_.head_dim();
    const int64_t QDIM = cfg_.num_attention_heads * HD;    // 896
    const int64_t KVDIM = cfg_.num_key_value_heads * HD;   // 128
    const int64_t INTER = cfg_.intermediate_size;
    const int64_t V = cfg_.vocab_size;

    load("model.embed_tokens.weight", embed_, V * H);
    // Tied embeddings (Qwen2.5): no lm_head.weight tensor exists; logits reuse
    // the embedding matrix. Only load a separate head when NOT tied.
    if (!cfg_.tie_word_embeddings) load("lm_head.weight", lm_head_, V * H);

    layers_.resize(static_cast<size_t>(cfg_.num_hidden_layers));
    for (int64_t i = 0; i < cfg_.num_hidden_layers; ++i) {
        Layer& L = layers_[static_cast<size_t>(i)];
        std::string p = "model.layers." + std::to_string(i) + ".";
        load(p + "input_layernorm.weight", L.in_ln, H);
        load(p + "self_attn.q_proj.weight", L.q_w, QDIM * H);
        load(p + "self_attn.k_proj.weight", L.k_w, KVDIM * H);
        load(p + "self_attn.v_proj.weight", L.v_w, KVDIM * H);
        // Qwen2 has Q/K/V projection biases (Llama does not) — load when present.
        load_optional(p + "self_attn.q_proj.bias", L.q_b, QDIM);
        load_optional(p + "self_attn.k_proj.bias", L.k_b, KVDIM);
        load_optional(p + "self_attn.v_proj.bias", L.v_b, KVDIM);
        load(p + "self_attn.o_proj.weight", L.o_w, H * QDIM);
        load(p + "post_attention_layernorm.weight", L.post_ln, H);
        load(p + "mlp.gate_proj.weight", L.gate_w, INTER * H);
        load(p + "mlp.up_proj.weight", L.up_w, INTER * H);
        load(p + "mlp.down_proj.weight", L.down_w, H * INTER);
    }
    load("model.norm.weight", final_ln_, H);
}

void Qwen2::forward_one(int32_t id, int64_t pos, KVCache& kv, float* logits_out) {
    arena_.reset();
    const int64_t H = cfg_.hidden_size;
    const int64_t HD = cfg_.head_dim();
    const int64_t NH = cfg_.num_attention_heads;
    const int64_t NKV = cfg_.num_key_value_heads;
    const int64_t QDIM = NH * HD;
    const int64_t KVDIM = NKV * HD;
    const int64_t INTER = cfg_.intermediate_size;
    const int64_t V = cfg_.vocab_size;
    const int64_t qpk = cfg_.q_per_kv();
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    float* h = arena_.alloc_n<float>(static_cast<size_t>(H));
    std::memcpy(h, &embed_[static_cast<size_t>(id) * static_cast<size_t>(H)],
                static_cast<size_t>(H) * sizeof(float));

    float* normed = arena_.alloc_n<float>(static_cast<size_t>(H));
    float* q = arena_.alloc_n<float>(static_cast<size_t>(QDIM));
    float* kcur = arena_.alloc_n<float>(static_cast<size_t>(KVDIM));
    float* vcur = arena_.alloc_n<float>(static_cast<size_t>(KVDIM));
    float* attn = arena_.alloc_n<float>(static_cast<size_t>(QDIM));
    float* o = arena_.alloc_n<float>(static_cast<size_t>(H));
    float* scores = arena_.alloc_n<float>(static_cast<size_t>(pos + 1));
    float* gate = arena_.alloc_n<float>(static_cast<size_t>(INTER));
    float* up = arena_.alloc_n<float>(static_cast<size_t>(INTER));
    float* ffn = arena_.alloc_n<float>(static_cast<size_t>(INTER));
    float* down = arena_.alloc_n<float>(static_cast<size_t>(H));

    for (int64_t li = 0; li < cfg_.num_hidden_layers; ++li) {
        const Layer& L = layers_[static_cast<size_t>(li)];

        // --- attention ---
        ops::rmsnorm(h, L.in_ln.data(), normed, H, cfg_.rms_norm_eps);
        ops::linear(normed, L.q_w.data(), L.q_b.empty() ? nullptr : L.q_b.data(), q, 1, H, QDIM);
        ops::linear(normed, L.k_w.data(), L.k_b.empty() ? nullptr : L.k_b.data(), kcur, 1, H, KVDIM);
        ops::linear(normed, L.v_w.data(), L.v_b.empty() ? nullptr : L.v_b.data(), vcur, 1, H, KVDIM);

        for (int64_t hh = 0; hh < NH; ++hh)
            ops::rope(q + hh * HD, HD, pos, cfg_.rope_theta);
        for (int64_t hh = 0; hh < NKV; ++hh)
            ops::rope(kcur + hh * HD, HD, pos, cfg_.rope_theta);

        std::memcpy(kv.k_at(li, pos), kcur, static_cast<size_t>(KVDIM) * sizeof(float));
        std::memcpy(kv.v_at(li, pos), vcur, static_cast<size_t>(KVDIM) * sizeof(float));

        for (int64_t qh = 0; qh < NH; ++qh) {
            int64_t kvh = qh / qpk;
            const float* qvec = q + qh * HD;
            for (int64_t t = 0; t <= pos; ++t)
                scores[t] = dot(qvec, kv.k_at(li, t) + kvh * HD, HD) * scale;
            ops::softmax(scores, pos + 1);
            float* outv = attn + qh * HD;
            for (int64_t d = 0; d < HD; ++d) outv[d] = 0.0f;
            for (int64_t t = 0; t <= pos; ++t) {
                float w = scores[t];
                const float* vt = kv.v_at(li, t) + kvh * HD;
                for (int64_t d = 0; d < HD; ++d) outv[d] += w * vt[d];
            }
        }

        ops::linear(attn, L.o_w.data(), nullptr, o, 1, QDIM, H);
        for (int64_t i = 0; i < H; ++i) h[i] += o[i];

        // --- FFN (SwiGLU) ---
        ops::rmsnorm(h, L.post_ln.data(), normed, H, cfg_.rms_norm_eps);
        ops::linear(normed, L.gate_w.data(), nullptr, gate, 1, H, INTER);
        ops::linear(normed, L.up_w.data(), nullptr, up, 1, H, INTER);
        ops::silu_mul(gate, up, ffn, INTER);
        ops::linear(ffn, L.down_w.data(), nullptr, down, 1, INTER, H);
        for (int64_t i = 0; i < H; ++i) h[i] += down[i];
    }

    ops::rmsnorm(h, final_ln_.data(), normed, H, cfg_.rms_norm_eps);
    // logits = hidden @ (tied embedding | lm_head)^T.
    ops::linear(normed, lm_head_weight(), nullptr, logits_out, 1, H, V);
}

void Qwen2::prefill(const int32_t* ids, int64_t n, KVCache& kv, float* logits_out) {
    for (int64_t i = 0; i < n; ++i)
        forward_one(ids[i], i, kv, logits_out);  // last iteration leaves last-pos logits
    kv.set_len(n);
}

void Qwen2::decode_step(int32_t id, KVCache& kv, float* logits_out) {
    int64_t pos = kv.len();
    forward_one(id, pos, kv, logits_out);
    kv.set_len(pos + 1);
}
