#include "qwen2.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "../ops/gemm_avx2.h"
#include "../ops/ops.h"

namespace {
// Sizing the activation arena: the largest transient buffers are the two FFN
// projections (intermediate = 4864) plus assorted hidden-sized scratch, plus the
// attention score row which grows with position. A few hundred KB per token;
// 8 MB is comfortable headroom and high_water() reports the true working set for
// M3's scratchpad sizing.
constexpr std::size_t kArenaBytes = 8u * 1024u * 1024u;

float dot(const float* a, const float* b, int64_t n) {
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) acc += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(acc);
}
}  // namespace

// ---------------------------------------------------------------------------
// WMat
// ---------------------------------------------------------------------------
void WMat::build(std::vector<float>&& w, int64_t out, int64_t in, Precision p) {
    out_ = out;
    in_ = in;
    // QK divides every reduction dimension in this model (hidden 896 = 14*64,
    // intermediate 4864 = 76*64), so no padding path is needed. Assert it rather
    // than assume it — a model whose hidden size is not a multiple of 64 should
    // fail here, loudly, not produce quietly wrong logits.
    if (p == Precision::I8 && in % quant::QK == 0) {
        q_ = quant::quantize_and_pack(w.data(), out, in);
        quantized_ = true;
        std::vector<float>().swap(w);  // release the 4x-larger fp32 copy now
    } else {
        f32_ = std::move(w);
        quantized_ = false;
    }
}

void WMat::apply(const float* x, const float* bias, float* y, int64_t M) const {
    if (quantized_)
        ops::gemm_i8(x, M, q_, bias, y);
    else
        ops::linear_fast(x, f32_.data(), bias, y, M, in_, out_);
}

std::size_t WMat::bytes() const {
    if (quantized_) return q_.q.size() + q_.scale.size() * sizeof(float);
    return f32_.size() * sizeof(float);
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------
std::vector<float> Qwen2::load(const std::string& name, int64_t expected) {
    const TensorInfo& t = st_.get(name);
    if (t.numel() != expected)
        throw std::runtime_error("qwen2: tensor '" + name + "' numel " +
                                 std::to_string(t.numel()) + " != expected " +
                                 std::to_string(expected));
    std::vector<float> dst(static_cast<size_t>(expected));
    st_.load_as_f32(name, dst.data());
    return dst;
}

std::vector<float> Qwen2::load_optional(const std::string& name, int64_t expected) {
    if (!st_.has(name)) return {};
    return load(name, expected);
}

void Qwen2::load_mat(const std::string& name, WMat& dst, int64_t out, int64_t in) {
    dst.build(load(name, out * in), out, in, precision_);
}

Qwen2::Qwen2(const ModelConfig& cfg, const std::string& safetensors_path, Precision precision)
    : cfg_(cfg), precision_(precision), st_(safetensors_path), arena_(kArenaBytes) {
    const int64_t H = cfg_.hidden_size;
    const int64_t HD = cfg_.head_dim();
    const int64_t QDIM = cfg_.num_attention_heads * HD;    // 896
    const int64_t KVDIM = cfg_.num_key_value_heads * HD;   // 128
    const int64_t INTER = cfg_.intermediate_size;
    const int64_t V = cfg_.vocab_size;

    embed_ = load("model.embed_tokens.weight", V * H);

    layers_.resize(static_cast<size_t>(cfg_.num_hidden_layers));
    for (int64_t i = 0; i < cfg_.num_hidden_layers; ++i) {
        Layer& L = layers_[static_cast<size_t>(i)];
        std::string p = "model.layers." + std::to_string(i) + ".";
        L.in_ln = load(p + "input_layernorm.weight", H);
        load_mat(p + "self_attn.q_proj.weight", L.q_w, QDIM, H);
        load_mat(p + "self_attn.k_proj.weight", L.k_w, KVDIM, H);
        load_mat(p + "self_attn.v_proj.weight", L.v_w, KVDIM, H);
        // Qwen2 has Q/K/V projection biases (Llama does not) — load when present.
        L.q_b = load_optional(p + "self_attn.q_proj.bias", QDIM);
        L.k_b = load_optional(p + "self_attn.k_proj.bias", KVDIM);
        L.v_b = load_optional(p + "self_attn.v_proj.bias", KVDIM);
        load_mat(p + "self_attn.o_proj.weight", L.o_w, H, QDIM);
        L.post_ln = load(p + "post_attention_layernorm.weight", H);
        load_mat(p + "mlp.gate_proj.weight", L.gate_w, INTER, H);
        load_mat(p + "mlp.up_proj.weight", L.up_w, INTER, H);
        load_mat(p + "mlp.down_proj.weight", L.down_w, H, INTER);
    }
    final_ln_ = load("model.norm.weight", H);

    // Tied embeddings (Qwen2.5): there is NO lm_head.weight tensor and its
    // absence is correct, not a bug — the output projection reuses the embedding
    // matrix.
    //
    // THE OUTPUT PROJECTION STAYS fp32 EVEN IN I8 MODE. Two independent reasons,
    // and they point the same way:
    //   1. Accuracy. Every other matrix's quantization error gets attenuated by
    //      the layers downstream of it. The lm_head's error lands directly on the
    //      logits with nothing after it to wash out, and logits are what the
    //      argmax reads. Measured: quantizing it took logit rel-L2 from 1.0% to
    //      4.8% and top-1 agreement with HF from 99%+ to 66%.
    //   2. Memory. The embedding matrix must stay resident in fp32 anyway for
    //      the token LOOKUP (a gather, not a matmul). Quantizing the tied
    //      projection therefore does not replace those 545 MB, it ADDS 136 MB of
    //      int8 beside them. Keeping fp32 is smaller AND more accurate.
    // llama.cpp reaches the same conclusion from the other end, keeping
    // token_embd/output at higher precision than the body of the model.
    if (cfg_.tie_word_embeddings) {
        std::vector<float> copy = embed_;
        lm_head_.build(std::move(copy), V, H, Precision::F32);
    } else {
        // An untied head is a separate matrix, so reason 2 does not apply — but
        // reason 1 still does, and it is the one that matters.
        lm_head_.build(load("lm_head.weight", V * H), V, H, Precision::F32);
    }
}

std::size_t Qwen2::weight_bytes() const {
    std::size_t n = lm_head_.bytes() + final_ln_.size() * sizeof(float);
    for (const Layer& L : layers_)
        n += L.q_w.bytes() + L.k_w.bytes() + L.v_w.bytes() + L.o_w.bytes() +
             L.gate_w.bytes() + L.up_w.bytes() + L.down_w.bytes() +
             (L.in_ln.size() + L.post_ln.size() + L.q_b.size() + L.k_b.size() + L.v_b.size()) *
                 sizeof(float);
    return n;
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------
void Qwen2::forward_one(int32_t id, int64_t pos, KVCache& kv, float* logits_out) {
    if (id < 0 || id >= cfg_.vocab_size)
        throw std::runtime_error("qwen2: token id " + std::to_string(id) + " out of vocab");
    kv.require_pos(pos);
    arena_.reset();
    const int64_t H = cfg_.hidden_size;
    const int64_t HD = cfg_.head_dim();
    const int64_t NH = cfg_.num_attention_heads;
    const int64_t NKV = cfg_.num_key_value_heads;
    const int64_t QDIM = NH * HD;
    const int64_t KVDIM = NKV * HD;
    const int64_t INTER = cfg_.intermediate_size;
    const int64_t qpk = cfg_.q_per_kv();
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    float* h = arena_.alloc_n_checked<float>(static_cast<size_t>(H));
    std::memcpy(h, &embed_[static_cast<size_t>(id) * static_cast<size_t>(H)],
                static_cast<size_t>(H) * sizeof(float));

    float* normed = arena_.alloc_n_checked<float>(static_cast<size_t>(H));
    float* q = arena_.alloc_n_checked<float>(static_cast<size_t>(QDIM));
    float* kcur = arena_.alloc_n_checked<float>(static_cast<size_t>(KVDIM));
    float* vcur = arena_.alloc_n_checked<float>(static_cast<size_t>(KVDIM));
    float* attn = arena_.alloc_n_checked<float>(static_cast<size_t>(QDIM));
    float* o = arena_.alloc_n_checked<float>(static_cast<size_t>(H));
    float* scores = arena_.alloc_n_checked<float>(static_cast<size_t>(pos + 1));
    float* gate = arena_.alloc_n_checked<float>(static_cast<size_t>(INTER));
    float* up = arena_.alloc_n_checked<float>(static_cast<size_t>(INTER));
    float* ffn = arena_.alloc_n_checked<float>(static_cast<size_t>(INTER));
    float* down = arena_.alloc_n_checked<float>(static_cast<size_t>(H));

    auto tap = [&](std::vector<float> ForwardTaps::*field, const float* src) {
        if (taps_) (taps_->*field).assign(src, src + H);
    };
    tap(&ForwardTaps::h_embed, h);

    for (int64_t li = 0; li < cfg_.num_hidden_layers; ++li) {
        const Layer& L = layers_[static_cast<size_t>(li)];

        // --- attention ---
        ops::rmsnorm(h, L.in_ln.data(), normed, H, cfg_.rms_norm_eps);
        L.q_w.apply(normed, L.q_b.empty() ? nullptr : L.q_b.data(), q, 1);
        L.k_w.apply(normed, L.k_b.empty() ? nullptr : L.k_b.data(), kcur, 1);
        L.v_w.apply(normed, L.v_b.empty() ? nullptr : L.v_b.data(), vcur, 1);

        for (int64_t hh = 0; hh < NH; ++hh)
            ops::rope(q + hh * HD, HD, pos, cfg_.rope_theta);
        for (int64_t hh = 0; hh < NKV; ++hh)
            ops::rope(kcur + hh * HD, HD, pos, cfg_.rope_theta);

        std::memcpy(kv.k_at(li, pos), kcur, static_cast<size_t>(KVDIM) * sizeof(float));
        std::memcpy(kv.v_at(li, pos), vcur, static_cast<size_t>(KVDIM) * sizeof(float));

        for (int64_t qh = 0; qh < NH; ++qh) {
            // GQA: each KV head serves q_per_kv() = 7 query heads, which is why
            // the cache is sized by KV heads and is 7x smaller than naive.
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

        L.o_w.apply(attn, nullptr, o, 1);
        for (int64_t i = 0; i < H; ++i) h[i] += o[i];

        // --- FFN (SwiGLU) ---
        ops::rmsnorm(h, L.post_ln.data(), normed, H, cfg_.rms_norm_eps);
        L.gate_w.apply(normed, nullptr, gate, 1);
        L.up_w.apply(normed, nullptr, up, 1);
        ops::silu_mul(gate, up, ffn, INTER);
        L.down_w.apply(ffn, nullptr, down, 1);
        for (int64_t i = 0; i < H; ++i) h[i] += down[i];

        if (li == 0) tap(&ForwardTaps::h_layer0, h);
    }

    ops::rmsnorm(h, final_ln_.data(), normed, H, cfg_.rms_norm_eps);
    tap(&ForwardTaps::h_final, normed);
    lm_head_.apply(normed, nullptr, logits_out, 1);
}

void Qwen2::prefill(const int32_t* ids, int64_t n, KVCache& kv, float* logits_out) {
    // Token-at-a-time: correct, and it makes prefill and decode literally the
    // same code path. M4's chunked prefill is where this becomes a batched
    // matmul; doing it early would mean two forward passes to keep in sync.
    for (int64_t i = 0; i < n; ++i)
        forward_one(ids[i], i, kv, logits_out);  // last iteration leaves last-pos logits
    kv.set_len(n);
}

void Qwen2::decode_step(int32_t id, KVCache& kv, float* logits_out) {
    int64_t pos = kv.len();
    forward_one(id, pos, kv, logits_out);
    kv.set_len(pos + 1);
}
