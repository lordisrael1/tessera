// End-to-end greedy inference driver:
//   run_infer --model <dir> --prompt "..." [--steps N] [--precision f32|i8]
// Loads config.json + tokenizer.json + model.safetensors, prefills the prompt,
// then greedily decodes N tokens, printing them as they come.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../src/model/config.h"
#include "../src/model/kv_cache.h"
#include "../src/model/qwen2.h"
#include "../src/tokenizer/bpe.h"

static int32_t argmax(const float* v, int64_t n) {
    int64_t best = 0;
    for (int64_t i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;
    return static_cast<int32_t>(best);
}

int main(int argc, char** argv) {
    std::string dir = "models/qwen2.5-0.5b";
    std::string prompt = "The capital of France is";
    int steps = 16;
    Precision prec = Precision::F32;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) dir = argv[++i];
        else if (!std::strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--precision") && i + 1 < argc)
            prec = std::strcmp(argv[++i], "i8") == 0 ? Precision::I8 : Precision::F32;
    }

    ModelConfig cfg = ModelConfig::from_json_file(dir + "/config.json");
    std::printf("%s\n", cfg.summary().c_str());

    Tokenizer tok = Tokenizer::from_file(dir + "/tokenizer.json");
    auto ids = tok.encode(prompt);
    std::printf("prompt = \"%s\"  (%zu tokens)\n", prompt.c_str(), ids.size());

    auto t0 = std::chrono::steady_clock::now();
    Qwen2 model(cfg, dir + "/model.safetensors", prec);
    auto t1 = std::chrono::steady_clock::now();
    std::printf("weights loaded in %.2f s  (%s, %.0f MB resident)\n",
                std::chrono::duration<double>(t1 - t0).count(),
                prec == Precision::I8 ? "int8" : "fp32",
                static_cast<double>(model.weight_bytes()) / 1e6);

    KVCache kv(cfg, 4096);
    std::vector<float> logits(static_cast<size_t>(cfg.vocab_size));

    model.prefill(ids.data(), static_cast<int64_t>(ids.size()), kv, logits.data());
    int32_t next = argmax(logits.data(), cfg.vocab_size);

    std::string out;
    std::printf("\n%s", prompt.c_str());
    auto t2 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        out += tok.decode({next});
        std::printf("%s", tok.decode({next}).c_str());
        std::fflush(stdout);
        model.decode_step(next, kv, logits.data());
        next = argmax(logits.data(), cfg.vocab_size);
    }
    double decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
    std::printf("\n\n[%d greedy tokens generated]\n", steps);

    // THE DECODE NUMBER docs/01-roofline.md quotes. Effective bandwidth is
    // weight_bytes / seconds-per-token: at M=1 every weight byte is read exactly
    // once per token and is used for exactly two flops, so this figure is
    // directly comparable to the measured STREAM roof — and the comparison is
    // the whole memory-bound argument, end to end rather than per kernel.
    double per_tok = decode_s / (steps ? steps : 1);
    std::printf("decode: %.1f ms/token (%.2f tok/s), effective weight bandwidth %.1f GB/s\n",
                per_tok * 1e3, 1.0 / per_tok,
                static_cast<double>(model.weight_bytes()) / per_tok / 1e9);
    // Reported now (not before inference) because the arena only fills during a
    // forward pass. This is the per-token activation working set that sizes the
    // M3 accelerator's on-chip scratchpad.
    std::printf("activation high-water = %.1f KB per token\n",
                static_cast<double>(model.activation_high_water()) / 1024.0);
    return 0;
}
