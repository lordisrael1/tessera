// End-to-end greedy inference driver:
//   run_infer --model <dir> --prompt "..." [--steps N]
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
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) dir = argv[++i];
        else if (!std::strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) steps = std::atoi(argv[++i]);
    }

    ModelConfig cfg = ModelConfig::from_json_file(dir + "/config.json");
    std::printf("%s\n", cfg.summary().c_str());

    Tokenizer tok = Tokenizer::from_file(dir + "/tokenizer.json");
    auto ids = tok.encode(prompt);
    std::printf("prompt = \"%s\"  (%zu tokens)\n", prompt.c_str(), ids.size());

    auto t0 = std::chrono::steady_clock::now();
    Qwen2 model(cfg, dir + "/model.safetensors");
    auto t1 = std::chrono::steady_clock::now();
    std::printf("weights loaded in %.2f s\n", std::chrono::duration<double>(t1 - t0).count());

    KVCache kv(cfg, 4096);
    std::vector<float> logits(static_cast<size_t>(cfg.vocab_size));

    model.prefill(ids.data(), static_cast<int64_t>(ids.size()), kv, logits.data());
    int32_t next = argmax(logits.data(), cfg.vocab_size);

    std::string out;
    std::printf("\n%s", prompt.c_str());
    for (int s = 0; s < steps; ++s) {
        out += tok.decode({next});
        std::printf("%s", tok.decode({next}).c_str());
        std::fflush(stdout);
        model.decode_step(next, kv, logits.data());
        next = argmax(logits.data(), cfg.vocab_size);
    }
    std::printf("\n\n[%d greedy tokens generated]\n", steps);
    // Reported now (not before inference) because the arena only fills during a
    // forward pass. This is the per-token activation working set that sizes the
    // M3 accelerator's on-chip scratchpad.
    std::printf("activation high-water = %.1f KB per token\n",
                static_cast<double>(model.activation_high_water()) / 1024.0);
    return 0;
}
