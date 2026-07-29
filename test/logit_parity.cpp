// ============================================================================
// logit_parity — THE CONSTITUTION OF THE REPO (bible §3.7, §10).
// ============================================================================
//
// Runs our from-scratch fp32 stack against logits dumped from HuggingFace and
// asserts:
//   * our tokenizer produces HF's exact prompt ids,
//   * the residual stream matches at three bisection points,
//   * prefill logits match to < 1e-3 max-abs,
//   * 64 greedy decode steps produce the EXACT same token ids,
//   * the logits after those 64 steps still match to < 1e-3 (i.e. no drift).
//
// Every later milestone — the INT8 kernel, the T1 simulator, tensor parallelism —
// has to leave this green. A milestone that moves logits past tolerance is not
// done, whatever else it achieved.
//
// SELF-SKIPPING: the goldens are ~3.5 MB and the weights are ~1 GB, so neither
// is committed. Absent either, this test prints SKIP and exits 0 so CI stays
// green while still failing loudly on any machine that HAS the artifacts.
// Regenerate with:
//     python3 tools/dump_logits.py --model models/qwen2.5-0.5b --out test/golden
// ============================================================================
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/core/json.h"
#include "../src/model/config.h"
#include "../src/model/kv_cache.h"
#include "../src/model/qwen2.h"
#include "../src/tokenizer/bpe.h"
#include "check.h"

namespace {

std::string env_or(const char* k, const char* dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::string(v) : std::string(dflt);
}

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return static_cast<bool>(f);
}

template <typename T>
std::vector<T> read_bin(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<T> out(static_cast<size_t>(n) / sizeof(T));
    f.read(reinterpret_cast<char*>(out.data()), n);
    return out;
}

JsonValue read_json(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return parse_json(ss.str());
}

struct Diff {
    double max_abs = 0.0;
    double rel_l2 = 0.0;
    size_t argmax_ours = 0, argmax_ref = 0;
};

Diff compare(const std::vector<float>& ours, const std::vector<float>& ref) {
    Diff d;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size() && i < ours.size(); ++i) {
        double e = std::fabs(static_cast<double>(ours[i]) - ref[i]);
        if (e > d.max_abs) d.max_abs = e;
        num += e * e;
        den += static_cast<double>(ref[i]) * ref[i];
        if (ours[i] > ours[d.argmax_ours]) d.argmax_ours = i;
        if (ref[i] > ref[d.argmax_ref]) d.argmax_ref = i;
    }
    d.rel_l2 = (den > 0) ? std::sqrt(num / den) : 0.0;
    return d;
}

// The bisection ladder: report ALL of them before asserting, so a failing run
// prints the whole picture instead of stopping at the first bad rung.
void rung(const char* name, const std::vector<float>& ours, const std::vector<float>& ref,
          double tol) {
    if (ref.empty()) return;
    if (ours.size() != ref.size()) {
        std::printf("    %-9s SIZE MISMATCH ours=%zu ref=%zu\n", name, ours.size(), ref.size());
        CHECK(ours.size() == ref.size());
        return;
    }
    Diff d = compare(ours, ref);
    std::printf("    %-9s max|d|=%.3e  relL2=%.3e  %s\n", name, d.max_abs, d.rel_l2,
                d.max_abs <= tol ? "ok" : "*** OVER TOLERANCE ***");
    CHECK(d.max_abs <= tol);
}

}  // namespace

int main() {
    const std::string model_dir = env_or("TESSERA_MODEL_DIR", "models/qwen2.5-0.5b");
    const std::string gold_dir = env_or("TESSERA_GOLDEN_DIR", "test/golden");
    // The gate the bible specifies. Overridable so a bisecting session can
    // loosen it deliberately and say so, rather than editing the source.
    const double tol = std::atof(env_or("TESSERA_PARITY_TOL", "1e-3").c_str());

    const std::string weights = model_dir + "/model.safetensors";
    if (!file_exists(weights) || !file_exists(gold_dir + "/p0_meta.json")) {
        std::printf("SKIP logit_parity: need %s and %s/p0_meta.json\n"
                    "     (fetch the model, then: python3 tools/dump_logits.py "
                    "--model %s --out %s)\n",
                    weights.c_str(), gold_dir.c_str(), model_dir.c_str(), gold_dir.c_str());
        return 0;
    }

    ModelConfig cfg = ModelConfig::from_json_file(model_dir + "/config.json");
    Tokenizer tok = Tokenizer::from_file(model_dir + "/tokenizer.json");
    Qwen2 model(cfg, weights);
    std::printf("logit_parity: tolerance %.1e max-abs on logits, EXACT on greedy tokens\n", tol);

    for (int pi = 0; pi < 3; ++pi) {
        const std::string pfx = gold_dir + "/p" + std::to_string(pi) + "_";
        if (!file_exists(pfx + "meta.json")) continue;
        JsonValue meta = read_json(pfx + "meta.json");
        const std::string prompt = meta["prompt"].as_string();

        std::vector<int32_t> ref_ids;
        for (const auto& v : meta["prompt_ids"].arr)
            ref_ids.push_back(static_cast<int32_t>(v.as_int()));

        std::printf("\n  [%d] %s\n", pi, prompt.c_str());

        // --- rung 0: the tokenizer ------------------------------------------
        // Run with HF's ids either way, so a tokenizer bug is reported as a
        // tokenizer bug rather than smeared across every downstream rung.
        std::vector<int32_t> our_ids = tok.encode(prompt);
        bool ids_match = (our_ids == ref_ids);
        std::printf("    tokenizer %s (%zu ids)\n", ids_match ? "exact" : "*** MISMATCH ***",
                    ref_ids.size());
        if (!ids_match) {
            std::printf("      ours:");
            for (int32_t i : our_ids) std::printf(" %d", i);
            std::printf("\n      ref :");
            for (int32_t i : ref_ids) std::printf(" %d", i);
            std::printf("\n");
        }
        CHECK(ids_match);

        // --- rungs 1-3: the residual stream, then prefill logits -------------
        KVCache kv(cfg, 4096);
        ForwardTaps taps;
        model.set_taps(&taps);
        std::vector<float> logits(static_cast<size_t>(cfg.vocab_size));
        model.prefill(ref_ids.data(), static_cast<int64_t>(ref_ids.size()), kv, logits.data());
        model.set_taps(nullptr);

        // Hidden states live on a much smaller scale than logits (|h| ~ 1-10 vs
        // |logits| ~ 20) and every later rung amplifies them, so they get a
        // tighter tolerance: a 1e-3 error at h_embed is already a bug.
        rung("h_embed", taps.h_embed, read_bin<float>(pfx + "h_embed.bin"), 1e-4);
        rung("h_layer0", taps.h_layer0, read_bin<float>(pfx + "h_layer0.bin"), 1e-3);
        rung("h_final", taps.h_final, read_bin<float>(pfx + "h_final.bin"), 1e-3);

        std::vector<float> ref_prefill = read_bin<float>(pfx + "prefill_logits.bin");
        rung("logits0", logits, ref_prefill, tol);

        // --- rung 4: 64 greedy steps, EXACT token match ----------------------
        std::vector<int32_t> ref_tokens = read_bin<int32_t>(pfx + "tokens.bin");
        auto argmax = [&] {
            int32_t best = 0;
            for (size_t i = 1; i < logits.size(); ++i)
                if (logits[i] > logits[static_cast<size_t>(best)]) best = static_cast<int32_t>(i);
            return best;
        };

        size_t matched = 0;
        int first_bad = -1;
        int32_t next = argmax();
        for (size_t s = 0; s < ref_tokens.size(); ++s) {
            // The decode happens at the TOP of the iteration (except the first,
            // which reuses the prefill logits). That leaves `logits` holding the
            // step-(n-1) row when the loop exits, which is exactly what
            // final_logits.bin contains — an off-by-one here would compare our
            // 65th forward against HF's 64th and report phantom drift.
            if (s > 0) {
                model.decode_step(next, kv, logits.data());
                next = argmax();
            }
            if (next == ref_tokens[s]) {
                ++matched;
            } else if (first_bad < 0) {
                first_bad = static_cast<int>(s);
                std::printf("    step %zu: ours=%d (%s) ref=%d (%s)\n", s, next,
                            tok.decode({next}).c_str(), ref_tokens[s],
                            tok.decode({ref_tokens[s]}).c_str());
            }
            // Note we feed OUR token onward, never the reference one. Feeding
            // the reference would resynchronise the sequence after every step
            // and hide exactly the compounding divergence this test exists for.
        }
        std::printf("    greedy   %zu/%zu exact%s\n", matched, ref_tokens.size(),
                    matched == ref_tokens.size() ? "" : "  *** DIVERGED ***");
        CHECK(matched == ref_tokens.size());

        // --- rung 5: no drift after 64 steps --------------------------------
        rung("logits64", logits, read_bin<float>(pfx + "final_logits.bin"), tol);

        std::string text;
        for (size_t s = 0; s < ref_tokens.size() && s < 12; ++s)
            text += tok.decode({ref_tokens[s]});
        std::printf("    -> \"%s...\"\n", text.c_str());
    }

    return test_summary();
}
