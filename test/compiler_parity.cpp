// ============================================================================
// compiler_parity — the M3 gate (bible §5, "Definition of done").
// ============================================================================
// Compiles a transformer decode step to the T1 ISA, runs it on the simulator,
// and asserts the logits are BIT-IDENTICAL to the CPU fp32 reference.
//
// Bit-identical, not close. Both paths use the same op order and the same
// double-accumulating dot product, so any difference at all is a real defect —
// a mistiled matmul, an SPM buffer aliased with a live one, a DMA stride off by
// one, a GQA head mapped to the wrong KV head. A tolerance-based check would
// pass through most of those.
//
// The model here is a SMALL synthetic config with random weights rather than
// Qwen2.5-0.5B. That is deliberate and it is not a weakening:
//   * every structural feature that can break is present — GQA with
//     q_per_kv > 1, QKV biases, SwiGLU, tied output projection, and a hidden
//     size that forces multi-tile projections;
//   * it runs in milliseconds, so it can gate every commit, which a 5-minute
//     full-model simulation never would;
//   * the full-model run is a separate opt-in (tools/run_sim) precisely because
//     a gate you skip is not a gate.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../src/accel/lower.h"
#include "../src/accel/reference.h"
#include "../src/accel/simulator.h"
#include "../src/ops/ops.h"
#include "check.h"

using namespace accel;

namespace {

ModelConfig tiny_config() {
    ModelConfig c;
    c.hidden_size = 64;
    c.intermediate_size = 128;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 2;  // q_per_kv = 2: exercises the GQA head mapping
    c.vocab_size = 48;
    c.max_position_embeddings = 128;
    c.rms_norm_eps = 1e-6f;
    c.rope_theta = 10000.0f;
    c.tie_word_embeddings = true;
    return c;
}

// The CPU reference lives in src/accel/reference.h so that this fast gate and
// the full-model tools/run_sim check the SAME semantics. It reads the same HBM
// image the simulator does, so there is one copy of the weights and no chance
// of the two paths being fed different numbers.

}  // namespace

int main() {
    ModelConfig cfg = tiny_config();
    const int64_t max_seq = 16;
    HbmLayout L = HbmLayout::build(cfg, max_seq);
    std::printf("  %s\n", L.summary().c_str());

    // One HBM image, shared by both paths.
    std::vector<float> hbm(static_cast<size_t>(L.total_floats), 0.0f);
    std::mt19937 rng(12345);
    std::normal_distribution<float> d(0.0f, 0.05f);
    for (auto& x : hbm) x = d(rng);
    // Layernorm gains near 1 so activations do not collapse.
    for (const auto& la : L.layers) {
        for (int64_t i = 0; i < cfg.hidden_size; ++i) {
            hbm[static_cast<size_t>(la.in_ln + i)] = 1.0f + 0.01f * d(rng);
            hbm[static_cast<size_t>(la.post_ln + i)] = 1.0f + 0.01f * d(rng);
        }
    }
    for (int64_t i = 0; i < cfg.hidden_size; ++i)
        hbm[static_cast<size_t>(L.final_ln + i)] = 1.0f + 0.01f * d(rng);

    T1Config t1;
    LowerOptions opt;
    opt.spm_bytes = t1.spm_bytes;
    // Small banks so even this tiny model needs several tiles per projection —
    // if the tiler is never exercised the test proves nothing about tiling.
    opt.weight_bank_bytes = 4 * 1024;

    // Decode several positions in sequence: position 0 exercises the pos+1 == 1
    // edge case, later positions exercise the KV gather with real strides.
    std::vector<float> kcache(static_cast<size_t>(cfg.num_hidden_layers * max_seq *
                                                  cfg.num_key_value_heads * cfg.head_dim()),
                              0.0f);
    std::vector<float> vcache(kcache.size(), 0.0f);

    Simulator sim(t1, hbm);
    LowerReport rep;
    bool printed = false;

    // The same token embeddings are replayed through prefill below, so they are
    // generated once and kept.
    const int64_t NTOK = 5;
    std::vector<std::vector<float>> toks(static_cast<size_t>(NTOK));
    for (auto& t : toks) {
        t.resize(static_cast<size_t>(cfg.hidden_size));
        for (auto& x : t) x = d(rng);
    }
    std::vector<float> last_decode_logits;

    for (int64_t pos = 0; pos < NTOK; ++pos) {
        // This step's token, written to the same HBM slot in both worlds.
        const std::vector<float>& tok = toks[static_cast<size_t>(pos)];
        for (int64_t i = 0; i < cfg.hidden_size; ++i) {
            sim.hbm()[static_cast<size_t>(L.act_in + i)] = tok[static_cast<size_t>(i)];
            hbm[static_cast<size_t>(L.act_in + i)] = tok[static_cast<size_t>(i)];
        }

        Program p = lower_decode_step(L, pos, opt, pos == 0 ? &rep : nullptr);
        sim.reset_stats();
        SimStatus st = sim.run(p);
        if (st != SimStatus::OK) std::printf("  SIM ERROR: %s\n", sim.error().c_str());
        CHECK(st == SimStatus::OK);

        std::vector<float> want(static_cast<size_t>(cfg.vocab_size));
        cpu_reference_decode(L, hbm.data(), pos, kcache.data(), vcache.data(), want.data());

        int bad = 0;
        for (int64_t i = 0; i < cfg.vocab_size; ++i) {
            float got = sim.hbm()[static_cast<size_t>(L.act_out + i)];
            if (got != want[static_cast<size_t>(i)]) {
                if (bad < 3)
                    std::printf("    pos %lld logit %lld: sim %.9g != cpu %.9g\n",
                                static_cast<long long>(pos), static_cast<long long>(i),
                                static_cast<double>(got),
                                static_cast<double>(want[static_cast<size_t>(i)]));
                ++bad;
            }
        }
        CHECK(bad == 0);

        if (!printed) {
            std::printf("  compiled decode step: %zu instructions, %lld matmuls -> %lld tiles\n",
                        p.size(), static_cast<long long>(rep.matmuls),
                        static_cast<long long>(rep.tiles));
            std::printf("  SPM high-water %lld B of %lld B\n",
                        static_cast<long long>(rep.spm_high_water),
                        static_cast<long long>(opt.spm_bytes));
            printed = true;
        }
        std::printf("  pos %lld: %s bit-exact vs CPU, %lld cycles, %.1f%% MAC util\n",
                    static_cast<long long>(pos), bad == 0 ? "OK" : "FAIL",
                    static_cast<long long>(sim.stats().cycles),
                    100.0 * sim.stats().mac_utilisation());
        if (pos == NTOK - 1)
            last_decode_logits.assign(
                sim.hbm().begin() + static_cast<long>(L.act_out),
                sim.hbm().begin() + static_cast<long>(L.act_out + cfg.vocab_size));
    }

    // -----------------------------------------------------------------------
    // PREFILL. The same NTOK tokens, compiled as chunks with m > 1.
    //
    // The gate is equality with the DECODE path, which the loop above just
    // proved against the CPU. That is a stronger check than another comparison
    // against a reference written alongside the feature: prefill has to land on
    // an answer that was computed by different instructions, a different tiling,
    // a different attention shape, and a causal mask that did not exist in the
    // decode program — and it has to land on it exactly.
    //
    // Two chunkings are tested. One chunk exercises base = 0; two chunks
    // exercise base > 0, where the causal ramp starts partway along and the
    // second chunk must read the KV the first one wrote.
    // -----------------------------------------------------------------------
    std::printf("\n  prefill (m > 1):\n");
    struct Chunking { const char* name; std::vector<int64_t> sizes; };
    const Chunking chunkings[] = {
        {"one chunk of 5", {5}},
        {"two chunks, 3 + 2", {3, 2}},
        {"five chunks of 1 (degenerate)", {1, 1, 1, 1, 1}},
    };

    for (const Chunking& ch : chunkings) {
        // Fresh machine and a fresh KV cache: nothing carries over from decode.
        Simulator ps(t1, hbm);
        for (int64_t r = 0; r < NTOK; ++r)
            for (int64_t i = 0; i < cfg.hidden_size; ++i) {
                const float v = toks[static_cast<size_t>(r)][static_cast<size_t>(i)];
                ps.hbm()[static_cast<size_t>(L.act_in + r * cfg.hidden_size + i)] = v;
                hbm[static_cast<size_t>(L.act_in + r * cfg.hidden_size + i)] = v;
            }
        std::vector<float> pk(kcache.size(), 0.0f), pv(kcache.size(), 0.0f);
        std::vector<float> want_pf(static_cast<size_t>(cfg.vocab_size));

        int64_t base = 0, cycles = 0, instrs = 0;
        LowerReport prep;
        for (int64_t n : ch.sizes) {
            Program pp = lower_prefill(L, base, n, opt, &prep);
            ps.reset_stats();
            SimStatus st = ps.run(pp);
            if (st != SimStatus::OK) std::printf("    SIM ERROR: %s\n", ps.error().c_str());
            CHECK(st == SimStatus::OK);
            cycles += ps.stats().cycles;
            instrs += ps.stats().instructions;
            cpu_reference_prefill(L, hbm.data(), base, n, pk.data(), pv.data(), want_pf.data());
            base += n;
        }

        int bad_cpu = 0, bad_dec = 0;
        for (int64_t i = 0; i < cfg.vocab_size; ++i) {
            const float got = ps.hbm()[static_cast<size_t>(L.act_out + i)];
            if (got != want_pf[static_cast<size_t>(i)]) ++bad_cpu;
            if (got != last_decode_logits[static_cast<size_t>(i)]) ++bad_dec;
        }
        CHECK(bad_cpu == 0);
        CHECK(bad_dec == 0);
        std::printf("    %-30s %5lld instrs %8lld cycles  vs CPU %s  vs decode %s\n",
                    ch.name, static_cast<long long>(instrs), static_cast<long long>(cycles),
                    bad_cpu == 0 ? "exact" : "DIFFER", bad_dec == 0 ? "exact" : "DIFFER");
        if (bad_dec)
            std::printf("      %d/%lld logits differ from the decode path\n", bad_dec,
                        static_cast<long long>(cfg.vocab_size));
    }

    // The compiler's decisions, as an artifact. This log IS the "programming
    // abstractions for rapid model porting" evidence — porting a new model means
    // reading this and checking the tiler did something sane.
    std::printf("\n  tiling decisions (first 6):\n");
    for (size_t i = 0; i < rep.tiling_log.size() && i < 6; ++i)
        std::printf("    %s\n", rep.tiling_log[i].c_str());

    // -----------------------------------------------------------------------
    // The scheduling study. Same program, four schedules, identical results —
    // and very different cycle counts. Correctness must be invariant to
    // scheduling; if it is not, the dependency masks are wrong.
    // -----------------------------------------------------------------------
    std::printf("\n  schedule                          cycles   MAC util   DMA stall\n");
    struct Variant { const char* name; bool pipeline; bool coarse; };
    const Variant variants[] = {
        {"serial, coarse deps (baseline)", false, true},
        {"serial, fine deps", false, false},
        {"double-buffered, coarse deps", true, true},
        {"double-buffered, fine deps", true, false},
    };
    std::vector<float> baseline_logits;
    for (const Variant& v : variants) {
        LowerOptions o = opt;
        o.software_pipeline = v.pipeline;
        o.coarse_dma_deps = v.coarse;
        Simulator s2(t1, hbm);
        for (int64_t i = 0; i < cfg.hidden_size; ++i)
            s2.hbm()[static_cast<size_t>(L.act_in + i)] = hbm[static_cast<size_t>(L.act_in + i)];
        Program p = lower_decode_step(L, 0, o, nullptr);
        CHECK(s2.run(p) == SimStatus::OK);
        std::printf("  %-32s %8lld    %5.1f%%      %5.1f%%\n", v.name,
                    static_cast<long long>(s2.stats().cycles),
                    100.0 * s2.stats().mac_utilisation(),
                    100.0 * s2.stats().dma_stall_fraction());

        std::vector<float> got(sim.hbm().begin() + static_cast<long>(L.act_out),
                               sim.hbm().begin() + static_cast<long>(L.act_out + cfg.vocab_size));
        std::vector<float> cur(s2.hbm().begin() + static_cast<long>(L.act_out),
                               s2.hbm().begin() + static_cast<long>(L.act_out + cfg.vocab_size));
        if (baseline_logits.empty()) baseline_logits = cur;
        for (size_t i = 0; i < cur.size(); ++i) CHECK(cur[i] == baseline_logits[i]);
    }

    return test_summary();
}
