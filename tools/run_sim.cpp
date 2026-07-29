// ============================================================================
// run_sim — the REAL Qwen2.5-0.5B, compiled to the T1 ISA, on the simulator.
// ============================================================================
// This is milestone 3's definition of done (bible §5): the fp32 model, lowered
// by our compiler to our ISA, runs on our simulator, is bit-exact against the
// CPU fp32 path, and the simulator reports cycles / DMA stalls / MAC
// utilisation PER LAYER.
//
// test/compiler_parity.cpp gates every commit on a tiny synthetic model because
// a gate you skip is not a gate. This is the other half: slow, opt-in, and run
// against the actual weights, where the tensor that forces tiling
// (up_proj, 4864x896 fp32 = 17.4 MB against an 8 MB scratchpad) is real rather
// than simulated by shrinking the bank budget.
//
// Memory: the HBM image is ~2.0 GB of fp32 and both paths read the SAME copy,
// which is the only reason this fits on a 7.6 GB WSL box. Runtime is minutes
// per token: the simulator's functional MATMUL is a scalar double-accumulating
// triple loop, on purpose (it is the thing being trusted).
//
//   run_sim [--model DIR] [--prompt "..."] [--steps N] [--max-seq N]
//           [--serial] [--fine-deps] [--disasm N]
// ============================================================================
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../src/accel/lower.h"
#include "../src/accel/reference.h"
#include "../src/accel/simulator.h"
#include "../src/core/safetensors.h"
#include "../src/model/config.h"
#include "../src/tokenizer/bpe.h"

using namespace accel;

namespace {

int32_t argmax(const float* v, int64_t n) {
    int64_t best = 0;
    for (int64_t i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;
    return static_cast<int32_t>(best);
}

// Fill the HBM image from the safetensors file, at the addresses the layout
// chose. This function IS the "port a new model" surface: it is the only place
// that knows HuggingFace tensor names.
void load_hbm(const SafeTensors& st, const HbmLayout& L, std::vector<float>& hbm) {
    const ModelConfig& cfg = L.cfg;
    const int64_t H = cfg.hidden_size, HD = cfg.head_dim();
    const int64_t QDIM = cfg.num_attention_heads * HD, KVDIM = cfg.num_key_value_heads * HD;
    const int64_t INTER = cfg.intermediate_size;

    auto get = [&](const std::string& name, uint64_t off) {
        st.load_as_f32(name, hbm.data() + off);
    };
    auto get_optional = [&](const std::string& name, uint64_t off, int64_t n) {
        if (st.has(name)) st.load_as_f32(name, hbm.data() + off);
        else for (int64_t i = 0; i < n; ++i) hbm[static_cast<size_t>(off) + static_cast<size_t>(i)] = 0.0f;
    };

    get("model.embed_tokens.weight", L.embed);
    for (int64_t li = 0; li < cfg.num_hidden_layers; ++li) {
        const std::string p = "model.layers." + std::to_string(li) + ".";
        const auto& la = L.layers[static_cast<size_t>(li)];
        get(p + "input_layernorm.weight", la.in_ln);
        get(p + "self_attn.q_proj.weight", la.q_w);
        get(p + "self_attn.k_proj.weight", la.k_w);
        get(p + "self_attn.v_proj.weight", la.v_w);
        get(p + "self_attn.o_proj.weight", la.o_w);
        // Qwen2 has QKV biases; a model without them gets zeros, which the
        // bias-seeded MATMUL then adds harmlessly. Same "load when present"
        // rule the CPU model follows — never a hardcoded model name.
        get_optional(p + "self_attn.q_proj.bias", la.q_b, QDIM);
        get_optional(p + "self_attn.k_proj.bias", la.k_b, KVDIM);
        get_optional(p + "self_attn.v_proj.bias", la.v_b, KVDIM);
        get(p + "post_attention_layernorm.weight", la.post_ln);
        get(p + "mlp.gate_proj.weight", la.gate_w);
        get(p + "mlp.up_proj.weight", la.up_w);
        get(p + "mlp.down_proj.weight", la.down_w);
        (void)INTER; (void)H;
    }
    get("model.norm.weight", L.final_ln);
}

// Which layer an instruction belongs to, parsed from its provenance note
// ("L7.mlp.up_proj matmul tile 3"). -1 means prologue/epilogue.
int64_t layer_of(const std::string& note) {
    if (note.size() < 2 || note[0] != 'L') return -1;
    size_t i = 1;
    int64_t v = 0;
    if (i >= note.size() || note[i] < '0' || note[i] > '9') return -1;
    while (i < note.size() && note[i] >= '0' && note[i] <= '9') {
        v = v * 10 + (note[i] - '0');
        ++i;
    }
    return (i < note.size() && note[i] == '.') ? v : -1;
}

struct Bucket {
    int64_t cycles = 0, mxu_busy = 0, dma_bytes = 0, macs = 0, instrs = 0;
    int64_t stall_dma = 0, stall_mxu = 0, stall_vpu = 0;
};

// Attribute one instruction's stall to a unit. `stall_class` carries the blame
// when a DEPENDENCY was binding; a structural hazard leaves it 0, and then the
// instruction's own unit is the thing that was busy. Same split the aggregate
// counters use — see the SimStats comment on why dependency and structural
// stalls must not be added together.
void charge_stall(Bucket& b, const Simulator::Event& ev, const Instr& in) {
    uint16_t blame = ev.stall_class;
    if (blame == 0) {
        switch (occupied_unit(in)) {
            case UNIT_DMA_IN:
            case UNIT_DMA_OUT: blame = DEP_DMA_A; break;
            case UNIT_MXU: blame = DEP_MXU; break;
            case UNIT_VPU: blame = DEP_VPU; break;
            default: return;
        }
    }
    if (blame & (DEP_DMA_A | DEP_DMA_B | DEP_DMA_OUT)) b.stall_dma += ev.stall_cycles;
    else if (blame & DEP_MXU) b.stall_mxu += ev.stall_cycles;
    else if (blame & DEP_VPU) b.stall_vpu += ev.stall_cycles;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = "models/qwen2.5-0.5b";
    std::string prompt = "The capital of France is";
    int steps = 4;
    int64_t max_seq = 64;
    LowerOptions opt;
    int disasm = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) dir = argv[++i];
        else if (!std::strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-seq") && i + 1 < argc) max_seq = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--serial")) opt.software_pipeline = false;
        else if (!std::strcmp(argv[i], "--fine-deps")) opt.coarse_dma_deps = false;
        else if (!std::strcmp(argv[i], "--disasm") && i + 1 < argc) disasm = std::atoi(argv[++i]);
    }

    ModelConfig cfg = ModelConfig::from_json_file(dir + "/config.json");
    std::printf("%s\n", cfg.summary().c_str());

    T1Config t1;
    opt.spm_bytes = t1.spm_bytes;
    HbmLayout L = HbmLayout::build(cfg, max_seq);
    std::printf("%s\n", L.summary().c_str());
    std::printf("T1: %lldx%lld MXU, %lld B SPM, DMA %lld B/cy + %lld cy latency, balance %.1f "
                "flop/byte\n",
                static_cast<long long>(t1.mxu_dim), static_cast<long long>(t1.mxu_dim),
                static_cast<long long>(t1.spm_bytes),
                static_cast<long long>(t1.dma_bytes_per_cycle),
                static_cast<long long>(t1.dma_latency_cycles), t1.balance());

    std::printf("loading %.2f GB HBM image...\n",
                static_cast<double>(L.total_floats) * 4.0 / 1e9);
    auto t0 = std::chrono::steady_clock::now();
    SafeTensors st(dir + "/model.safetensors");
    std::vector<float> hbm(static_cast<size_t>(L.total_floats), 0.0f);
    load_hbm(st, L, hbm);
    std::printf("loaded in %.1f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    Tokenizer tok = Tokenizer::from_file(dir + "/tokenizer.json");
    std::vector<int32_t> ids = tok.encode(prompt);
    std::printf("prompt = \"%s\" (%zu tokens)\n", prompt.c_str(), ids.size());

    Simulator sim(t1, std::move(hbm));
    sim.set_trace(true);

    // The CPU reference keeps its OWN KV cache, so agreement is not an artifact
    // of the two paths sharing state.
    const int64_t KVDIM = cfg.num_key_value_heads * cfg.head_dim();
    std::vector<float> kcache(static_cast<size_t>(cfg.num_hidden_layers * max_seq * KVDIM), 0.0f);
    std::vector<float> vcache(kcache.size(), 0.0f);
    std::vector<float> want(static_cast<size_t>(cfg.vocab_size));

    std::map<int64_t, Bucket> per_layer;
    int64_t total_mismatch = 0;
    std::string generated;

    const int64_t nsteps = static_cast<int64_t>(ids.size()) + steps;
    for (int64_t pos = 0; pos < nsteps && pos < max_seq; ++pos) {
        // The host writes the token embedding; that is the ONE thing the chip
        // does not do for itself (an embedding lookup is a gather, not a matmul).
        const int32_t id = (pos < static_cast<int64_t>(ids.size()))
                               ? ids[static_cast<size_t>(pos)]
                               : argmax(want.data(), cfg.vocab_size);
        if (pos >= static_cast<int64_t>(ids.size())) generated += tok.decode({id});
        for (int64_t i = 0; i < cfg.hidden_size; ++i)
            sim.hbm()[static_cast<size_t>(L.act_in + i)] =
                sim.hbm()[static_cast<size_t>(L.embed + id * cfg.hidden_size + i)];

        LowerReport rep;
        auto tl = std::chrono::steady_clock::now();
        Program p = lower_decode_step(L, pos, opt, &rep);
        double lower_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count();

        if (disasm && pos == 0) {
            std::printf("\n--- disassembly (first %d of %zu) ---\n%s\n", disasm, p.size(),
                        disassemble(p, static_cast<size_t>(disasm)).c_str());
        }

        sim.reset_stats();
        auto ts = std::chrono::steady_clock::now();
        SimStatus st_ = sim.run(p);
        double sim_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
        if (st_ != SimStatus::OK) {
            std::printf("SIM ERROR: %s\n", sim.error().c_str());
            return 1;
        }

        // Bit-exactness against the CPU, on the same HBM image.
        cpu_reference_decode(L, sim.hbm().data(), pos, kcache.data(), vcache.data(), want.data());
        int64_t bad = 0;
        double maxdev = 0.0;
        for (int64_t i = 0; i < cfg.vocab_size; ++i) {
            const float got = sim.hbm()[static_cast<size_t>(L.act_out + i)];
            const float ref = want[static_cast<size_t>(i)];
            if (got != ref) {
                ++bad;
                maxdev = std::max(maxdev, std::fabs(static_cast<double>(got) - ref));
            }
        }
        total_mismatch += bad;

        const SimStats& s = sim.stats();
        std::printf("pos %2lld  %s  %8lld instrs  %10lld cycles (%.2f ms @1GHz)  "
                    "MAC util %5.1f%%  DMA stall %5.1f%%   [compile %.2fs sim %.1fs]\n",
                    static_cast<long long>(pos),
                    bad == 0 ? "BIT-EXACT" : "MISMATCH ",
                    static_cast<long long>(s.instructions), static_cast<long long>(s.cycles),
                    static_cast<double>(s.cycles) / 1e6, 100.0 * s.mac_utilisation(),
                    100.0 * s.dma_stall_fraction(), lower_s, sim_s);
        if (bad) std::printf("        %lld/%lld logits differ, max |d| = %.3g\n",
                             static_cast<long long>(bad), static_cast<long long>(cfg.vocab_size),
                             maxdev);

        if (pos == 0) {
            std::printf("  tiling: %lld matmuls -> %lld tiles, %.1f MB of weights streamed, "
                        "SPM high-water %lld B of %lld B\n",
                        static_cast<long long>(rep.matmuls), static_cast<long long>(rep.tiles),
                        static_cast<double>(rep.weight_bytes_streamed) / 1e6,
                        static_cast<long long>(rep.spm_high_water),
                        static_cast<long long>(opt.spm_bytes));
            for (size_t i = 0; i < rep.tiling_log.size() && i < 8; ++i)
                std::printf("    %s\n", rep.tiling_log[i].c_str());

            // PER-LAYER ATTRIBUTION. The front end is in-order, so the interval
            // between consecutive issue cycles partitions the run exactly — no
            // double counting, and the buckets sum to the total.
            const auto& ev = sim.events();
            for (size_t i = 0; i < ev.size(); ++i) {
                const int64_t next = (i + 1 < ev.size()) ? ev[i + 1].issue_cycle : ev[i].end_cycle;
                Bucket& b = per_layer[layer_of(p.notes[static_cast<size_t>(ev[i].index)])];
                b.cycles += next - ev[i].issue_cycle;
                ++b.instrs;
                const Instr& in = p.code[static_cast<size_t>(ev[i].index)];
                charge_stall(b, ev[i], in);
                if (in.op == Op::MATMUL) {
                    b.mxu_busy += ev[i].end_cycle - ev[i].issue_cycle;
                    b.macs += static_cast<int64_t>(in.arg[3]) * in.arg[4] * in.arg[5];
                } else if (in.op == Op::DMA_LOAD || in.op == Op::DMA_STORE) {
                    b.dma_bytes += static_cast<int64_t>(in.arg[3]) * in.arg[4] * 4;
                }
            }
        }
    }

    std::printf("\ngenerated: \"%s\"\n", generated.c_str());
    std::printf("%s\n", total_mismatch == 0
                            ? "M3 GATE: every logit bit-identical to the CPU fp32 path."
                            : "M3 GATE FAILED: see mismatches above.");

    std::printf("\nper-layer breakdown of decode step 0 (%s, %s deps):\n",
                opt.software_pipeline ? "double-buffered" : "serial",
                opt.coarse_dma_deps ? "coarse" : "fine");
    std::printf("  %-16s %7s %11s %8s %8s %8s %8s %8s\n", "region", "instrs", "cycles",
                "MXU busy", "array%", "DMA MB", "stallDMA", "stallMXU");
    int64_t tot_cycles = 0;
    Bucket total;
    for (const auto& kv : per_layer) {
        tot_cycles += kv.second.cycles;
        total.macs += kv.second.macs;
        total.mxu_busy += kv.second.mxu_busy;
        total.dma_bytes += kv.second.dma_bytes;
    }
    for (const auto& kv : per_layer) {
        const Bucket& b = kv.second;
        char name[32];
        if (kv.first < 0) std::snprintf(name, sizeof(name), "prologue+lm_head");
        else std::snprintf(name, sizeof(name), "layer %lld", static_cast<long long>(kv.first));
        const double c = static_cast<double>(b.cycles ? b.cycles : 1);
        std::printf("  %-16s %7lld %11lld %7.1f%% %7.2f%% %8.1f %7.1f%% %7.1f%%\n", name,
                    static_cast<long long>(b.instrs), static_cast<long long>(b.cycles),
                    100.0 * static_cast<double>(b.mxu_busy) / c,
                    // ARRAY EFFICIENCY: of the MACs the 32x32 array COULD have
                    // retired while it was busy, what fraction were real?
                    b.mxu_busy ? 100.0 * static_cast<double>(b.macs) /
                                     (static_cast<double>(b.mxu_busy) * 1024.0) : 0.0,
                    static_cast<double>(b.dma_bytes) / 1e6,
                    100.0 * static_cast<double>(b.stall_dma) / c,
                    100.0 * static_cast<double>(b.stall_mxu) / c);
    }
    std::printf("  %-16s %7s %11lld\n", "TOTAL", "", static_cast<long long>(tot_cycles));

    // THE NUMBER THAT MATTERS, and the reason "MAC utilisation" alone lies.
    // The array is 32x32 and decode feeds it m=1, so 31 of 32 rows idle: the
    // array can be BUSY almost all the time while retiring 1/32 of the MACs it
    // is capable of. Both figures are printed together so neither can be quoted
    // without the other.
    const T1Config& c = t1;
    const double array_eff = total.mxu_busy
        ? static_cast<double>(total.macs) / (static_cast<double>(total.mxu_busy) * 1024.0) : 0.0;
    const int64_t dma_floor = total.dma_bytes / c.dma_bytes_per_cycle;
    std::printf("\n  array occupancy      %.1f%% of cycles the MXU was busy\n"
                "  array EFFICIENCY     %.2f%% of the MACs it could have retired while busy\n"
                "  -> m=1 uses %.0f of %lld rows: decode is limited by the SHAPE of the array,\n"
                "     not by its size and not by DMA (the weight stream alone would need\n"
                "     only ~%lld cycles, %.1fx less than the %lld we spend).\n",
                100.0 * static_cast<double>(total.mxu_busy) / static_cast<double>(tot_cycles),
                100.0 * array_eff, array_eff * static_cast<double>(c.mxu_dim),
                static_cast<long long>(c.mxu_dim), static_cast<long long>(dma_floor),
                static_cast<double>(tot_cycles) / static_cast<double>(dma_floor),
                static_cast<long long>(tot_cycles));
    return total_mismatch == 0 ? 0 : 1;
}
