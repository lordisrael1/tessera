#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "../model/config.h"
#include "isa.h"

// ============================================================================
// The T1 compiler: ModelConfig + weights -> one straight-line T1 program.
// ============================================================================
// Passes, in order (docs/04-compiler.md is the prose version):
//
//   1. HBM LAYOUT      Decide where every weight tensor lives in device memory.
//                      Deterministic and config-derived, so the same model
//                      always produces the same addresses and a program can be
//                      diffed between builds.
//   2. TILING          Split every matmul so that its operands fit an SPM bank.
//                      The FFN projections are the ones that force this: the
//                      up_proj weight for Qwen2.5-0.5B is 4864x896 fp32 = 17.4 MB
//                      against an 8 MB scratchpad, so it CANNOT be resident and
//                      the tiling pass is not optional cleverness, it is what
//                      makes the model runnable at all.
//   3. SPM ALLOCATION  Linear-scan over the straight-line program, reusing
//                      offsets once a buffer's last read has been emitted.
//   4. DMA SCHEDULING  Software pipelining: issue tile N+1's weight load before
//                      tile N's MATMUL, alternating SPM banks, so the DMA engine
//                      and the MXU overlap instead of taking turns.
//   5. ENCODE          Flatten to Instr[].
//
// There is no general graph IR and there should not be one. A transformer
// forward pass is a straight line of known shape; a lowering that walks that
// line directly is ~600 lines, and every one of them is about the accelerator
// rather than about graph plumbing.
// ============================================================================

namespace accel {

// Where each tensor sits in the simulated HBM, in FLOAT units.
//
// One flat address space, computed once. `total_floats` is what the simulator's
// HBM vector must be sized to, and the fact that it is exactly derivable from
// ModelConfig is the property that makes "port a new model" mean "write a new
// layout", not "rewrite the compiler".
struct HbmLayout {
    struct LayerAddrs {
        uint64_t in_ln = 0, post_ln = 0;
        uint64_t q_w = 0, k_w = 0, v_w = 0, o_w = 0;
        uint64_t q_b = 0, k_b = 0, v_b = 0;
        uint64_t gate_w = 0, up_w = 0, down_w = 0;
    };

    ModelConfig cfg;
    uint64_t embed = 0;       // [vocab, hidden]
    uint64_t final_ln = 0;    // [hidden]
    std::vector<LayerAddrs> layers;
    uint64_t kv_k = 0, kv_v = 0;   // [layers, max_seq, kv_dim]
    uint64_t act_in = 0;      // [hidden]  the token's embedding, written by host
    uint64_t act_out = 0;     // [vocab]   logits, written back by the program
    uint64_t total_floats = 0;
    int64_t max_seq = 0;

    static HbmLayout build(const ModelConfig& cfg, int64_t max_seq);

    uint64_t k_at(int64_t layer, int64_t pos) const;
    uint64_t v_at(int64_t layer, int64_t pos) const;
    std::string summary() const;
};

struct LowerOptions {
    // Pass 4. Off => a load and its consumer take strict turns, which is the
    // honest "before" number for the double-buffering chart.
    bool software_pipeline = true;

    // Reproduce the coarse-scoreboard design of ISA stance 3: treat both SPM
    // banks as one dependency class, so a consumer of bank A also waits for the
    // in-flight load into bank B. Default TRUE because that is what the ISA spec
    // describes; flip it to measure what the coarseness costs. This is the
    // bottleneck M7 is meant to rediscover from a trace — the flag exists so the
    // claim can be quantified now rather than assumed.
    bool coarse_dma_deps = true;

    // Total scratchpad the compiler may allocate. Must match the T1Config the
    // program will run on — the simulator will reject out-of-range addresses,
    // which is exactly how a mismatch announces itself.
    int64_t spm_bytes = 8 * 1024 * 1024;
    // Bytes of SPM the tiler may use for ONE weight bank. Two banks plus the
    // activation working set must fit spm_bytes.
    int64_t weight_bank_bytes = 2 * 1024 * 1024;

    bool emit_notes = true;  // provenance strings for the disassembly / trace
};

// Statistics the tiler reports, so "the compiler made these decisions" is an
// artifact rather than a claim (JD: "programming abstractions ... to rapidly
// iterate on model porting").
struct LowerReport {
    int64_t matmuls = 0;
    int64_t tiles = 0;
    int64_t weight_bytes_streamed = 0;
    int64_t spm_high_water = 0;
    std::vector<std::string> tiling_log;
};

// Compile one decode step at absolute position `pos`.
// Precondition: the host has written the token embedding to layout.act_in and
// the KV cache holds positions [0, pos).
Program lower_decode_step(const HbmLayout& layout, int64_t pos, const LowerOptions& opt,
                          LowerReport* report = nullptr);

}  // namespace accel
