#include "lower.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace accel {

// ---------------------------------------------------------------------------
// Pass 1 — HBM layout
// ---------------------------------------------------------------------------
HbmLayout HbmLayout::build(const ModelConfig& cfg, int64_t max_seq) {
    HbmLayout L;
    L.cfg = cfg;
    L.max_seq = max_seq;

    const int64_t H = cfg.hidden_size;
    const int64_t HD = cfg.head_dim();
    const int64_t QDIM = cfg.num_attention_heads * HD;
    const int64_t KVDIM = cfg.num_key_value_heads * HD;
    const int64_t INTER = cfg.intermediate_size;
    const int64_t V = cfg.vocab_size;

    uint64_t off = 0;
    auto place = [&off](uint64_t n) {
        uint64_t a = off;
        off += n;
        return a;
    };

    L.embed = place(static_cast<uint64_t>(V) * static_cast<uint64_t>(H));
    L.layers.resize(static_cast<size_t>(cfg.num_hidden_layers));
    for (auto& la : L.layers) {
        la.in_ln = place(static_cast<uint64_t>(H));
        la.q_w = place(static_cast<uint64_t>(QDIM) * static_cast<uint64_t>(H));
        la.q_b = place(static_cast<uint64_t>(QDIM));
        la.k_w = place(static_cast<uint64_t>(KVDIM) * static_cast<uint64_t>(H));
        la.k_b = place(static_cast<uint64_t>(KVDIM));
        la.v_w = place(static_cast<uint64_t>(KVDIM) * static_cast<uint64_t>(H));
        la.v_b = place(static_cast<uint64_t>(KVDIM));
        la.o_w = place(static_cast<uint64_t>(H) * static_cast<uint64_t>(QDIM));
        la.post_ln = place(static_cast<uint64_t>(H));
        la.gate_w = place(static_cast<uint64_t>(INTER) * static_cast<uint64_t>(H));
        la.up_w = place(static_cast<uint64_t>(INTER) * static_cast<uint64_t>(H));
        la.down_w = place(static_cast<uint64_t>(H) * static_cast<uint64_t>(INTER));
    }
    L.final_ln = place(static_cast<uint64_t>(H));
    L.kv_k = place(static_cast<uint64_t>(cfg.num_hidden_layers) *
                   static_cast<uint64_t>(max_seq) * static_cast<uint64_t>(KVDIM));
    L.kv_v = place(static_cast<uint64_t>(cfg.num_hidden_layers) *
                   static_cast<uint64_t>(max_seq) * static_cast<uint64_t>(KVDIM));
    // Room for a whole prefill chunk of token embeddings, not just one. Decode
    // uses row 0 and ignores the rest.
    L.act_in = place(static_cast<uint64_t>(max_seq) * static_cast<uint64_t>(H));
    L.act_out = place(static_cast<uint64_t>(V));
    L.total_floats = off;
    return L;
}

uint64_t HbmLayout::k_at(int64_t layer, int64_t pos) const {
    int64_t kvdim = cfg.num_key_value_heads * cfg.head_dim();
    return kv_k + static_cast<uint64_t>((layer * max_seq + pos) * kvdim);
}
uint64_t HbmLayout::v_at(int64_t layer, int64_t pos) const {
    int64_t kvdim = cfg.num_key_value_heads * cfg.head_dim();
    return kv_v + static_cast<uint64_t>((layer * max_seq + pos) * kvdim);
}

std::string HbmLayout::summary() const {
    std::ostringstream os;
    os << "HBM layout: " << total_floats << " floats ("
       << (static_cast<double>(total_floats) * 4.0 / 1e6) << " MB), " << layers.size()
       << " layers, max_seq " << max_seq;
    return os.str();
}

// ---------------------------------------------------------------------------
// Pass 3 helper — SPM allocation.
//
// A bump allocator with an explicit mark/release, which is all a straight-line
// program needs. The M1 arena discipline is what makes this sufficient: because
// the CPU forward pass already proved the per-token working set is a few tens of
// KB and is reused identically every layer, the compiler can allocate the
// activation region once, outside the layer loop, and reuse it 24 times.
// (Measured on the real model: 79.1 KB/token — see run_infer's high-water
// report. That number was collected in M1 for exactly this moment.)
// ---------------------------------------------------------------------------
class SpmAllocator {
public:
    explicit SpmAllocator(int64_t capacity_bytes) : cap_(capacity_bytes) {}

    uint32_t alloc(int64_t floats, const char* what) {
        int64_t bytes = floats * 4;
        int64_t p = (off_ + 63) & ~int64_t{63};  // 64-byte aligned, like the CPU arena
        if (p + bytes > cap_) {
            std::ostringstream os;
            os << "SPM exhausted allocating " << bytes << " B for " << what;
            throw std::runtime_error(os.str());
        }
        off_ = p + bytes;
        high_ = std::max(high_, off_);
        return static_cast<uint32_t>(p);
    }
    int64_t mark() const { return off_; }
    void release(int64_t m) { off_ = m; }
    int64_t high_water() const { return high_; }

private:
    int64_t cap_, off_ = 0, high_ = 0;
};

namespace {

struct Emitter {
    Program prog;
    const LowerOptions* opt;

    void emit(Instr in, std::string note) {
        prog.emit(in, opt->emit_notes ? std::move(note) : std::string{});
    }

    // Dependency mask for "the data this consumer needs has landed in `bank`".
    //
    // Under coarse_dma_deps we return BOTH bank bits, which is the ISA's stated
    // (and deliberately flawed) design: a consumer of bank A also ends up
    // waiting for whatever is in flight into bank B, destroying the overlap the
    // pipelining pass just built. LowerOptions carries the switch so the cost is
    // a measurement, not an anecdote.
    uint16_t wait_bank(int bank) const {
        if (opt->coarse_dma_deps) return DEP_DMA_IN;
        return bank == 0 ? DEP_DMA_A : DEP_DMA_B;
    }
    static uint8_t bank_flag(int bank) { return bank == 0 ? 0 : kFlagBankB; }
};

Instr dma_load(uint64_t hbm, uint32_t spm, int64_t rows, int64_t cols, int64_t row_stride,
               int bank) {
    Instr in;
    in.op = Op::DMA_LOAD;
    in.flags = (bank == 0) ? 0 : kFlagBankB;
    in.set_hbm_addr(0, hbm);
    in.arg[2] = spm;
    in.arg[3] = static_cast<uint32_t>(rows);
    in.arg[4] = static_cast<uint32_t>(cols);
    in.arg[5] = static_cast<uint32_t>(row_stride);
    return in;
}

Instr dma_store(uint64_t hbm, uint32_t spm, int64_t rows, int64_t cols, int64_t row_stride) {
    Instr in;
    in.op = Op::DMA_STORE;
    in.set_hbm_addr(0, hbm);
    in.arg[2] = spm;
    in.arg[3] = static_cast<uint32_t>(rows);
    in.arg[4] = static_cast<uint32_t>(cols);
    in.arg[5] = static_cast<uint32_t>(row_stride);
    return in;
}

Instr vec3(Op op, uint32_t a, uint32_t b, uint32_t c, int64_t len, float imm = 0.0f) {
    Instr in;
    in.op = op;
    in.arg[0] = a;
    in.arg[1] = b;
    in.arg[2] = c;
    in.arg[3] = static_cast<uint32_t>(len);
    in.imm_f = imm;
    in.dep_mask = DEP_VPU;  // VPU ops are serialised against each other
    return in;
}

}  // namespace

// ---------------------------------------------------------------------------
// Passes 2 + 4 + 5 — tile a projection, pipeline its loads, encode.
//
// Y[1, OUT] = X[1, IN] . W[OUT, IN]^T (+ bias). The weight is streamed in tiles
// of `rows_per_tile` output rows; the activation X is already resident (it is
// hidden-sized, i.e. kilobytes).
//
// THE PIPELINE. With software_pipeline on we prefetch tile t+1's weights into
// the OTHER bank before issuing tile t's MATMUL, so the DMA engine is refilling
// while the MXU works:
//
//     LOAD t0 -> bank A
//     LOAD t1 -> bank B          <-- issued before the first MATMUL
//     MATMUL t0 (waits on A)     <-- overlaps with the t1 load
//     LOAD t2 -> bank A          (waits on MXU: t0 has drained bank A)
//     MATMUL t1 (waits on B)
//     ...
// Without it, each MATMUL simply waits for the load immediately before it, and
// the two units take turns. Both schedules are emitted from this one function so
// the comparison is apples to apples.
// ---------------------------------------------------------------------------
// `M` is the number of token rows. X is [M, IN] and Y is [M, OUT], both
// row-major, so every tile computes a COLUMN SLICE of Y.
//
// THE ONE THING THAT CHANGES AT m > 1. MATMUL writes C packed as [m, n]. At
// M=1 a tile's n outputs are exactly the slice of Y they belong to, so C can
// point straight into Y. At M>1 they are not: the tile's [M, n] block has to be
// interleaved into Y's [M, OUT] rows. So the tile writes a packed scratch and a
// 2-D strided V_COPY places it. That copy is ~3% of the tile's matmul cycles,
// which is the price of not giving the MXU's writeback port an output stride.
// A single-tile projection skips it entirely — C is already the whole of Y.
static void lower_projection(Emitter& E, SpmAllocator& spm, uint64_t w_hbm,
                             uint64_t bias_hbm, bool has_bias, uint32_t x_spm, uint32_t y_spm,
                             int64_t IN, int64_t OUT, LowerReport* rep, const char* what,
                             int64_t M = 1) {
    const LowerOptions& opt = *E.opt;
    // TILING. One tile's weights must fit a bank; the tile is a whole number of
    // output rows because the MXU consumes B as [n, k] row-major.
    int64_t rows_per_tile = std::max<int64_t>(1, opt.weight_bank_bytes / (IN * 4));
    rows_per_tile = std::min(rows_per_tile, OUT);
    const int64_t ntiles = (OUT + rows_per_tile - 1) / rows_per_tile;

    if (rep) {
        rep->matmuls += 1;
        rep->tiles += ntiles;
        rep->weight_bytes_streamed += OUT * IN * 4;
        std::ostringstream os;
        os << what << " [" << IN << "->" << OUT << "]  weight " << (OUT * IN * 4 / 1024)
           << " KB -> " << ntiles << " tile(s) of " << rows_per_tile << " rows ("
           << (rows_per_tile * IN * 4 / 1024) << " KB/bank)";
        rep->tiling_log.push_back(os.str());
    }

    const int64_t mark = spm.mark();
    uint32_t wbuf[2] = {spm.alloc(rows_per_tile * IN, "weight bank A"),
                        spm.alloc(rows_per_tile * IN, "weight bank B")};

    // BIAS FIRST, INTO THE OUTPUT BUFFER. Qwen2 has Q/K/V biases where Llama does
    // not, and the flag comes from the config, never from a hardcoded model name.
    //
    // The bias is DMA'd straight into Y and every tile then MATMULs with
    // kFlagAccumulate, so the MXU's wide accumulator starts at the bias and
    // rounds once — bit-identical to ops::linear seeding `acc` with it. The
    // obvious alternative (matmul, then a V_ADD of the bias) rounds twice and
    // disagrees with the CPU in the last ulp, which is precisely the class of
    // "close enough" that compiler_parity refuses.
    // THE FIRST WRITE TO Y IS THE DANGEROUS ONE. Y may still be being READ by an
    // outbound DMA issued earlier: k_proj and v_proj write the same kcur/vcur
    // buffers that the previous layer's "store K/V" streamed to HBM, and lm_head
    // writes the logits buffer that the previous step's "store logits" read. So
    // the first write of every projection waits on DEP_DMA_OUT as well as on the
    // MXU and VPU that produced Y's previous contents.
    //
    // In this model the gap happens to be enormous — a whole attention block and
    // FFN separate the store from the next writer — so nothing would break
    // today. That is exactly why it has to be a declared dependency rather than
    // a lucky schedule: the simulator executes functionally at ISSUE time, so an
    // anti-dependency violation is invisible to it (wrong timing, never wrong
    // values), and the next scheduling pass that hoists anything would turn this
    // from latent to silent-wrong. Stance 2 says the compiler owns placement;
    // owning it means declaring it.
    const uint16_t kFirstWriteDeps = static_cast<uint16_t>(DEP_MXU | DEP_VPU | DEP_DMA_OUT);

    // A tile's packed [M, rows_per_tile] landing zone, when Y is wider than one
    // tile. Allocated once and reused by every tile of this projection.
    const bool scatter = (M > 1 && ntiles > 1);
    const uint32_t tile_buf = scatter ? spm.alloc(M * rows_per_tile, "matmul tile out") : 0;

    // Bias goes into whatever the MATMUL will accumulate on top of. With M > 1
    // it must appear in every token row: a DMA with row_stride 0 re-reads the
    // same OUT floats M times, which is a broadcast descriptor and costs the
    // engine nothing extra.
    if (has_bias && !scatter) {
        Instr ld = dma_load(bias_hbm, y_spm, M, OUT, 0, 0);
        ld.dep_mask = kFirstWriteDeps;
        E.emit(ld, std::string(what) + " bias load -> Y");
    }

    auto issue_load = [&](int64_t t, int bank) {
        int64_t r0 = t * rows_per_tile;
        int64_t rows = std::min(rows_per_tile, OUT - r0);
        Instr in = dma_load(w_hbm + static_cast<uint64_t>(r0 * IN), wbuf[bank], rows, IN, IN,
                            bank);
        // The load must not overwrite a bank the MXU is still reading, so EVERY
        // weight load waits on DEP_MXU. From tile 2 on that is the obvious
        // anti-dependency against tile t-2's matmul. For tiles 0 and 1 it is
        // less obvious and just as necessary: the banks are at fixed SPM offsets
        // reused by every projection, so this projection's prologue load would
        // otherwise land on top of the PREVIOUS projection's still-executing
        // tail matmul. Program order does not save us — the front end issues and
        // moves on, which is the entire point of the machine.
        //
        // This one is invisible to the simulator: its functional layer executes
        // at issue time, so an anti-dependency violation shows up as wrong
        // TIMING, never as wrong VALUES. It has to be got right by reasoning,
        // and the cost (no DMA/MXU overlap across a projection boundary) is real
        // and is quoted in docs/04-compiler.md.
        in.dep_mask = DEP_MXU;
        E.emit(in, std::string(what) + " load tile " + std::to_string(t) + " -> bank " +
                       (bank ? "B" : "A"));
    };

    auto issue_matmul = [&](int64_t t, int bank) {
        int64_t r0 = t * rows_per_tile;
        int64_t rows = std::min(rows_per_tile, OUT - r0);
        // Where this tile's packed [M, rows] result lands.
        const uint32_t c_spm = scatter ? tile_buf : static_cast<uint32_t>(y_spm + r0 * 4);

        if (has_bias && scatter) {
            // Broadcast this tile's slice of the bias into the scratch, so the
            // accumulator still starts at the bias and rounds once.
            Instr ld = dma_load(bias_hbm + static_cast<uint64_t>(r0), c_spm, M, rows, 0, 0);
            ld.dep_mask = static_cast<uint16_t>(DEP_MXU | DEP_VPU);
            E.emit(ld, std::string(what) + " bias -> tile " + std::to_string(t));
        }

        Instr mm;
        mm.op = Op::MATMUL;
        mm.arg[0] = x_spm;
        mm.arg[1] = wbuf[bank];
        mm.arg[2] = c_spm;
        mm.arg[3] = static_cast<uint32_t>(M);             // m: tokens in this chunk
        mm.arg[4] = static_cast<uint32_t>(rows);          // n
        mm.arg[5] = static_cast<uint32_t>(IN);            // k
        // Seeded by the bias sitting in C, when there is one.
        if (has_bias) mm.flags |= kFlagAccumulate;
        mm.dep_mask = static_cast<uint16_t>(E.wait_bank(bank) | DEP_VPU);
        // The bias load is inbound DMA like the weights, but it is not tagged to
        // this tile's bank, so a fine-grained mask would not name it. One engine
        // and in-order issue already order it ahead of every tile load; saying so
        // explicitly costs nothing here and survives a future second DMA engine.
        if (has_bias) mm.dep_mask = static_cast<uint16_t>(mm.dep_mask | DEP_DMA_IN);
        // Without a bias, tile 0's matmul IS the first write to Y — see
        // kFirstWriteDeps above. Later tiles write disjoint slices and are
        // ordered behind tile 0, so gating the first one is sufficient.
        if (!has_bias && t == 0) mm.dep_mask = static_cast<uint16_t>(mm.dep_mask | kFirstWriteDeps);
        E.emit(mm, std::string(what) + " matmul tile " + std::to_string(t));

        if (scatter) {
            // Place the packed [M, rows] tile into Y's column slice [r0, r0+rows).
            Instr cp;
            cp.op = Op::V_COPY;
            cp.arg[0] = tile_buf;
            cp.arg[1] = static_cast<uint32_t>(y_spm + r0 * 4);
            cp.arg[2] = static_cast<uint32_t>(rows);   // cols
            cp.arg[3] = static_cast<uint32_t>(M);      // rows
            cp.arg[4] = static_cast<uint32_t>(rows);   // packed source stride
            cp.arg[5] = static_cast<uint32_t>(OUT);    // destination row stride
            cp.dep_mask = static_cast<uint16_t>(DEP_MXU | DEP_VPU | DEP_DMA_OUT);
            E.emit(cp, std::string(what) + " place tile " + std::to_string(t));
        }
    };

    if (opt.software_pipeline && ntiles > 1) {
        issue_load(0, 0);
        issue_load(1, 1);
        for (int64_t t = 0; t < ntiles; ++t) {
            int bank = static_cast<int>(t & 1);
            // MATMUL FIRST, THEN THE PREFETCH. Tile t+2 reuses tile t's bank, so
            // emitting its load before this matmul would hand the MXU tile t+2's
            // weights — and because the simulator's functional layer runs in
            // program order, it would do so silently and deterministically. The
            // prefetch still overlaps: it flies while tile t+1's matmul (the
            // OTHER bank) runs, which is all double buffering ever promised.
            issue_matmul(t, bank);
            if (t + 2 < ntiles) issue_load(t + 2, bank);
        }
    } else {
        for (int64_t t = 0; t < ntiles; ++t) {
            int bank = static_cast<int>(t & 1);
            issue_load(t, bank);
            issue_matmul(t, bank);
        }
    }

    spm.release(mark);
}

// How many floats of scratchpad a prefill chunk of M tokens needs, by the same
// arithmetic the allocator below performs. The allocator remains the ground
// truth — it throws if this is wrong — but a caller wants to size a chunk
// without compiling one first.
static int64_t prefill_spm_floats(const HbmLayout& L, const LowerOptions& opt, int64_t base,
                                  int64_t M) {
    const ModelConfig& cfg = L.cfg;
    const int64_t H = cfg.hidden_size, HD = cfg.head_dim();
    const int64_t QDIM = cfg.num_attention_heads * HD;
    const int64_t KVDIM = cfg.num_key_value_heads * HD;
    const int64_t INTER = cfg.intermediate_size, KEYS = base + M;
    // Widest tile scratch: rows_per_tile is largest when the reduction dim is
    // smallest, i.e. the hidden-sized projections.
    const int64_t rpt = std::max<int64_t>(1, opt.weight_bank_bytes / (H * 4));

    const int64_t per_token = 4 * H + 2 * QDIM + 2 * KVDIM + 3 * INTER + 2 * HD;
    const int64_t fixed = H + cfg.vocab_size + KEYS * HD;
    return fixed + per_token * M + M * KEYS + M * rpt;
}

int64_t max_prefill_tokens(const HbmLayout& L, const LowerOptions& opt) {
    // Two weight banks are carved out of the same scratchpad, plus slack for
    // 64-byte alignment padding across ~16 allocations.
    const int64_t budget = opt.spm_bytes - 2 * opt.weight_bank_bytes - 4096;
    int64_t best = 0;
    for (int64_t m = 1; m <= L.max_seq; ++m)
        if (prefill_spm_floats(L, opt, 0, m) * 4 <= budget) best = m;
    return std::max<int64_t>(1, best);
}

Program lower_prefill(const HbmLayout& L, int64_t base, int64_t ntokens,
                      const LowerOptions& opt, LowerReport* rep) {
    const ModelConfig& cfg = L.cfg;
    const int64_t H = cfg.hidden_size;
    const int64_t HD = cfg.head_dim();
    const int64_t NH = cfg.num_attention_heads;
    const int64_t NKV = cfg.num_key_value_heads;
    const int64_t QDIM = NH * HD;
    const int64_t KVDIM = NKV * HD;
    const int64_t INTER = cfg.intermediate_size;
    const int64_t V = cfg.vocab_size;
    const int64_t qpk = cfg.q_per_kv();
    const int64_t M = ntokens;
    const int64_t KEYS = base + M;  // keys visible to the last token of the chunk

    Emitter E;
    E.opt = &opt;
    SpmAllocator spm(opt.spm_bytes);

    // Every activation is [M, feature] row-major, so a token is a contiguous
    // row and the vector ops' (rows, stride) form applies directly.
    uint32_t h = spm.alloc(M * H, "h");
    uint32_t normed = spm.alloc(M * H, "normed");
    uint32_t ln_w = spm.alloc(H, "layernorm weight");
    uint32_t q = spm.alloc(M * QDIM, "q");
    uint32_t kcur = spm.alloc(M * KVDIM, "k");
    uint32_t vcur = spm.alloc(M * KVDIM, "v");
    uint32_t attn = spm.alloc(M * QDIM, "attn out");
    uint32_t o = spm.alloc(M * H, "o proj out");
    uint32_t scores = spm.alloc(M * KEYS, "scores [M, KEYS]");
    uint32_t qh = spm.alloc(M * HD, "one head's Q, packed");
    uint32_t ah = spm.alloc(M * HD, "one head's attention out");
    uint32_t krow = spm.alloc(KEYS * HD, "K or V rows for one head");
    uint32_t gate = spm.alloc(M * INTER, "gate");
    uint32_t up = spm.alloc(M * INTER, "up");
    uint32_t ffn = spm.alloc(M * INTER, "ffn");
    uint32_t down = spm.alloc(M * H, "down");

    {
        // The host wrote M embeddings contiguously; one descriptor fetches them.
        Instr ld = dma_load(L.act_in + static_cast<uint64_t>(base * H), h, M, H, H, 0);
        E.emit(ld, "load token embeddings");
    }

    const uint16_t kNormDeps = static_cast<uint16_t>(DEP_DMA_IN | DEP_VPU | DEP_MXU);

    for (int64_t li = 0; li < cfg.num_hidden_layers; ++li) {
        const auto& la = L.layers[static_cast<size_t>(li)];
        const std::string lp = "L" + std::to_string(li) + ".";

        {
            Instr ld = dma_load(la.in_ln, ln_w, 1, H, H, 0);
            ld.dep_mask = DEP_VPU;  // ln_w is still being read by the last norm
            E.emit(ld, lp + "input_layernorm load");
            Instr rn = vec3(Op::V_RMSNORM, h, ln_w, normed, H, cfg.rms_norm_eps);
            rn.arg[4] = static_cast<uint32_t>(M);  // one normalisation per token
            rn.arg[5] = static_cast<uint32_t>(H);
            rn.dep_mask = kNormDeps;
            E.emit(rn, lp + "input_layernorm");
        }

        lower_projection(E, spm, la.q_w, la.q_b, true, normed, q, H, QDIM, rep,
                         (lp + "q_proj").c_str(), M);
        lower_projection(E, spm, la.k_w, la.k_b, true, normed, kcur, H, KVDIM, rep,
                         (lp + "k_proj").c_str(), M);
        lower_projection(E, spm, la.v_w, la.v_b, true, normed, vcur, H, KVDIM, rep,
                         (lp + "v_proj").c_str(), M);

        {
            // Row r is the token at absolute position base + r; the VPU walks
            // the rows and steps the angle itself.
            Instr rq;
            rq.op = Op::V_ROPE;
            rq.arg[0] = q;
            rq.arg[1] = static_cast<uint32_t>(HD);
            rq.arg[2] = static_cast<uint32_t>(base);
            rq.arg[3] = static_cast<uint32_t>(NH);
            rq.arg[4] = static_cast<uint32_t>(M);
            rq.arg[5] = static_cast<uint32_t>(QDIM);
            rq.imm_f = cfg.rope_theta;
            rq.dep_mask = static_cast<uint16_t>(DEP_VPU | DEP_MXU | DEP_DMA_IN);
            E.emit(rq, lp + "rope(q)");

            Instr rk = rq;
            rk.arg[0] = kcur;
            rk.arg[3] = static_cast<uint32_t>(NKV);
            rk.arg[5] = static_cast<uint32_t>(KVDIM);
            E.emit(rk, lp + "rope(k)");
        }

        {
            // The whole chunk's K and V append to the cache in one descriptor
            // each — M rows of KVDIM, contiguous on both sides.
            Instr sk = dma_store(L.k_at(li, base), kcur, M, KVDIM, KVDIM);
            sk.dep_mask = static_cast<uint16_t>(DEP_VPU | DEP_DMA_OUT);
            E.emit(sk, lp + "store K");
            Instr sv = dma_store(L.v_at(li, base), vcur, M, KVDIM, KVDIM);
            sv.dep_mask = static_cast<uint16_t>(DEP_VPU | DEP_DMA_OUT);
            E.emit(sv, lp + "store V");
        }

        for (int64_t qi = 0; qi < NH; ++qi) {
            const int64_t kvh = qi / qpk;
            const std::string hd = lp + "head " + std::to_string(qi) + " ";

            // Pack this head's Q out of [M, QDIM] into a contiguous [M, HD], so
            // the MATMUL's A operand has row stride k as the ISA requires.
            Instr cq;
            cq.op = Op::V_COPY;
            cq.arg[0] = static_cast<uint32_t>(q + qi * HD * 4);
            cq.arg[1] = qh;
            cq.arg[2] = static_cast<uint32_t>(HD);
            cq.arg[3] = static_cast<uint32_t>(M);
            cq.arg[4] = static_cast<uint32_t>(QDIM);
            cq.arg[5] = static_cast<uint32_t>(HD);
            cq.dep_mask = static_cast<uint16_t>(DEP_VPU | DEP_MXU);
            E.emit(cq, hd + "pack Q");

            Instr ldk = dma_load(L.k_at(li, 0) + static_cast<uint64_t>(kvh * HD), krow, KEYS,
                                 HD, KVDIM, 0);
            ldk.dep_mask = static_cast<uint16_t>(DEP_DMA_OUT | DEP_MXU);
            E.emit(ldk, hd + "gather K");

            // scores[M, KEYS] = Qh[M, HD] . K[KEYS, HD]^T
            Instr mm;
            mm.op = Op::MATMUL;
            mm.arg[0] = qh;
            mm.arg[1] = krow;
            mm.arg[2] = scores;
            mm.arg[3] = static_cast<uint32_t>(M);
            mm.arg[4] = static_cast<uint32_t>(KEYS);
            mm.arg[5] = static_cast<uint32_t>(HD);
            mm.dep_mask = static_cast<uint16_t>(DEP_DMA_IN | DEP_VPU);
            E.emit(mm, hd + "QK^T");

            Instr sc;
            sc.op = Op::V_SCALE;
            sc.arg[0] = scores;
            sc.arg[1] = scores;
            sc.arg[2] = static_cast<uint32_t>(M * KEYS);
            sc.imm_f = 1.0f / std::sqrt(static_cast<float>(HD));
            sc.dep_mask = static_cast<uint16_t>(DEP_MXU | DEP_VPU);
            E.emit(sc, hd + "scale");

            // Causal softmax: row r covers base+r+1 keys, the rest is zeroed so
            // the AV matmul below can read the full rectangle.
            Instr sm;
            sm.op = Op::V_SOFTMAX;
            sm.arg[0] = scores;
            sm.arg[1] = static_cast<uint32_t>(base + 1);
            sm.arg[2] = static_cast<uint32_t>(M);
            sm.arg[3] = static_cast<uint32_t>(KEYS);
            sm.dep_mask = DEP_VPU;
            E.emit(sm, hd + "causal softmax");

            Instr ldv = dma_load(L.v_at(li, 0) + static_cast<uint64_t>(kvh * HD), krow, KEYS,
                                 HD, KVDIM, 0);
            ldv.dep_mask = static_cast<uint16_t>(DEP_MXU);
            E.emit(ldv, hd + "gather V");

            // attn_h[M, HD] = scores[M, KEYS] . V[KEYS, HD], V in [k, n] order.
            Instr av;
            av.op = Op::MATMUL;
            av.arg[0] = scores;
            av.arg[1] = krow;
            av.arg[2] = ah;
            av.arg[3] = static_cast<uint32_t>(M);
            av.arg[4] = static_cast<uint32_t>(HD);
            av.arg[5] = static_cast<uint32_t>(KEYS);
            av.flags |= kFlagBIsKN;
            av.dep_mask = static_cast<uint16_t>(DEP_DMA_IN | DEP_VPU);
            E.emit(av, hd + "AV");

            Instr ca;
            ca.op = Op::V_COPY;
            ca.arg[0] = ah;
            ca.arg[1] = static_cast<uint32_t>(attn + qi * HD * 4);
            ca.arg[2] = static_cast<uint32_t>(HD);
            ca.arg[3] = static_cast<uint32_t>(M);
            ca.arg[4] = static_cast<uint32_t>(HD);
            ca.arg[5] = static_cast<uint32_t>(QDIM);
            ca.dep_mask = static_cast<uint16_t>(DEP_MXU | DEP_VPU);
            E.emit(ca, hd + "place into attn");
        }

        lower_projection(E, spm, la.o_w, 0, false, attn, o, QDIM, H, rep,
                         (lp + "o_proj").c_str(), M);
        {
            Instr add = vec3(Op::V_ADD, o, h, 0, M * H);
            add.arg[2] = static_cast<uint32_t>(M * H);
            add.dep_mask = static_cast<uint16_t>(DEP_MXU | DEP_VPU);
            E.emit(add, lp + "residual (attn)");
        }

        {
            Instr ld = dma_load(la.post_ln, ln_w, 1, H, H, 0);
            ld.dep_mask = DEP_VPU;
            E.emit(ld, lp + "post_attention_layernorm load");
            Instr rn = vec3(Op::V_RMSNORM, h, ln_w, normed, H, cfg.rms_norm_eps);
            rn.arg[4] = static_cast<uint32_t>(M);
            rn.arg[5] = static_cast<uint32_t>(H);
            rn.dep_mask = kNormDeps;
            E.emit(rn, lp + "post_attention_layernorm");
        }
        lower_projection(E, spm, la.gate_w, 0, false, normed, gate, H, INTER, rep,
                         (lp + "gate_proj").c_str(), M);
        lower_projection(E, spm, la.up_w, 0, false, normed, up, H, INTER, rep,
                         (lp + "up_proj").c_str(), M);
        {
            Instr sm = vec3(Op::V_SILU_MUL, gate, up, ffn, M * INTER);
            sm.dep_mask = static_cast<uint16_t>(DEP_MXU | DEP_VPU);
            E.emit(sm, lp + "silu*up");
        }
        lower_projection(E, spm, la.down_w, 0, false, ffn, down, INTER, H, rep,
                         (lp + "down_proj").c_str(), M);
        {
            Instr add = vec3(Op::V_ADD, down, h, 0, M * H);
            add.arg[2] = static_cast<uint32_t>(M * H);
            add.dep_mask = static_cast<uint16_t>(DEP_MXU | DEP_VPU);
            E.emit(add, lp + "residual (ffn)");
        }
    }

    // ONLY THE LAST TOKEN GETS LOGITS. Prefill exists to fill the KV cache; the
    // other M-1 rows would each cost a 545 MB sweep of the embedding matrix to
    // produce a distribution nobody samples.
    {
        Instr ld = dma_load(L.final_ln, ln_w, 1, H, H, 0);
        ld.dep_mask = DEP_VPU;
        E.emit(ld, "model.norm load");
        Instr rn = vec3(Op::V_RMSNORM, static_cast<uint32_t>(h + (M - 1) * H * 4), ln_w, normed,
                        H, cfg.rms_norm_eps);
        rn.dep_mask = kNormDeps;
        E.emit(rn, "model.norm (last token)");
    }

    uint32_t logits = spm.alloc(V, "logits");
    lower_projection(E, spm, L.embed, 0, false, normed, logits, H, V, rep, "lm_head", 1);
    {
        Instr st = dma_store(L.act_out, logits, 1, V, V);
        st.dep_mask = static_cast<uint16_t>(DEP_MXU | DEP_DMA_OUT);
        E.emit(st, "store logits");
    }

    {
        Instr halt;
        halt.op = Op::HALT;
        halt.dep_mask = DEP_ALL;
        E.emit(halt, "halt");
    }

    if (rep) rep->spm_high_water = spm.high_water();
    return std::move(E.prog);
}

Program lower_decode_step(const HbmLayout& L, int64_t pos, const LowerOptions& opt,
                          LowerReport* rep) {
    const ModelConfig& cfg = L.cfg;
    const int64_t H = cfg.hidden_size;
    const int64_t HD = cfg.head_dim();
    const int64_t NH = cfg.num_attention_heads;
    const int64_t NKV = cfg.num_key_value_heads;
    const int64_t QDIM = NH * HD;
    const int64_t KVDIM = NKV * HD;
    const int64_t INTER = cfg.intermediate_size;
    const int64_t V = cfg.vocab_size;
    const int64_t qpk = cfg.q_per_kv();

    Emitter E;
    E.opt = &opt;
    // ONE allocator for the whole 8 MB scratchpad: activations first (they live
    // for the entire program), then each projection's two weight banks carved
    // out and released around it. high_water() therefore reports the true peak,
    // which is the number docs/03-isa-spec.md uses to justify the 8 MB.
    SpmAllocator spm(opt.spm_bytes);

    // --- persistent activation region (allocated once, reused every layer) ---
    uint32_t h = spm.alloc(H, "h");
    uint32_t normed = spm.alloc(H, "normed");
    uint32_t ln_w = spm.alloc(H, "layernorm weight");
    uint32_t q = spm.alloc(QDIM, "q");
    uint32_t kcur = spm.alloc(KVDIM, "k");
    uint32_t vcur = spm.alloc(KVDIM, "v");
    uint32_t attn = spm.alloc(QDIM, "attn out");
    uint32_t o = spm.alloc(H, "o proj out");
    uint32_t scores = spm.alloc(pos + 1, "scores");
    uint32_t krow = spm.alloc((pos + 1) * HD, "K rows for one head");
    uint32_t gate = spm.alloc(INTER, "gate");
    uint32_t up = spm.alloc(INTER, "up");
    uint32_t ffn = spm.alloc(INTER, "ffn");
    uint32_t down = spm.alloc(H, "down");

    // The host has already written the token embedding to HBM; fetch it.
    {
        Instr ld = dma_load(L.act_in, h, 1, H, H, 0);
        E.emit(ld, "load token embedding");
    }

    for (int64_t li = 0; li < cfg.num_hidden_layers; ++li) {
        const auto& la = L.layers[static_cast<size_t>(li)];
        const std::string lp = "L" + std::to_string(li) + ".";

        // ---- attention ----
        {
            Instr ld = dma_load(la.in_ln, ln_w, 1, H, H, 0);
            E.emit(ld, lp + "input_layernorm load");
            Instr rn = vec3(Op::V_RMSNORM, h, ln_w, normed, H, cfg.rms_norm_eps);
            rn.dep_mask = static_cast<uint16_t>(DEP_DMA_IN | DEP_VPU);
            E.emit(rn, lp + "input_layernorm");
        }

        lower_projection(E, spm, la.q_w, la.q_b, true, normed, q, H, QDIM, rep,
                         (lp + "q_proj").c_str());
        lower_projection(E, spm, la.k_w, la.k_b, true, normed, kcur, H, KVDIM, rep,
                         (lp + "k_proj").c_str());
        lower_projection(E, spm, la.v_w, la.v_b, true, normed, vcur, H, KVDIM, rep,
                         (lp + "v_proj").c_str());

        {
            Instr rq;
            rq.op = Op::V_ROPE;
            rq.arg[0] = q;
            rq.arg[1] = static_cast<uint32_t>(HD);
            rq.arg[2] = static_cast<uint32_t>(pos);
            rq.arg[3] = static_cast<uint32_t>(NH);
            rq.imm_f = cfg.rope_theta;
            rq.dep_mask = static_cast<uint16_t>(DEP_VPU | DEP_MXU | DEP_DMA_IN);
            E.emit(rq, lp + "rope(q)");

            Instr rk = rq;
            rk.arg[0] = kcur;
            rk.arg[3] = static_cast<uint32_t>(NKV);
            E.emit(rk, lp + "rope(k)");
        }

        // Append this position's K/V to the cache in HBM. The KV cache lives in
        // device memory, not the scratchpad — at max_seq it is far too large to
        // be resident, and this is exactly the traffic that makes decode
        // memory-bound on T1 just as it is on the CPU.
        {
            Instr sk = dma_store(L.k_at(li, pos), kcur, 1, KVDIM, KVDIM);
            sk.dep_mask = static_cast<uint16_t>(DEP_VPU | DEP_DMA_OUT);
            E.emit(sk, lp + "store K");
            Instr sv = dma_store(L.v_at(li, pos), vcur, 1, KVDIM, KVDIM);
            sv.dep_mask = static_cast<uint16_t>(DEP_VPU | DEP_DMA_OUT);
            E.emit(sv, lp + "store V");
        }

        // GQA attention, one query head at a time. Head-at-a-time keeps the
        // score buffer to (pos+1) floats and matches the CPU reference's op
        // order exactly, which is what preserves bit-exactness.
        for (int64_t qh = 0; qh < NH; ++qh) {
            int64_t kvh = qh / qpk;

            // Gather this KV head's K rows for positions [0, pos]. The 2-D
            // strided DMA does the gather: (pos+1) runs of HD floats, stride
            // KVDIM. No host-side repacking, which is the entire argument for
            // giving the DMA engine strides.
            Instr ldk = dma_load(L.k_at(li, 0) + static_cast<uint64_t>(kvh * HD), krow, pos + 1,
                                 HD, KVDIM, 0);
            ldk.dep_mask = static_cast<uint16_t>(DEP_DMA_OUT | DEP_MXU);
            E.emit(ldk, lp + "head " + std::to_string(qh) + " gather K");

            Instr mm;
            mm.op = Op::MATMUL;
            mm.arg[0] = static_cast<uint32_t>(q + qh * HD * 4);
            mm.arg[1] = krow;
            mm.arg[2] = scores;
            mm.arg[3] = 1;
            mm.arg[4] = static_cast<uint32_t>(pos + 1);
            mm.arg[5] = static_cast<uint32_t>(HD);
            mm.dep_mask = static_cast<uint16_t>(DEP_DMA_IN | DEP_VPU);
            E.emit(mm, lp + "head " + std::to_string(qh) + " QK^T");

            Instr sc;
            sc.op = Op::V_SCALE;
            sc.arg[0] = scores;
            sc.arg[1] = scores;
            sc.arg[2] = static_cast<uint32_t>(pos + 1);
            sc.imm_f = 1.0f / std::sqrt(static_cast<float>(HD));
            sc.dep_mask = static_cast<uint16_t>(DEP_MXU | DEP_VPU);
            E.emit(sc, lp + "head " + std::to_string(qh) + " scale");

            Instr sm;
            sm.op = Op::V_SOFTMAX;
            sm.arg[0] = scores;
            sm.arg[1] = static_cast<uint32_t>(pos + 1);
            sm.dep_mask = DEP_VPU;
            E.emit(sm, lp + "head " + std::to_string(qh) + " softmax");

            // Same gather for V, then attn[head] = scores . V.
            Instr ldv = dma_load(L.v_at(li, 0) + static_cast<uint64_t>(kvh * HD), krow, pos + 1,
                                 HD, KVDIM, 0);
            ldv.dep_mask = static_cast<uint16_t>(DEP_MXU);
            E.emit(ldv, lp + "head " + std::to_string(qh) + " gather V");

            // attn[1, HD] = scores[1, pos+1] . Vrows[pos+1, HD], and MATMUL
            // wants B as [n, k]. Vrows is [pos+1, HD] = [k, n], so this one
            // needs V^T... which the systolic array gets for free by feeding
            // rows as columns. Model it as: B = krow viewed as [HD, pos+1].
            Instr av;
            av.op = Op::MATMUL;
            av.arg[0] = scores;
            av.arg[1] = krow;
            av.arg[2] = static_cast<uint32_t>(attn + qh * HD * 4);
            av.arg[3] = 1;
            av.arg[4] = static_cast<uint32_t>(HD);
            av.arg[5] = static_cast<uint32_t>(pos + 1);
            // krow currently holds V as [pos+1, HD] = [k, n], so the B operand
            // is in the transposed orientation. kFlagBIsKN consumes it directly
            // instead of paying for a scratchpad transpose per head per layer.
            av.flags |= kFlagBIsKN;
            av.dep_mask = static_cast<uint16_t>(DEP_DMA_IN | DEP_VPU);
            E.emit(av, lp + "head " + std::to_string(qh) + " AV");
        }

        lower_projection(E, spm, la.o_w, 0, false, attn, o, QDIM, H, rep,
                         (lp + "o_proj").c_str());
        {
            Instr add = vec3(Op::V_ADD, o, h, 0, H);
            add.arg[2] = static_cast<uint32_t>(H);
            add.dep_mask = static_cast<uint16_t>(DEP_MXU | DEP_VPU);
            E.emit(add, lp + "residual (attn)");
        }

        // ---- FFN ----
        {
            Instr ld = dma_load(la.post_ln, ln_w, 1, H, H, 0);
            E.emit(ld, lp + "post_attention_layernorm load");
            Instr rn = vec3(Op::V_RMSNORM, h, ln_w, normed, H, cfg.rms_norm_eps);
            rn.dep_mask = static_cast<uint16_t>(DEP_DMA_IN | DEP_VPU);
            E.emit(rn, lp + "post_attention_layernorm");
        }
        lower_projection(E, spm, la.gate_w, 0, false, normed, gate, H, INTER, rep,
                         (lp + "gate_proj").c_str());
        lower_projection(E, spm, la.up_w, 0, false, normed, up, H, INTER, rep,
                         (lp + "up_proj").c_str());
        {
            Instr sm = vec3(Op::V_SILU_MUL, gate, up, ffn, INTER);
            sm.dep_mask = static_cast<uint16_t>(DEP_MXU | DEP_VPU);
            E.emit(sm, lp + "silu*up");
        }
        lower_projection(E, spm, la.down_w, 0, false, ffn, down, INTER, H, rep,
                         (lp + "down_proj").c_str());
        {
            Instr add = vec3(Op::V_ADD, down, h, 0, H);
            add.arg[2] = static_cast<uint32_t>(H);
            add.dep_mask = static_cast<uint16_t>(DEP_MXU | DEP_VPU);
            E.emit(add, lp + "residual (ffn)");
        }
    }

    // ---- final norm + tied output projection ----
    {
        Instr ld = dma_load(L.final_ln, ln_w, 1, H, H, 0);
        E.emit(ld, "model.norm load");
        Instr rn = vec3(Op::V_RMSNORM, h, ln_w, normed, H, cfg.rms_norm_eps);
        rn.dep_mask = static_cast<uint16_t>(DEP_DMA_IN | DEP_VPU);
        E.emit(rn, "model.norm");
    }

    // The output projection reuses the embedding matrix (tie_word_embeddings).
    // At [896 -> 151936] this is 545 MB of fp32 weights for ONE token: the
    // single biggest DMA in the program, and on its own a demonstration that
    // decode on T1 is bandwidth-bound exactly as it is on the CPU.
    uint32_t logits = spm.alloc(V, "logits");
    lower_projection(E, spm, L.embed, 0, false, normed, logits, H, V, rep, "lm_head");
    {
        Instr st = dma_store(L.act_out, logits, 1, V, V);
        st.dep_mask = static_cast<uint16_t>(DEP_MXU | DEP_DMA_OUT);
        E.emit(st, "store logits");
    }

    {
        Instr halt;
        halt.op = Op::HALT;
        halt.dep_mask = DEP_ALL;  // drain everything before reporting done
        E.emit(halt, "halt");
    }

    if (rep) rep->spm_high_water = spm.high_water();
    return std::move(E.prog);
}

}  // namespace accel
