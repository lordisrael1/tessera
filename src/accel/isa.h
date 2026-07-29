#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

// ============================================================================
// The T1 ISA — a fixed-function transformer accelerator that does not exist.
// ============================================================================
// The normative spec is docs/03-isa-spec.md. This header is the machine-readable
// half of it and the two must agree.
//
// THREE DESIGN STANCES, each of which is a bet worth defending:
//
// 1. NO REGISTERS, NO BRANCHES, NO LOOPS. A transformer forward pass is a
//    straight line: the shapes are known from config.json, the layer count is
//    known, nothing is data-dependent except the position, and position is a
//    compile-time constant once you bucket sequence lengths. So the compiler
//    unrolls everything and the program is a flat instruction array. The chip
//    spends zero area on branch prediction, register renaming, or an
//    instruction cache hierarchy — that area becomes MACs instead. This is the
//    fixed-function bet stated in silicon.
//
// 2. SOFTWARE-MANAGED MEMORY. There are no caches and no coherence. Every byte
//    that moves between HBM and the scratchpad does so because a DMA_LOAD or
//    DMA_STORE said so. The compiler owns data placement, and being wrong about
//    it is a correctness bug rather than a performance one. This single decision
//    generates most of the interesting compiler work in M3 (tiling, SPM
//    allocation, double buffering) and it is why an accelerator can hit high
//    utilisation on a workload where a cache-based CPU cannot: the machine never
//    speculates, never evicts something it still needed, and never pays a
//    read-for-ownership on a line it is about to overwrite whole.
//
// 3. COARSE SCOREBOARD CLASSES, ON PURPOSE. Dependencies are expressed as a
//    small bitmask of unit classes, not as per-buffer dataflow. This is cheap in
//    hardware and it is WRONG in a specific, interesting way: two independent
//    DMA streams share the DMA class and therefore serialise against each other
//    even when they touch disjoint memory. The flaw is left in deliberately.
//    M7's profiler is supposed to find it, and "I found it in the trace, here is
//    the before/after" is the point of having built the profiler at all.
// ============================================================================

namespace accel {

enum class Op : uint8_t {
    NOP = 0,
    DMA_LOAD,    // HBM -> SPM, 2-D strided
    DMA_STORE,   // SPM -> HBM, 2-D strided
    MATMUL,      // SPM: C[m,n] (+)= A[m,k] * B[n,k]^T   (B is row-major [n,k])
    V_RMSNORM,   // SPM: dst[i] = src[i] * rsqrt(mean(src^2)+eps) * weight[i]
    V_SOFTMAX,   // SPM: in-place softmax over `len` elements
    V_ROPE,      // SPM: in-place rotary embedding, split-half convention
    V_SILU_MUL,  // SPM: dst[i] = silu(gate[i]) * up[i]
    V_ADD,       // SPM: dst[i] += src[i]              (residual)
    V_COPY,      // SPM: dst[i]  = src[i]
    V_SCALE,     // SPM: dst[i]  = src[i] * imm_f
    BARRIER,     // wait for every class in dep_mask to be idle
    HALT,
};

// Scoreboard dependency classes. An instruction issues only once every class in
// its dep_mask has retired all the work assigned to it.
//
// Inbound DMA has TWO classes, one per SPM bank, so a double-buffered schedule
// can say "wait for bank A's load, while bank B's load is still in flight".
// Whether the compiler actually uses them separately is a compiler OPTION
// (LowerOptions::coarse_dma_deps) precisely so the cost of the coarse design can
// be measured rather than asserted — see docs/04-compiler.md.
//
// Classes are about DEPENDENCE, not about hardware. There is still only ONE DMA
// engine: two loads tagged to different banks are independent for correctness
// but still serialise on the engine. The simulator models both effects
// separately, which is the only way the resulting stall attribution means
// anything.
enum Dep : uint16_t {
    DEP_NONE = 0,
    DEP_DMA_A = 1u << 0,    // inbound DMA into bank A
    DEP_DMA_B = 1u << 1,    // inbound DMA into bank B
    DEP_DMA_OUT = 1u << 2,  // outbound DMA
    DEP_MXU = 1u << 3,      // the systolic array
    DEP_VPU = 1u << 4,      // the vector unit
    DEP_DMA_IN = DEP_DMA_A | DEP_DMA_B,
    DEP_ALL = 0x1Fu,
};
constexpr int kNumDepClasses = 5;

// A fixed 32-byte instruction record.
//
// Addresses: SPM operands are BYTE OFFSETS into the scratchpad (32-bit is ample
// for 8 MB). HBM addresses need more than 32 bits, so they occupy an aligned
// PAIR of arg slots, low word first — see hbm_addr()/set_hbm_addr().
struct Instr {
    Op op = Op::NOP;
    uint8_t flags = 0;       // per-op; see kFlag* below
    uint16_t dep_mask = 0;   // classes this instruction waits on
    uint32_t arg[6] = {0, 0, 0, 0, 0, 0};
    float imm_f = 0.0f;      // eps, rope theta, scale factor

    uint64_t hbm_addr(int i) const {
        return static_cast<uint64_t>(arg[i]) | (static_cast<uint64_t>(arg[i + 1]) << 32);
    }
    void set_hbm_addr(int i, uint64_t a) {
        arg[i] = static_cast<uint32_t>(a & 0xFFFFFFFFu);
        arg[i + 1] = static_cast<uint32_t>(a >> 32);
    }
};
static_assert(sizeof(Instr) == 32, "T1 instructions are fixed 32-byte records");

// MATMUL: seed the accumulator from C instead of from zero. This is what lets a
// reduction dimension larger than one SPM tile be split across several MATMULs.
//
// THE ROUNDING IS PART OF THE CONTRACT: the existing C is read INTO the MXU's
// wide accumulator, so the result is fp32(double(C) + sum(A.B)) — one rounding,
// at the end, not one per MATMUL. A cell that rounded its partial before adding
// would make a k-split matmul answer differently from an unsplit one, i.e. would
// make the compiler's tiling decisions visible in the numerics. Making tiling
// value-invariant is the property test/compiler_parity.cpp exists to assert.
//
// It also gives bias for free: DMA the bias vector into C, then MATMUL with this
// flag, and the machine computes fp32(bias + sum(x.w)) — exactly what
// ops::linear does when it seeds `acc` with the bias. No V_ADD, no second pass
// over the output, and no 1-ULP disagreement with the CPU reference.
constexpr uint8_t kFlagAccumulate = 1u << 0;
// DMA_LOAD: this load targets bank B, so its completion satisfies DEP_DMA_B
// rather than DEP_DMA_A. Set by the software-pipelining pass.
constexpr uint8_t kFlagBankB = 1u << 1;
// MATMUL: the B operand is laid out [k, n] instead of the default [n, k].
//
// This is not a software convenience. A systolic array is fed from two edges,
// and which edge a matrix enters by is a wiring choice, not a data movement —
// so consuming both orientations costs the hardware nothing. Exposing it costs
// the COMPILER a great deal: attention needs Q.K^T (B as [n,k]) and then
// scores.V (B as [k,n]) back to back on the same buffer. Without this flag the
// compiler would have to materialise a transposed copy of V in the scratchpad
// for every head of every layer, which is pure DMA traffic in the one place the
// machine can least afford it.
constexpr uint8_t kFlagBIsKN = 1u << 2;

// Which dependency class an instruction's COMPLETION satisfies. Every
// instruction produces exactly one (or none).
inline uint16_t produced_class(const Instr& in) {
    switch (in.op) {
        case Op::DMA_LOAD: return (in.flags & kFlagBankB) ? DEP_DMA_B : DEP_DMA_A;
        case Op::DMA_STORE: return DEP_DMA_OUT;
        case Op::MATMUL: return DEP_MXU;
        case Op::V_RMSNORM:
        case Op::V_SOFTMAX:
        case Op::V_ROPE:
        case Op::V_SILU_MUL:
        case Op::V_ADD:
        case Op::V_COPY:
        case Op::V_SCALE: return DEP_VPU;
        default: return DEP_NONE;
    }
}

// Which physical unit an instruction OCCUPIES. Both inbound DMA classes share
// one engine — that is the distinction between dependence and structure.
enum Unit { UNIT_DMA_IN = 0, UNIT_DMA_OUT = 1, UNIT_MXU = 2, UNIT_VPU = 3, UNIT_NONE = -1 };
inline int occupied_unit(const Instr& in) {
    switch (in.op) {
        case Op::DMA_LOAD: return UNIT_DMA_IN;
        case Op::DMA_STORE: return UNIT_DMA_OUT;
        case Op::MATMUL: return UNIT_MXU;
        case Op::V_RMSNORM:
        case Op::V_SOFTMAX:
        case Op::V_ROPE:
        case Op::V_SILU_MUL:
        case Op::V_ADD:
        case Op::V_COPY:
        case Op::V_SCALE: return UNIT_VPU;
        default: return UNIT_NONE;
    }
}

// A program is a flat instruction array plus the HBM image it reads from.
// No entry points, no call stack, no relocation: execution starts at 0 and runs
// until HALT.
struct Program {
    std::vector<Instr> code;
    // Human-readable provenance for each instruction ("layer 3 mlp.up tile 2/8").
    // Not part of the ISA — it is what makes the disassembly and the M7 trace
    // legible, and it is the "programming abstractions for rapid model porting"
    // artifact the JD asks for.
    std::vector<std::string> notes;

    void emit(const Instr& i, std::string note = {}) {
        code.push_back(i);
        notes.push_back(std::move(note));
    }
    size_t size() const { return code.size(); }
};

const char* op_name(Op op);
std::string disassemble(const Program& p, size_t max_lines = 0);

// ---------------------------------------------------------------------------
// Machine parameters. One struct so the simulator, the compiler and the docs
// cannot drift apart, and so a "what if the scratchpad were bigger" experiment
// is a field change rather than a rewrite.
// ---------------------------------------------------------------------------
struct T1Config {
    // MXU: a 32x32 systolic array of fp32 MACs, one MAC per cell per cycle.
    int64_t mxu_dim = 32;
    // VPU: 32 fp32 lanes.
    int64_t vpu_lanes = 32;
    // SPM: 8 MB software-managed scratchpad, two banks for double buffering.
    int64_t spm_bytes = 8 * 1024 * 1024;
    int64_t spm_banks = 2;
    // DMA: 256 B/cycle sustained, fixed 400-cycle latency. The latency is the
    // whole reason double buffering exists; make it zero and the compiler's
    // software-pipelining pass becomes untestable.
    int64_t dma_bytes_per_cycle = 256;
    int64_t dma_latency_cycles = 400;
    double clock_ghz = 1.0;

    // Derived roofs, so docs/03-isa-spec.md quotes computed values.
    double peak_flops_per_cycle() const {
        return static_cast<double>(mxu_dim * mxu_dim * 2);  // 2048
    }
    double peak_gflops() const { return peak_flops_per_cycle() * clock_ghz; }
    double dma_gbytes_per_s() const {
        return static_cast<double>(dma_bytes_per_cycle) * clock_ghz;
    }
    // Machine balance: flop per byte at which the chip stops being memory-bound.
    double balance() const { return peak_flops_per_cycle() / static_cast<double>(dma_bytes_per_cycle); }
};

}  // namespace accel
