// ============================================================================
// sim_isa — hand-counted cycle tests for the T1 simulator (bible §5.3).
// ============================================================================
// "Do not skip these; debugging the compiler on a broken simulator is despair."
//
// Every expected cycle count below is derived by hand from the timing model in
// docs/03-isa-spec.md and written out in the comment above the assertion. If a
// test fails, either the model changed (update the doc in the same commit) or
// the simulator is wrong. Both are things you want to hear about immediately.
// ============================================================================
#include <cmath>
#include <cstdio>
#include <vector>

#include "../src/accel/isa.h"
#include "../src/accel/simulator.h"
#include "check.h"

using namespace accel;

namespace {

T1Config cfg() {
    T1Config c;  // defaults: 32x32 MXU, 32-lane VPU, 256 B/cy DMA, 400 cy latency
    return c;
}

Instr load(uint64_t hbm, uint32_t spm, int64_t rows, int64_t cols, int64_t stride,
           uint16_t dep = DEP_NONE, uint8_t flags = 0) {
    Instr in;
    in.op = Op::DMA_LOAD;
    in.flags = flags;
    in.dep_mask = dep;
    in.set_hbm_addr(0, hbm);
    in.arg[2] = spm;
    in.arg[3] = static_cast<uint32_t>(rows);
    in.arg[4] = static_cast<uint32_t>(cols);
    in.arg[5] = static_cast<uint32_t>(stride);
    return in;
}

Instr halt() {
    Instr in;
    in.op = Op::HALT;
    in.dep_mask = DEP_ALL;
    return in;
}

}  // namespace

// 1) DMA timing: 400 cycles of latency + bytes/256.
//    64 floats = 256 bytes = 1 cycle of transfer -> 401 cycles busy.
//    Issue at 0, HALT waits on everything -> ends at 401, +1 for the HALT = 402.
static void test_dma_cycles() {
    std::vector<float> hbm(1024, 1.0f);
    Simulator sim(cfg(), hbm);
    Program p;
    p.emit(load(0, 0, 1, 64, 64), "load 64 floats");
    p.emit(halt(), "halt");
    CHECK(sim.run(p) == SimStatus::OK);
    std::printf("  [dma] 256 B load -> %lld cycles\n", static_cast<long long>(sim.stats().cycles));
    CHECK(sim.stats().cycles == 402);
    CHECK(sim.stats().dma_bytes_in == 256);
}

// 2) MATMUL timing: dims round UP to the 32x32 array.
//    m=1,n=32,k=32 -> mm=32,nn=32,kk=32 -> steady = 1*1*32 = 32, fill = 64 -> 96.
//    The load is 1024 floats = 4096 B, so it costs 400 + 4096/256 = 416 cycles,
//    NOT the 401 of a single 256 B beat. MATMUL waits on DEP_DMA_A and runs
//    416..512; HALT waits on everything, issues at 512, +1 -> 513.
static void test_matmul_cycles() {
    std::vector<float> hbm(4096, 0.0f);
    Simulator sim(cfg(), hbm);
    Program p;
    p.emit(load(0, 0, 1, 1024, 1024), "load A+B");
    Instr mm;
    mm.op = Op::MATMUL;
    mm.arg[0] = 0;                 // A at SPM 0
    mm.arg[1] = 32 * 4;            // B after it
    mm.arg[2] = 4096 * 4;          // C somewhere clear
    mm.arg[3] = 1;
    mm.arg[4] = 32;
    mm.arg[5] = 32;
    mm.dep_mask = DEP_DMA_A;
    p.emit(mm, "matmul 1x32x32");
    p.emit(halt(), "halt");
    CHECK(sim.run(p) == SimStatus::OK);
    std::printf("  [mxu] 1x32x32 matmul -> total %lld cycles (mxu busy %lld)\n",
                static_cast<long long>(sim.stats().cycles),
                static_cast<long long>(sim.stats().mxu_busy_cycles));
    CHECK(sim.stats().mxu_busy_cycles == 96);
    CHECK(sim.stats().cycles == 513);
    CHECK(sim.stats().mac_ops == 1 * 32 * 32);
}

// 3) THE PLANTED FLAW, measured. Two loads into different banks followed by a
//    consumer of bank A. Each load is 256 B = 400 + 1 = 401 cycles, and they
//    serialise on the single DMA engine (a STRUCTURAL hazard, not a dependency):
//    load A occupies 0..401, load B 401..802.
//      * With FINE dependencies the consumer waits only for load A, which
//        landed at 401. The front end issues one instruction per cycle and load
//        B did not issue until 401 itself, so the MATMUL issues at 402.
//      * With the COARSE mask it also waits for load B and issues at 802 —
//        400 cycles late, for no reason at all. THAT is the cost of ISA stance 3
//        and it is asserted below against the instruction's issue cycle.
//    The END-TO-END cost is smaller (96 cycles, one MXU occupancy), because HALT
//    drains the bank-B load either way: delaying the MATMUL past that load only
//    exposes the MATMUL's own 96 cycles. Both numbers are asserted, because
//    quoting the 401 as if it were the run's cost would be exactly the kind of
//    unattributed stall accounting this simulator exists to prevent.
static void test_coarse_vs_fine_deps() {
    std::vector<float> hbm(8192, 1.0f);

    auto build = [&](uint16_t consumer_dep) {
        Program p;
        p.emit(load(0, 0, 1, 64, 64, DEP_NONE, 0), "load -> bank A");
        p.emit(load(1024, 8192, 1, 64, 64, DEP_NONE, kFlagBankB), "load -> bank B");
        Instr mm;
        mm.op = Op::MATMUL;
        mm.arg[0] = 0;
        mm.arg[1] = 0;
        mm.arg[2] = 65536;
        mm.arg[3] = 1;
        mm.arg[4] = 32;
        mm.arg[5] = 32;
        mm.dep_mask = consumer_dep;
        p.emit(mm, "consume bank A");
        p.emit(halt(), "halt");
        return p;
    };

    Simulator fine(cfg(), hbm);
    fine.set_trace(true);
    CHECK(fine.run(build(DEP_DMA_A)) == SimStatus::OK);
    Simulator coarse(cfg(), hbm);
    coarse.set_trace(true);
    CHECK(coarse.run(build(DEP_DMA_IN)) == SimStatus::OK);

    // Instruction 2 is the MATMUL in both programs.
    int64_t f_issue = fine.events()[2].issue_cycle, c_issue = coarse.events()[2].issue_cycle;
    int64_t f = fine.stats().cycles, c = coarse.stats().cycles;
    std::printf("  [scoreboard] MATMUL issues at %lld (fine) vs %lld (coarse): the coarse mask "
                "delays it %lld cycles = one full DMA\n",
                static_cast<long long>(f_issue), static_cast<long long>(c_issue),
                static_cast<long long>(c_issue - f_issue));
    std::printf("  [scoreboard] end to end %lld vs %lld cycles (+%lld: HALT drains bank B "
                "regardless, so only the MXU's own 96 cycles are exposed)\n",
                static_cast<long long>(f), static_cast<long long>(c),
                static_cast<long long>(c - f));
    CHECK(f_issue == 402);
    CHECK(c_issue == 802);
    CHECK(c - f == 96);
    // And the coarse run must attribute those cycles to a DEPENDENCY stall, not
    // a structural one — the whole point of splitting the counters.
    CHECK(coarse.stats().stall_dep_dma > fine.stats().stall_dep_dma);
}

// 4) Functional correctness of the vector ops, against the same scalar formulas
//    the CPU path uses. Value-exactness is what lets the compiler be tested
//    against the M1 oracle instead of against a tolerance.
static void test_vector_ops_values() {
    std::vector<float> hbm(256, 0.0f);
    for (int i = 0; i < 64; ++i) hbm[static_cast<size_t>(i)] = static_cast<float>(i % 7) - 3.0f;
    for (int i = 0; i < 64; ++i) hbm[static_cast<size_t>(64 + i)] = 1.0f + 0.01f * static_cast<float>(i);

    Simulator sim(cfg(), hbm);
    Program p;
    p.emit(load(0, 0, 1, 64, 64), "x");
    p.emit(load(64, 256, 1, 64, 64), "gamma");

    Instr rn;
    rn.op = Op::V_RMSNORM;
    rn.arg[0] = 0;      // src
    rn.arg[1] = 256;    // weight
    rn.arg[2] = 512;    // dst
    rn.arg[3] = 64;
    rn.imm_f = 1e-6f;
    rn.dep_mask = DEP_DMA_IN;
    p.emit(rn, "rmsnorm");
    p.emit(halt(), "halt");
    CHECK(sim.run(p) == SimStatus::OK);

    // Same computation, same order (double accumulation), on the host.
    double sumsq = 0.0;
    for (int i = 0; i < 64; ++i) sumsq += static_cast<double>(hbm[static_cast<size_t>(i)]) *
                                          hbm[static_cast<size_t>(i)];
    float inv = static_cast<float>(1.0 / std::sqrt(sumsq / 64.0 + 1e-6f));
    for (int i = 0; i < 64; ++i) {
        float want = hbm[static_cast<size_t>(i)] * inv * hbm[static_cast<size_t>(64 + i)];
        CHECK(sim.spm()[static_cast<size_t>(128 + i)] == want);  // 512 bytes / 4
    }
}

// 5) MATMUL in both B orientations must agree with the obvious host loops, bit
//    for bit. kFlagBIsKN is what lets attention do Q.K^T then scores.V without
//    materialising a transpose (see isa.h).
static void test_matmul_orientations() {
    const int M = 2, N = 3, K = 4;
    std::vector<float> hbm(64, 0.0f);
    for (int i = 0; i < M * K; ++i) hbm[static_cast<size_t>(i)] = 0.5f * static_cast<float>(i + 1);
    for (int i = 0; i < N * K; ++i)
        hbm[static_cast<size_t>(16 + i)] = -0.25f * static_cast<float>(i + 2);

    auto run = [&](bool b_is_kn) {
        Simulator sim(cfg(), hbm);
        Program p;
        p.emit(load(0, 0, 1, M * K, M * K), "A");
        p.emit(load(16, 256, 1, N * K, N * K), "B");
        Instr mm;
        mm.op = Op::MATMUL;
        mm.flags = b_is_kn ? kFlagBIsKN : 0;
        mm.arg[0] = 0;
        mm.arg[1] = 256;
        mm.arg[2] = 512;
        mm.arg[3] = M;
        mm.arg[4] = N;
        mm.arg[5] = K;
        mm.dep_mask = DEP_DMA_IN;
        p.emit(mm, "matmul");
        p.emit(halt(), "halt");
        CHECK(sim.run(p) == SimStatus::OK);
        std::vector<float> out(M * N);
        for (int i = 0; i < M * N; ++i) out[static_cast<size_t>(i)] =
            sim.spm()[static_cast<size_t>(128 + i)];
        return out;
    };

    const float* A = hbm.data();
    const float* B = hbm.data() + 16;
    auto nk = run(false);
    auto kn = run(true);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double s1 = 0.0, s2 = 0.0;
            for (int p = 0; p < K; ++p) {
                s1 += static_cast<double>(A[i * K + p]) * B[j * K + p];
                s2 += static_cast<double>(A[i * K + p]) * B[p * N + j];
            }
            CHECK(nk[static_cast<size_t>(i * N + j)] == static_cast<float>(s1));
            CHECK(kn[static_cast<size_t>(i * N + j)] == static_cast<float>(s2));
        }
}

// 6) A bad SPM address must be caught, not scribbled over. Software-managed
//    memory means the compiler owns placement, so this is the error class a
//    compiler bug produces and it has to be loud.
static void test_bad_address_is_caught() {
    std::vector<float> hbm(64, 0.0f);
    Simulator sim(cfg(), hbm);
    Program p;
    p.emit(load(0, 0xFFFFFF00u, 1, 64, 64), "load past the end of SPM");
    p.emit(halt(), "halt");
    CHECK(sim.run(p) == SimStatus::BAD_PROGRAM);
    CHECK(!sim.error().empty());
    std::printf("  [guard] %s\n", sim.error().c_str());
}

// 7) A DMA that never completes must hit the deadline rather than hang. M4's
//    runtime turns this into a per-request failure; the mechanism is here.
static void test_stuck_dma_hits_deadline() {
    std::vector<float> hbm(256, 0.0f);
    Simulator sim(cfg(), hbm);
    FaultConfig f;
    f.stuck_dma_at_instr = 0;
    f.deadline_cycles = 100000;
    sim.set_faults(f);

    Program p;
    p.emit(load(0, 0, 1, 64, 64), "load that never lands");
    Instr mm;
    mm.op = Op::MATMUL;
    mm.arg[0] = 0; mm.arg[1] = 0; mm.arg[2] = 4096;
    mm.arg[3] = 1; mm.arg[4] = 32; mm.arg[5] = 32;
    mm.dep_mask = DEP_DMA_A;
    p.emit(mm, "consumer");
    p.emit(halt(), "halt");
    CHECK(sim.run(p) == SimStatus::DEADLINE_EXCEEDED);
    std::printf("  [fault] stuck DMA -> %s at cycle %lld\n", sim.error().c_str(),
                static_cast<long long>(sim.stats().cycles));
}

int main() {
    test_dma_cycles();
    test_matmul_cycles();
    test_coarse_vs_fine_deps();
    test_vector_ops_values();
    test_matmul_orientations();
    test_bad_address_is_caught();
    test_stuck_dma_hits_deadline();
    return test_summary();
}
