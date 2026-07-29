#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "isa.h"

// ============================================================================
// The T1 simulator: CYCLE-APPROXIMATE, VALUE-EXACT.
// ============================================================================
// Two layers that deliberately do not talk to each other:
//
//   FUNCTIONAL. Every instruction computes its real result, immediately, on a
//   flat float scratchpad and a flat float "HBM". MATMUL is the same triple loop
//   as ops::gemm_ref. Because the op ORDER is identical to the CPU reference,
//   the floats come out BIT-IDENTICAL — which is what makes
//   test/sim_isa.cpp able to assert == against the CPU path and what makes the
//   M1 oracle a valid gate on compiled code. A simulator that is merely close
//   cannot tell you whether your compiler's tiling is correct.
//
//   TIMING. Each unit (MXU, VPU, DMA) has a busy-until cycle. Instructions issue
//   IN ORDER; an instruction whose dep_mask names a busy class stalls the front
//   end and the stall is ATTRIBUTED to that class. Nothing about the timing
//   model feeds back into the values.
//
// The stall attribution is the raw material for M7's profiler. It is recorded
// from day one even though nothing visualises it yet, because retrofitting
// per-stall provenance into a simulator later means re-deriving it from a log.
// ============================================================================

namespace accel {

// Where the cycles went. Everything here is per-run and resettable.
struct SimStats {
    int64_t cycles = 0;
    int64_t instructions = 0;

    int64_t mxu_busy_cycles = 0;
    int64_t vpu_busy_cycles = 0;
    int64_t dma_busy_cycles = 0;

    // Front-end stall cycles, split by WHY.
    //
    // `stall_dep_*` is a real data dependency: we are waiting for a result.
    // `stall_struct_*` is a structural hazard: the result was ready but the unit
    // was still busy with earlier work. Keeping them apart is what makes the
    // numbers actionable — a dependency stall says "schedule earlier", a
    // structural stall says "you are bandwidth-bound, scheduling will not help".
    // Collapsing both into one "DMA stall" figure is how you spend a week
    // software-pipelining a loop that was already at the memory roof.
    int64_t stall_dep_dma = 0;
    int64_t stall_dep_mxu = 0;
    int64_t stall_dep_vpu = 0;
    int64_t stall_struct_dma = 0;
    int64_t stall_struct_mxu = 0;
    int64_t stall_struct_vpu = 0;

    int64_t dma_bytes_in = 0;
    int64_t dma_bytes_out = 0;
    int64_t mac_ops = 0;  // multiply-accumulates actually issued

    // MAC utilisation: the fraction of the run in which the systolic array was
    // doing arithmetic. The single number that says whether a schedule is any
    // good.
    double mac_utilisation() const {
        return cycles ? static_cast<double>(mxu_busy_cycles) / static_cast<double>(cycles) : 0.0;
    }
    int64_t total_stall() const {
        return stall_dep_dma + stall_dep_mxu + stall_dep_vpu + stall_struct_dma +
               stall_struct_mxu + stall_struct_vpu;
    }
    double dma_stall_fraction() const {
        return cycles ? static_cast<double>(stall_dep_dma + stall_struct_dma) /
                            static_cast<double>(cycles)
                      : 0.0;
    }
    std::string summary(const T1Config& cfg) const;
};

// Fault injection (used in earnest by M4, wired in now so the simulator does not
// need reopening later).
struct FaultConfig {
    // Instruction index whose DMA never completes. -1 disables.
    int64_t stuck_dma_at_instr = -1;
    // Simulated cycles after which the run aborts with a deadline error.
    int64_t deadline_cycles = -1;
};

enum class SimStatus { OK, DEADLINE_EXCEEDED, BAD_PROGRAM };

class Simulator {
public:
    Simulator(const T1Config& cfg, std::vector<float> hbm);

    // Runs from instruction 0 to HALT. Deterministic and single-threaded.
    SimStatus run(const Program& p);

    const SimStats& stats() const { return stats_; }
    void reset_stats() { stats_ = SimStats{}; }
    const T1Config& config() const { return cfg_; }

    std::vector<float>& hbm() { return hbm_; }
    const std::vector<float>& hbm() const { return hbm_; }
    const std::vector<float>& spm() const { return spm_; }

    void set_faults(const FaultConfig& f) { faults_ = f; }
    const std::string& error() const { return error_; }

    // Per-instruction timeline, kept only when enabled. This is what M7's
    // Chrome-trace emitter will serialise; recording it is cheap and having it
    // from the start means the trace and the cycle counts can never disagree.
    struct Event {
        int64_t index;
        Op op;
        int64_t issue_cycle;
        int64_t end_cycle;
        int64_t stall_cycles;
        uint16_t stall_class;  // which Dep bits were responsible
    };
    void set_trace(bool on) { trace_ = on; }
    const std::vector<Event>& events() const { return events_; }

private:
    // Unit models. Returns the number of cycles the unit is occupied.
    int64_t matmul_cycles(int64_t m, int64_t n, int64_t k) const;
    int64_t dma_cycles(int64_t bytes) const;
    int64_t vpu_cycles(int64_t elems, int64_t fixed) const;

    bool exec_functional(const Instr& in, int64_t index);
    float* spm_at(uint32_t byte_off, int64_t elems);
    const float* spm_at_c(uint32_t byte_off, int64_t elems) const;

    T1Config cfg_;
    std::vector<float> hbm_;
    std::vector<float> spm_;
    SimStats stats_;
    FaultConfig faults_;
    std::string error_;
    bool trace_ = false;
    std::vector<Event> events_;

    // When each physical unit next goes idle (structural hazards).
    int64_t unit_free_[4] = {0, 0, 0, 0};
    // When the work producing each dependency class has completed.
    int64_t class_ready_[kNumDepClasses] = {0, 0, 0, 0, 0};
};

}  // namespace accel
