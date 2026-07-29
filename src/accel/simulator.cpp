#include "simulator.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>

namespace accel {

const char* op_name(Op op) {
    switch (op) {
        case Op::NOP: return "NOP";
        case Op::DMA_LOAD: return "DMA_LOAD";
        case Op::DMA_STORE: return "DMA_STORE";
        case Op::MATMUL: return "MATMUL";
        case Op::V_RMSNORM: return "V_RMSNORM";
        case Op::V_SOFTMAX: return "V_SOFTMAX";
        case Op::V_ROPE: return "V_ROPE";
        case Op::V_SILU_MUL: return "V_SILU_MUL";
        case Op::V_ADD: return "V_ADD";
        case Op::V_COPY: return "V_COPY";
        case Op::V_SCALE: return "V_SCALE";
        case Op::BARRIER: return "BARRIER";
        case Op::HALT: return "HALT";
    }
    return "?";
}

std::string disassemble(const Program& p, size_t max_lines) {
    std::ostringstream os;
    size_t n = max_lines ? std::min(max_lines, p.code.size()) : p.code.size();
    for (size_t i = 0; i < n; ++i) {
        const Instr& in = p.code[i];
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%5zu  %-11s dep=%04x  %8u %8u %8u %8u %8u %8u", i,
                      op_name(in.op), in.dep_mask, in.arg[0], in.arg[1], in.arg[2], in.arg[3],
                      in.arg[4], in.arg[5]);
        os << buf;
        if (i < p.notes.size() && !p.notes[i].empty()) os << "   ; " << p.notes[i];
        os << "\n";
    }
    if (max_lines && p.code.size() > max_lines)
        os << "  ... " << (p.code.size() - max_lines) << " more\n";
    return os.str();
}

std::string SimStats::summary(const T1Config& cfg) const {
    std::ostringstream os;
    double us = static_cast<double>(cycles) / (cfg.clock_ghz * 1000.0);
    os << "  cycles          " << cycles << "  (" << us << " us @ " << cfg.clock_ghz << " GHz)\n"
       << "  instructions    " << instructions << "\n"
       << "  MAC ops         " << mac_ops << "\n"
       << "  MXU busy        " << mxu_busy_cycles << "  (" << (100.0 * mac_utilisation())
       << "% utilisation)\n"
       << "  VPU busy        " << vpu_busy_cycles << "\n"
       << "  DMA busy        " << dma_busy_cycles << "  (" << dma_bytes_in << " B in, "
       << dma_bytes_out << " B out)\n"
       << "  stall (dep)     dma " << stall_dep_dma << "  mxu " << stall_dep_mxu << "  vpu "
       << stall_dep_vpu << "\n"
       << "  stall (struct)  dma " << stall_struct_dma << "  mxu " << stall_struct_mxu
       << "  vpu " << stall_struct_vpu << "\n"
       << "  DMA stall share " << (100.0 * dma_stall_fraction()) << "% of run\n";
    return os.str();
}

// ---------------------------------------------------------------------------

Simulator::Simulator(const T1Config& cfg, std::vector<float> hbm)
    : cfg_(cfg), hbm_(std::move(hbm)) {
    spm_.assign(static_cast<size_t>(cfg_.spm_bytes / 4), 0.0f);
}

// TIMING MODELS. Deliberately simple closed forms — the goal is to reproduce the
// SHAPE of the machine's behaviour (fixed costs, pipeline fill, bandwidth
// limits), not to model a pipeline that does not exist yet. Every formula here
// is restated in docs/03-isa-spec.md; if you change one, change both.

int64_t Simulator::matmul_cycles(int64_t m, int64_t n, int64_t k) const {
    // A 32x32 systolic array retires one 32x32 * 32x1 column per cycle once
    // full, and takes (rows + cols) cycles to fill and drain. Tiles are rounded
    // UP to the array dimension, because a 3x3 matmul occupies the array just as
    // long as a 32x32 one does — which is exactly why the compiler's tiling pass
    // must not emit skinny tiles, and why leaving that cost visible matters.
    const int64_t d = cfg_.mxu_dim;
    auto up = [d](int64_t v) { return ((v + d - 1) / d) * d; };
    int64_t mm = up(m), nn = up(n), kk = up(k);
    int64_t steady = (mm / d) * (nn / d) * kk;  // one k-step per cycle per tile pair
    int64_t fill = d + d;                       // pipeline fill + drain
    return steady + fill;
}

int64_t Simulator::dma_cycles(int64_t bytes) const {
    return cfg_.dma_latency_cycles + (bytes + cfg_.dma_bytes_per_cycle - 1) /
                                         cfg_.dma_bytes_per_cycle;
}

int64_t Simulator::vpu_cycles(int64_t elems, int64_t fixed) const {
    return fixed + (elems + cfg_.vpu_lanes - 1) / cfg_.vpu_lanes;
}

// Why an SPM operand was rejected. Under stance 2 the compiler owns placement,
// so this error class IS how a compiler bug surfaces — and "out of range" when
// the real problem is a misaligned offset sends you hunting the wrong bug. The
// two failures have completely different causes (a bad size calculation vs a
// byte/float unit mix-up), so they get different messages.
const char* Simulator::spm_fault(uint32_t byte_off, int64_t elems) const {
    if (byte_off % 4 != 0) return "SPM address is not 4-byte aligned";
    if (elems < 0) return "SPM operand has negative length";
    return "SPM address out of range";
}

float* Simulator::spm_at(uint32_t byte_off, int64_t elems) {
    if (byte_off % 4 != 0) return nullptr;
    int64_t idx = byte_off / 4;
    if (idx < 0 || idx + elems > static_cast<int64_t>(spm_.size())) return nullptr;
    return spm_.data() + idx;
}

const float* Simulator::spm_at_c(uint32_t byte_off, int64_t elems) const {
    if (byte_off % 4 != 0) return nullptr;
    int64_t idx = byte_off / 4;
    if (idx < 0 || idx + elems > static_cast<int64_t>(spm_.size())) return nullptr;
    return spm_.data() + idx;
}

// The functional layer. Returns false on a malformed instruction — which, given
// stance 2 (software-managed memory), is how a compiler bug in SPM allocation
// surfaces: as an out-of-range address, immediately, rather than as silently
// corrupted activations three layers later.
bool Simulator::exec_functional(const Instr& in, int64_t index) {
    auto fail = [&](const char* why) {
        std::ostringstream os;
        os << "instr " << index << " (" << op_name(in.op) << "): " << why;
        error_ = os.str();
        return false;
    };

    switch (in.op) {
        case Op::NOP:
        case Op::BARRIER:
        case Op::HALT:
            return true;

        case Op::DMA_LOAD:
        case Op::DMA_STORE: {
            // 2-D strided: `rows` runs of `cols` floats, HBM rows separated by
            // `row_stride` floats, SPM side packed. Strides are what let the
            // compiler fetch a tile of a big matrix without the host repacking
            // anything — the accelerator equivalent of a gather that costs
            // nothing extra because the DMA engine walks descriptors anyway.
            uint64_t hbm_off = in.hbm_addr(0);
            uint32_t spm_off = in.arg[2];
            int64_t rows = in.arg[3], cols = in.arg[4], row_stride = in.arg[5];
            if (rows <= 0 || cols <= 0) return fail("non-positive rows/cols");
            float* s = spm_at(spm_off, rows * cols);
            if (!s) return fail(spm_fault(spm_off, rows * cols));
            if (hbm_off + static_cast<uint64_t>((rows - 1) * row_stride + cols) > hbm_.size())
                return fail("HBM address out of range");

            if (in.op == Op::DMA_LOAD) {
                if (faults_.stuck_dma_at_instr == index) return true;  // silently never lands
                for (int64_t r = 0; r < rows; ++r)
                    std::copy_n(hbm_.data() + hbm_off + r * row_stride, cols, s + r * cols);
            } else {
                for (int64_t r = 0; r < rows; ++r)
                    std::copy_n(s + r * cols, cols, hbm_.data() + hbm_off + r * row_stride);
            }
            return true;
        }

        case Op::MATMUL: {
            // C[m,n] (+)= A[m,k] . B[n,k]^T. B is row-major [n,k] — the same
            // weight-stationary layout HuggingFace stores nn.Linear.weight in,
            // so no transpose ever happens anywhere in this stack.
            uint32_t a_off = in.arg[0], b_off = in.arg[1], c_off = in.arg[2];
            int64_t m = in.arg[3], n = in.arg[4], k = in.arg[5];
            if (m <= 0 || n <= 0 || k <= 0) return fail("non-positive matmul dims");
            const float* A = spm_at_c(a_off, m * k);
            const float* B = spm_at_c(b_off, n * k);  // same element count either way
            float* C = spm_at(c_off, m * n);
            if (!A) return fail(spm_fault(a_off, m * k));
            if (!B) return fail(spm_fault(b_off, n * k));
            if (!C) return fail(spm_fault(c_off, m * n));

            const bool acc = (in.flags & kFlagAccumulate) != 0;
            const bool b_kn = (in.flags & kFlagBIsKN) != 0;
            for (int64_t i = 0; i < m; ++i)
                for (int64_t j = 0; j < n; ++j) {
                    // Double accumulation, in the same order as ops::linear, so
                    // the simulated result is bit-identical to the CPU path.
                    // This is the whole reason the compiler can be tested
                    // against the M1 oracle rather than against a tolerance.
                    // The accumulator is seeded from C when kFlagAccumulate is
                    // set, and it is seeded IN THE WIDE FORMAT — see isa.h. So a
                    // k-split matmul rounds once, like the unsplit one, and a
                    // bias-seeded C reproduces ops::linear exactly.
                    double s = acc ? static_cast<double>(C[i * n + j]) : 0.0;
                    if (b_kn)
                        for (int64_t p = 0; p < k; ++p)
                            s += static_cast<double>(A[i * k + p]) * B[p * n + j];
                    else
                        for (int64_t p = 0; p < k; ++p)
                            s += static_cast<double>(A[i * k + p]) * B[j * k + p];
                    C[i * n + j] = static_cast<float>(s);
                }
            stats_.mac_ops += m * n * k;
            return true;
        }

        case Op::V_RMSNORM: {
            uint32_t src = in.arg[0], w = in.arg[1], dst = in.arg[2];
            int64_t len = in.arg[3];
            const float* x = spm_at_c(src, len);
            const float* g = spm_at_c(w, len);
            float* y = spm_at(dst, len);
            if (!x) return fail(spm_fault(src, len));
            if (!g) return fail(spm_fault(w, len));
            if (!y) return fail(spm_fault(dst, len));
            double sumsq = 0.0;  // double, exactly as ops::rmsnorm does
            for (int64_t i = 0; i < len; ++i) sumsq += static_cast<double>(x[i]) * x[i];
            float inv = static_cast<float>(
                1.0 / std::sqrt(sumsq / static_cast<double>(len) + in.imm_f));
            for (int64_t i = 0; i < len; ++i) y[i] = x[i] * inv * g[i];
            return true;
        }

        case Op::V_SOFTMAX: {
            uint32_t src = in.arg[0];
            int64_t len = in.arg[1];
            float* x = spm_at(src, len);
            if (!x) return fail(spm_fault(src, len));
            if (len <= 0) return true;
            float mx = x[0];
            for (int64_t i = 1; i < len; ++i) mx = std::max(mx, x[i]);
            double sum = 0.0;
            for (int64_t i = 0; i < len; ++i) {
                float e = std::exp(x[i] - mx);
                x[i] = e;
                sum += e;
            }
            float inv = static_cast<float>(1.0 / sum);
            for (int64_t i = 0; i < len; ++i) x[i] *= inv;
            return true;
        }

        case Op::V_ROPE: {
            uint32_t src = in.arg[0];
            int64_t head_dim = in.arg[1], pos = in.arg[2], heads = in.arg[3];
            float* x = spm_at(src, head_dim * heads);
            if (!x) return fail(spm_fault(src, head_dim * heads));
            for (int64_t h = 0; h < heads; ++h) {
                float* v = x + h * head_dim;
                int64_t half = head_dim / 2;
                for (int64_t i = 0; i < half; ++i) {
                    // Split-half convention (Qwen), not interleaved. Getting
                    // this wrong produces fluent-looking garbage, which is what
                    // the bisection ladder exists to catch.
                    double freq = 1.0 / std::pow(static_cast<double>(in.imm_f),
                                                 (2.0 * static_cast<double>(i)) /
                                                     static_cast<double>(head_dim));
                    double angle = static_cast<double>(pos) * freq;
                    float c = static_cast<float>(std::cos(angle));
                    float s = static_cast<float>(std::sin(angle));
                    float a = v[i], b = v[i + half];
                    v[i] = a * c - b * s;
                    v[i + half] = a * s + b * c;
                }
            }
            return true;
        }

        case Op::V_SILU_MUL: {
            uint32_t g = in.arg[0], u = in.arg[1], d = in.arg[2];
            int64_t len = in.arg[3];
            const float* gp = spm_at_c(g, len);
            const float* up = spm_at_c(u, len);
            float* dp = spm_at(d, len);
            if (!gp) return fail(spm_fault(g, len));
            if (!up) return fail(spm_fault(u, len));
            if (!dp) return fail(spm_fault(d, len));
            for (int64_t i = 0; i < len; ++i) {
                float x = gp[i];
                dp[i] = (x / (1.0f + std::exp(-x))) * up[i];
            }
            return true;
        }

        case Op::V_ADD:
        case Op::V_COPY:
        case Op::V_SCALE: {
            uint32_t s = in.arg[0], d = in.arg[1];
            int64_t len = in.arg[2];
            const float* sp = spm_at_c(s, len);
            float* dp = spm_at(d, len);
            if (!sp) return fail(spm_fault(s, len));
            if (!dp) return fail(spm_fault(d, len));
            for (int64_t i = 0; i < len; ++i) {
                if (in.op == Op::V_ADD) dp[i] += sp[i];
                else if (in.op == Op::V_COPY) dp[i] = sp[i];
                else dp[i] = sp[i] * in.imm_f;
            }
            return true;
        }
    }
    return fail("unknown opcode");
}

SimStatus Simulator::run(const Program& p) {
    error_.clear();
    events_.clear();
    for (int64_t& b : unit_free_) b = 0;
    for (int64_t& c : class_ready_) c = 0;
    int64_t now = 0;

    for (size_t i = 0; i < p.code.size(); ++i) {
        const Instr& in = p.code[i];
        const int unit = occupied_unit(in);

        // ---- issue -------------------------------------------------------
        // In-order front end. An instruction issues at the latest of:
        //   * now (program order),
        //   * every dependency class named in dep_mask being satisfied,
        //   * its own physical unit being free (structural hazard).
        // The stall is charged to whichever of those was the binding constraint,
        // and DEPENDENCE and STRUCTURE are counted separately — see the comment
        // on SimStats. Computing the attribution here rather than reconstructing
        // it from a log later is what makes M7's profiler nearly free.
        int64_t dep_ready = now;
        uint16_t blame_class = 0;
        for (int bit = 0; bit < kNumDepClasses; ++bit) {
            if (!(in.dep_mask & (1u << bit))) continue;
            if (class_ready_[bit] > dep_ready) {
                dep_ready = class_ready_[bit];
                blame_class = static_cast<uint16_t>(1u << bit);
            }
        }
        int64_t struct_ready = (unit >= 0) ? unit_free_[unit] : now;

        int64_t ready = std::max({now, dep_ready, struct_ready});
        int64_t dep_stall = std::max<int64_t>(0, std::min(dep_ready, ready) - now);
        int64_t struct_stall = std::max<int64_t>(0, ready - std::max(now, dep_ready));

        if (dep_stall > 0) {
            switch (blame_class) {
                case DEP_DMA_A:
                case DEP_DMA_B:
                case DEP_DMA_OUT: stats_.stall_dep_dma += dep_stall; break;
                case DEP_MXU: stats_.stall_dep_mxu += dep_stall; break;
                case DEP_VPU: stats_.stall_dep_vpu += dep_stall; break;
                default: break;
            }
        }
        if (struct_stall > 0) {
            switch (unit) {
                case UNIT_DMA_IN:
                case UNIT_DMA_OUT: stats_.stall_struct_dma += struct_stall; break;
                case UNIT_MXU: stats_.stall_struct_mxu += struct_stall; break;
                case UNIT_VPU: stats_.stall_struct_vpu += struct_stall; break;
                default: break;
            }
        }
        uint16_t blame = dep_stall >= struct_stall ? blame_class : 0;
        int64_t stall = ready - now;
        now = ready;

        if (faults_.deadline_cycles >= 0 && now > faults_.deadline_cycles) {
            error_ = "deadline exceeded";
            stats_.cycles = now;
            return SimStatus::DEADLINE_EXCEEDED;
        }

        if (!exec_functional(in, static_cast<int64_t>(i))) {
            stats_.cycles = now;
            return SimStatus::BAD_PROGRAM;
        }
        ++stats_.instructions;

        // ---- occupy the unit ---------------------------------------------
        int64_t dur = 1;
        switch (in.op) {
            case Op::DMA_LOAD: {
                int64_t bytes = static_cast<int64_t>(in.arg[3]) * in.arg[4] * 4;
                dur = dma_cycles(bytes);
                stats_.dma_bytes_in += bytes;
                stats_.dma_busy_cycles += dur;
                // A stuck DMA occupies the engine forever, so everything
                // downstream of it hangs. M4's runtime is what notices, via the
                // deadline; the point of injecting it here is that the failure
                // looks exactly like a real one rather than like an exception.
                if (faults_.stuck_dma_at_instr == static_cast<int64_t>(i))
                    dur = std::numeric_limits<int64_t>::max() / 4;
                break;
            }
            case Op::DMA_STORE: {
                int64_t bytes = static_cast<int64_t>(in.arg[3]) * in.arg[4] * 4;
                dur = dma_cycles(bytes);
                stats_.dma_bytes_out += bytes;
                stats_.dma_busy_cycles += dur;
                break;
            }
            case Op::MATMUL:
                dur = matmul_cycles(in.arg[3], in.arg[4], in.arg[5]);
                stats_.mxu_busy_cycles += dur;
                break;
            case Op::V_RMSNORM:
                dur = vpu_cycles(in.arg[3], 8);  // + reduction latency
                stats_.vpu_busy_cycles += dur;
                break;
            case Op::V_SOFTMAX:
                dur = vpu_cycles(static_cast<int64_t>(in.arg[1]) * 3, 8);  // max, exp, normalise
                stats_.vpu_busy_cycles += dur;
                break;
            case Op::V_ROPE:
                dur = vpu_cycles(static_cast<int64_t>(in.arg[1]) * in.arg[3], 4);
                stats_.vpu_busy_cycles += dur;
                break;
            case Op::V_SILU_MUL:
                dur = vpu_cycles(in.arg[3], 4);
                stats_.vpu_busy_cycles += dur;
                break;
            case Op::V_ADD:
            case Op::V_COPY:
            case Op::V_SCALE:
                dur = vpu_cycles(in.arg[2], 1);
                stats_.vpu_busy_cycles += dur;
                break;
            case Op::BARRIER:
            case Op::NOP:
                dur = 1;
                break;
            case Op::HALT:
                stats_.cycles = now + 1;
                if (trace_)
                    events_.push_back({static_cast<int64_t>(i), in.op, now, now + 1, stall, blame});
                return SimStatus::OK;
        }

        const int64_t end = now + dur;
        if (unit >= 0) unit_free_[unit] = end;
        uint16_t prod = produced_class(in);
        for (int bit = 0; bit < kNumDepClasses; ++bit)
            if (prod & (1u << bit)) class_ready_[bit] = std::max(class_ready_[bit], end);

        if (trace_) events_.push_back({static_cast<int64_t>(i), in.op, now, end, stall, blame});

        // The front end issues one instruction per cycle and moves on; only a
        // dep_mask or a busy unit makes it wait. That is what allows a DMA to
        // overlap with the MATMUL that follows it — the point of double
        // buffering, and the reason the dep classes have to be fine enough to
        // express "the OTHER bank's load, not this one".
        now += 1;
    }

    // Fell off the end without a HALT: drain outstanding units so the reported
    // cycle count still means something.
    for (int64_t b : unit_free_) now = std::max(now, b);
    stats_.cycles = now;
    error_ = "program has no HALT";
    return SimStatus::BAD_PROGRAM;
}

}  // namespace accel
