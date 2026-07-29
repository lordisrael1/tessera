# SOHU-SIM: The Project Bible

**A complete C++ inference stack for an accelerator that doesn't exist.**

Working title: `tessera`.
Target: Etched-class inference infrastructure roles. Secondary market: Modal, Baseten, Together, Fireworks, vLLM-adjacent teams, ML-systems groups at labs.

> **New session? Read this file for the *plan*, then read [PROGRESS.md](PROGRESS.md)
> for the *state*.** This file changes rarely; PROGRESS.md changes every session.
> Rule 4 below (write the doc the week the thing happens) is why `docs/` exists.

---

## 0. Charter

### 0.1 One-sentence pitch

> I designed a fixed-function transformer accelerator on paper, wrote a cycle-approximate simulator for it in C++, and then built the entire software stack above it — compiler, runtime, hand-written collectives, MoE routing, and profiler — until a real 0.5B-parameter model ran end to end and I could tell you exactly which cycle it stalled on and why.

### 0.2 Thesis of the writeup

Inference decode is memory-bandwidth-starved, not compute-starved. You will *derive* this from your own measurements on your own machine (roofline, Milestone 2), then *design around it* in silicon (the ISA's explicit scratchpad + DMA, Milestone 3), then *fight it again* at the cluster level (collectives, Milestone 5). One thesis, three altitudes. That arc is the whole story.

### 0.3 JD mapping — every bullet gets an artifact

| Etched JD bullet | Your artifact | Milestone |
|---|---|---|
| "Support porting state-of-the-art models to our architecture" | Qwen2.5-0.5B lowered to a custom ISA via your own compiler pass | M3 |
| "Build programming abstractions and testing capabilities to rapidly iterate on model porting" | Logit-parity oracle wired into CTest; op-level golden tests | M1 |
| "Build, enhance, and scale our runtime … state management, robust error handling" | KV-cache manager, continuous batching, DMA fault injection | M4 |
| "multi-node inference, intra-node execution" | Tensor parallelism across processes: shm → TCP → io_uring | M5 |
| "Optimize routing and communication layers using our collectives" | Hand-written all-reduce / all-gather / reduce-scatter / all-to-all | M5, M6 |
| "performance profiling and debugging tools to identify bottlenecks" | Per-instruction Chrome-trace profiler + one documented bottleneck fix | M7 |
| "Proficiency in C++ or Rust" | The whole repo, sanitizer-clean C++20 | all |
| "accelerator architectures" | You designed one | M3 |
| "high-speed interconnects" | Your collectives layer is a software model of exactly this problem | M5 |
| Nice-to-have: "kernel-level and user-space networking stacks" | TCP sockets vs io_uring implementations of the same collective | M5 |
| Nice-to-have: "MoE" | 8-expert top-2 FFN with all-to-all dispatch, load-imbalance study | M6 |
| Nice-to-have: "extensive SIMD optimizations" | AVX2 INT8 GEMM, roofline-benchmarked vs llama.cpp | M2 |

Print this table. When a milestone tempts you to wander, check whether the wandering adds a row. If not, cut it.

### 0.4 Rules of the project

1. **No inference dependencies.** No BLAS, no Eigen, no llama.cpp code, no ONNX runtime. `<immintrin.h>`, the C++20 standard library, POSIX, and io_uring (via raw syscalls or liburing — liburing is acceptable, it's a syscall wrapper, not an inference crutch).
2. **Every milestone regresses against the M1 oracle.** If logits drift past tolerance, the milestone is not done.
3. **Sanitizer-clean, always.** ASan+UBSan build in CI from day one. TSan for M5.
4. **Write the doc entry the week the thing happens.** `docs/NN-topic.md`, committed with the code. The engineering log is half the artifact.
5. **Ship early, ship publicly.** M2's benchmark post goes out before M3 starts. Don't wait for the cathedral.

---

## 1. The machine (know your silicon)

HP EliteBook 845 G8 — AMD Ryzen 5 PRO 5650U (Cezanne, Zen 3).

| Property | Value | Consequence |
|---|---|---|
| Cores / threads | 6C / 12T | Thread pool of 6; test whether SMT helps (it usually won't for GEMM) |
| ISA extensions | AVX2, FMA3, **no AVX-512, no VNNI** | INT8 dot product via `maddubs` idiom; 256-bit vectors everywhere |
| L1d / L2 / L3 | 32 KB / 512 KB per core / 16 MB shared | Cache-blocking parameters for GEMM; L2 is your "scratchpad rehearsal" |
| Memory | 16 GB (15.3 usable) dual-channel DDR4-3200 | ~51.2 GB/s theoretical; expect ~35–42 GB/s measured (STREAM it) |
| TDP | 15–25 W mobile | Thermal throttling within seconds of sustained FMA. Benchmark best-of-N *and* sustained; report both |
| Disk free | ~32 GB | Model weights ≤ 1.1B params; `ccache`; build dir on a path you clean; `git gc` occasionally |
| OS | Windows 11 Pro → **WSL2 Ubuntu 24.04** | perf, hugepages, io_uring, pthread affinity all live here |

### 1.1 WSL2 setup checklist

```bash
wsl --install -d Ubuntu-24.04
# Inside WSL:
sudo apt update && sudo apt install -y \
  build-essential g++-13 clang-17 cmake ninja-build ccache \
  linux-tools-generic liburing-dev git python3-pip
pip3 install --user torch transformers safetensors numpy --index-url https://download.pytorch.org/whl/cpu
```

`.wslconfig` on the Windows side (`C:\Users\<you>\.wslconfig`):

```ini
[wsl2]
memory=12GB        # leave Windows 4 GB or it will swap and lie to your benchmarks
processors=12
swap=0             # swap poisons latency measurements; fail loudly instead
```

**perf inside WSL2:** the packaged `perf` often mismatches the WSL kernel. If `perf stat` complains, build perf from the WSL2 kernel source (github.com/microsoft/WSL2-Linux-Kernel, `tools/perf`). Hardware PMU counters work on recent WSL2; if `perf stat -e cycles` returns `<not supported>`, fall back to `perf stat -e task-clock` plus your own rdtsc harness — and note it in the docs. Do this check in week 1, not week 6.

Verify the flags reach the compiler:

```bash
g++ -march=znver3 -dM -E - </dev/null | grep -E "__AVX2__|__FMA__|__AVX512"
# want AVX2 and FMA defined, no AVX512 lines
```

---

## 2. Toolchain, repo, CI

### 2.1 Repo layout (final form — grows into this)

```
tessera/
  CMakeLists.txt
  cmake/            toolchain-znver3.cmake  sanitizers.cmake
  src/
    core/           tensor.h  arena.h  align.h  safetensors.{h,cpp}  threadpool.{h,cpp}
    tokenizer/      bpe.{h,cpp}                      # M1
    ops/            gemm_avx2.{h,cpp}  gemm_ref.cpp  quant.{h,cpp}
                    rmsnorm.cpp  rope.cpp  softmax.cpp  silu.cpp
    model/          config.{h,cpp}  qwen2.{h,cpp}  kv_cache.{h,cpp}
    accel/          isa.h  encode.{h,cpp}  simulator.{h,cpp}         # M3
                    lower.{h,cpp}  tiling.{h,cpp}  schedule.{h,cpp}
    runtime/        engine.{h,cpp}  batching.{h,cpp}  faults.{h,cpp} # M4
    comm/           collective.h  shm_ring.{h,cpp}  tcp.{h,cpp}      # M5
                    uring.{h,cpp}  topology.{h,cpp}
    moe/            router.{h,cpp}  expert_parallel.{h,cpp}          # M6
    prof/           trace.{h,cpp}  counters.h                        # M7
  tools/            run_infer.cpp  dump_logits.py  compare_logits.cpp
                    trace_view.md  stream_bench.cpp
  bench/            gemm_bench.cpp  roofline.cpp  decode_bench.cpp
  test/             logit_parity.cpp  gemm_vs_ref.cpp  op_golden.cpp
                    collective_fuzz.cpp  sim_isa.cpp
  docs/             00-charter.md  01-roofline.md  02-int8-saturation.md
                    03-isa-spec.md  04-compiler.md  05-runtime.md
                    06-collectives.md  07-moe.md  08-bottleneck-hunt.md
  .github/workflows/ci.yml
```

### 2.2 Root CMakeLists.txt (starting point)

```cmake
cmake_minimum_required(VERSION 3.25)
project(tessera CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_program(CCACHE ccache)
if(CCACHE)
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
endif()

add_compile_options(-Wall -Wextra -Wshadow -Wconversion -g)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -march=znver3 -mtune=znver3 -fno-math-errno)
endif()

option(TESSERA_SANITIZE "ASan+UBSan" OFF)
if(TESSERA_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -O1)
  add_link_options(-fsanitize=address,undefined)
endif()

option(TESSERA_TSAN "ThreadSanitizer" OFF)
if(TESSERA_TSAN)
  add_compile_options(-fsanitize=thread -O1 -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

add_subdirectory(src)
add_subdirectory(bench)
enable_testing()
add_subdirectory(test)
```

Three build trees, always warm:

```bash
cmake -B build/rel  -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake -B build/asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTESSERA_SANITIZE=ON
cmake -B build/tsan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTESSERA_TSAN=ON
```

### 2.3 CI (GitHub Actions, free tier)

`ci.yml`: build rel + asan on ubuntu-latest, run `ctest` on both. CI runners have no AVX2 guarantee issues (all GitHub runners support AVX2) but **different cache sizes and no znver3** — so CI uses `-march=x86-64-v3` and your GEMM must have a runtime-dispatched generic fallback tested there. Perf numbers never come from CI; correctness always does.

### 2.4 Core headers you write once and never touch again

`src/core/align.h`:

```cpp
#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>

inline void* aligned_alloc64(std::size_t bytes) {
    void* p = std::aligned_alloc(64, (bytes + 63) & ~std::size_t{63});
    if (!p) throw std::bad_alloc{};
    return p;
}
struct AlignedFree { void operator()(void* p) const noexcept { std::free(p); } };
```

`src/core/tensor.h` — non-owning view, shapes derived at runtime from config:

```cpp
#pragma once
#include <array>
#include <cstdint>
#include <cassert>

template <typename T>
struct View {
    T* data = nullptr;
    std::array<int64_t, 4> shape{1, 1, 1, 1};   // pad leading dims with 1
    std::array<int64_t, 4> stride{0, 0, 0, 0};  // in elements

    int64_t numel() const {
        return shape[0] * shape[1] * shape[2] * shape[3];
    }
    T& at(int64_t a, int64_t b, int64_t c, int64_t d) const {
        return data[a * stride[0] + b * stride[1] + c * stride[2] + d * stride[3]];
    }
    static View contiguous(T* p, std::array<int64_t, 4> s) {
        View v; v.data = p; v.shape = s;
        v.stride = {s[1] * s[2] * s[3], s[2] * s[3], s[3], 1};
        return v;
    }
};

using F32View = View<float>;
using I8View  = View<int8_t>;
```

`src/core/arena.h` — the discipline that makes M3 possible:

```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include "align.h"

class Arena {
    std::byte* base_;
    std::size_t cap_, off_ = 0, high_water_ = 0;
public:
    explicit Arena(std::size_t cap)
        : base_(static_cast<std::byte*>(aligned_alloc64(cap))), cap_(cap) {}
    ~Arena() { std::free(base_); }
    Arena(const Arena&) = delete;

    void* alloc(std::size_t bytes, std::size_t align = 64) {
        std::size_t p = (off_ + align - 1) & ~(align - 1);
        if (p + bytes > cap_) return nullptr;   // caller handles; no throw in hot path
        off_ = p + bytes;
        if (off_ > high_water_) high_water_ = off_;
        return base_ + p;
    }
    void reset() { off_ = 0; }
    std::size_t high_water() const { return high_water_; }  // sizing data for M3 scratchpad
};
```

`high_water()` is not decoration: after M1 runs, it tells you the real activation working set per token, which is the number you use to size the fictional chip's scratchpad in M3. The project feeds itself.

**Rule: no `malloc`/`new` in any per-token path.** Weights: one mmap. Activations: one Arena, reset per token. KV cache: one up-front slab. If you feel the urge to `std::vector` inside the forward pass, you're about to make M3 a rewrite.

---

## 3. Milestone 1 — Reference forward pass (weeks 1–3)

**Definition of done:** `./run_infer --model qwen2.5-0.5b --prompt "The capital of France is"` produces greedy tokens identical to HuggingFace for ≥ 64 steps, and `ctest -R logit_parity` passes with max-abs-diff < 1e-3 on three fixed prompts.

### 3.1 Model acquisition (mind the disk)

Qwen2.5-0.5B-Instruct: `model.safetensors` ≈ 1 GB (bf16 on disk), `tokenizer.json` ≈ 7 MB, `config.json`. Download once via Python on the WSL side, keep exactly one copy, path in an env var. You'll convert bf16 → fp32 at load time in C++ (bf16 → fp32 is a 16-bit left shift into the high half of a float — write it, don't import it):

```cpp
inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
```

fp32 weights resident ≈ 2 GB. Fine in 12 GB of WSL RAM.

### 3.2 Safetensors reader (~150 lines, `src/core/safetensors.cpp`)

Format: 8-byte little-endian u64 = header length, then a JSON header mapping tensor name → `{dtype, shape, data_offsets}`, then the raw blob. Write a tiny recursive-descent JSON parser for exactly this subset (strings, numbers, arrays, objects — ~120 lines) rather than importing nlohmann. `mmap` the file `PROT_READ`, hand out views into it. Loading a 1 GB model should take milliseconds, not seconds; if it doesn't, you copied when you should have mapped.

### 3.3 Tokenizer — the part every plan forgets

Qwen2 uses byte-level BPE. You need: encode (prompt → ids) and decode (ids → text). This is real work — budget 3–4 days.

- Parse `tokenizer.json`: vocab (token → id) and merges list.
- Byte-level pretokenization: Qwen's pretokenizer regex is GPT-2-style. Implementing the full regex in C++ is misery; the pragmatic, defensible shortcut is to **precompute pretokenized pieces offline** for your test prompts via a 10-line Python script, and implement the merge loop + vocab lookup + byte-decoder in C++. Document the boundary honestly in `docs/`. For the interview, what matters is that the merge algorithm and the byte-fallback are yours.
- Greedy merge loop: repeatedly find the highest-priority adjacent pair present in the merges table; a linked-list-of-pieces with a heap of candidate merges is the clean O(n log n) shape.
- Decode: id → token string → byte-level unmap → UTF-8. Get the Ġ (U+0120 space marker) handling right or your output will look drunk.

Golden test: 20 strings (ASCII, Unicode, emoji, code snippets) encoded by HF, asserted equal in CTest.

### 3.4 Architecture — derive, don't hardcode

Read `config.json` at startup into:

```cpp
struct ModelConfig {
    int64_t hidden_size, intermediate_size;
    int64_t num_hidden_layers, num_attention_heads, num_key_value_heads;
    int64_t vocab_size, max_position_embeddings;
    float   rms_norm_eps, rope_theta;
    bool    tie_word_embeddings;
};
```

The two traps, named in advance:

1. **GQA.** `num_key_value_heads` (2) ≪ `num_attention_heads` (14) on the 0.5B. Q is [14, 64] per token; K and V are [2, 64]. Each KV head serves 7 query heads. Your KV cache is sized by *KV* heads — 7× smaller than the naive layout. Your attention loop indexes `kv_head = q_head / (n_q_heads / n_kv_heads)`.
2. **Tied embeddings.** `tie_word_embeddings: true` — there is no `lm_head.weight` tensor; the output projection is the input embedding matrix transposed. Your safetensors reader will not find the tensor you expect; that's correct, not a bug.

Also: Qwen2 attention has **biases on Q/K/V projections** (unusual — Llama doesn't). Load them.

### 3.5 The ops (fp32, scalar first, `src/ops/`)

Each op gets: a scalar reference implementation, a golden test against values dumped from PyTorch (`tools/dump_logits.py` can dump intermediate activations too — dump layer 0's post-attention hidden state and bisect from there when logits diverge), and *later* a vectorized version tested against the scalar one.

- `rmsnorm`: `x * rsqrt(mean(x²) + eps) * weight`. Accumulate the mean in **double** — this is where your first 1e-3 failure will come from if you don't.
- `rope`: rotate pairs `(x[i], x[i + d/2])` — Qwen uses the split-half convention, not interleaved. Getting this wrong produces plausible-looking garbage, which is why the layer-0 activation dump exists.
- `softmax`: subtract rowmax, exp, normalize. Causal mask = only iterate valid columns; don't materialize a mask tensor.
- `silu(x) * gate` for the FFN (SwiGLU): `x / (1 + exp(-x)) * g`.
- `gemm_ref`: the naive triple loop. Slow is fine; it's the oracle for M2.

### 3.6 Forward pass shape

```
embed → 24 × [ rmsnorm → QKV proj (+bias) → rope → attention (GQA, causal, KV cache)
               → O proj → residual → rmsnorm → gate/up proj → silu·mul → down proj → residual ]
      → final rmsnorm → logits = hidden @ embed.T
```

Two entry points from day one, because M4 depends on the split existing:

```cpp
// prefill: process N prompt tokens, fill KV cache, return last-position logits
void prefill(const int32_t* ids, int64_t n, KVCache& kv, float* logits_out);
// decode: one token in, one logits row out
void decode_step(int32_t id, KVCache& kv, float* logits_out);
```

### 3.7 The oracle

`tools/dump_logits.py`: three fixed prompts → HF forward → save fp32 logits + first-64-greedy-token ids to `.bin`. `test/logit_parity.cpp`: run your stack, compare. Tolerance 1e-3 max-abs on logits, exact match on greedy tokens. **This test is the constitution of the repo.** Every subsequent milestone must leave it green (quantized milestones get a second, looser oracle — see M2.6).

**Doc entry:** `docs/00-charter.md` (this file's §0, adapted) and notes on GQA/tied-embedding/rope-convention traps as you hit them.

---

## 4. Milestone 2 — AVX2 INT8 GEMM + roofline (weeks 4–6)

**Definition of done:** INT8 GEMM ≥ 70% of your *measured* (not paper) compute roof on prefill-shaped matmuls; roofline chart published; decode measured and shown memory-bound; blog post shipped.

### 4.1 Quantization scheme (`src/ops/quant.{h,cpp}`)

Weight-only INT8, symmetric, per-block along the reduction (K) dimension:

- Block size **QK = 64**. Per block: `scale = max(|w|) / 127`, `q = round(w / scale)`, stored as `int8_t q[64]` + `float scale` (or fp16 scale later if you want the exercise).
- Activations quantized on the fly per row-block to **uint8** asymmetric (zero-point 128 shift trick) — because `maddubs` wants u8 × s8.

### 4.2 Why QK=64 is not arbitrary — the saturation math

`_mm256_maddubs_epi16(a_u8, w_s8)` computes pairwise `a*w + a*w` into **saturating int16**. Worst case per pair: `255·127 + 255·127 = 64,770` — but int16 saturates at 32,767. So even a *single pairwise add* can saturate with extreme values.

The escape: after the zero-point shift, quantized activations in a well-scaled block are nowhere near 255 in magnitude on average, but you cannot rely on "on average" for correctness. The robust idiom drains to int32 every iteration:

```cpp
// one 32-byte chunk of the K dimension
__m256i a  = _mm256_loadu_si256(pa);            // u8 ×32
__m256i w  = _mm256_loadu_si256(pw);            // s8 ×32
__m256i p16 = _mm256_maddubs_epi16(a, w);       // s16 ×16 (saturating!)
__m256i p32 = _mm256_madd_epi16(p16, ones16);   // s32 ×8  (safe)
acc = _mm256_add_epi32(acc, p32);
```

and you bound the residual saturation risk by (a) the zero-point shift keeping |a| small and (b) QK=64 keeping per-block dynamic range tight. Then **prove it empirically**: quantize every weight matrix in the model, compute max |pairwise product sum| per block, assert < 32767 in a test. If any block fails, drop to QK=32 for that tensor. This test-instead-of-hope move is `docs/02-int8-saturation.md` and it is a genuinely good interview story.

Zero-point correction: with `a' = a + 128`, `Σ a'w = Σ aw + 128·Σw`. Precompute `Σw` per block at quantization time, subtract at the end. Free.

### 4.3 Kernel structure (`src/ops/gemm_avx2.cpp`)

C[M,N] += A[M,K] · B[K,N], B pre-packed.

- **Register tile 4×3**: 12 ymm int32 accumulators + a-broadcast + b-loads + ones16 = fits in 16 ymm with no spill. (Try 6×2 as well; measure, don't argue.)
- **Packing:** B packed once into K-major panels of width 3·8 int32-lanes-worth, 64-byte aligned, laid out so the inner loop's loads are purely sequential. Pack cost amortizes over M.
- **Cache blocking:** Kc sized so the packed B panel + A slice sit in L2 (512 KB): start Kc=1024, Nc=192, Mc=64 and sweep. Put the sweep results in the doc.
- **Prefetch:** `_mm_prefetch(ptr, _MM_HINT_T0)` on A two iterations ahead. Measure the delta; on Zen 3 the hardware prefetcher may make it a wash — reporting a null result honestly is good signal.
- **Threading:** row-block work queue over your `ThreadPool` (6 pinned threads via `pthread_setaffinity_np`). Compare 6 vs 12 threads; explain the SMT result in the doc (port contention on the two FMA/vector-ALU pipes).
- **Runtime dispatch:** `gemm()` checks cpuid once; falls back to `gemm_ref` off-AVX2 (this is what CI exercises).

### 4.4 The two ceilings

Measure, don't quote:

1. **Compute roof:** write a dependency-free FMA loop (or the int8 madd chain), pin one core, run 10 s, count ops. Multiply by 6 for the multicore roof — then measure the multicore version too and watch thermals eat it. Record both 10-second and 5-minute sustained numbers; the gap *is* the mobile-silicon story.
2. **Memory roof:** write `tools/stream_bench.cpp` (triad: `a[i] = b[i] + s*c[i]`, arrays ≫ L3, nontemporal stores variant too). Expect mid-30s to low-40s GB/s against 51.2 theoretical.

### 4.5 The chart

Roofline: x = arithmetic intensity (ops/byte), y = GOP/s, log-log. Plot:

- prefill GEMM points (M = 128, 512, 2048 prompt tokens) — should climb toward the compute roof,
- decode GEMV points (M = 1) — pinned to the bandwidth diagonal,
- llama.cpp on the same machine, same model, same quant class (`-t 6`, Q8_0) as the honesty benchmark.

Beating llama.cpp is not the goal and probably won't fully happen; being within 15–25% with a from-scratch kernel *and being able to explain every point on the chart* is the goal.

### 4.6 Second oracle

Quantization moves logits > 1e-3, so add `logit_parity_int8`: tolerance on logits loosened to what you measure (record it), but **top-1 token match ≥ 99%** over 200 greedy steps across the three prompts, and perplexity delta on a fixed 10 KB text < 0.5%. The fp32 path stays exact forever.

**Ship:** blog post — "A from-scratch INT8 GEMM on a laptop, and what it teaches you about why inference hardware exists." Roofline chart front and center. This post is your first public artifact; link it in every application.

---

## 5. Milestone 3 — The accelerator: ISA, simulator, compiler (weeks 7–11)

**Definition of done:** the fp32 model, lowered by your compiler to your ISA, runs on your simulator and passes the M1 oracle bit-for-bit against your own CPU fp32 path (same order of operations ⇒ identical floats); simulator reports cycles, DMA stalls, and MAC utilization per layer.

This is the milestone that makes the project *about Etched* instead of about llama.cpp. Design decisions below are defaults — deviating is fine if you write down why.

### 5.1 The fictional chip: "T1"

One core ("tile") for now — M5 adds more.

| Unit | Spec | Rationale |
|---|---|---|
| MXU | 32×32 systolic array, fp32 MACs, 1 MAC/cell/cycle | Big enough that tiling is real, small enough to simulate fast |
| VPU | 32-lane fp32 vector unit: add, mul, rsqrt, exp approx, max, reduce | Enough for rmsnorm/softmax/silu/rope |
| SPM | 8 MB scratchpad, software-managed, 2 banks (double-buffer) | Size it against M1's `high_water()` + a weight tile budget; 8 MB forces real tiling on the FFN |
| DMA | 1 engine, HBM↔SPM, 256 B/cycle, fixed 400-cycle latency, strided 2-D transfers | Latency forces double-buffering; strides let you fetch tiles without host repacking |
| SEQ | in-order scoreboard: an instruction issues when its declared deps retire | Simple, but exposes pipeline bubbles you'll hunt in M7 |
| Clock | 1 GHz nominal | Makes cycles ↔ ns trivial |

Derived roofs (put these in `docs/03-isa-spec.md`): compute 32·32·2 = 2048 flop/cycle = 2 Tflop/s; DMA 256 GB/s. The chip's machine balance is ~8 flop/byte — now *decode on T1* is memory-bound by design, same thesis, new altitude.

### 5.2 ISA (`src/accel/isa.h`)

Fixed 32-byte instruction records in a flat buffer ("program"), plus a const-pool. Roughly:

```cpp
enum class Op : uint8_t {
    DMA_LOAD,    // hbm_addr, spm_addr, rows, cols, row_stride  (2-D)
    DMA_STORE,
    MATMUL,      // spm A [m,k], spm B [k,n], spm C [m,n], accumulate flag
    V_RMSNORM,   // spm src, spm weight, len, eps
    V_SOFTMAX,   // spm src, rows, cols (causal handled by cols per row)
    V_ROPE,      // spm src, head_dim, position
    V_SILU_MUL,  // spm gate, spm up, len
    V_ADD,       // residual
    BARRIER,     // fence a dependency class
    HALT,
};

struct Instr {
    Op    op;
    uint8_t flags;
    uint16_t dep_mask;      // scoreboard classes this waits on
    uint32_t arg[6];        // addresses in SPM are byte offsets; HBM addrs are 64-bit via pairs
};
```

Design stances worth defending in the doc:

- **No registers, no branches.** A transformer forward pass is a straight line; the compiler unrolls everything. This is the "fixed-function, Bitter-Lesson" bet stated in silicon — say so.
- **Software-managed memory.** No caches, no coherence. Every byte movement is an explicit `DMA_LOAD`. The compiler owns correctness of data placement. This single decision generates 80% of the interesting compiler work.
- **Scoreboard classes, not full dataflow.** `dep_mask` bits like {DMA_A, DMA_B, MXU, VPU, DMA_OUT} — coarse, cheap, and the false-dependency stalls it causes are *exactly* what your M7 profiler will catch. Leave the flaw in on purpose; fixing it is M7's documented optimization.

### 5.3 Simulator (`src/accel/simulator.cpp`)

Cycle-approximate, not cycle-accurate — model *time*, compute *values* directly:

- Functional layer: each instruction executes its math on a `std::vector<float> spm(8MB/4)` and a mmap'd "HBM" (the weight file + activation slabs). MATMUL is your `gemm_ref` on SPM views — which means functional correctness is trivially bit-identical to the CPU path if op order matches. That's the point.
- Timing layer: each unit (MXU, VPU, DMA) has a `busy_until` cycle. Instruction latency models: MATMUL ≈ pipeline fill (m + k + n) + max(m·k·n / 1024, …) cycles — keep it simple, write the formula in the doc; DMA = 400 + bytes/256; VPU = len/32 + fixed cost.
- Scoreboard: issue in order; an instr with unmet `dep_mask` stalls the front end — record *why* it stalled (which class) per stall. This attribution is the raw material for the M7 profiler; log it from day one even though you won't visualize it until later.
- Determinism: single-threaded sim. Speed target: ≥ 1 token/s simulated for the 0.5B — if slower, your MATMUL functional path needs the M2 kernel under the hood (legitimate: functional result, modeled timing).

Sanity tests (`test/sim_isa.cpp`): hand-written 10-instruction programs with cycle counts asserted by hand. Do not skip these; debugging the compiler on a broken simulator is despair.

### 5.4 Compiler (`src/accel/lower.cpp`, `tiling.cpp`, `schedule.cpp`)

Input: `ModelConfig` + weight map. Output: one program per (phase, seq-length-bucket) — prefill programs for bucketed lengths {128, 512, 2048} and one decode program parameterized by position.

Passes, in order:

1. **Graph build:** the fixed op list from §3.6, as a vector of high-level ops with symbolic tensor refs. (No general graph IR — resist. It's a straight line.)
2. **Tiling:** every MATMUL split to ≤ 32×32×K_tile chunks such that A-tile + B-tile + C-tile + the *next* prefetched pair fit in one SPM bank (double-buffer budget = 4 MB per bank). FFN up-proj [hidden → intermediate] is the tensor that won't fit; it drives the whole design. Emit the tiling decisions to a debug log — this log is your "programming abstractions for rapid porting" evidence.
3. **SPM allocation:** linear-scan over the straight-line program; reuse offsets when lifetimes end. Assert the M1 arena discipline pays off here (it will).
4. **DMA scheduling:** software pipelining — issue tile N+1's loads before tile N's MATMUL retires, alternating banks. This is the pass that moves DMA-stall% from ~60 to ~15 and it's one chart in the writeup: stall% before/after pipelining.
5. **Encode** to `Instr[]`.

**Doc entries:** `03-isa-spec.md` (a real spec — formats, semantics, timing model, memory map) and `04-compiler.md` (passes + the double-buffering chart). Written well, the ISA spec alone is a portfolio piece.

---

## 6. Milestone 4 — The runtime (weeks 12–15)

**Definition of done:** an engine that accepts concurrent generation requests, continuously batches them through simulated-T1 decode steps, manages KV memory under pressure, and survives injected DMA faults with per-request (not process-wide) failure.

### 6.1 Engine shape (`src/runtime/engine.{h,cpp}`)

```cpp
struct Request {
    uint64_t id;
    std::vector<int32_t> prompt;
    int max_new_tokens;
    // filled in by the engine:
    enum class State { QUEUED, PREFILLING, DECODING, DONE, FAILED } state;
    KVHandle kv;
    std::vector<int32_t> output;
};
```

Single scheduler thread owns all state (no locks in v1 — document this as a deliberate simplicity/latency trade); worker interaction is submit/complete queues. The loop each iteration:

1. Admit new requests if a KV allocation succeeds.
2. Choose phase: run one prefill (chunked at 256 tokens so decode latency isn't destroyed — this is chunked prefill, name it) **or** one batched decode step for all DECODING requests.
3. Submit the program to the simulator; on completion, sample greedy, append, retire finished requests, free KV.

### 6.2 KV cache management (`src/model/kv_cache.cpp` grows up)

Paged, vLLM-style but honest about being simplified: fixed pages of 64 tokens × layers × kv_heads × head_dim, a free list, per-request page tables. Eviction policy v1: refuse admission when full (backpressure) rather than preempting — then add preemption (drop lowest-progress request, requeue with prompt) as a measured improvement. Chart: throughput vs. concurrent requests at page-pool sizes {tight, comfortable}.

### 6.3 Batching study

Decode-batched GEMV becomes a skinny GEMM — arithmetic intensity rises with batch. On simulated T1, sweep batch = 1…32 and plot tokens/s and per-token latency. You will reproduce the canonical throughput/latency frontier on hardware *you invented* — that chart is `docs/05-runtime.md`'s centerpiece and possibly the writeup's second-best figure after the roofline.

### 6.4 Fault injection (`src/runtime/faults.{h,cpp}`)

The JD says "robust error handling" — make it demonstrable:

- Simulator gains a fault hook: with probability p (or at a scripted cycle), a DMA never completes, or completes with a poisoned checksum.
- Runtime response: per-program deadline (cycles budget × 1.5) → on expiry, mark the batch's in-flight requests FAILED with a typed error, reset simulator state, continue serving everyone else. No crash, no hang, no zombie KV pages (assert the free-list is balanced in the test).
- `test/`: scripted fault at a known cycle; assert exactly the affected requests fail and a subsequent request succeeds.

This section is small in code and large in interview value: "what happens when a DMA never completes" has a concrete, tested answer in your repo.

---

## 7. Milestone 5 — Tensor parallelism & hand-written collectives (weeks 16–20)

**Definition of done:** the model sharded across 2 and 4 *processes* (each owning one simulated T1), producing oracle-identical output, over three interchangeable transports — shared memory, TCP, io_uring — with a latency/bandwidth microbenchmark table for each collective on each transport.

### 7.1 Sharding plan (Megatron-style, adapted to GQA)

- Attention: shard by query head. 14 heads over 4 ranks is uneven (4/4/3/3) — handle it; uneven sharding is realistic and mildly painful, which is why it's good. KV heads (2) replicate on ranks that need them (document the alternative — KV-head sharding — and why 2 heads over 4 ranks forces replication anyway).
- FFN: column-parallel up/gate, row-parallel down ⇒ one all-reduce per FFN, one per attention block: 2 all-reduces per layer, the textbook pattern.
- Embedding/logits: replicate (0.5B is small; note the vocab-parallel alternative).

Rank 0 is the driver: tokenize, broadcast ids, gather logits, sample, broadcast next id. Determinism rule: fixed reduction order (rank order) in all-reduce so floats are bit-stable and the oracle stays exact.

### 7.2 Collective interface (`src/comm/collective.h`)

```cpp
struct Communicator {
    virtual void all_reduce(float* buf, size_t n) = 0;      // sum, in-place
    virtual void all_gather(const float* in, size_t n, float* out) = 0;
    virtual void reduce_scatter(const float* in, float* out, size_t n_per_rank) = 0;
    virtual void broadcast(void* buf, size_t bytes, int root) = 0;
    virtual void all_to_all(const float* in, float* out, size_t n_per_rank) = 0;  // M6
    virtual ~Communicator() = default;
};
```

Algorithms (write both, select by size — this is what NCCL does and saying so is the point):

- Small buffers: recursive doubling / direct exchange (latency-optimal).
- Large buffers: **ring** all-reduce = reduce-scatter + all-gather, 2(p−1)/p · bytes per link (bandwidth-optimal). Derive the formula in the doc, then *measure* that shm hits it and TCP doesn't, and explain the gap (per-message syscall + copy costs).

Correctness harness before any transport work: `collective_fuzz.cpp` — random sizes, random data, compare against a serial reference reduction, all three transports, under TSan. Non-negotiable; debugging TP through wrong-answer logits without this is weeks of pain.

### 7.3 The three transports

1. **`shm_ring.cpp`** — `memfd_create` + `mmap` shared segments; per-link SPSC ring with C++20 `std::atomic<size_t>::wait/notify` for doorbells; acquire/release fences you justify comment-by-comment. This is "intra-node execution."
2. **`tcp.cpp`** — nonblocking sockets, `TCP_NODELAY`, your own length-prefixed framing, an epoll loop. Naive first; then fix the obvious (coalesce small sends, reuse buffers) and show the before/after.
3. **`uring.cpp`** — same protocol on io_uring (liburing): registered buffers, multishot recv, ≥ N inflight ops. The chart that matters: p50/p99 latency vs message size, TCP-epoll vs io_uring, small messages — where completion-batching visibly wins. This is the JD's "kernel-level and user-space networking stacks" bullet made concrete.

All-reduce microbenchmark table (bytes × transport × algorithm → GB/s and µs) goes in `docs/06-collectives.md`.

### 7.4 Integration with the simulator

Each rank simulates its T1 shard; collectives happen at program boundaries (decode step: run program → all-reduce activations → continue). Add modeled interconnect time to the simulated clock: bytes / link_bw + hop_latency, parameters in `topology.{h,cpp}`. Now your M7 traces show compute *and* communication on one timeline — which is the actual daily work of the team you're applying to.

---

## 8. Milestone 6 — MoE (weeks 21–23)

**Definition of done:** FFN replaced by 8 experts, top-2 routing, experts sharded across ranks with your all-to-all; a load-imbalance study with numbers.

### 8.1 Model surgery

No pretrained 0.5B MoE fits the disk budget cleanly, so build the MoE variant synthetically and be upfront that this milestone is about *systems*, not model quality:

- Clone the dense FFN weights into 8 experts, add small Gaussian noise (σ = 0.01 · weight-std) so routing isn't degenerate; router = new random linear [hidden → 8].
- Quality oracle is replaced by an invariance test: with router forced to always pick expert 0 with weight 1.0, output must equal the dense model exactly. That single test catches 90% of dispatch/combine bugs.

### 8.2 Routing + dispatch (`src/moe/`)

- Router: logits → top-2 → softmax over the 2 → scale expert outputs.
- Expert parallelism: expert e lives on rank e mod p. Per decode batch: count tokens per expert, all-to-all token activations to expert owners, run expert FFNs, all-to-all back, combine with gate weights. Capacity factor C: each expert accepts ≤ C · (tokens·2/8); overflow tokens dropped-through with residual only — count them.
- The study: run with (a) noise-broken random-ish router and (b) an adversarial router (all tokens → 2 experts). Chart tokens/s and per-rank busy% for both. The imbalance cliff you measure is the real reason aux-load-balancing losses exist in MoE training — connect that dot explicitly in `docs/07-moe.md`.

---

## 9. Milestone 7 — Profiler & the bottleneck hunt (weeks 24–26)

**Definition of done:** every simulator instruction and every collective emits Chrome-trace events; one bottleneck found *via the profiler*, fixed, and written up with before/after traces.

### 9.1 Trace emission (`src/prof/trace.{h,cpp}`)

Chrome Trace Event Format (JSON array of `{name, ph:"X", ts, dur, pid, tid, args}`) — viewable in Perfetto with zero UI work from you. Map: pid = rank, tid = unit (MXU / VPU / DMA / NET), ts = simulated cycles (µs at 1 GHz). `args` carries: bytes moved, MAC utilization, stall class from the scoreboard. Counter events (`ph:"C"`) for SPM occupancy and KV pages in use.

### 9.2 Derived metrics (per layer, per phase)

MAC utilization = active-MXU cycles / total; DMA stall% by class; pipeline-bubble cycles from the scoreboard's false-dependency log (planted in M5.2 — pay it off now); network wait% per decode step.

### 9.3 The hunt (the writeup's climax)

You planted at least three real bottlenecks:

1. Coarse scoreboard classes serialize independent DMA streams (M3 §5.2) → fix: split classes / per-buffer deps → measure DMA overlap gain.
2. Decode all-reduce latency dominates small-batch TP decode (M5) → fix: fuse the two per-layer all-reduces where legal, or overlap layer-N comms with layer-N+1 compute → measure.
3. MoE stragglers under imbalance (M6) → fix: capacity + a second all-to-all round for overflow, or expert replication for hot experts → measure.

Pick whichever trace looks worst, fix it, show the two traces side by side in `docs/08-bottleneck-hunt.md`. "I found it in the profiler, here's the before/after" is the exact sentence the JD's fourth bullet wants to hear.

---

## 10. Testing constitution (applies to every milestone)

| Layer | Test | Gate |
|---|---|---|
| Ops | golden vs PyTorch dumps; vectorized vs scalar | every commit |
| Model fp32 | logit parity < 1e-3, greedy-64 exact | every commit |
| Model int8 | top-1 ≥ 99% / 200 steps; ppl delta < 0.5% | every commit post-M2 |
| ISA | hand-counted cycle tests | post-M3 |
| Compiler | lowered-vs-CPU bit-exact (fp32) | post-M3 |
| Collectives | fuzz vs serial reference, ×3 transports, TSan | post-M5 |
| MoE | forced-expert-0 ≡ dense | post-M6 |
| Runtime | scripted-fault isolation; KV free-list balance | post-M4 |
| Memory | ASan+UBSan full test suite | CI, every push |

Perf numbers are never asserted in CI (thermals + shared runners = flakes); perf regressions are caught by a local `bench/` script you run before tagging each milestone, results committed as CSV.

---

## 11. Publication plan

| When | Artifact | Where |
|---|---|---|
| End M2 | "From-scratch INT8 GEMM on a laptop: a roofline story" | Blog + repo README + X/LinkedIn |
| End M3 | The T1 ISA spec (docs/03 polished) + "Compiling a transformer to a chip that doesn't exist" | Blog |
| End M5 | Collectives microbenchmark writeup (shm vs TCP vs io_uring) | Blog |
| End M7 | The capstone: full-stack writeup, the three charts (roofline, batching frontier, before/after trace) | Blog, submitted to HN; PDF version attached to applications |
| Continuous | docs/ engineering log | repo |

README structure: pitch paragraph (§0.1) → the three charts → architecture diagram → build/run in 5 commands → milestone log with links. Recruiters see the README for 40 seconds; engineers read `docs/03-isa-spec.md` for 20 minutes. Serve both.

### Repo hygiene that doubles as signal

- Conventional commits, PR-sized changes even solo (you review your own diffs — say so in the README, it lands well).
- `clang-format` + `clang-tidy` config committed; zero warnings at `-Wall -Wextra -Wshadow -Wconversion`.
- LICENSE: MIT or Apache-2.0.
- Weights never committed; `tools/fetch_model.sh` + checksums.

---

## 12. Schedule (part-time, honest)

| Weeks | Milestone | Ship |
|---|---|---|
| 1–3 | M1 reference pass + tokenizer + oracle | repo public, docs/00 |
| 4–6 | M2 GEMM + roofline | **blog post 1** |
| 7–11 | M3 ISA + simulator + compiler | **blog post 2**, ISA spec |
| 12–15 | M4 runtime | batching-frontier chart |
| 16–20 | M5 TP + collectives | **blog post 3** |
| 21–23 | M6 MoE | imbalance study |
| 24–26 | M7 profiler + hunt | **capstone post** |

~6 months part-time end to end. **Apply to Etched and the broader inference-infra list after M3** — a working compiler-to-custom-ISA plus a published roofline post is already a differentiated application, and the remaining milestones become "here's what I'm building next" talking points that improve with every interview cycle. Do not wait for M7 to apply.

Cut lines if life happens, in order of least regret: M6 (MoE) → the io_uring transport (keep shm + TCP) → M4 preemption (keep backpressure). Never cut: the oracle, the roofline, the ISA spec, the profiler-found fix.

---

## 13. Deviations from this bible (living section)

Reality has already disagreed with the plan in a few places. Each deviation is
recorded here with its reason, per the rule at the bottom of this file.

| § | Bible says | We do | Why |
|---|---|---|---|
| 4.1 | Activations → **uint8** asymmetric (zero-point +128) for `maddubs` | Activations → **signed int8** in [−127,127]; AVX2 uses the `_mm256_sign_epi8` trick to make `\|a\|` the unsigned operand and `w·sign(a)` the signed one | Provably removes saturation instead of bounding it empirically: each int16 partial is ≤ 2·127² = 32258 < 32767. Also deletes the `Σw` zero-point correction entirely. See `docs/02-int8-saturation.md`. |
| 4.4 | `tools/stream_bench.cpp` | `bench/stream_bench.cpp` | Benchmarks belong in `bench/`; `tools/` is for things you run to produce artifacts. |
| 1.1 | WSL `memory=12GB` | ~7.6 GB visible | Host has 15.3 GB and no `.wslconfig` tuning yet. Weights are fp32-resident (~2 GB) so this still fits; revisit if M4 batching pushes it. |
| 2.2 | `-march=znver3` | `-march=native` locally, `-march=x86-64-v3` in CI | `native` resolves to znver3 on the target laptop and keeps the tree portable. |
| 3.3 | Precompute pretokenized pieces offline in Python | Hand-written C++ pretokenizer, exact for ASCII | Keeps `run_infer` self-contained. The approximation is tested against an HF golden dump (`test/golden/tokenizer.txt`) so its exact boundary is *measured*, not assumed. |
| 4.1 | Block size **QK = 64** | **QK = 32** | 64 scored 94.8% top-1 vs HF and 4.33% logit rel-L2; 32 scores 98.4% and 3.28%. Costs ~20% of GEMM throughput (measured, both configs, same machine), which we pay because decode is memory-bound anyway. `docs/02` §2.4. |
| 4.1 | Weight-only INT8 | **W8A8**, activations quantized per QK block | Per-*row* activation scales are catastrophic on transformers (outlier features): 14.6% top-1. Per-block confines each outlier to its own block and costs nothing in the kernel. `docs/02` §2.3. |
| 4.1 | Everything quantized | **The output projection stays fp32** | Its error lands directly on the logits with nothing downstream to attenuate it (top-1 99%+ → 66%), and since embeddings are tied it must stay fp32 for the token lookup anyway — quantizing would *add* 136 MB beside the 545 MB, not replace them. |
| 4.5 | Roofline chart ≥70% of the compute roof; llama.cpp comparison | **24.9% of roof**; llama.cpp not run | Not rescoped, decomposed: `docs/01` attributes a 20% ceiling to dequantize instructions (predicted +20% for QK=64, measured +19.6%), **refutes** the weight-bandwidth hypothesis with an MR sweep, and labels the remaining ~2× unattributed. The llama.cpp honesty benchmark is still owed. |
| 4.6 | INT8 top-1 **≥ 99%** over 200 steps | **≥ 96%**, with perplexity < 2% as the real gate | We score 98.4%, and the residual is activation quantization — model-prep work (SmoothQuant/AWQ/QuaRot), not kernel work. The metric is also flag-sensitive: 98.4% at `-march=native`, 96.9% with `-ffp-contract=off` *and* in the ASan tree (identical digit for digit — FMA contraction, not the sanitizer). Perplexity moves 0.05 points across all three, so it carries the gate. `docs/02` §2.2. |
| 5.2 | `MATMUL ... accumulate flag` | Accumulate seeds the MXU accumulator **in the wide format** | `fp32(double(C) + Σ)`, one rounding. Otherwise a k-split matmul answers differently from an unsplit one, i.e. tiling decisions become visible in the numerics — and biases (which `ops::linear` folds into its double accumulator) cannot be reproduced bit-exactly. `docs/03` §4.3. |
| 5.4 | Programs per (phase, seq-length bucket), prefill at {128, 512, 2048} | **Decode only**, `m` always 1 | M3's gate is bit-exactness, and decode is the phase the whole project's thesis is about. `docs/04` §4 shows prefill is where T1's performance is (m rounds up to 32, so 32 tokens cost the same cycles as 1) — which makes it M4 work, not a footnote here. |

---

*Version 1.0 — July 2026. This document is the spec. When reality disagrees with it, update the document in the same commit that changes the plan.*
