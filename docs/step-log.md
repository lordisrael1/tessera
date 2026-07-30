# Step log — plain English, newest first

A running "what we did and why" so you (and future sessions) never face a blank
page. Brutal and honest: what worked, what's fake, what's left.

---

## 2026-07-30 — Session 5: prefill, and the 32× the timing model promised

### The headline
**Thirty-two tokens for the price of one, bit-exact.** One prefill chunk of 32
tokens costs **25,466,046 cycles** against **808,077,440** for the same 32
tokens as decode steps — **31.7×** — and the last token's logits are identical
to the decode path's, bit for bit. Array efficiency goes from **2.88% to
67.8%**.

That number was not discovered, it was *collected*. Session 4 ended by noting
that `matmul_cycles` rounds `m` up to the array dimension, so a batch of 32
should cost about what a batch of 1 costs, and wrote it down as the reason to
build M4. This session went and took it.

### Why it was mostly free
The ISA already specified `m > 1`, the simulator already timed it, the tiler
already split the output dimension, and `kFlagAccumulate` had already been given
its wide-accumulator rounding contract while fixing the bias bug. Four things
still had to be built:

1. **A row dimension on the vector ops.** `V_RMSNORM`, `V_ROPE`, `V_SOFTMAX` and
   `V_COPY` grew `rows` and a stride. Every one of them encodes "one row" as
   **0**, which is why not a single decode program changed by a byte and why the
   decode measurements came back at *exactly* 25,242,004 and 18,361,229 cycles
   afterwards. Backward compatibility you can check with a diff of a cycle count
   is worth more than backward compatibility you assert.
2. **Causal masking as a hardware feature.** `V_SOFTMAX` row `r` now covers
   `first_len + r` elements and **zeroes the rest**. The zeroing is the whole
   trick: it lets one `[M, KEYS] · [KEYS, HD]` matmul do the AV product for every
   query row, because masked entries contribute exactly `0.0` and adding `0.0`
   to a double accumulator is a no-op. For a 32-token chunk that is 21,504
   instructions replaced by 336. The bible's original ISA sketch had this all
   along — *"causal handled by cols per row"* — and it took building prefill to
   understand what that line was for.
3. **A 2-D strided `V_COPY`.** MATMUL writes a tile packed as `[m, n]`; at M=1
   that happens to be exactly the slice of `Y` it belongs to, and at M>1 it is
   not. Rather than put an output stride on the systolic array's writeback port,
   the VPU places the tile. It costs ~3% of the tile's matmul cycles.
4. **Bias broadcast via `row_stride = 0`.** With the accumulator seeded from C,
   the bias must appear in all M rows. A DMA descriptor with a zero row stride
   re-reads the same OUT floats M times. No new opcode, no cost.

### The test caught a bug — in the test's own reference
The gate is not "prefill matches a reference written next to it". It is
**"prefill matches the decode path"**, which session 4 had already proved
against the CPU and, through `logit_parity`, against HuggingFace. Prefill has to
land on that answer using different instructions, a different tiling, a
different attention shape, and a causal mask that did not exist in the decode
program.

First run: one chunk of 5 was exact, but 3+2 and five-chunks-of-1 disagreed
**with the CPU reference while agreeing with decode exactly**. When the machine
agrees with an independently-verified path and the reference does not, the
reference is what is wrong — and it was: `cpu_reference_prefill` read the chunk's
embeddings from row 0 regardless of `base`, while the compiled DMA descriptor
correctly read from `act_in + base*H`. Two gates disagreeing is how you find out
which one is lying.

All three chunkings are now bit-identical to each other, to the CPU, and to
decode.

### The most useful thing the simulator has said yet
`max_prefill_tokens()` reports **42 tokens** for Qwen2.5-0.5B on T1, and at M=32
the scratchpad is 88% full. The bible asked for prefill buckets of {128, 512,
2048}; **they are not reachable on this machine**, and no scheduling trick
changes that — it is 8 MB against ~81 KB of activations per token. A long prompt
is several chunks, which is exactly what production systems call chunked
prefill, and `lower_prefill` takes a `base` so chunks compose.

That is a *design* answer rather than a tuning one: **if you want longer prefill
chunks on T1, you buy scratchpad, not MACs.** It is the first time the simulated
hardware has pushed back on the plan and been right.

### Honest edge on the headline
67.8% is not 96%. A single projection tile at M=32 measures 95.9% efficient; the
chunk average is dragged down by the **output projection, which still runs at
m=1** and is 26% of the cycles. That is correct rather than lazy — prefill only
needs the last token's logits, and the other 31 rows would each cost a 545 MB
sweep of the embedding matrix to produce a distribution nobody samples. The
inefficiency that remains is inherent to what prefill is *for*, and it now sits
in one identifiable instruction instead of being smeared across the model.

---

## 2026-07-29 — Session 4: M3 lands — the real model runs on the fake chip

> **Gap in this log.** Session 3 is missing: it produced the M1 numerical oracle
> (`dump_logits.py`, `logit_parity`), all of M2's kernel work (`quant`,
> `gemm_avx2`, `threadpool`, `linear_f32`), the benchmarks, and the first cut of
> M3's `isa.h` / `simulator.cpp` / `lower.cpp` — and never wrote an entry. This
> entry documents what that code turned out to *do* once it was run and
> measured, but the reasoning behind writing it is not recorded anywhere and is
> not recoverable. Write the log the day the thing happens.

### The headline
**Qwen2.5-0.5B, compiled to our own ISA and executed on our own simulator,
produces logits bit-identical to the CPU fp32 path at every position, and
answers "The capital of France is" with " Paris."** 4461 instructions, 25.24 M
simulated cycles, 1975.8 MB streamed through an 8 MB scratchpad the compiler
manages by hand. That is milestone 3's definition of done, on the real model
rather than a toy.

### It did not start green
The session opened by running the tests, which is how the honest version of this
log always starts. Two were red, and both were real bugs, not flaky tests.

**Bug 1 — a write-after-read in the software pipeline.** The prefetch of tile
t+2 was emitted *before* the MATMUL of tile t, and those two share an SPM bank.
So the MXU computed on the wrong weights. The failure mode is the nasty one:
deterministic, silent, no crash, same wrong answer every run — the functional
simulator executes in program order, so it faithfully reproduced the bug.
`compiler_parity` caught it as 48 of 48 logits wrong. Fixing it meant emitting
the MATMUL first (which is what the function's own comment had always described
— the code and its documentation had disagreed since it was written), and giving
**every** weight load the `DEP_MXU` anti-dependency, including the prologue
loads: the banks are at fixed offsets reused by every projection, so a
projection's first load can otherwise land on top of the previous projection's
still-running tail matmul. That second half is invisible to the simulator —
its functional layer runs at issue time, so an anti-dependency violation shows
up as wrong *timing*, never as wrong *values*. It had to be reasoned out.

**Bug 2 — one ulp, in the bias.** After bug 1 the logits went from garbage to
almost-right: differing in the last bit. `ops::linear` seeds its **double**
accumulator with the bias; the compiler was adding the bias afterwards, in
float, with a `V_ADD`. Two roundings instead of one.

The tempting fix is to loosen the test. The right fix was to notice the ISA was
under-specified: `kFlagAccumulate` now seeds the MXU accumulator from C *in the
wide format*, so the machine computes `fp32(bias + Σx·w)` in one rounding, and
the compiler simply DMAs the bias into the output buffer before the tiles. This
deleted an instruction per biased projection, and — more importantly — made
k-split matmuls value-invariant, which is the property `compiler_parity`'s
four-schedule check actually asserts. A rounding rule that was implicit is now
written down in `docs/03` §4.3.

**And three wrong hand-counts in `test/sim_isa.cpp`.** The cycle assertions were
derived by hand when the test was written, and two of them were simply wrong
arithmetic: one forgot that a 1024-float load is 4096 bytes and not 256, another
forgot the front end issues one instruction per cycle. The simulator was right
both times. Worth saying plainly: the tests were wrong, not the code.

### The benchmarks were worse than useless
The committed `bench/results/*.csv` from last session were taken while a build
was running. The compute roof came out **45% low** (230 GFLOP/s against 431 on a
quiet machine) and one GEMM point read 35.9 GOP/s where the identical binary
does 64.4. Nothing looked wrong — the numbers were plausible and internally
consistent, which is exactly why this is dangerous. Everything was re-measured
in one idle-machine pass, now scripted as `bench/run_all.sh` so it is
reproducible rather than remembered. Lesson for the file: **a benchmark taken on
a busy machine is worse than no benchmark, because it looks authoritative.**

### What the numbers turned out to say
- **Decode is memory-bound by ~35×.** 48.9 ms/token INT8, of which the
  arithmetic accounts for 1.4 ms. Compute is idle 97% of the time.
- **INT8 bought exactly the bytes it removed and nothing else:** 2.09× fewer
  bytes, 2.09× faster, and *identical* bandwidth utilisation (19.3 → 19.4 GB/s).
- **The M2 ≥70%-of-roof gate is missed** — we reach 24.9%. Written up rather
  than rescoped. The op-count model explains a 20% ceiling and predicted that
  QK=64 would be ~20% faster; it measured +19.6% on lm_head, which is the
  strongest evidence the model is right. A tuning sweep then **refuted** the
  obvious next hypothesis: quadrupling weight reuse (MR 1→4) buys only 8.6%, so
  the kernel is not weight-bandwidth-bound. The remaining 2× is unattributed and
  is labelled as such.
- **On T1, decode is not bandwidth-bound at all** — it is bound by the *shape*
  of the systolic array. "MAC utilisation 66.4%" looks fine until you compute
  array **efficiency**: 2.88%, because m=1 uses 1 of 32 rows. The weight stream
  alone would need 7.7 M cycles; we spend 25.2 M. Since `matmul_cycles` rounds m
  up to 32, **a batch of 32 tokens would cost the same cycles as a batch of 1**.
  M3's measurement wrote M4's justification without being asked to.
- **The planted ISA flaw is worth 27% of the runtime.** Software pipelining
  takes a decode step from 25.24 M to 18.36 M cycles — but only with fine-grained
  dependencies. Under the coarse scoreboard the ISA actually specifies, double
  buffering is worth *exactly zero*: the same cycle count, to the cycle, as not
  doing it. The optimisation is not degraded, it is annihilated. That table is
  M7's before/after, sitting ready.

### The gate we lowered, and the flag that made us
`logit_parity_int8` failed in the ASan tree at 96.9% against a 98% bar, while
passing at 98.4% in Release. Rather than guess, we built Release with
`-ffp-contract=off` — and it reproduced the ASan numbers **digit for digit**. So
the whole spread is FMA contraction in `ops::linear`'s double accumulator, not
the sanitizer: three near-ties out of 192 land differently. The build with the
*worse* top-1 has the *better* rel-L2, which is the tell that top-1 over 192
samples is a noisy metric. The hard gate moved onto perplexity (stable to 0.05
points) and top-1 dropped to 96% with the reasoning written into the test. It
still catches the things it should: quantizing the lm_head scores 66%, QK=64
scores 94.8%.

Also worth recording: the ASan tree earned its keep here by finding a threshold
that only passed because of an optimisation flag.

### Written this session
`docs/01-roofline.md`, `docs/02-int8-saturation.md`, `docs/03-isa-spec.md`,
`docs/04-compiler.md`, `tools/run_sim.cpp` (the full-model gate, which a code
comment already claimed existed), `src/accel/reference.h` (one CPU reference
shared by the fast gate and the full-model tool, so they cannot drift), and
`bench/run_all.sh` + `bench/sim_schedules.sh`.

### Then an audit pass, with the tests already green
Green tests are the *start* of a review, not the end of one. Reading the
dependency masks against the emitted program turned up a third latent bug of the
same family as the first two:

**`kcur`, `vcur` and `logits` are read by a `DMA_STORE`, and the next thing to
write them waited only on `DEP_MXU | DEP_VPU` — never on `DEP_DMA_OUT`.** The
next layer's `k_proj` writes the same buffer the previous layer streamed to HBM.
Today the gap is a whole attention block plus an FFN, so it cannot bite; the
declared dependency did not cover it, which means only the schedule was keeping
it correct. Declaring it properly measured **zero extra cycles** (25,242,004
before and after, to the cycle) because the store has always long since retired
— the correct dependency was free, and only luck was making the wrong one work.
Stance 2 says the compiler owns placement; owning it means declaring it, not
observing that it happens to work.

Also fixed: **the simulator reported a misaligned SPM address as "out of
range"**, which is the wrong bug — a bad size calculation and a byte/float unit
mix-up have nothing to do with each other, and the ISA doc promises this error
class is loud *and* accurate because it is how a compiler bug surfaces. Each
operand now reports its own reason, with a test for the misalignment path.

**And we finally pointed ThreadSanitizer at the thread pool.** ASan does not
detect races, so every INT8 path had been running six threads with no race
checking at all. TSan on `test_gemm` (which sweeps MR × panel-block × prefetch ×
{1, all} threads) is **clean**. Worth writing down because it cost ten minutes:
TSan aborts immediately on this kernel with "unexpected memory mapping" — that
is ASLR entropy, not a bug in the code. `setarch $(uname -m) -R` fixes it.

And three numbers in the docs did not survive being checked against a fresh run
(per-layer array efficiency 2.87 → 2.84%, tile range 4–9 → 2–8, two rows of the
STREAM table mistranscribed). All corrected. Writing a number down is not the
same as measuring it, which is the same lesson as the busy-machine benchmarks,
one level up.

### What is REAL vs what is still FAKE

**Real, measured, and gated by a test that runs:**
- The fp32 forward pass. `logit_parity` holds it to 1.98e-05 against HuggingFace
  with 64/64 greedy tokens exact, and the bisection ladder agrees at every tap.
- The INT8 path: kernel, quantization, packing, threading. `test_gemm` asserts
  every kernel generation is bit-identical (21,501 checks), the saturation bound
  is proven *and* measured, and `logit_parity_int8` gates quality end to end.
- The T1 simulator and compiler. The real 0.5B, lowered and executed, is
  bit-identical to the CPU at every position — checked by two independent gates
  sharing one reference.
- Every number in `docs/01`–`04`. Re-measured on a quiet machine this session,
  raw output committed, and the ones that disagreed with a fresh run were fixed
  rather than kept.

**Real but narrower than it sounds:**
- "The model runs on the accelerator" means **decode, batch 1, fp32, one tile,
  no interconnect**. Prefill is not lowered. There is no INT8 datapath on T1.
- `run_sim` is checked against our own CPU path, not against HuggingFace. The
  chain to HF runs through `logit_parity`, which covers the same ops — but the
  T1 path is not *directly* anchored to an external oracle.
- The per-layer stall split charges a mixed dependency+structural stall entirely
  to the dependency class. The aggregate counters split them properly; the
  per-layer view is an approximation.

**Still fake or absent, and named so nobody trips over it:**
- **No llama.cpp comparison.** Until it exists, "our INT8 GEMM" has no external
  reference point, and the 24.9%-of-roof number cannot be put in context.
- **The last 2× in the micro-kernel is unexplained.** Not "probably X" —
  unattributed. The MR sweep already killed the obvious hypothesis.
- **The tokenizer pretokenizer is still the documented approximation** (exact
  for ASCII, tested against an HF dump).
- **M4–M7 do not exist.** `FaultConfig` and the trace buffer are wired in as
  hooks with no consumer; they are scaffolding, deliberately, and they are not
  evidence of a runtime or a profiler.

---

## 2026-07-24 — Session 2: M1 forward pass runs, model says "Paris"

### The headline
A from-scratch C++ Qwen2.5-0.5B forward pass runs end to end and generates
correct text: **"The capital of France is" -> "Paris. It is the largest city in
Europe and the third"**. A second prompt: "Once upon a time" -> "there was a
young man named John". Fluent, factual output is strong evidence every op is
right — a wrong RoPE convention or GQA head mapping yields garbage, not this.

### What we built
- **Tokenizer** (`src/tokenizer/bpe.{h,cpp}`): the real byte-level BPE — the
  256-byte->unicode map + inverse, the HF merge-loop algorithm, vocab/merges
  loaded from tokenizer.json. The ONE documented shortcut is the pretokenizer
  regex (unicode \p{L}/\p{N} are misery in C++): we hand-write a splitter that's
  exact for ASCII (all three oracle prompts are digit-free) and treats UTF-8
  multibyte as letters. Tested offline (synthetic fixture proves the merge loop)
  AND against the real 151936 vocab.
- **`ops::linear`**: HF stores nn.Linear.weight as [OUT,IN]; this is
  weight-stationary Y[o]=dot(X,W[o,:])+b[o], so we never transpose a projection.
- **KV cache** (`src/model/kv_cache.{h,cpp}`): one up-front slab, sized by
  num_key_value_heads (2), not attention heads (14) — the 7x GQA saving.
- **Qwen2 forward** (`src/model/qwen2.{h,cpp}`): embed -> 24x[rmsnorm -> QKV(+bias)
  -> rope -> GQA causal attention -> o_proj -> residual -> rmsnorm -> SwiGLU ->
  residual] -> final rmsnorm -> tied logits. Activations come from one Arena,
  reset per token; `high_water()` reports **79.1 KB/token** — the real number
  that will size M3's scratchpad. Two entry points from day one: prefill() and
  decode_step().
- **Config-driven loader** (per the flow diagram): conditional Q/K/V biases
  (load "when present"), conditional lm_head (load only when NOT tied). Nothing
  hardcoded that the config already states.
- **`run_infer`** driver ties tokenizer + config + model into greedy generation.

### The download saga (brutal, honest)
The model download was a fight. The naive `curl -o` **silently truncated** the
weights at ~72% (connection dropped; the file looked "done"). We caught it by
computing the byte-exact expected size (988,097,824) from the safetensors header
itself, then resumed. `curl -C -` failed too (HTTP/2 stream CANCEL on this
flaky link). Fixed with a `wget -c` retry LOOP in a script file (nested-quote
hell through PowerShell->WSL->bash kept breaking inline commands) that grinds
through drops until byte-exact. `fetch_model.sh` now has a size sanity gate so a
truncated model can never silently pass again. Lesson worth keeping: **verify
downloads against a known-exact size, never trust "it finished."**

### What is REAL vs still MISSING
- **Real & working:** tokenizer, KV cache, forward pass, run_infer. Build clean
  (zero warnings at -Wall -Wextra -Wshadow -Wconversion), ctest 3/3 green.
- **Still missing for M1 to be officially done:** the NUMERICAL oracle
  (logit parity < 1e-3 vs HuggingFace). Needs PyTorch (~2 GB, not installed).
  Coherent text is qualitative; the charter demands the numeric gate. That is
  the one and only remaining M1 task. See PROGRESS.md.

### Perf notes (reference path, expected-slow)
~32 s weight load (scalar bf16->f32 of 494M params), ~1.4 s/token scalar
forward. Fine for a reference/oracle; making it fast IS Milestone 2.

---

## 2026-07-24 — Session 1: bootstrap (M0) + M1 plumbing

### The reality we hit
The machine had **no C++ compiler at all** — no g++, clang, MSVC, cmake, ninja,
make. The bible assumes a fully set-up WSL Ubuntu box; that box did not exist.
So step one was not code, it was tools.

### What we did, in order

1. **Probed the machine.** Confirmed: Windows 11, Ryzen 5 PRO 5650U (Zen 3),
   15.3 GB RAM, 32 GB disk free. Git ✅, Python 3.13 ✅, compilers ❌.
   Only WSL distro present was Docker Desktop's.

2. **Kicked off two toolchain paths in parallel** so we weren't blocked:
   - `wsl --install -d Ubuntu-24.04` (the bible's reference target; needed later
     for io_uring in M5). Installed, but not yet initialized — first launch wants
     a UNIX username.
   - winget installs of **Ninja, CMake, LLVM (clang++)** for a Windows-native
     build we could use *today*. Our core code has `_WIN32` branches (mmap via
     `MapViewOfFile`) precisely so this works.

3. **Wrote the foundation — the "write once, never touch again" layer (§2.4):**
   - `align.h` (64-byte aligned alloc), `arena.h` (bump allocator with
     `high_water()` — the number that will size the M3 chip's scratchpad),
     `tensor.h` (non-owning 4-D view).
   - `bf16.h` — bf16→f32 is just a 16-bit shift; we wrote it instead of importing
     it, per the no-dependencies rule. Threw in a correct f16→f32 too.
   - `json.h` — a ~200-line hand-written JSON parser. The bible bans nlohmann;
     safetensors headers and config.json are a tiny JSON subset, so we own it.
   - `safetensors.{h,cpp}` — memory-maps the model file once (zero copy) and hands
     out views. The ONE copy we allow is bf16→f32 at load time.

4. **M1 plumbing:**
   - `config.{h,cpp}` — reads config.json, DERIVES all geometry. Encodes the two
     Qwen traps as first-class helpers: `q_per_kv()` (GQA) and the
     `tie_word_embeddings` flag (no separate lm_head).
   - `ops.{h,cpp}` — scalar reference rmsnorm / rope / softmax / silu_mul /
     gemm_ref. rmsnorm sums in **double** (the classic 1e-3 parity trap). rope
     uses the **split-half** convention Qwen needs, not interleaved.

5. **Tests that actually assert things:**
   - `test_core.cpp` — alignment, arena reset/overflow/high-water, bf16 bit
     patterns, JSON parse (nested, escapes, numbers), tensor strides.
   - `test_ops.cpp` — rmsnorm vs hand math, rope identity at pos 0 + norm
     preservation, softmax sums-to-1 + large-input stability, silu, gemm 2×3@3×2.
   All dependency-free (`test/check.h`), wired into CTest.

6. **Toolchain fight (the honest version).** The winget installs of CMake/LLVM on
   Windows **stalled** — two winget processes hung for 20+ minutes with empty logs,
   nothing installed (parallel Store installs contend; likely an invisible
   elevation wait). We killed them and pivoted to **WSL Ubuntu-24.04**, which had
   finished registering. Ran it as root (`-u root`, sidestepping the first-run user
   prompt) and installed g++ 13.3.0 / cmake 3.28.3 / ninja 1.11.1 via apt in one
   shot. This is the bible's target anyway — the detour cost time but landed us
   where we wanted to be.

7. **Built and tested for real.** Configured + built release and ASan+UBSan trees
   in WSL against `/mnt/c/.../SOHU`. **Both `ctest` runs: 2/2 passed, zero
   warnings, zero sanitizer complaints.** The core library is not just written, it
   is proven.

### What is REAL vs what is still FAKE

- **Real & tested:** the core library and both test binaries (once the compiler
  finishes installing and we run `ctest`).
- **Not done yet:** tokenizer, KV cache, the assembled forward pass, and the
  HuggingFace oracle. Without the oracle, "M1" is not M1 — it's just parts. That
  is the honest state and it's written down in `PROGRESS.md`.

### Why this order
The oracle needs the model download + PyTorch; the tokenizer is 3–4 days on its
own. Building the tested, portable core FIRST means the next session opens the
repo to green tests and a precise to-do list instead of a blank page — which is
the whole point for a long solo project.
