# 04 — The compiler: a transformer, lowered to T1

> Milestone 3. `src/accel/lower.cpp` turns a `ModelConfig` and a weight file into
> one straight-line T1 program. This document is the prose version of its passes,
> and every number in it comes from `tools/run_sim` on the real Qwen2.5-0.5B or
> from `test/compiler_parity.cpp`.

**Status: the M3 gate passes on the real model.** Qwen2.5-0.5B, lowered to the
T1 ISA and run on the simulator, produces logits **bit-identical** to the CPU
fp32 path at every position, and answers "The capital of France is" with
" Paris."

```
$ ./build/rel/tools/run_sim --steps 1
HBM layout: 494382208 floats (1977.53 MB), 24 layers, max_seq 32
pos  0  BIT-EXACT   4461 instrs   25242004 cycles (25.24 ms @1GHz)  MAC util 66.4%  DMA stall 33.6%
  tiling: 169 matmuls -> 1052 tiles, 1975.8 MB of weights streamed,
          SPM high-water 4885824 B of 8388608 B
...
M3 GATE: every logit bit-identical to the CPU fp32 path.
```

---

## 1. Why there is no graph IR

A transformer forward pass is a straight line. There is no control flow to
analyse, no operator fusion search worth doing at this scale, and no unknown
shapes — everything is derivable from `config.json` and the position. A lowering
that walks that line directly is ~550 lines, and every one of them is about the
accelerator rather than about graph plumbing.

The cost of that decision is real and worth stating: there is no place to hang a
generic optimisation, and a model with a different topology needs new lowering
code rather than a new pattern. The bet is that for one architecture family
that trade is overwhelmingly correct, and the evidence is that a decode step
compiles in **under 10 ms** — the compile time does not appear in any
measurement in this document because it is unmeasurably small next to a 25 ms
simulated token.

## 2. The passes

### Pass 1 — HBM layout

`HbmLayout::build` assigns every tensor a fixed address in one flat fp32 space,
deterministically from `ModelConfig` (order and map in `docs/03` §7). For
Qwen2.5-0.5B that is **1977.53 MB**, and the same model always produces the same
addresses, so two builds of a program can be diffed.

This pass is the entire "port a new model" surface. Together with the tensor-name
mapping in `tools/run_sim.cpp`, it is what porting means here.

### Pass 2 — Tiling

Every projection is `Y[1, OUT] = X[1, IN] · W[OUT, IN]ᵀ (+ bias)`. The
activation is hidden-sized (kilobytes, resident); the weight is streamed in tiles
of whole output rows, because the MXU consumes B as `[n, k]` row-major.

```
rows_per_tile = clamp(weight_bank_bytes / (IN * 4), 1, OUT)
```

The tiler is not optional cleverness — it is what makes the model runnable at
all. `up_proj` is 4864 × 896 fp32 = **17.4 MB against an 8 MB scratchpad**. It
cannot be resident, ever. The real decisions it makes:

```
L0.q_proj    [896->896]   3136 KB -> 2 tiles of 585 rows (2047 KB/bank)
L0.k_proj    [896->128]    448 KB -> 1 tile  of 128 rows
L0.gate_proj [896->4864]  17024 KB -> 9 tiles of 585 rows (2047 KB/bank)
L0.down_proj [4864->896]  17024 KB -> 9 tiles of 107 rows (2033 KB/bank)
```

Note `down_proj`: same 17 MB of weights, same 9 tiles, but **107 rows per tile
instead of 585**, because its reduction dimension is 4864 rather than 896 and a
bank holds a fixed number of *bytes*, not rows. The tiler derives that from the
shape rather than being told. Across the whole model: **169 matmuls become 1052
tiles**, streaming **1975.8 MB** per token.

This log is emitted by design (`LowerReport::tiling_log`). Porting a model means
reading it and checking the tiler did something sane, which is the "programming
abstractions to rapidly iterate on model porting" artifact in concrete form.

### Pass 3 — SPM allocation

A bump allocator with mark/release over a straight-line program. The persistent
activation region is allocated **once**, outside the layer loop, and reused by
all 24 layers; each projection's two weight banks are carved out and released
around it.

That works because M1 already proved the per-token activation working set is
tiny and identical every layer — 79.2 KB measured by the arena's `high_water()`.
Measured peak on the real model: **4,885,824 B of 8,388,608 B**, i.e. **58% of
the scratchpad**, nearly all of it the two 2 MB weight banks plus the 608 KB
logit buffer. The M1 arena discipline paying off here was a prediction made in
milestone 1; this is the number that confirms it.

### Pass 4 — DMA scheduling (software pipelining)

Prologue-loads tiles 0 and 1 into alternating banks, then for each tile: issue
the MATMUL, then prefetch tile t+2 into the bank that matmul just drained.

```
LOAD t0 -> A
LOAD t1 -> B
MATMUL t0 (bank A)      LOAD t2 -> A   flies while MATMUL t1 runs
MATMUL t1 (bank B)      LOAD t3 -> B
...
```

**The order matters and getting it wrong is silent.** The first version of this
pass emitted the prefetch *before* the matmul, so the MXU was handed tile t+2's
weights while computing tile t. That is a write-after-read violation, and
because the simulator's functional layer executes in program order it produced
wrong logits **deterministically** — the same wrong answer every run, no
crash, no warning. `compiler_parity` caught it as 48 of 48 logits differing.

The anti-dependency that makes it safe is `DEP_MXU` on **every** weight load,
including the two prologue loads. That last part is subtler: the banks live at
fixed SPM offsets reused by every projection, so this projection's prologue load
would otherwise land on top of the *previous* projection's still-executing tail
matmul. Program order does not save you — the front end issues and moves on,
which is the whole point of the machine.

And this one is **invisible to the simulator**: its functional layer executes at
issue time, so an anti-dependency violation shows up as wrong timing, never as
wrong values. It has to be got right by reasoning. The cost — no DMA/MXU overlap
across a projection boundary — is real and is paid deliberately.

### Pass 5 — Encode

Flatten to `Instr[]`. A decode step for the real model is **4461 instructions**,
carrying a provenance string each (`"L7.mlp.up_proj matmul tile 3"`) that the
disassembler, the per-layer report, and M7's trace all key off.

## 3. The double-buffering study — and the planted flaw, quantified

Four schedules, the real model, position 0. All four produce **bit-identical**
logits; correctness is invariant to scheduling, which is what a correct set of
dependency masks means. (`bench/sim_schedules.sh`, output in
`bench/results/sim_schedules.txt`.)

| schedule | cycles | ms @1GHz | MXU occupancy | DMA stall |
|---|--:|--:|--:|--:|
| serial, coarse deps | 25,242,004 | 25.24 | 66.4% | 33.6% |
| serial, fine deps | 25,242,004 | 25.24 | 66.4% | 33.6% |
| double-buffered, **coarse** deps *(the ISA as specified)* | 25,242,004 | 25.24 | 66.4% | 33.6% |
| double-buffered, **fine** deps | **18,361,229** | **18.36** | **91.3%** | **10.6%** |

Software pipelining is worth **27.3% of the runtime**, taking DMA stall from
33.6% to 10.6% and MXU occupancy from 66.4% to 91.3%.

**And under the ISA as actually specified it is worth exactly nothing — the same
cycle count, to the cycle, as not doing it at all.**

That is ISA stance 3 (`docs/03` §1) presenting its bill. The coarse scoreboard
gives both SPM banks one dependency class, so a MATMUL waiting on bank A also
waits on the in-flight load into bank B — which is precisely the load that
double buffering exists to overlap. The optimisation is not degraded, it is
**annihilated**, and the three identical rows in that table are what
annihilation looks like in a measurement.

The flaw stays in. M7's job is to find it in a trace and fix it, and the
before/after is already sitting here waiting: 25.24 ms → 18.36 ms, one bit in a
dependency mask. `test/sim_isa.cpp` isolates the same effect down to a
four-instruction program where the false dependency costs exactly 400 cycles.

## 4. What the simulator says about T1 itself

The per-layer report (`tools/run_sim`) for one decode step:

| region | instrs | cycles | MXU busy | array eff. | DMA MB | stall DMA | stall MXU |
|---|--:|--:|--:|--:|--:|--:|--:|
| prologue + lm_head | 525 | 6,673,108 | 66.5% | 3.00% | 545.2 | 33.5% | 66.5% |
| each of 24 layers | 164 | 773,704 | 66.3% | 2.84% | 59.7 | 33.6% | 66.3% |
| **total** | **4461** | **25,242,004** | 66.4% | **2.88%** | 1975.8 | | |

The layers are identical to the cycle, which is the expected shape of a program
with no data-dependent control flow, and a useful smoke test: an outlier layer
would mean the compiler treated one differently.

**The output projection is 26% of the whole decode step** (6.67 M of 25.24 M
cycles) and 28% of the DMA traffic, to produce 151,936 logits of which one is
used. Same finding as `docs/01` §"where the bytes go", from a completely
different measurement — which is the useful kind of agreement.

### The number that reframes the milestone

`MAC utilisation 66.4%` looks respectable and is nearly meaningless on its own.
The 32 × 32 array is fed `m = 1`, so **31 of its 32 rows are idle**:

- **array occupancy: 66.4%** — the fraction of cycles the MXU was busy;
- **array efficiency: 2.88%** — the fraction of the MACs it *could* have retired
  while busy that were real work.

Both are printed together, deliberately, so neither can be quoted without the
other.

This changes what T1's decode bottleneck actually is. The weight stream alone
needs 1975.8 MB / 256 B per cycle = **7.72 M cycles**; we spend 25.24 M. On this
chip, at batch 1, decode is **3.3× above the memory floor and limited by the
shape of the systolic array**, not by bandwidth and not by the array's size.

That is a different answer from the laptop's, and it is not a contradiction —
it is the same fact seen through different hardware. A CPU at M=1 runs a GEMV at
memory bandwidth. A systolic array at M=1 runs a GEMV at 1/32 of its width. The
laptop is bandwidth-starved; T1, having fixed bandwidth by design (256 GB/s and
an explicit scratchpad), immediately exposes the *next* constraint underneath.

And the fix is not a better schedule. `matmul_cycles(m, n, k)` rounds `m` up to
32, so **a batch of 32 tokens should cost the same cycles as a batch of 1** —
32× the work for free.

That was a prediction when this section was first written. §5 went and collected
it: **31.7×, measured, bit-exact.** Batching stops being a throughput nicety and
becomes the highest-value thing the machine is asking for.

## 5. Prefill: the 32× that was sitting in the timing model

§4 ended by observing that `matmul_cycles` rounds `m` up to the array
dimension, so a batch of 32 tokens should cost about the same as a batch of 1.
That was a prediction. `lower_prefill` is the experiment, and this is the
result — same machine, same weights, same 32 tokens, **verified bit-identical**:

| | 32 decode steps | 1 prefill chunk | ratio |
|---|--:|--:|--:|
| cycles | 808,077,440 | **25,466,046** | **31.7×** |
| time @ 1 GHz | 808.08 ms | 25.47 ms | |
| instructions | 142,752 | 5,901 | 24.2× |
| DMA traffic | ~63.2 GB | **1.99 GB** | 31.8× |
| array efficiency | 2.88% | **67.77%** | 23.5× |
| last token's logits | — | **bit-exact vs decode** | |

**Thirty-two tokens for the price of one, and the arithmetic is identical to the
last bit.** The weights are streamed once instead of thirty-two times, which is
where the DMA ratio comes from, and the systolic array stops running 31 of its
32 rows empty, which is where the cycle ratio comes from.

### Why 67.8% and not 96%

A projection tile at M=32 is `matmul_cycles(32, 585, 896)` = 17,088 cycles for
16,773,120 MACs against a capacity of 17,498,112 — **95.9% efficient**. The
whole-chunk figure is 67.8% because of one instruction: **the output projection
still runs at m=1**, and it is 26% of the cycles.

That is correct, not a missed optimisation. Prefill exists to fill the KV cache;
only the *last* token's logits are ever read, and producing the other 31 rows
would cost a 545 MB sweep of the embedding matrix each to compute distributions
nobody samples. So the residual inefficiency is inherent to what prefill is for,
and it is concentrated in exactly one place — which is a much better position
than having it smeared across every matmul in the model.

### What had to change

Three things, and only the first was expected:

1. **The projections take an `M`.** `X` is `[M, IN]`, `Y` is `[M, OUT]`, and each
   tile computes a *column slice* of `Y`. MATMUL writes `C` packed as `[m, n]`,
   which at M=1 happens to be exactly the slice it belongs to and at M>1 is not.
   So a tile writes a packed scratch and a 2-D strided `V_COPY` places it —
   about 3% of the tile's matmul cycles, and the price of not putting an output
   stride on the systolic array's writeback port. Single-tile projections skip
   it: `C` is already the whole of `Y`.

2. **Causal masking became a hardware feature.** `V_SOFTMAX` gained rows and a
   causal ramp that zeroes past each row's boundary — which the bible's original
   ISA sketch had already called for (*"causal handled by cols per row"*). The
   zeroing is what lets a single `[M, KEYS] · [KEYS, HD]` matmul do the AV
   product for every query row at once, because the masked entries contribute
   exactly `0.0` and adding `0.0` to a double accumulator is a no-op. That last
   detail is why the reference can sum only the valid prefix and still match bit
   for bit.

3. **The bias needed broadcasting.** With `kFlagAccumulate` seeding the
   accumulator from `C`, the bias has to appear in all M token rows. A DMA
   descriptor with **`row_stride = 0`** re-reads the same OUT floats M times —
   a broadcast that costs the engine nothing and needed no new opcode.

### The scratchpad sets the chunk size, and that is the real constraint

The activation working set scales with M (~81 KB per token for this model —
the same 79.2 KB the M1 arena measured, plus attention scratch), against a fixed
8 MB scratchpad shared with two 2 MB weight banks. `max_prefill_tokens()`
computes the bound from the same arithmetic the allocator uses:

**42 tokens for Qwen2.5-0.5B on T1.** At M=32 the high-water is 7,421,952 B of
8,388,608 B — 88% full.

So the bible's prefill buckets of {128, 512, 2048} are *not reachable on this
machine*, and no amount of scheduling changes that: it is 8 MB against 81 KB per
token. A long prompt is several chunks, which is exactly what production systems
call chunked prefill, and `lower_prefill` takes a `base` so chunks compose. The
test covers 5 tokens as one chunk, as 3+2, and as five chunks of 1 — all three
bit-identical to each other and to decode.

This is the most useful thing the simulator has said so far, because it is a
*design* answer rather than a tuning one: **if you want longer prefill chunks on
T1, you buy scratchpad, not MACs.**

## 6. Testing

Two gates, deliberately different in cost:

- **`test/compiler_parity.cpp`** — a tiny synthetic model (hidden 64, 2 layers,
  GQA with `q_per_kv = 2`, QKV biases, SwiGLU, tied output projection) with the
  bank budget shrunk to 4 KB so even that model needs 2–8 tiles per projection.
  Runs in ~40 ms, so it gates every commit, and it asserts bit-exactness at five
  positions plus identical logits across all four schedules. A gate you skip is
  not a gate.
- **`tools/run_sim`** — the real 0.5B, 2 GB HBM image, ~0.5 s of simulation per
  token. Opt-in, because a 5-minute gate would get skipped.

Both check against the *same* CPU reference (`src/accel/reference.h`), so they
cannot drift into testing different semantics. That reference is written
directly against `ops::` and shares no code with the compiler — a reference that
shares code with the thing it checks proves nothing.

## 7. What the compiler does not do

- **No whole-prompt prefill.** Chunks are capped at 42 tokens by the scratchpad
  (§5), so a long prompt is several programs. Composing them is the caller's job
  today; scheduling them against a KV cache is M4's.
- **No batching across *requests*.** `lower_prefill` batches tokens of one
  sequence. Continuous batching — many sequences at different positions sharing
  one matmul — is M4, and §5 is its justification.
- **No k-splitting.** Tiles split the output dimension only. A projection whose
  *reduction* dimension exceeds a bank would need it; none in this model does
  (`down_proj`'s 4864 still fits at 107 rows per tile). `kFlagAccumulate` exists
  and is specified to make it value-exact when it arrives — and prefill now
  exercises that seeding path for every biased projection, so the rounding
  contract is tested rather than merely written down.
- **No SPM residency across layers.** Every layer re-streams its own weights,
  which is correct for decode (they are read once each anyway) and would be
  wasteful for prefill.
- **No instruction scheduling beyond the DMA pipeline.** Vector ops are
  serialised against each other by a blanket `DEP_VPU`, which is conservative.
  The measured VPU contribution is small enough that this has not been worth
  revisiting — but "small enough" here means "not measured separately", and that
  is the honest state of it.
