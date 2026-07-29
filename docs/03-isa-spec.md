# 03 — The T1 ISA: specification

> Milestone 3. This is the normative spec for a fixed-function transformer
> accelerator that does not exist. `src/accel/isa.h` is the machine-readable half
> of it; **if the two disagree, that is a bug in one of them.** Every timing
> formula here is the one `src/accel/simulator.cpp` implements, and
> `test/sim_isa.cpp` asserts the numbers in the worked examples by hand.

---

## 1. What T1 is, and what it refuses to be

T1 is one core ("tile") that runs one transformer forward pass. It has no
registers, no branches, no caches, no coherence protocol, and no operating
system. It has a systolic array, a vector unit, a scratchpad, one DMA engine,
and an in-order front end with a five-bit scoreboard.

Three design stances generate everything else. Each is a bet, and each is
falsifiable.

### Stance 1 — No registers, no branches, no loops

A transformer forward pass is a straight line. The shapes come from
`config.json`, the layer count is fixed, and nothing is data-dependent except
the position — and position becomes a compile-time constant the moment you
bucket sequence lengths. So the compiler unrolls **everything** and a program is
a flat array of instructions executed from index 0 to `HALT`.

What this buys: no branch predictor, no register renaming, no instruction cache
hierarchy, no reorder buffer. That area becomes MACs.

What it costs: program size grows with sequence length, and every distinct
position needs its own program (or a program keyed on a bucketed length).
Measured, for Qwen2.5-0.5B: one decode step is **4461 instructions = 143 KB**,
streamed once against 1976 MB of weights — 0.007% overhead. Cheap here. For a
128 K-context model it would not be, and that is a real limit of the design
rather than an oversight.

### Stance 2 — Software-managed memory

There are no caches. Every byte that crosses between HBM and the scratchpad does
so because a `DMA_LOAD` or `DMA_STORE` said so, at an address the compiler
computed. Data placement is a **correctness** property, not a performance one:
get it wrong and you read stale weights, not slow ones.

This single decision generates most of the interesting compiler work — tiling,
scratchpad allocation, double buffering — and it is why an accelerator can hold
high utilisation where a cache-based CPU cannot. The machine never speculates,
never evicts a line it still needed, and never pays read-for-ownership on a line
it is about to overwrite whole. (That last one is not hypothetical: it is
literally the 25.9 → 30.2 GB/s gap measured on the laptop in `docs/01`.)

### Stance 3 — Coarse scoreboard classes, deliberately

Dependencies are a five-bit mask of *unit classes*, not per-buffer dataflow.
This is cheap in hardware and it is **wrong in a specific, interesting way**:
two independent DMA streams share a class and serialise against each other even
when they touch disjoint memory.

The flaw is left in on purpose. M7's profiler is supposed to rediscover it from
a trace, and "I found it in the trace, here is the before/after" is the reason
for building a profiler at all. It is already quantified twice — once in
`test/sim_isa.cpp` as a 400-cycle false dependency, and once in
`test/compiler_parity.cpp` where it erases the entire benefit of double
buffering. See `docs/04-compiler.md`.

---

## 2. Machine parameters

From `T1Config` (`src/accel/isa.h`), which is the single source these numbers
come from, so the docs and the simulator cannot drift:

| Unit | Parameter | Value |
|---|---|---|
| MXU | systolic array | 32 × 32 fp32 MACs, 1 MAC/cell/cycle |
| VPU | vector lanes | 32 fp32 |
| SPM | scratchpad | 8 MB, 2 banks, software-managed |
| DMA | engines | **1**, 256 B/cycle sustained, 400-cycle fixed latency, 2-D strided |
| SEQ | front end | in-order, 1 instruction/cycle, 5-class scoreboard |
| — | clock | 1 GHz nominal (so 1 cycle = 1 ns) |

### Derived roofs

- **Compute:** 32·32·2 = **2048 flop/cycle = 2.048 TFLOP/s**
- **Bandwidth:** 256 B/cycle = **256 GB/s**
- **Machine balance:** 2048/256 = **8 flop/byte**

Decode reads each fp32 weight once and does two flops with it: **0.5 flop/byte**,
sixteen times below balance. **Decode is memory-bound on T1 by design, exactly as
it is on the laptop.** Same thesis, new altitude — and this time the numbers were
chosen deliberately rather than sampled from a store shelf, which is what makes
them arguable: a 32×32 array against 256 B/cycle is a machine that admits its
workload is bandwidth-starved and spends its area accordingly.

### Why 8 MB of scratchpad

Two measurements pin it:

- The per-token **activation** working set of the real model is **79.2 KB**,
  measured by the M1 arena's `high_water()` (`tools/run_infer` prints it). That
  is 1% of the scratchpad. Activations are not the problem.
- The FFN `up_proj` weight for Qwen2.5-0.5B is 4864 × 896 fp32 = **17.4 MB**,
  which does not fit in 8 MB and never will. So weight **tiling is mandatory**,
  not an optimisation — the machine cannot run the model without it.

8 MB is therefore the smallest size that makes the interesting problem
unavoidable while leaving room for two weight banks plus the activation set.

---

## 3. Instruction format

A fixed 32-byte record. No variable length, no prefixes, no const pool
indirection.

```
 byte 0      1        2..3          4..27            28..31
+--------+--------+-----------+------------------+-----------+
|  op    | flags  | dep_mask  |  arg[0..5] u32   |   imm_f   |
+--------+--------+-----------+------------------+-----------+
```

- `op` — the opcode (§4).
- `flags` — per-op bits (§3.2).
- `dep_mask` — the dependency classes this instruction waits on (§5).
- `arg[6]` — operands. **SPM addresses are byte offsets** (32 bits is ample for
  8 MB). **HBM addresses need more than 32 bits**, so they occupy an aligned
  *pair* of slots, low word first (`hbm_addr(i)` / `set_hbm_addr(i, a)`).
- `imm_f` — the one float immediate: `eps`, RoPE `theta`, or a scale factor.

`static_assert(sizeof(Instr) == 32)` holds the layout.

### 3.1 A program

```cpp
struct Program {
    std::vector<Instr>       code;   // execution starts at 0, ends at HALT
    std::vector<std::string> notes;  // provenance, e.g. "L3.up_proj matmul tile 2"
};
```

There are no entry points, no call stack, and no relocation. `notes` is **not
part of the ISA** — it is what makes the disassembly and the M7 trace legible,
and it is the "programming abstractions for rapid model porting" artifact.

### 3.2 Flags

| Bit | Name | Applies to | Meaning |
|--:|---|---|---|
| 0 | `kFlagAccumulate` | MATMUL | Seed the accumulator from `C` instead of zero — **in the wide format**. See §4.3. |
| 1 | `kFlagBankB` | DMA_LOAD | This load's completion satisfies `DEP_DMA_B`, not `DEP_DMA_A`. |
| 2 | `kFlagBIsKN` | MATMUL | The `B` operand is laid out `[k, n]` instead of the default `[n, k]`. |

`kFlagBIsKN` deserves its justification here rather than in a code comment. A
systolic array is fed from two edges; which edge a matrix enters by is a **wiring
choice, not a data movement**, so consuming both orientations costs the hardware
nothing. Exposing it saves the compiler a great deal: attention needs `Q·Kᵀ`
(B as `[n,k]`) and then `scores·V` (B as `[k,n]`) back to back on the same
buffer. Without the flag the compiler would materialise a transposed copy of V
in the scratchpad for every head of every layer — pure DMA traffic in the one
place the machine can least afford it.

---

## 4. Opcodes and semantics

`NOP`, `BARRIER`, `HALT` do no arithmetic. `BARRIER` waits for its `dep_mask`
and retires; `HALT` ends the program (and the run's cycle count is its issue
cycle + 1).

All SPM operands are byte offsets and must be 4-byte aligned and in range; a
violation is `SimStatus::BAD_PROGRAM` with a message naming the instruction, not
a silent scribble. Under stance 2 that error class *is* how a compiler bug in
scratchpad allocation announces itself, so it must be loud.

### 4.1 DMA_LOAD / DMA_STORE

| arg | meaning |
|--:|---|
| 0,1 | HBM address (floats, 64-bit pair) |
| 2 | SPM byte offset |
| 3 | `rows` |
| 4 | `cols` |
| 5 | `row_stride` (floats, HBM side) |

`rows` runs of `cols` floats. The HBM side advances by `row_stride` per row; the
**SPM side is packed**. This is what lets the compiler gather a strided tile —
say one KV head's rows out of an interleaved cache — without the host repacking
anything, and it costs the DMA engine nothing because it walks descriptors
anyway.

`DMA_LOAD` satisfies `DEP_DMA_A` or `DEP_DMA_B` (per `kFlagBankB`);
`DMA_STORE` satisfies `DEP_DMA_OUT`. Both occupy the **same physical engine**.

### 4.2 The vector ops

| op | arg0 | arg1 | arg2 | arg3 | imm_f |
|---|---|---|---|---|---|
| `V_RMSNORM` | src | weight | dst | len | eps |
| `V_SOFTMAX` | src (in place) | len | — | — | — |
| `V_ROPE` | src (in place) | head_dim | pos | heads | theta |
| `V_SILU_MUL` | gate | up | dst | len | — |
| `V_ADD` | src | dst (`dst += src`) | len | — | — |
| `V_COPY` | src | dst | len | — | — |
| `V_SCALE` | src | dst | len | — | factor |

The arithmetic is specified to the last bit, because bit-exactness against the
CPU path is the M3 gate:

- **RMSNORM** accumulates the sum of squares in **double**, then
  `inv = float(1/sqrt(mean + eps))`, then `y[i] = x[i] * inv * g[i]` in that
  association. (The double accumulator is the classic 1e-3 parity trap.)
- **SOFTMAX** subtracts the max, exponentiates in float, sums in **double**, and
  multiplies by `float(1/sum)` — it does not divide.
- **ROPE** uses the **split-half** convention (Qwen), not interleaved:
  `v[i], v[i+half]` rotate together, with the angle computed in double and the
  sin/cos rounded to float before use.
- **SILU_MUL** is `(x / (1 + exp(-x))) * up[i]`, all in float.

These are character-for-character the formulas in `src/ops/ops.cpp`. That is not
laziness; it is the specification. A simulator that is merely *close* to the CPU
path cannot tell you whether your compiler's tiling is correct.

### 4.3 MATMUL

| arg | meaning |
|--:|---|
| 0 | A, SPM byte offset — `[m, k]` |
| 1 | B, SPM byte offset — `[n, k]`, or `[k, n]` with `kFlagBIsKN` |
| 2 | C, SPM byte offset — `[m, n]` |
| 3,4,5 | `m`, `n`, `k` |

`C[i][j] = fp32( seed + Σ_p double(A[i][p]) · B[...] )`, where `seed` is
`double(C[i][j])` when `kFlagAccumulate` is set and `0.0` otherwise.

**The rounding is part of the contract.** The accumulation is double and there is
**one** rounding, at the end — not one per MATMUL. A cell that rounded its
partial before accumulating would make a k-split matmul answer differently from
an unsplit one, i.e. would make the compiler's tiling decisions visible in the
numerics. Tiling must be value-invariant, and `test/compiler_parity.cpp` asserts
exactly that by running four different schedules and demanding identical bits.

This also gives **bias for free**: DMA the bias vector into `C`, then MATMUL with
`kFlagAccumulate`, and the machine computes `fp32(bias + Σ x·w)` — which is
precisely what `ops::linear` does when it seeds its accumulator with the bias.
The obvious alternative (matmul, then a `V_ADD` of the bias) rounds twice and
disagrees with the CPU in the last ulp. That is not a hypothetical: it is the bug
that made `compiler_parity` fail before this contract was written down.

---

## 5. The scoreboard

Five dependency classes:

| bit | class | satisfied by |
|--:|---|---|
| 0 | `DEP_DMA_A` | inbound DMA into bank A |
| 1 | `DEP_DMA_B` | inbound DMA into bank B |
| 2 | `DEP_DMA_OUT` | outbound DMA |
| 3 | `DEP_MXU` | MATMUL |
| 4 | `DEP_VPU` | any vector op |

An instruction issues at the **latest** of: program order; every class in its
`dep_mask` being satisfied; and its own physical unit being free.

**Classes are about dependence; units are about structure, and they are not the
same set.** There are four units — `DMA_IN`, `DMA_OUT`, `MXU`, `VPU` — and both
inbound DMA classes map onto the *single* `DMA_IN` engine. Two loads tagged to
different banks are independent for correctness but still take turns on the wire.
The simulator models the two effects separately, and that is the only reason its
stall attribution means anything:

- `stall_dep_*` — waiting for a **result**. Fix: schedule it earlier.
- `stall_struct_*` — the result was ready, the **unit** was busy. Fix: nothing.
  You are at the roof; scheduling will not help.

Collapsing both into one "DMA stall" figure is how you spend a week software-
pipelining a loop that was already bandwidth-bound.

---

## 6. Timing model

Cycle-**approximate**, value-**exact**. The functional layer computes real
results immediately; the timing layer accounts cycles; **neither feeds the
other**. Values never depend on the schedule (see §4.3), and that invariant is
what makes the M1 oracle a valid gate on compiled code.

```
dma_cycles(bytes)      = 400 + ceil(bytes / 256)

matmul_cycles(m,n,k)   = (ceil32(m)/32) * (ceil32(n)/32) * ceil32(k)   [steady]
                       + 64                                            [fill+drain]

vpu_cycles(elems,fixed) = fixed + ceil(elems / 32)
    V_RMSNORM  fixed=8, elems=len        V_SOFTMAX  fixed=8, elems=3*len
    V_ROPE     fixed=4, elems=head_dim*heads
    V_SILU_MUL fixed=4, elems=len        V_ADD/COPY/SCALE fixed=1, elems=len

front end: 1 instruction issued per cycle
```

Dimensions round **up** to the array: a 3×3 matmul occupies the MXU exactly as
long as a 32×32 one. That is not a modelling shortcut, it is what the hardware
does — and leaving the cost visible is what stops the compiler from emitting
skinny tiles and calling it a win.

### Worked examples (asserted in `test/sim_isa.cpp`)

1. **A 256 B load.** `400 + 256/256 = 401` cycles busy. Issue at 0, HALT waits on
   everything and issues at 401, so the run is **402 cycles**.
2. **A 1024-float load feeding a 1×32×32 matmul.** The load is 4096 B →
   `400 + 16 = 416`. The MXU takes `(1)·(1)·32 + 64 = 96`, running 416→512. HALT
   issues at 512: **513 cycles**.
3. **The planted flaw.** Two 256 B loads into different banks, then a consumer of
   bank A. The loads serialise on the one engine (0→401, 401→802). With a fine
   mask the MATMUL issues at **402**; with the coarse mask it waits for bank B
   and issues at **802** — 400 cycles late for no reason. End to end the run
   costs only 96 cycles more, because HALT drains bank B either way. **Both
   numbers are asserted**, because quoting the 400 as if it were the run's cost
   would be precisely the unattributed stall accounting this simulator exists to
   prevent.

---

## 7. Memory map

`HbmLayout::build` assigns one flat fp32 address space, deterministically from
`ModelConfig`, in this order:

```
embed [vocab, hidden]
for each layer:
    in_ln, q_w, q_b, k_w, k_b, v_w, v_b, o_w, post_ln, gate_w, up_w, down_w
final_ln
kv_k  [layers, max_seq, kv_dim]     kv_v [layers, max_seq, kv_dim]
act_in  [hidden]      <- the host writes the token embedding here
act_out [vocab]       <- the program writes logits here
```

Determinism is the point: the same model always produces the same addresses, so
two builds of a program can be diffed. The KV cache lives in **HBM, not the
scratchpad** — at `max_seq` it is far too large to be resident, and it is exactly
the traffic that makes decode memory-bound on T1 just as on the CPU.

Porting a new model means writing a new layout, not touching the compiler.

---

## 8. Errors and fault injection

`SimStatus` is `OK`, `BAD_PROGRAM`, or `DEADLINE_EXCEEDED`.

`BAD_PROGRAM` covers an SPM or HBM address out of range, a misaligned SPM
address, non-positive dimensions, an unknown opcode, and a program that runs off
the end without `HALT`.

`FaultConfig` injects two failures that M4's runtime will need:

- `stuck_dma_at_instr` — that load never lands *and* occupies the engine
  forever, so everything downstream hangs. The failure looks exactly like a real
  one instead of like an exception.
- `deadline_cycles` — the run aborts once the simulated clock passes it. This is
  the mechanism M4 turns into per-request fault isolation.

---

## 9. What the simulator records

Per run: cycles, instructions, MAC ops, per-unit busy cycles, bytes in/out, and
six stall counters (dependency and structural, for DMA/MXU/VPU). Derived:
`mac_utilisation()` — the fraction of the run in which the systolic array was
doing arithmetic, which is the single number that says whether a schedule is any
good — and `dma_stall_fraction()`.

With tracing enabled it also keeps a per-instruction timeline (`issue_cycle`,
`end_cycle`, `stall_cycles`, and **which class was to blame**). Nothing
visualises it yet; M7 will. It is recorded from day one because retrofitting
per-stall provenance into a simulator later means re-deriving it from a log, and
because the trace and the counters can never disagree if they come from the same
place.

---

## 10. What this spec does not have

Stated plainly, so nobody has to discover them by reading the source:

- **No prefill programs.** Only `lower_decode_step` exists; `m` is always 1. The
  ISA supports `m > 1` and the simulator times it, but nothing emits it yet.
- **No INT8 datapath.** T1 is fp32 end to end. The M2 measurements say an INT8
  array would be the single biggest win available, and the extrapolated
  `vpdpbusd` number in `docs/01` sizes it — but bit-exactness against the fp32
  oracle is what makes M3 checkable, and having both at once was not worth the
  loss of that gate this milestone.
- **No multi-tile / no interconnect.** M5.
- **No instruction fetch model.** Programs are assumed resident; the front end
  never stalls on code.
- **One clock domain, no DVFS, no power model.**
