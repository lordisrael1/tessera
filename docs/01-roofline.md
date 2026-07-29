# 01 — The roofline: why decode is memory-bound

> Milestone 2. Every number here was measured on the target laptop by code in
> `bench/`, and the raw output is committed in `bench/results/`. Nothing is
> quoted from a spec sheet unless it is labelled *theoretical*.

**The claim:** transformer decode on this machine is starved of memory
bandwidth, not of arithmetic — by a factor of about thirty. Everything after
this milestone (a custom ISA with an explicit scratchpad, then collectives) is a
response to that one measurement.

---

## A word on methodology, before any numbers

The first version of these benchmarks was run while a compile was happening in
another window. The compute roof came out **45% low** (fp32 230 GFLOP/s against
431 on a quiet machine) and one GEMM point read 35.9 GOP/s against 64.4 for the
identical binary. Nothing about the runs looked wrong; the CSVs were plausible,
internally consistent, and useless.

Everything below was therefore re-measured in one pass by `bench/run_all.sh` on
an idle machine, so the ceilings and the points that are compared against them
come from the same conditions. **A benchmark taken on a busy machine is worse
than no benchmark: it looks authoritative and it is wrong.** Run-to-run
variation on a quiet machine is about ±5% (two consecutive INT8 roof
measurements: 995 and 1048 GOP/s), and no claim here rests on a margin smaller
than that.

## The machine

HP EliteBook 845 G8 — AMD Ryzen 5 PRO 5650U (Cezanne, Zen 3), 6 cores / 12
threads, AVX2 + FMA3, **no AVX-512 and no VNNI**. L1d 32 KB, L2 512 KB per core,
L3 16 MB shared. 16 GB DDR4-3200 dual channel, 51.2 GB/s *theoretical*. 15–25 W.

The two missing ISA features matter later and are worth naming now: without VNNI
an INT8 multiply-accumulate costs three instructions instead of one, and with 16
vector registers instead of 32 the GEMM micro-kernel can keep only a couple of
activation rows resident. Both show up as measured gaps below.

## Ceiling 1 — memory (`bench/stream_bench.cpp`)

STREAM triad (`a[i] = b[i] + s*c[i]`) over 192 MiB arrays — 12× L3, so every
access is DRAM. Full table in `bench/results/stream.txt`.

| threads | triad, STREAM-counted | triad, true traffic | triad, non-temporal |
|--:|--:|--:|--:|
| 1 | 16.0 | 21.4 | 22.6 |
| 2 | 19.5 | 26.0 | 30.5 |
| 3 | 19.9 | 26.5 | **31.8** |
| 6 | **20.0** | **26.6** | 29.9 |
| 12 | 19.1 | 25.4 | 28.6 |

*(GB/s. "True traffic" counts the read-for-ownership: a store to `a[i]` first
fetches that line, so DRAM moves 32 B per iteration where STREAM's convention
counts 24. The non-temporal variant bypasses the RFO, which is why it is the one
column where the two conventions agree.)*

**Memory roof: 26.6 GB/s with ordinary stores, 31.8 GB/s with non-temporal
stores — 52% and 62% of theoretical.** For weight streaming, which is pure reads
with no RFO, the higher figure is the relevant ceiling.

**Two to three threads saturate DRAM.** Going from 3 to 12 threads buys nothing
(26.5 → 25.4 GB/s; it gets slightly *worse*). Remember this: it is why the decode
path gets almost nothing from six cores, and it is the first hint that a chip
built for this workload does not need many of them either.

## Ceiling 2 — compute (`bench/peak_bench.cpp`)

Dependency-free FMA / integer-MAC chains, 10-second bursts
(`bench/results/peak.txt`):

| kernel | 1 thread | 6 threads | 12 threads (SMT) | 6T scaling |
|---|--:|--:|--:|--:|
| fp32 FMA (`vfmadd231ps`) | 104.5 | **431.4** | 552.6 | 4.13× |
| INT8, as we run it | 236.1 | **995.4** | 1248.1 | 4.22× |
| INT8 without `vpsignb` | 237.2 | 975.7 | 1203.8 | 4.11× |

*(GFLOP/s and GOP/s, counting 2 ops per MAC.)*

Three results worth pausing on:

- **The saturation-proof INT8 idiom is free.** Ours is four instructions per 32
  MACs (`vpsignb`, `vpmaddubsw`, `vpmaddwd`, `vpaddd`) against three for the
  plain unsigned form, and the two measure within 2% of each other in both
  directions (236.1 vs 237.2 single-thread; 995.4 vs 975.7 all-core). The likely
  reason is port placement — `vpsignb` is a shuffle-class op that is not
  contending with the multiply chain — but that is a hypothesis, not a
  measurement; confirming it needs a `perf stat` port-utilisation read. What is
  measured, and is what matters, is that **provable freedom from int16
  saturation costs no throughput.** See `docs/02`.
- **SMT buys 25%, not 100%.** 6 → 12 threads takes INT8 from 995 to 1248 GOP/s.
  The vector pipes are already busy; the second thread per core only fills
  issue-slot gaps.
- **A single-uop `vpdpbusd` would give ~2927 GOP/s** by op-count extrapolation.
  That number is **not measured** — Zen 3 has no VNNI. It is printed to size the
  prize that a fixed-function INT8 datapath collects for free, which is the
  bridge from M2 to M3.

**Sustained** (60 s, all cores, fp32): starts at 442 GFLOP/s, sags to 378 at
t=40 s, recovers to 411, and ends **7.1% below** the first window. So the mobile
thermal story is real but modest: this part holds ~90% of its burst roof
indefinitely rather than collapsing. Every ratio in this document uses the burst
roof, which makes the kernel numbers look *worse* than using the sustained one —
the conservative direction.

## Machine balance

The ridge point — the arithmetic intensity at which a kernel stops being
memory-bound — follows from the two ceilings:

| precision | compute roof | memory roof | balance |
|---|--:|--:|--:|
| fp32 | 431.4 GFLOP/s | 26.6 GB/s | **16.2 flop/byte** |
| INT8 | 995.4 GOP/s | 26.6 GB/s | **37.4 op/byte** |

Below 37 ops per byte, an INT8 kernel cannot reach the compute roof on this
machine no matter how good it is.

## The GEMM points (`bench/gemm_bench.cpp`)

The four projection shapes Qwen2.5-0.5B actually contains, at the batch sizes
prefill and decode actually use. Arithmetic intensity is computed, not assumed:

```
ops   = 2*M*K*N
bytes = N*K  (int8 weights) + N*(K/QK)*4 (fp32 block scales)
      + M*K*4 (activations in) + M*N*4 (results out)
```

Best INT8 result per point, from `bench/results/gemm.csv` (QK=32):

| shape | M | AI (op/B) | INT8 GOP/s | fp32 GOP/s | % of INT8 roof |
|---|--:|--:|--:|--:|--:|
| qkv/o [896→896] | 1 | 1.76 | 49.0 | 6.4 | 4.9% |
| gate/up [896→4864] | 1 | 1.77 | 58.0 | 14.7 | 5.8% |
| down [4864→896] | 1 | 1.77 | 52.7 | 21.5 | 5.3% |
| lm_head [896→151936] | 1 | 1.77 | 53.5 | 15.7 | 5.4% |
| qkv/o | 128 | 112.9 | 110.4 | 55.6 | 11.1% |
| down | 128 | 142.1 | 126.8 | 55.5 | 12.7% |
| gate/up | 128 | 142.1 | 187.8 | 55.2 | 18.9% |
| gate/up | 512 | 267.2 | 205.4 | 61.4 | 20.6% |
| lm_head | 128 | 150.6 | **247.5** | 59.9 | **24.9%** |

**The same kernel, on the same weights, moves from one side of the ridge to the
other purely because of M.** At M=1 the weights are read once and used twice; at
M=512 they are read once and used a thousand times. That is the thesis in one
table, and it is why batching is the first thing every serving system does.

### Honest accounting: the ≥70% gate is missed

The bible's M2 definition of done is "INT8 GEMM ≥ 70% of the *measured* compute
roof on prefill-shaped matmuls". **We reach 24.9%.** Calling that done would be
a lie, so here is the decomposition and what each piece is worth.

**1. Dequantisation is not free (a 20% ceiling, confirmed).** The peak kernel is
four instructions per 32 MACs and touches no memory. The real kernel must also
convert each block's int32 lanes to float and FMA them by `a_scale*w_scale` —
two more instructions per block. At QK=32 that is 6 ops per 32 MACs against the
peak loop's 4, so 67% of peak is the ceiling before anything else goes wrong.

The model makes a falsifiable prediction: QK=64 amortises those two instructions
over twice the MACs (10 ops per 64 MACs vs 12), so it should be ~20% faster.
Measured on the same quiet machine (`gemm_qk64.csv` vs `gemm.csv`):

| point | QK=32 | QK=64 | delta |
|---|--:|--:|--:|
| lm_head M=128, 6T | 247.5 | 296.0 | +19.6% |
| gate/up M=512, 6T | 205.4 | 272.2 | +32.5% |
| gate/up M=128, 6T | 187.8 | 284.4 | +51.4% |
| gate/up M=128, 1T | 66.7 | 85.6 | +28.3% |

The prediction lands almost exactly on `lm_head` (+19.6% vs +20% predicted) and
is a **lower bound** everywhere else. The op count does not explain the +51%, so
something else is also improving; the leading candidate is the accumulator
spill. The micro-kernel holds `MR×NRP = 8` float accumulators plus 8 int32 ones
against 16 ymm registers, so the float set spills, and it is touched **once per
block** — halving the block count halves that spill traffic. Stated as a
hypothesis, because the experiment that would settle it (a spill-free MR=1
variant at both block sizes) has not been run.

We ship QK=32 and pay this, because it buys 3.6 points of top-1 agreement with
HuggingFace. That trade is `docs/02`.

**2. It is NOT weight-bandwidth. The sweep says so.** The obvious next
hypothesis is that with MR=2 each weight byte feeds only 2 MACs before being
re-read, making the kernel L2-bandwidth-bound. **That hypothesis is wrong**, and
the tuning sweep in `bench/results/gemm.txt` is what killed it — quadrupling the
weight reuse barely moves the number:

| MR (activation rows per weight load) | best GOP/s, gate/up M=128, 1T |
|--:|--:|
| 1 | 62.7 |
| 2 | 64.5 |
| 4 | **68.1** |

Going from MR=1 to MR=4 is 4× the arithmetic per weight byte for **+8.6%**. A
bandwidth-bound kernel would have scaled far closer to linearly. So the limit is
inside the micro-kernel — instruction issue, the spills above, loop overhead —
not the memory system. That is worth knowing precisely because it says which
optimisations would be a waste of a week.

The same sweep explains the `mr = (M >= 4) ? 2 : 1` heuristic in the kernel. At
M=1 the higher tiles have no real rows to work on, so they replay row 0 and
discard the result: MR=1 gets 58.5 GOP/s, MR=2 gets 31.6, MR=4 gets 16.7 — an
almost exact 2× and 4× waste. The heuristic is not a guess; it is this table.

**3. What is left.** The op-count model says ~140 GOP/s per core should be
reachable; we measure 66.8 single-thread. The remaining ~2× is unexplained by
any model in this document, and the honest statement is that it is
unattributed. The next experiment is `perf stat` on uops-per-port, which has not
been run.

## Decode, end to end (`tools/run_infer`, `bench/results/decode.txt`)

The kernel benchmarks are per-matmul. This is the whole model, 32 greedy tokens:

| precision | weight bytes per token | ms/token | tok/s | effective bandwidth |
|---|--:|--:|--:|--:|
| fp32 | 1976 MB | 102.4 | 9.77 | 19.3 GB/s |
| INT8 | 947 MB | 48.9 | **20.5** | **19.4 GB/s** |

Read that table twice. **INT8 is 2.09× faster and it moved 2.09× fewer bytes.**
The bandwidth utilisation is *identical* (19.3 vs 19.4 GB/s). Nothing about the
arithmetic got faster; quantization bought exactly the bytes it removed and not
one thing more, because the arithmetic was never the constraint.

The same run against the ceilings:

- **Arithmetic:** 0.99 GOP per token. At the measured roofs (INT8 body at
  995 GOP/s, fp32 lm_head at 431 GFLOP/s) the maths takes **1.4 ms**. We spend
  **48.9 ms**. The compute units are idle **97%** of the time.
- **Memory:** 947 MB per token at the 26.6 GB/s roof is **35.6 ms**; at the
  read-side 31.8 GB/s roof, 29.8 ms. We spend 48.9 ms — **73% of the achievable
  memory roof, and 3% of the compute roof.**

**Decode is memory-bound by roughly 35×, and the remaining 27% of headroom is a
bandwidth problem, not an arithmetic one.** That is the thesis, measured end to
end rather than argued.

### Where the bytes go, and the biggest single lever

Of the 947 MB read per token, **545 MB (57%) is the output projection**, which
stays fp32 deliberately: it is tied to the embedding table, which must remain
fp32 for the token lookup anyway, and quantizing it costs more accuracy than any
other matrix (top-1 vs HuggingFace 99%+ → 66%; see `docs/02` §2.3). So **more
than half of decode's memory traffic produces 151,936 logits of which we use
exactly one.** The fixes are structural rather than numerical — a smaller
vocabulary head, or not materialising all the logits — and they belong to a
later milestone. The remaining 402 MB is the 24 transformer layers, already
INT8.

### The roofline

Log-log, INT8 ops. `▲` = measured GEMM points, `●` = the whole-model decode step.

```
 GOP/s
  1000 |........................................................... 995 INT8 compute roof
       |                                        ......
   500 |                                 .......         ridge at 37.4 op/B
       |                          .......             ▲(151, 248)
   200 |                   .......               ▲(142,188)  ▲(267,205)
       |             ......                ▲(113,110)
   100 |       ......
       |  ..▲(1.8, 58)  gate/up M=1
    50 | .▲(1.8, 53)    lm_head M=1 — on the diagonal
       |.●(1.8, 20)     whole model decode
    20 |
       +----------------------------------------------------------------
        1        3       10       30      100      300     1000   op/byte
        \____ memory-bound ____/         ridge      \___ compute-bound ___/
```

The dotted diagonal is the 26.6 GB/s memory roof. Everything at M=1 sits on or
under it; everything at M≥128 has climbed off it and is limited by the kernel
instead.

The M=1 points deserve a distinction the chart cannot show. The three small
projections (qkv, gate/up, down) have weights of 0.9–4.9 MB, which **fit in the
16 MB L3** — they never touch DRAM at all, so their ceiling is not the diagonal
and their limit is latency and thread fan-out. `lm_head` at M=1 is the only
decode matmul big enough (153 MB) to be genuinely DRAM-bound, and it achieves
**30.2 GB/s = 95% of the machine's read-side memory roof while reaching 5.4% of
its compute roof.** That single point is the clearest measurement in the
project.

## What is NOT in this document

- **A llama.cpp comparison.** The bible asks for one as an honesty benchmark. It
  has not been run, so no claim is made about how this kernel compares to a
  mature one.
- **A full Kc/Nc/Mc cache-blocking sweep.** Only MR, the L2 panel block, and
  prefetch were swept.
- **Prefill at M=2048.** The bench stops at M=512.
- **Any explanation of the last 2× in the micro-kernel.** Named above as
  unattributed rather than papered over.

## Reproducing

```bash
cmake -B build/rel -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build/rel
bench/run_all.sh              # all four artifacts, one quiet-machine pass
```

Benchmarks never run in CI: shared runners and unknown thermals make the numbers
meaningless. Run them locally, on mains power, with nothing else running, and
commit the CSV.
