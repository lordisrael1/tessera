# Charter — SOHU-SIM / `tessera`

> A complete C++ inference stack for an accelerator that doesn't exist.

This file is the condensed, authoritative charter. The full original spec (the
"bible") is the source of truth for milestone details; this captures the parts a
working session needs at a glance. **When reality disagrees with the plan, update
this doc in the same change that alters the plan.**

## One-sentence pitch

I designed a fixed-function transformer accelerator on paper, wrote a
cycle-approximate simulator for it in C++, and then built the entire software
stack above it — compiler, runtime, hand-written collectives, MoE routing, and
profiler — until a real 0.5B-parameter model ran end to end and I could tell you
exactly which cycle it stalled on and why.

## Thesis (one story, three altitudes)

Inference decode is **memory-bandwidth-starved, not compute-starved.**
1. *Derive it* from measurements on this machine (roofline, M2).
2. *Design around it* in silicon (the ISA's explicit scratchpad + DMA, M3).
3. *Fight it again* at the cluster level (collectives, M5).

## Rules of the project (non-negotiable)

1. **No inference dependencies.** No BLAS, Eigen, llama.cpp, ONNX. Allowed:
   `<immintrin.h>`, the C++20 stdlib, POSIX, io_uring (liburing ok).
2. **Every milestone regresses against the M1 oracle.** Logits drift past
   tolerance ⇒ the milestone is not done.
3. **Sanitizer-clean always.** ASan+UBSan in CI from day one; TSan for M5.
4. **Write the doc the week the thing happens.** `docs/NN-topic.md`.
5. **Ship early, ship publicly.** M2's blog post ships before M3 starts.

## The target machine

HP EliteBook 845 G8 — AMD Ryzen 5 PRO 5650U (Cezanne, Zen 3).
6C/12T, AVX2+FMA3 (no AVX-512, no VNNI), L1d 32KB / L2 512KB per core / L3 16MB,
16GB DDR4-3200 (~51.2 GB/s theoretical, expect ~35–42 measured), 15–25W (thermal
throttling within seconds of sustained FMA — benchmark best-of-N *and* sustained).

## Milestones

| # | Name | Definition of done | Weeks |
|---|---|---|---|
| M1 | Reference forward pass | Greedy tokens identical to HuggingFace ≥64 steps; `logit_parity` < 1e-3 on 3 prompts | 1–3 |
| M2 | AVX2 INT8 GEMM + roofline | INT8 GEMM ≥70% of measured compute roof; roofline chart; decode shown memory-bound | 4–6 |
| M3 | The accelerator (ISA + sim + compiler) | fp32 model lowered to custom ISA runs on the simulator, bit-exact vs CPU fp32; per-layer cycles/DMA-stall/MAC-util reported | 7–11 |
| M4 | Runtime | Continuous batching, paged KV, DMA fault isolation per-request | 12–15 |
| M5 | Tensor parallelism + collectives | Sharded across 2/4 processes, oracle-identical, over shm/TCP/io_uring | 16–20 |
| M6 | MoE | 8-expert top-2, all-to-all dispatch, load-imbalance study | 21–23 |
| M7 | Profiler + bottleneck hunt | Chrome-trace per instruction; one profiler-found fix, before/after | 24–26 |

**Apply to jobs after M3.** Do not wait for M7.

Cut order if life happens (least regret first): M6 → io_uring transport →
M4 preemption. **Never cut:** the oracle, the roofline, the ISA spec, the
profiler-found fix.

## The two M1 traps (named in advance)

1. **GQA.** `num_key_value_heads` (2) ≪ `num_attention_heads` (14). Q is [14,64]
   per token; K,V are [2,64]. Each KV head serves 7 query heads. KV cache sized
   by KV heads. `kv_head = q_head / (n_q_heads / n_kv_heads)`.
2. **Tied embeddings.** `tie_word_embeddings: true` — there is no `lm_head.weight`;
   output projection is the input embedding matrix transposed. The reader not
   finding that tensor is correct, not a bug.

Also: Qwen2 attention has **biases on Q/K/V projections** (Llama doesn't). Load them.
RoPE uses the **split-half** convention. RMSNorm mean accumulated in **double**.
