# tessera — a full inference stack for a chip that doesn't exist

**A from-scratch C++20 transformer inference stack, built from the silicon up:**
a fixed-function accelerator designed on paper, a cycle-approximate simulator for
it, and the entire software stack above — compiler, runtime, hand-written
collectives, MoE routing, and a profiler — until a real 0.5B model runs end to
end and you can point at the exact cycle it stalled on.

> Working title `tessera`. Target: Etched-class inference-infrastructure roles.

## The thesis

Inference decode is **memory-bandwidth-starved, not compute-starved.** This repo
proves it three times, at three altitudes: measured on a laptop (roofline),
designed around in a custom ISA, and fought again across a simulated cluster.

## Status

| Milestone | What | State |
|---|---|---|
| **M0** | Toolchain, repo scaffold, core headers, green tests | ✅ done |
| **M1** | Reference fp32 forward pass + tokenizer + HuggingFace oracle | ✅ done — max abs logit deviation **1.98e-05**, 64/64 greedy tokens exact |
| **M2** | AVX2 INT8 GEMM + roofline | ✅ done, with [one honest miss](docs/01-roofline.md) — 24.9% of the compute roof, not the 70% the plan asked for |
| **M3** | The accelerator: ISA + simulator + compiler | ✅ done — the real 0.5B, compiled to our ISA, **bit-identical** to the CPU on the simulator |
| M4 | Runtime: continuous batching, paged KV, fault isolation | ⏳ next |
| M5 | Tensor parallelism + hand-written collectives (shm/TCP/io_uring) | ⏳ |
| M6 | MoE: 8-expert top-2, all-to-all dispatch | ⏳ |
| M7 | Profiler + one documented bottleneck fix | ⏳ |

## What has actually been measured

Everything below is measured on one laptop (Ryzen 5 PRO 5650U) by code in this
repo, on an idle machine. Raw output is committed in `bench/results/`.

| | |
|---|---|
| memory roof / compute roof | 26.6 GB/s · 995 GOP/s INT8 |
| decode, whole model, INT8 | 48.9 ms/token — **compute idle 97% of the time** |
| what INT8 bought | 2.09× fewer bytes, 2.09× faster, *identical* bandwidth use |
| the model on the T1 simulator | 4461 instructions, 25.24 M cycles, bit-exact |
| T1 systolic array at batch 1 | busy 66.4% of cycles, retiring **2.88%** of the MACs it could |
| the deliberately-flawed scoreboard | costs **27%** of the runtime — double buffering under it is worth *zero* |

The thesis, in one line: **decode is memory-bandwidth-starved by ~35× on the
laptop; fix the bandwidth in silicon and the next wall is the shape of the array,
which is why batching exists.**

## Build (5 commands)

The reference path is WSL2 Ubuntu (the bible's target). A Windows-native build
also works — the core is portable and has `_WIN32` branches for mmap.

```bash
# from the repo root
cmake -B build/rel  -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/rel
ctest --test-dir build/rel --output-on-failure
# sanitizer tree (Linux/clang):
cmake -B build/asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTESSERA_SANITIZE=ON
ctest --test-dir build/asan --output-on-failure
```

On Windows with the winget toolchain, see `docs/step-log.md` for the exact
`cmake` invocation (it points CMake at clang++ and Ninja).

## Layout

```
src/core/    tensor, arena, align, bf16, json, safetensors, threadpool
src/model/   config, kv_cache, qwen2 (the forward pass, fp32 and INT8)
src/ops/     scalar reference ops (the oracle), AVX2 INT8 GEMM, quantization
src/accel/   isa.h, simulator, lower (the compiler), reference.h
test/        dependency-free harness; logit_parity is the constitution
bench/       stream / peak / gemm benchmarks + run_all.sh; results committed
docs/        00-charter, step-log, and one NN-topic.md per milestone
tools/       fetch_model.sh, dump_logits.py, run_infer, run_sim
```

## Docs to read

- `docs/00-charter.md` — the plan, condensed.
- **`docs/01-roofline.md`** — why decode is memory-bound, measured three ways.
- **`docs/02-int8-saturation.md`** — the saturation proof, and what quantization
  really costs (including the three choices that each moved accuracy >10 points).
- **`docs/03-isa-spec.md`** — the T1 ISA: formats, semantics, timing model, memory
  map, and the three design stances worth defending.
- **`docs/04-compiler.md`** — the passes, the double-buffering study, and the
  measurement that turned into M4's justification.
- `docs/step-log.md` — plain-English log of every step, newest first, including
  the bugs and the benchmarks that had to be thrown away.
- `PROGRESS.md` — handoff for the next work session.

## Try it

```bash
./build/rel/tools/run_infer --precision i8 --steps 32   # the model, on the CPU
./build/rel/tools/run_sim   --steps 1                   # the model, on the chip
bench/run_all.sh                                        # reproduce every number
```

## Rules

No inference dependencies (no BLAS/Eigen/llama.cpp). C++20 stdlib, `<immintrin.h>`,
POSIX, io_uring only. Every milestone must keep the M1 logit-parity test green.
Weights are never committed — fetch them with `tools/fetch_model.sh`.
