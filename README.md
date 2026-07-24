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
| **M0** | Toolchain, repo scaffold, core headers, green tests | ✅ done (see `docs/step-log.md`) |
| **M1** | Reference fp32 forward pass + tokenizer + HuggingFace oracle | 🚧 in progress |
| M2 | AVX2 INT8 GEMM + roofline | ⏳ |
| M3 | The accelerator: ISA + simulator + compiler | ⏳ |
| M4 | Runtime: continuous batching, paged KV, fault isolation | ⏳ |
| M5 | Tensor parallelism + hand-written collectives (shm/TCP/io_uring) | ⏳ |
| M6 | MoE: 8-expert top-2, all-to-all dispatch | ⏳ |
| M7 | Profiler + one documented bottleneck fix | ⏳ |

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
src/core/    tensor, arena, align, bf16, json, safetensors  — the foundation
src/model/   config (derives all geometry from config.json)
src/ops/     scalar reference ops (the oracle for M2/M3)
test/        dependency-free test harness + unit tests
docs/        00-charter (the plan), step-log (what we did), NN-topic per milestone
tools/       fetch_model.sh, dump_logits.py (the oracle generator)
```

## Docs to read

- `docs/00-charter.md` — the plan, condensed.
- `docs/step-log.md` — plain-English log of every step taken, newest first.
- `PROGRESS.md` — machine-readable handoff for the next work session.

## Rules

No inference dependencies (no BLAS/Eigen/llama.cpp). C++20 stdlib, `<immintrin.h>`,
POSIX, io_uring only. Every milestone must keep the M1 logit-parity test green.
Weights are never committed — fetch them with `tools/fetch_model.sh`.
