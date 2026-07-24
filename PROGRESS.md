# PROGRESS — session handoff

**Read this first if you are a new session picking up this repo cold.**
This file is the single source of truth for "where are we and what's next."
Keep it updated at the end of every work session.

Last updated: 2026-07-24

---

## TL;DR state

- **Milestone M0 (bootstrap): DONE.** Repo scaffold, portable core library,
  dependency-free tests, toolchain (WSL g++/cmake/ninja), CI, docs.
- **Milestone M1 (reference forward pass): FUNCTIONALLY WORKING.** The full
  Qwen2.5-0.5B fp32 forward pass runs end to end and generates correct text:
  `"The capital of France is"` -> `"Paris. It is the largest city in Europe..."`.
  Tokenizer, KV cache, GQA attention, split-half RoPE, SwiGLU, tied-embedding
  logits all implemented and qualitatively validated.
  - **Remaining for M1 to be officially "done":** the NUMERICAL oracle
    (`dump_logits.py` + `logit_parity` test, <1e-3 vs HuggingFace). That needs
    PyTorch (~2 GB), which is not installed yet. Coherent generation is strong
    evidence of correctness but is NOT the oracle the charter requires.
  - Model is downloaded and byte-exact (988,097,824 bytes). Perf: scalar path
    ~1.4 s/token, ~32 s weight load (bf16->f32). Both are fine for a reference
    path; M2 makes them fast. Activation high-water = 79.1 KB/token (feeds M3).

## Environment (verified 2026-07-24)

- Host: Windows 11, HP EliteBook 845 G8, Ryzen 5 PRO 5650U (Zen 3), 15.3 GB RAM,
  ~30 GB disk free after installs.
- **No compiler pre-existed.** The winget path (CMake/LLVM/Ninja) stalled — winget
  hung on parallel Store installs. We PIVOTED to WSL, which is the bible's target
  anyway.
- **WSL Ubuntu-24.04 is installed AND working as root.** Toolchain installed via
  apt: **g++ 13.3.0, cmake 3.28.3, ninja 1.11.1, ccache, git.** Run everything
  with `wsl -d Ubuntu-24.04 -u root -- bash -lc "cd /mnt/c/Users/user/Desktop/SOHU && ..."`.
  (First interactive launch would want a UNIX user; we sidestep that with `-u root`.
   Optional: create a normal user later, not required to build.)
- **BUILD IS PROVEN GREEN** — release + ASan both pass `ctest` (2/2), zero warnings.
- Note: building on `/mnt/c` works but is slower than a native ext4 path. Fine for
  now; if builds feel slow later, clone into `~/tessera` inside WSL.
- Docker Desktop's WSL distro exists; ignore it.
- CMake (winget) may also have landed on Windows at `C:\Program Files\CMake`;
  irrelevant — we build in WSL.

## What exists and is TESTED

- `src/core/align.h` — 64B aligned alloc (Win + POSIX).
- `src/core/arena.h` — bump allocator with `high_water()` (feeds M3 SPM sizing).
- `src/core/tensor.h` — non-owning 4-D `View<T>`.
- `src/core/bf16.h` — bf16/f16 → f32.
- `src/core/json.h` — hand-written JSON parser (no nlohmann).
- `src/core/safetensors.{h,cpp}` — mmap'd zero-copy reader, `load_as_f32`.
- `src/model/config.{h,cpp}` — `ModelConfig::from_json_file`, GQA/tie helpers.
  Verified against the real Qwen2.5-0.5B config.json.
- `src/ops/ops.{h,cpp}` — scalar rmsnorm, rope (split-half), softmax, silu_mul,
  gemm_ref, **linear** (HF [OUT,IN] weight-stationary). **These are the oracle.**
- `src/tokenizer/bpe.{h,cpp}` — byte-level BPE. Merge loop + byte tables fully
  real; pretokenizer is the one documented approximation. Validated against the
  real 151936-token vocab (`"The capital of France is"` -> 5 tokens, id0=785).
- `src/model/kv_cache.{h,cpp}` — one up-front slab, sized by KV heads (GQA).
- `src/model/qwen2.{h,cpp}` — the forward pass. `prefill()` + `decode_step()`.
  Config-driven: conditional biases (load when present) + conditional lm_head
  (only when NOT tied). Generates correct text end to end.
- `tools/run_infer.cpp` — greedy inference driver (built as `run_infer`).
- `test/test_*.cpp` — core, ops, tokenizer. All green via `ctest` (3/3).

## What does NOT exist yet — THE single remaining M1 task

**The numerical oracle.** Everything else in M1 is built and working. What's left:

1. **Install PyTorch + transformers (CPU)** — user's call, ~2 GB:
   `pip3 install --break-system-packages torch transformers safetensors huggingface_hub --index-url https://download.pytorch.org/whl/cpu`
2. **Run `tools/dump_logits.py`** (already written) to emit HF fp32 logits +
   greedy tokens for 3 fixed prompts into `test/golden/`.
3. **Write `test/logit_parity.cpp`** — run our stack, assert max-abs < 1e-3 on
   logits and exact greedy-token match. **This test is the constitution of the
   repo** and the official M1 gate. Wire it into CTest so it self-skips when the
   golden files are absent (keeps CI green without the model).

Optional polish (not blocking M1): SIMD/perf is M2; the tokenizer pretokenizer
approximation should be cross-checked against a HF token-id golden dump
(`tools/dump_tokenizer_golden.py`, not yet written) once transformers is present.

## How to build (THE working path — WSL Ubuntu)

From a Windows PowerShell or the WSL shell:

```powershell
wsl -d Ubuntu-24.04 -u root -- bash -lc "cd /mnt/c/Users/user/Desktop/SOHU && \
  cmake -B build/rel -G Ninja -DCMAKE_BUILD_TYPE=Release && \
  cmake --build build/rel && \
  ctest --test-dir build/rel --output-on-failure"
```

Sanitizer tree (also passing):

```powershell
wsl -d Ubuntu-24.04 -u root -- bash -lc "cd /mnt/c/Users/user/Desktop/SOHU && \
  cmake -B build/asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTESSERA_SANITIZE=ON && \
  cmake --build build/asan && ctest --test-dir build/asan --output-on-failure"
```

If a plain `wsl` opens the Ubuntu shell interactively and asks to create a user,
just close it and use the `-u root` form above.

## Downloads the user still needs to grab

- **Qwen2.5-0.5B-Instruct** (~1 GB) — via `tools/fetch_model.sh` (to be written).
- **PyTorch + transformers (CPU)** in a Python env — only needed to generate the
  oracle (`dump_logits.py`). ~2 GB. Not needed to build/test the C++ core.

## Conventions

- No git commits by the automation — the USER commits. Leave the working tree
  staged-free unless asked.
- Update `docs/step-log.md` (newest first) and this file at the end of a session.
- Keep the M1 oracle green forever once it exists.
