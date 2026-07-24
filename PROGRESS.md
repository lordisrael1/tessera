# PROGRESS — session handoff

**Read this first if you are a new session picking up this repo cold.**
This file is the single source of truth for "where are we and what's next."
Keep it updated at the end of every work session.

Last updated: 2026-07-24

---

## TL;DR state

- **Milestone M0 (bootstrap): essentially done.** Repo scaffold, portable core
  library, and a dependency-free test suite exist. Toolchain being installed.
- **Milestone M1 (reference forward pass): started.** Core plumbing done
  (safetensors reader, config loader, scalar ops). Not yet done: tokenizer,
  the assembled forward pass, and the HuggingFace oracle.

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
- `src/ops/ops.{h,cpp}` — scalar rmsnorm, rope (split-half), softmax, silu_mul,
  gemm_ref. **These are the oracle.**
- `test/test_core.cpp`, `test/test_ops.cpp` — unit tests, all with known-value
  assertions. Run via `ctest`.

## What does NOT exist yet (the M1 to-do list, in order)

1. **`tools/fetch_model.sh`** — download Qwen2.5-0.5B-Instruct (model.safetensors
   ~1GB, tokenizer.json, config.json) into `models/qwen2.5-0.5b/`. NOT written yet.
2. **Verify config loader** against the real `config.json` (needs the download).
3. **Tokenizer** (`src/tokenizer/bpe.{h,cpp}`) — byte-level BPE encode/decode.
   Budget 3–4 days. Pragmatic shortcut: precompute pretokenized pieces in Python
   for test prompts; implement the merge loop + byte-decoder in C++. NOT started.
4. **KV cache** (`src/model/kv_cache.{h,cpp}`) — sized by KV heads. NOT started.
5. **Attention + forward pass** (`src/model/qwen2.{h,cpp}`) — assemble the 24-layer
   forward with `prefill()` and `decode_step()` entry points. NOT started.
6. **The oracle** (`tools/dump_logits.py` + `test/logit_parity.cpp`) — dump HF
   fp32 logits + greedy tokens for 3 prompts; assert max-abs < 1e-3 and exact
   greedy match. **This test is the constitution of the repo.** NOT started.

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
