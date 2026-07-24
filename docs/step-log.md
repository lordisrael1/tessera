# Step log — plain English, newest first

A running "what we did and why" so you (and future sessions) never face a blank
page. Brutal and honest: what worked, what's fake, what's left.

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
