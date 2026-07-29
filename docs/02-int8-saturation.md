# 02 — INT8: the saturation trap, and what quantization actually costs

> Milestone 2. Every number here comes from a test or a benchmark in this repo,
> named at the point of use, and is reproducible with the commands at the bottom.

Two questions, both answered by measurement rather than by argument:

1. **Can the INT8 kernel silently produce garbage?** The AVX2 idiom everyone
   uses accumulates into *saturating* int16. We prove it cannot saturate — by
   construction, then by measurement on real weights.
2. **What does INT8 cost in model quality?** Not "about 1%": the exact top-1
   agreement, logit deviation, and perplexity delta against HuggingFace, plus the
   three design choices that each moved those numbers by more than 10 points.

---

## 1. The saturation trap

The workhorse instruction is `_mm256_maddubs_epi16(a, w)`: it multiplies 32
**unsigned** bytes by 32 **signed** bytes and adds *adjacent pairs* into 16
signed int16 lanes. The addition **saturates**.

The bible's original scheme (§4.1) quantizes activations to uint8 with a +128
zero-point shift, because `maddubs` demands an unsigned operand. Then the worst
case per pair is

```
255*127 + 255*127 = 64,770        int16 saturates at 32,767
```

A *single pairwise add* can overflow, and when it does the result is silently
clamped. No exception, no NaN — just wrong logits, in a kernel that passes every
smoke test because the average case is nowhere near the limit. This is the
failure mode that eats a week.

### The escape: keep the unsigned operand ≤ 127

We quantize activations to **signed** int8 in [−127, 127] and let `vpsignb` move
the sign onto the weight:

```cpp
__m256i w_sgn = _mm256_sign_epi8(w, a_raw);      // w * sign(a)
__m256i p16   = _mm256_maddubs_epi16(a_abs, w_sgn);  // |a| is 0..127 -> unsigned
__m256i p32   = _mm256_madd_epi16(p16, ones);        // drain to int32, always safe
```

`|a| ≤ 127` and `|w| ≤ 127`, so each int16 lane is bounded by

```
2 * 127 * 127 = 32,258  <  32,767
```

**Saturation is now impossible, by construction, for every input** — not
"unlikely for well-scaled blocks". It also deletes the `Σw` zero-point
correction term entirely, since there is no zero point any more.

### Measured, not just argued (`test/test_gemm.cpp`)

Proving a bound in a comment is worth little; the test computes the worst
pairwise partial over every block of a real quantized matrix, under both schemes:

```
[saturation] signed scheme worst |int16 partial| = 31245  (proven bound 32258)
[saturation] +128 zero-point scheme worst        = 63001  (int16 limit 32767)
```

Two things to take from that:

- The signed scheme reaches **31,245** — 97% of its proven bound. The bound is
  not slack; it is nearly tight, which means "it's fine in practice" would have
  been a genuinely dangerous thing to assume.
- The +128 scheme reaches **63,001**, which is **1.9× past the int16 limit**.
  It does not merely risk saturation on adversarial data; it saturates on the
  ordinary weights of an ordinary model. The deviation from the bible was
  necessary, not stylistic.

### What it costs: nothing

`vpsignb` adds a fourth instruction per 32 MACs (against three for the plain
unsigned form). Measured on this machine (`bench/results/peak.txt`):

| kernel | 1 thread | 6 threads |
|---|--:|--:|
| `vpsignb`+`vpmaddubsw`+`vpmaddwd`+`vpaddd` | 210.5 | **1047.8** |
| `vpmaddubsw`+`vpmaddwd`+`vpaddd` | 212.2 | 1021.6 |

*(GOP/s.)* The two are within 3% of each other in both directions — the extra
instruction is free within measurement noise, most likely because `vpsignb`
issues on a port the multiply chain is not contending for. **A provably
saturation-free kernel costs nothing.** That is the whole result.

---

## 2. What quantization costs the model

Scheme: **W8A8**, symmetric, per-block along the reduction dimension, block size
QK=32. Weights quantized and packed once at load; activations quantized on the
fly, per block. The output projection stays fp32 (§2.3).

The gate is `test/logit_parity_int8.cpp`, which sweeps 3 prompts × 64
teacher-forced steps against HuggingFace goldens. Teacher forcing matters: with
free-running decode, one disagreement at step 12 makes steps 13–63 compare two
different sentences, so the score measures "steps until first divergence" while
looking like an error rate.

### Headline (release build, `-O3 -march=native`)

| metric | fp32 | INT8 |
|---|--:|--:|
| teacher-forced top-1 vs HF | **100.0%** | **98.4%** |
| max abs logit deviation vs HF | 1.98e-05 | 5.48e-01 |
| mean relative L2 vs HF logits | 1.09e-06 | 3.28e-02 |
| mean token cross-entropy | 0.4708 | 0.4744 |
| resident weights | 1976 MB | 947 MB |

**Perplexity ratio INT8/fp32 over 192 tokens: 1.0036 (+0.36%).**
Free-running divergence step, per prompt: **13, never, 30**.

So: logits move by 3.3% in relative L2 — thirty thousand times the fp32 path's
deviation — and the model still picks the same token 98.4% of the time and is
0.36% worse at predicting the reference's own choices. That gap between "the
numbers moved a lot" and "the decisions barely moved" is the entire reason
quantization works, and it is why the fp32 gate (`<1e-3`) would be a meaningless
test to apply here.

The fp32 control is in the same run precisely so this cannot be fudged: it
scores 100.0% against the same goldens, so anything the INT8 column shows is
quantization, not our stack.

### 2.1 The bible asked for ≥99%. We get 98.4%.

The gap is real and it is **activation** quantization, not weights. Evidence:

- weight-only error on the same matrices is 0.53% relative L2
  (`test_gemm`'s `[i8 vs fp32]` check);
- a load-time scale search that measurably improved weight reconstruction moved
  the end-to-end number by less than 0.1 points.

The techniques that close it — SmoothQuant's activation/weight scale migration,
AWQ, a Hadamard rotation as in QuaRot — all attack activation outliers, and all
are model-preparation work rather than kernel work. That is a deliberate scope
call for this milestone, recorded here and in the bible's deviation table rather
than absorbed by quietly lowering a bar.

### 2.2 Why the gate is 96% and not 98%

Top-1 agreement is a count of argmax decisions over 192 samples, so it moves in
steps of 0.52 points and flips whenever a near-tie lands the other way. It is
therefore sensitive to the **last bits of the fp32 activations** feeding the
quantizer — which depend on compiler flags. Same source, same machine, same
weights:

| build | top-1 | rel-L2 | perplexity | divergence steps |
|---|--:|--:|--:|---|
| `-O3 -march=native` | 98.4% | 3.28e-02 | +0.36% | 13, never, 30 |
| `-O3 -march=native -ffp-contract=off` | 96.9% | 3.14e-02 | +0.41% | 8, never, 30 |
| `-O1` (the ASan tree) | 96.9% | 3.14e-02 | +0.41% | 8, never, 30 |

The ASan tree matches the no-contraction build **digit for digit**, which
identifies the cause exactly: FMA contraction in `ops::linear`'s double
accumulator, not the sanitizer. `a*b + c` becomes `fma(a, b, c)` under
`-march=native`, one rounding instead of two, and three of 192 near-ties land
differently as a result.

Note that the build with the **worse** top-1 has the **better** rel-L2. Three
flipped near-ties are noise, not quality. So the hard gate goes on the metrics
that are stable across all three builds — perplexity varies by 0.05 points — and
top-1 keeps a bar low enough to survive flag sensitivity while still catching a
real regression. It still has teeth: quantizing the output projection (§2.3)
drops it to 66%, which this gate catches by 30 points.

This is also why the ASan tree is not a rubber stamp. It found a real defect —
a threshold that only passed because of an optimisation flag.

### 2.3 Three choices that each moved the number by >10 points

Every one of these was measured end to end, not reasoned about:

| choice | naive version | ours | effect |
|---|---|---|---|
| activation scale granularity | one scale per row | one per QK block | 13.5% → 3.3% rel-L2; top-1 **14.6% → 98.4%** |
| output projection precision | INT8 like everything else | fp32 | rel-L2 4.8% → 1.0%; top-1 **66% → 99%+** |
| block size | QK=64 | QK=32 | rel-L2 4.33% → 3.28%; top-1 **94.8% → 98.4%** |

**Per-row activation scales are catastrophic, and this is the one to remember.**
LLM activations have outlier features: a handful of channels one to two orders of
magnitude above the rest, in every layer, consistently. A single row-wide scale
is set by those outliers, so every ordinary channel collapses into two or three
int8 levels and the model is destroyed — 14.6% top-1 agreement is barely better
than noise. Per-block scales confine each outlier's damage to its own block. In
the kernel this costs *nothing*: the dequantize multiplier was already
`a_scale * w_scale[b]`; it simply becomes `a_scale[b] * w_scale[b]`.

**The output projection stays fp32** for two independent reasons that point the
same way. Accuracy: every other matrix's error is attenuated by the layers after
it, while the lm_head's error lands directly on the logits with nothing
downstream to wash it out. Memory: the embedding table must stay fp32 anyway for
the token lookup (a gather, not a matmul), and since Qwen2.5 ties the two,
quantizing the projection would not replace those 545 MB — it would **add** 136
MB of int8 beside them. Keeping fp32 is smaller *and* more accurate. llama.cpp
reaches the same conclusion from the other end, keeping `token_embd`/`output` at
higher precision than the body.

### 2.4 QK=32 vs QK=64: paying 20% of throughput for 3.6 points

The bible starts at QK=64 and says to drop to 32 if the numbers demand it. They
did. But the smaller block doubles the number of dequantize FMAs per MAC, and
that is measurable:

Both configurations were measured on the same quiet machine, on this code —
QK=64 by rebuilding with the constant changed, not by quoting an old run:

| | QK=64 | QK=32 | delta |
|---|--:|--:|--:|
| logit rel-L2 vs HF | 4.33% | **3.28%** | −1.05 pts |
| teacher-forced top-1 | 94.8% | **98.4%** | **+3.6 pts** |
| perplexity vs fp32 | +0.58% | **+0.36%** | −0.22 pts |
| free-running divergence step | 11, 19, 4 | **13, never, 30** | much later |
| INT8 GEMM, lm_head M=128, 6T | 296.0 GOP/s | 247.5 | −16.4% |
| INT8 GEMM, gate/up M=128, 6T | 284.4 GOP/s | 187.8 | −34.0% |
| scale storage | 1/64 of weights | 1/32 of weights | 2× |

The divergence row is the one that would show up in a demo: at QK=64 one prompt
goes off the rails at step 4, while at QK=32 two of three prompts hold for 30+
steps and one never diverges at all.

The throughput side is worked through in `docs/01-roofline.md` §"the ≥70% gate
is missed": the op-count model predicts QK=64 should be ~20% faster because it
amortises the same two dequantize instructions over twice the MACs, and the
measurement agrees. **We ship QK=32 and pay that, because 3.6 points of top-1
agreement is worth more than 20% of a kernel that is not the bottleneck
anyway** — decode is memory-bound (that is the entire point of `docs/01`), and
in the memory-bound regime the QK=32 scales cost bandwidth, not arithmetic. The
scale array is 1/32 of the weight bytes, so QK=32 moves 3% more bytes per token
than QK=64: that, not the instruction count, is the real price at M=1.

QK must be a multiple of 32 regardless: the micro-kernel consumes a block in
32-byte chunks, so a smaller block would leave the vector registers half empty.

---

## 3. What INT8 actually bought, end to end

From `bench/results/decode.txt`, the whole model, 32 greedy tokens:

| precision | bytes/token | ms/token | tok/s | effective bandwidth |
|---|--:|--:|--:|--:|
| fp32 | 1976 MB | 102.4 | 9.77 | 19.3 GB/s |
| INT8 | 947 MB | 48.9 | 20.5 | 19.4 GB/s |

**2.09× fewer bytes, 2.09× faster, and identical bandwidth utilisation.**
Quantization bought exactly the bytes it removed and nothing else — the
arithmetic was never the constraint. Which is the thesis of `docs/01`, arrived at
from the other direction.

---

## 4. Reproducing

```bash
cmake -B build/rel -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build/rel
./build/rel/test/test_gemm                       # saturation proof + kernel parity
ctest --test-dir build/rel -R logit_parity_int8 -V   # the quality gate
./build/rel/bench/peak_bench --sustain 60        # the sign-fixup cost
```

The quality gate needs the weights (`tools/fetch_model.sh`) and the goldens
(`tools/dump_logits.py`, which needs PyTorch). Without them it self-skips, so a
fresh clone and CI stay green while a fully-provisioned machine gets a real gate.
