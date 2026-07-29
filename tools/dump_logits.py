#!/usr/bin/env python3
"""Generate the M1 ORACLE: HuggingFace fp32 logits + greedy token ids.

This is the constitution of the repo (bible §3.7). Every milestone's C++ output
is compared against these dumps. Run once (needs torch + transformers, CPU fine):

    pip install torch transformers safetensors --index-url \
        https://download.pytorch.org/whl/cpu
    python3 tools/dump_logits.py --model models/qwen2.5-0.5b --out test/golden

Emits, per prompt i, into <out>/:
    p<i>_prefill_logits.bin  float32 [vocab]  logits at the LAST PROMPT position.
                             This is exactly what our prefill() returns, so it is
                             the primary <1e-3 gate.
    p<i>_final_logits.bin    float32 [vocab]  logits after the last greedy step.
                             Catches drift that accumulates over 64 decode steps.
    p<i>_tokens.bin          int32   [steps]  the greedy continuation.
    p<i>_h_embed.bin         float32 [hidden] embedding output, last prompt pos.
    p<i>_h_layer0.bin        float32 [hidden] residual stream after layer 0.
    p<i>_h_final.bin         float32 [hidden] after the final RMSNorm.
    p<i>_meta.json           prompt text, prompt ids, shapes.

The three hidden-state dumps are the BISECTION LADDER: when logits diverge, you
compare embed -> layer0 -> final in that order and the first mismatch names the
broken op (bad embedding lookup / bad attention-or-FFN / bad final norm).

Design note: we deliberately re-run a FULL forward over the growing sequence at
every step instead of using HF's KV cache. Mathematically identical, but it makes
the reference independent of whichever cache API this transformers version ships,
and it means a C++ KV-cache bug shows up as a divergence rather than being
mirrored on both sides.
"""
import argparse
import json
import os

import numpy as np

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a land far away,",
]


def w(path, arr):
    with open(path, "wb") as f:
        f.write(np.ascontiguousarray(arr).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="test/golden")
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    torch.set_num_threads(os.cpu_count() or 1)

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    except TypeError:  # transformers < 5 spelled it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    cfg = model.config
    print(f"model: {cfg.num_hidden_layers}L hidden={cfg.hidden_size} "
          f"vocab={cfg.vocab_size} tie={cfg.tie_word_embeddings}")

    for i, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids
        prompt_ids = ids[0].tolist()

        # --- step 0: the prefill reference (+ the bisection ladder) ---
        out = model(ids, output_hidden_states=True)
        prefill_logits = out.logits[0, -1, :].float().numpy().astype(np.float32)
        hs = out.hidden_states  # (n_layers+1) x [1, seq, hidden]; hs[0] = embeddings
        h_embed = hs[0][0, -1, :].float().numpy().astype(np.float32)
        h_layer0 = hs[1][0, -1, :].float().numpy().astype(np.float32)

        # What we want is the vector that gets fed to the lm_head, i.e. AFTER the
        # final RMSNorm. Whether hs[-1] is pre- or post-norm has flipped between
        # transformers versions, so DERIVE it rather than trusting either: the
        # correct vector is the one whose projection reproduces the logits.
        head_w = model.get_output_embeddings().weight  # [vocab, hidden]; tied here
        cand = {"hs_last": hs[-1][0, -1, :].float(),
                "norm(hs_last)": model.model.norm(hs[-1])[0, -1, :].float()}
        ref = torch.from_numpy(prefill_logits)
        errs = {k: float((v @ head_w.T.float() - ref).abs().max()) for k, v in cand.items()}
        pick = min(errs, key=errs.get)
        if errs[pick] > 1e-2:
            raise RuntimeError(f"cannot identify post-norm hidden state: {errs}")
        if i == 0:
            print(f"     post-norm hidden state = {pick} (max|logit err| {errs[pick]:.2e})")
        h_final = cand[pick].numpy().astype(np.float32)

        # --- greedy continuation ---
        greedy = []
        cur = ids
        final_logits = prefill_logits
        for _ in range(args.steps):
            logits = model(cur).logits[0, -1, :].float()
            final_logits = logits.numpy().astype(np.float32)
            nxt = int(torch.argmax(logits))
            greedy.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)

        p = os.path.join(args.out, f"p{i}_")
        w(p + "prefill_logits.bin", prefill_logits)
        w(p + "final_logits.bin", final_logits)
        w(p + "tokens.bin", np.array(greedy, dtype=np.int32))
        w(p + "h_embed.bin", h_embed)
        w(p + "h_layer0.bin", h_layer0)
        w(p + "h_final.bin", h_final)
        with open(p + "meta.json", "w") as f:
            json.dump({
                "prompt": prompt,
                "prompt_ids": prompt_ids,
                "vocab": int(prefill_logits.shape[0]),
                "hidden": int(h_embed.shape[0]),
                "steps": args.steps,
            }, f, indent=2)

        print(f"[{i}] {prompt!r}")
        print(f"     prompt_ids = {prompt_ids}")
        print(f"     greedy[:8] = {greedy[:8]}  -> {tok.decode(greedy[:8])!r}")

    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump({"prompts": PROMPTS, "steps": args.steps}, f, indent=2)
    print("Oracle written to", args.out)


if __name__ == "__main__":
    main()
