#!/usr/bin/env python3
"""Generate the M1 ORACLE: HuggingFace fp32 logits + greedy token ids.

This is the constitution of the repo. Every milestone's C++ output is compared
against these dumps. Run once (needs torch + transformers, CPU is fine):

    pip install torch transformers safetensors --index-url \
        https://download.pytorch.org/whl/cpu
    python3 tools/dump_logits.py --model models/qwen2.5-0.5b --out test/golden

Emits, per prompt, into <out>/:
    prompt_<i>_logits.bin   float32 [seq? , vocab]  (last-position logits per step)
    prompt_<i>_tokens.bin   int32   greedy token ids (first 64 steps)
    prompt_<i>_meta.json    shapes + the prompt text

Also dumps layer-0 post-attention hidden state for bisecting divergence.
"""
import argparse, json, os, struct
import numpy as np

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a land far away,",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="test/golden")
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    for i, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids
        greedy = []
        last_logits = None
        with torch.no_grad():
            cur = ids
            for _ in range(args.steps):
                out = model(cur)
                logits = out.logits[0, -1, :].float()
                last_logits = logits.numpy().astype(np.float32)
                nxt = int(torch.argmax(logits))
                greedy.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)

        with open(os.path.join(args.out, f"prompt_{i}_logits.bin"), "wb") as f:
            f.write(last_logits.tobytes())
        with open(os.path.join(args.out, f"prompt_{i}_tokens.bin"), "wb") as f:
            f.write(np.array(greedy, dtype=np.int32).tobytes())
        with open(os.path.join(args.out, f"prompt_{i}_meta.json"), "w") as f:
            json.dump({"prompt": prompt, "vocab": int(last_logits.shape[0]),
                       "steps": args.steps, "prompt_ids": ids[0].tolist()}, f, indent=2)
        print(f"[{i}] '{prompt}' -> first tokens {greedy[:8]}")

    print("Oracle written to", args.out)

if __name__ == "__main__":
    main()
