#!/usr/bin/env bash
# Download Qwen2.5-0.5B-Instruct into models/qwen2.5-0.5b/.
# Weights are NEVER committed (see .gitignore). Run this once.
#
# Usage:  bash tools/fetch_model.sh
# Needs:  python3 with `huggingface_hub`  (pip install huggingface_hub)
#         OR curl/wget as a fallback (uncommented below).
set -euo pipefail

DEST="models/qwen2.5-0.5b"
REPO="Qwen/Qwen2.5-0.5B-Instruct"
mkdir -p "$DEST"

echo ">> Fetching $REPO into $DEST"

# Preferred: huggingface_hub (handles resume, hashing).
if python3 -c "import huggingface_hub" 2>/dev/null; then
  python3 - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, dest = sys.argv[1], sys.argv[2]
for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
          "generation_config.json", "model.safetensors"):
    try:
        p = hf_hub_download(repo_id=repo, filename=f, local_dir=dest)
        print("  ok:", f)
    except Exception as e:
        print("  skip:", f, "->", e)
PY
else
  echo "huggingface_hub not found; falling back to curl from the HF resolve URL."
  BASE="https://huggingface.co/$REPO/resolve/main"
  for f in config.json tokenizer.json tokenizer_config.json generation_config.json model.safetensors; do
    echo "  curl $f"
    curl -L --fail -o "$DEST/$f" "$BASE/$f" || echo "  (failed: $f)"
  done
fi

echo ">> Done. Point the loader at $DEST"
echo "   du -h $DEST/model.safetensors"
