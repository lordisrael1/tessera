#!/usr/bin/env bash
# CPU-only PyTorch + transformers for the M1 oracle. ~2 GB. Retries because the
# connection on this box drops mid-transfer (see the model-download saga).
set -x
# WSL Ubuntu ships python3 but not pip; install it first (idempotent).
if ! command -v pip3 >/dev/null; then
  apt-get update -qq && apt-get install -y -qq python3-pip python3-venv
fi
for attempt in $(seq 1 8); do
  pip3 install --break-system-packages --retries 10 --timeout 120 \
    torch transformers safetensors huggingface_hub \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple && break
  echo "pip attempt $attempt failed; retrying..."
  sleep 5
done
python3 -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)" \
  && echo "TORCH_OK" || echo "TORCH_FAILED"
