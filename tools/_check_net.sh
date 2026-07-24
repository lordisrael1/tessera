#!/usr/bin/env bash
set -u
echo "curl:   $(command -v curl || echo MISSING)"
echo "wget:   $(command -v wget || echo MISSING)"
python3 - <<'PY'
try:
    import huggingface_hub  # noqa
    print("hub:    ok")
except Exception as e:
    print("hub:    missing (%s)" % type(e).__name__)
PY
# tiny connectivity probe (HEAD the HF resolve endpoint)
if command -v curl >/dev/null; then
  code=$(curl -s -o /dev/null -w '%{http_code}' -I -L \
    https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json || echo ERR)
  echo "hf http: $code"
fi
