#!/usr/bin/env bash
# Robust resume loop for the flaky-connection case. Each wget -c attempt resumes
# the partial model.safetensors; the loop retries until the byte-exact target
# (computed from the safetensors header) is reached.
cd /mnt/c/Users/user/Desktop/SOHU/models/qwen2.5-0.5b || exit 1
URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors"
TARGET=988097824
for attempt in $(seq 1 60); do
  have=$(stat -c %s model.safetensors 2>/dev/null || echo 0)
  if [ "$have" -ge "$TARGET" ]; then
    echo "COMPLETE at attempt $attempt: $have bytes"
    break
  fi
  pct=$(( have * 100 / TARGET ))
  echo "attempt $attempt: have $have / $TARGET (${pct}%)"
  wget -c --tries=5 --retry-connrefused --waitretry=3 --timeout=60 \
       --no-http-keep-alive -q "$URL"
  sleep 2
done
have=$(stat -c %s model.safetensors 2>/dev/null || echo 0)
echo "FINAL $have / $TARGET"
[ "$have" -ge "$TARGET" ] && echo "OK_COMPLETE" || echo "STILL_INCOMPLETE"
