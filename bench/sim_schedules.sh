#!/usr/bin/env bash
# The double-buffering study of docs/04-compiler.md, on the REAL model.
# Four schedules, identical logits, different cycle counts.
set -euo pipefail
cd "$(dirname "$0")/.."
B="${1:-build/rel}"
run() {
  echo "--- $1 ---"
  shift
  "$B/tools/run_sim" --steps 0 --max-seq 32 "$@" 2>&1 \
    | grep -E '^pos  0|array EFF|array occ|GATE|weight stream'
}
run "double-buffered, coarse deps (the ISA as specified)"
run "double-buffered, fine deps" --fine-deps
run "serial, coarse deps" --serial
run "serial, fine deps" --serial --fine-deps
