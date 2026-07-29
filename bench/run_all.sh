#!/usr/bin/env bash
# Re-measure every M2 number in one pass, on a QUIET machine.
#
# The whole point of bench/results/*.csv is that docs/01-roofline.md can quote
# them, so they must all come from the same undisturbed run. A CSV taken while a
# build was running is worse than no CSV: it looks authoritative and is wrong.
# (We learned this the hard way — an earlier gate/up 1-thread point read 35.9
# GOP/s against 64.4 for the identical binary on an idle machine.)
#
# Usage:  bench/run_all.sh [build-dir]     (default build/rel)
set -euo pipefail
cd "$(dirname "$0")/.."
B="${1:-build/rel}"
mkdir -p bench/results

echo "### stream ###"
"$B/bench/stream_bench" --csv bench/results/stream.csv | tee bench/results/stream.txt
echo "### peak ###"
# --sustain is not optional for the writeup: the burst number alone hides
# whether a 15 W part can hold the roof, which is half the mobile-silicon story.
"$B/bench/peak_bench" --sustain 60 --csv bench/results/peak.csv \
  | tee bench/results/peak.txt
echo "### gemm ###"
"$B/bench/gemm_bench" --secs 1.0 --sweep --csv bench/results/gemm.csv \
  | tee bench/results/gemm.txt
echo "### end-to-end decode ###"
{
  "$B/tools/run_infer" --precision i8  --steps 32
  "$B/tools/run_infer" --precision f32 --steps 32
} | tee bench/results/decode.txt
echo "### done ###"
