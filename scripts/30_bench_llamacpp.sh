#!/usr/bin/env bash
# llama.cpp sweep. Usage: scripts/30_bench_llamacpp.sh [bench,ppl,fit,serve,eval,lora]
. "$(dirname "$0")/common.sh"
require_jetson; require_venv; check_power_mode
docker rm -f vllm-bench >/dev/null 2>&1 || true   # never benchmark with the other runtime resident
[ -f "$RESULTS/data/heldout.txt" ] || "$PY" "$ROOT/src/data.py" --preset "$DATASET_PRESET" --split test \
  --n "$EVAL_SAMPLES" --dump-text "$RESULTS/data/heldout.txt"
"$PY" "$ROOT/src/sweep_llamacpp.py" --parts "${1:-bench,ppl,fit,serve,eval,lora}" 2>&1 | tee -a "$OUT/sweep_llamacpp.log"
