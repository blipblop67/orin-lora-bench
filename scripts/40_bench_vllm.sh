#!/usr/bin/env bash
# vLLM sweep. Usage: scripts/40_bench_vllm.sh [fit,serve,eval]
# Each config is a fresh container start (2-5 min on Orin Nano incl. CUDA-graph capture).
. "$(dirname "$0")/common.sh"
require_jetson; require_venv; check_power_mode
docker rm -f lcpp-bench >/dev/null 2>&1 || true
"$PY" "$ROOT/src/sweep_vllm.py" --parts "${1:-fit,serve,eval}" 2>&1 | tee -a "$OUT/sweep_vllm.log"
