#!/usr/bin/env bash
# Shared helpers. Source from other scripts:  . "$(dirname "$0")/common.sh"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT
# shellcheck source=/dev/null
. "$ROOT/config.env"
# shellcheck source=/dev/null
[ -f "$ROOT/config.local.env" ] && . "$ROOT/config.local.env"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
RESULTS="$ROOT/results"
OUT="$ROOT/outputs"
mkdir -p "$RESULTS" "$OUT"

# Export scalars + flatten arrays so the Python sweeps can read them from the environment.
export BASE_MODEL DATASET_PRESET TRAIN_SAMPLES EVAL_SAMPLES LORA_RANK LORA_ALPHA LR BATCH GRAD_ACCUM \
       LLAMA_CPP_IMAGE VLLM_IMAGE LLAMA_CPP_BIN_DIR HF_CACHE LCPP_PORT LCPP_BENCH_DEPTHS \
       FIT_PROBE_MAX_CTX PPL_CHUNKS VLLM_PORT VLLM_BENCH_MAX_LEN VLLM_BENCH_GPU_UTIL VLLM_EXTRA_ARGS \
       VLLM_STARTUP_TIMEOUT BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_CONCURRENCY BENCH_NUM_PROMPTS \
       LCPP_PARALLEL DROP_CACHES SERVE_RUN
export GGUF_QUANTS_S="${GGUF_QUANTS[*]}"
export LCPP_CTX_LIST_S="${LCPP_CTX_LIST[*]}"
export LCPP_KV_TYPES_S="${LCPP_KV_TYPES[*]}"
export VLLM_MAX_LENS_S="${VLLM_MAX_LENS[*]}"
export VLLM_GPU_UTILS_S="${VLLM_GPU_UTILS[*]}"
export HF_HOME="$HF_CACHE"

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

require_jetson() {
  grep -qi "nvidia jetson" /proc/device-tree/model 2>/dev/null \
    || die "Not running on a Jetson (/proc/device-tree/model). Run this on the Orin Nano."
}

require_venv() {
  [ -x "$PY" ] || die "No venv at $VENV. Run scripts/00_setup.sh first."
}

# Warn (don't fail) when not in a MAXN* power mode; numbers are not comparable otherwise.
check_power_mode() {
  command -v nvpmodel >/dev/null 2>&1 || { warn "nvpmodel not found"; return 0; }
  local mode
  mode="$(nvpmodel -q 2>/dev/null | awk -F': ' '/NV Power Mode/ {print $2; exit}')"
  case "$mode" in
    MAXN*) log "Power mode: $mode" ;;
    *) warn "Power mode is '${mode:-unknown}', not MAXN/MAXN_SUPER. Run: sudo scripts/power_max.sh" ;;
  esac
}

# Run a llama.cpp tool (llama-bench, llama-quantize, llama-perplexity, ...) natively or in the container.
lcpp() {
  local tool="$1"; shift
  if [ -n "${LLAMA_CPP_BIN_DIR}" ]; then
    "${LLAMA_CPP_BIN_DIR}/${tool}" "$@"
  else
    docker run --rm --runtime nvidia --network host \
      -v "$ROOT:$ROOT" -w "$ROOT" "$LLAMA_CPP_IMAGE" "$tool" "$@"
  fi
}
