#!/usr/bin/env bash
# Training sweep: runs every TRAIN_MATRIX entry in its own process so memory is clean
# between runs, and records OOM / OOM-killer deaths as results instead of aborting.
# Usage: scripts/10_train.sh            # full matrix
#        scripts/10_train.sh 2          # only the 2nd entry (1-based)
. "$(dirname "$0")/common.sh"
require_jetson; require_venv; check_power_mode
# Unified memory: blocks PyTorch's caching allocator keeps reserved are lost to the whole system. The 0.5B run
# reserved 3.8 GB for 2.0 GB of live tensors; expandable segments map memory in place instead of caching
# fixed-size blocks, which cuts that fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

entries=("${TRAIN_MATRIX[@]}")
if [ $# -ge 1 ]; then entries=("${TRAIN_MATRIX[$(( $1 - 1 ))]}"); fi

for e in "${entries[@]}"; do
  IFS='|' read -r model mode seq steps <<<"$e"
  name="$(basename "$model")_${mode}_s${seq}"
  log "Training $name (steps=$steps)"
  # Start each run from a clean baseline: on L4T < r39 the previous run's freed CUDA memory can sit in caches.
  "$PY" "$ROOT/src/memmon.py" --drop-caches >/dev/null
  set +e
  "$PY" "$ROOT/src/train_lora.py" --model "$model" --mode "$mode" --max-seq-len "$seq" --max-steps "$steps" \
    2>&1 | tee "$OUT/train_${name}.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then
    # 137 = SIGKILL (kernel OOM killer on unified memory), anything else = crash
    status=$([ "$rc" -eq 137 ] && echo "oom_killed" || echo "crashed_rc$rc")
    warn "$name: $status"
    "$PY" - "$name" "$model" "$mode" "$seq" "$status" <<'EOF'
import json, sys, os
n, m, mode, seq, st = sys.argv[1:]
with open(os.path.join(os.environ["ROOT"], "results", "train.jsonl"), "a") as f:
    f.write(json.dumps({"kind": "train", "run": n, "model": m, "mode": mode, "max_seq_len": int(seq), "status": st}) + "\n")
EOF
  fi
  sleep 5
done
log "Training records: results/train.jsonl"
