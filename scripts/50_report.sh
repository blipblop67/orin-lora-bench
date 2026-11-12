#!/usr/bin/env bash
# Build results/report.md + results/plots/*.png from every results/*.jsonl
. "$(dirname "$0")/common.sh"
require_venv
"$PY" "$ROOT/src/report.py" --results "$RESULTS"
