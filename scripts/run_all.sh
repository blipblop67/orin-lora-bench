#!/usr/bin/env bash
# Whole pipeline, end to end. Expect several hours on an Orin Nano; each stage can be re-run alone.
set -euo pipefail
D="$(dirname "$0")"
[ -x "$D/../.venv/bin/python" ] || bash "$D/00_setup.sh"
bash "$D/10_train.sh"
bash "$D/20_export.sh"
bash "$D/30_bench_llamacpp.sh"
bash "$D/40_bench_vllm.sh"
bash "$D/50_report.sh"
