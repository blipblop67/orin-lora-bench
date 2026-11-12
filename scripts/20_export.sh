#!/usr/bin/env bash
# Export the chosen adapter (SERVE_RUN) for serving:
#   outputs/merged/                 merged bf16 HF checkpoint (vLLM, GGUF source)
#   outputs/gguf/ft-F16.gguf        + ft-<quant>.gguf for every GGUF_QUANTS entry
#   outputs/gguf/base-*.gguf        un-tuned baseline (Q8_0) and base for runtime-LoRA (Q4_K_M)
#   outputs/gguf/lora-F16.gguf      adapter as GGUF, for llama-server --lora
#   outputs/w4a16/                  optional (--w4a16): GPTQ W4A16 checkpoint for vLLM
. "$(dirname "$0")/common.sh"
require_venv

W4A16=0; [ "${1:-}" = "--w4a16" ] && W4A16=1
ADAPTER="$OUT/train/$SERVE_RUN/adapter"
[ -d "$ADAPTER" ] || die "Adapter not found: $ADAPTER (train it first or change SERVE_RUN)"
CONV="$ROOT/third_party/llama.cpp"
[ -f "$CONV/convert_hf_to_gguf.py" ] || die "llama.cpp converters missing; run scripts/00_setup.sh"
G="$OUT/gguf"; mkdir -p "$G"

log "Merging $SERVE_RUN"
rm -rf "$OUT/merged"
"$PY" "$ROOT/src/merge_adapter.py" --adapter "$ADAPTER" --out "$OUT/merged"

BASE_DIR="$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$BASE_MODEL'))")"
grep -q "\"$BASE_MODEL\"" "$ADAPTER/adapter_config.json" \
  || warn "Adapter base differs from BASE_MODEL=$BASE_MODEL; base/LoRA GGUFs will use BASE_MODEL."

log "HF -> GGUF F16"
"$PY" "$CONV/convert_hf_to_gguf.py" "$OUT/merged" --outtype f16 --outfile "$G/ft-F16.gguf"
"$PY" "$CONV/convert_hf_to_gguf.py" "$BASE_DIR" --outtype f16 --outfile "$G/base-F16.gguf"
log "LoRA adapter -> GGUF"
"$PY" "$CONV/convert_lora_to_gguf.py" "$ADAPTER" --base "$BASE_DIR" --outtype f16 --outfile "$G/lora-F16.gguf"

for q in "${GGUF_QUANTS[@]}"; do
  log "Quantising ft -> $q"
  lcpp llama-quantize "$G/ft-F16.gguf" "$G/ft-$q.gguf" "$q"
done
lcpp llama-quantize "$G/base-F16.gguf" "$G/base-Q8_0.gguf" Q8_0
lcpp llama-quantize "$G/base-F16.gguf" "$G/base-Q4_K_M.gguf" Q4_K_M

if [ "$W4A16" = 1 ]; then
  log "W4A16 (GPTQ via llm-compressor) in a separate venv"
  QV="$ROOT/.venv-quant"
  if [ ! -x "$QV/bin/python" ]; then
    python3 -m venv "$QV"
    "$QV/bin/pip" install -U pip >/dev/null
    "$QV/bin/pip" install --index-url "$JETSON_PIP_INDEX" --extra-index-url https://pypi.org/simple torch "numpy<2"
    "$QV/bin/pip" install llmcompressor datasets
  fi
  "$QV/bin/python" "$ROOT/src/quantize_w4a16.py" --model "$OUT/merged" --out "$OUT/w4a16"
fi

log "Sizes"
"$PY" - <<'EOF'
import glob, json, os
root = os.environ["ROOT"]
rows = []
for p in sorted(glob.glob(f"{root}/outputs/gguf/*.gguf")):
    rows.append({"kind": "model_size", "file": os.path.basename(p), "size_mb": round(os.path.getsize(p) / 2**20)})
for d in ("merged", "w4a16"):
    fs = glob.glob(f"{root}/outputs/{d}/*.safetensors")
    if fs:
        rows.append({"kind": "model_size", "file": d, "size_mb": round(sum(map(os.path.getsize, fs)) / 2**20)})
with open(f"{root}/results/model_sizes.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n"); print(f"{r['file']:28s} {r['size_mb']:>7} MB")
EOF
