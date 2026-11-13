#!/usr/bin/env bash
# One-time setup on the Orin Nano (JetPack 6.2): device report, venv with Jetson torch,
# training deps, container pulls, llama.cpp converter checkout, dataset cache.
. "$(dirname "$0")/common.sh"
require_jetson

log "Device"
tr -d '\0' < /proc/device-tree/model; echo
head -n1 /etc/nv_tegra_release || true
grep -E 'MemTotal|SwapTotal' /proc/meminfo
check_power_mode
grep -q "R36" /etc/nv_tegra_release 2>/dev/null \
  || warn "Expected L4T R36 (JetPack 6.x). Container tags / pip index in config.env may need changing."

# ---- Memory hygiene hints (not applied automatically) ----
avail_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
log "MemAvailable: ${avail_mb} MB"
if systemctl is-active --quiet gdm3 2>/dev/null || systemctl is-active --quiet gdm 2>/dev/null; then
  warn "Desktop (gdm) is running and costs ~0.8-1.2 GB. For training/fit tests consider:"
  warn "  sudo systemctl set-default multi-user.target && sudo reboot   (undo: graphical.target)"
fi
if swapon --show 2>/dev/null | grep -q zram; then
  warn "zram swap is active. For 1.5B training a real swapfile on NVMe is safer, e.g.:"
  warn "  sudo systemctl disable nvzramconfig && sudo fallocate -l 8G /ssd/swapfile && sudo chmod 600 /ssd/swapfile"
  warn "  sudo mkswap /ssd/swapfile && sudo swapon /ssd/swapfile   (add to /etc/fstab to persist)"
fi

# ---- Docker ----
command -v docker >/dev/null || die "docker not installed (sudo apt install nvidia-container docker.io)"
docker info >/dev/null 2>&1 \
  || die "Cannot reach docker as $USER. Run: sudo usermod -aG docker $USER, then log out and back in (or reboot)."
docker info 2>/dev/null | grep -qi 'runtimes.*nvidia' \
  || warn "docker does not list the nvidia runtime; set \"default-runtime\": \"nvidia\" in /etc/docker/daemon.json"

log "Pulling containers"
docker pull "$LLAMA_CPP_IMAGE"
docker pull "$VLLM_IMAGE"

# ---- Python venv ----
log "Creating venv at $VENV"
sudo apt-get install -y python3-venv python3-dev git >/dev/null
python3 -m venv "$VENV"
"$PY" -m pip install -U pip wheel >/dev/null
rm -f "$VENV/.setup-ok"

# pip has no index priority: with PyPI as an extra index it picks the newest torch anywhere, and PyPI's
# aarch64 wheels (e.g. 2.14+cu130) are newer than the Jetson builds but need a newer driver than JetPack 6.
# So look up the newest version on the Jetson index and pin to exactly that.
jetson_version() {
  "$PY" -m pip index versions "$1" --pre --index-url "$JETSON_PIP_INDEX" 2>/dev/null \
    | sed -n "s/^$1 (\(.*\))$/\1/p"
}
TORCH_VER="${TORCH_VERSION:-$(jetson_version torch)}"
[ -n "$TORCH_VER" ] || die "Could not list torch versions on $JETSON_PIP_INDEX"
log "Installing torch==$TORCH_VER from $JETSON_PIP_INDEX"
"$PY" -m pip install --index-url "$JETSON_PIP_INDEX" --extra-index-url https://pypi.org/simple \
  "torch==$TORCH_VER" "numpy<2"

log "Installing bitsandbytes (needed for QLoRA)"
BNB_VER="$(jetson_version bitsandbytes)"
if [ -n "$BNB_VER" ]; then
  "$PY" -m pip install --index-url "$JETSON_PIP_INDEX" --extra-index-url https://pypi.org/simple \
    "bitsandbytes==$BNB_VER" \
    || warn "bitsandbytes install failed: qlora runs will be skipped/fail; lora runs still work."
else
  warn "No bitsandbytes on $JETSON_PIP_INDEX: qlora runs will be skipped/fail; lora runs still work."
fi

# Keep the Jetson builds (and the numpy 1.x they were compiled against) when the remaining
# requirements pull in torch or numpy as a dependency.
CONSTRAINTS="$VENV/jetson-constraints.txt"
"$PY" -m pip freeze | grep -iE '^(torch|bitsandbytes|numpy)==' > "$CONSTRAINTS" || true
"$PY" -m pip install -r "$ROOT/requirements.txt" -c "$CONSTRAINTS"

log "Verifying CUDA in torch"
if ! "$PY" - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  {p.name}  sm_{p.major}{p.minor}")
assert (p.major, p.minor) == (8, 7), "expected sm_87 (Orin)"
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); (x @ x).sum().item()
EOF
then
  warn "If the error mentions libcusparseLt.so or libcudss.so, see README > Troubleshooting."
  die "torch CUDA check failed"
fi
"$PY" -c "import bitsandbytes as b; print('bitsandbytes', b.__version__)" 2>/dev/null || warn "bitsandbytes not importable"

# ---- llama.cpp converters (pure Python; the binaries come from the container) ----
if [ ! -d "$ROOT/third_party/llama.cpp" ]; then
  log "Cloning llama.cpp ($LLAMA_CPP_REF) for convert_hf_to_gguf.py / convert_lora_to_gguf.py"
  git clone --depth 1 --branch "$LLAMA_CPP_REF" https://github.com/ggml-org/llama.cpp "$ROOT/third_party/llama.cpp"
fi
# --no-deps: gguf-py declares numpy>=2.2.6, which breaks the Jetson torch build; it works fine on numpy 1.26
# and its other deps (pyyaml, tqdm, requests) already come from requirements.txt.
"$PY" -m pip install --no-deps "$ROOT/third_party/llama.cpp/gguf-py"

log "Checking llama.cpp container tools"
for t in llama-server llama-bench llama-quantize llama-perplexity; do
  lcpp "$t" --help >/dev/null 2>&1 && echo "  ok  $t" || warn "  $t not runnable from $LLAMA_CPP_IMAGE (set LLAMA_CPP_BIN_DIR?)"
done

# ---- Data + base model cache ----
log "Caching dataset and base model"
"$PY" "$ROOT/src/data.py" --preset "$DATASET_PRESET" --split test --n "$EVAL_SAMPLES" \
  --dump-text "$RESULTS/data/heldout.txt"
"$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$BASE_MODEL'))"

"$PY" "$ROOT/src/memmon.py" --device-meta > "$RESULTS/device.json"
touch "$VENV/.setup-ok"   # run_all.sh only skips setup when this exists
log "Setup complete. Device info saved to results/device.json"
