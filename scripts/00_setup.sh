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
docker info 2>/dev/null | grep -qi 'runtimes.*nvidia' \
  || warn "docker does not list the nvidia runtime; set \"default-runtime\": \"nvidia\" in /etc/docker/daemon.json"
id -nG | grep -qw docker || warn "User not in docker group: sudo usermod -aG docker \$USER && newgrp docker"

log "Pulling containers"
docker pull "$LLAMA_CPP_IMAGE"
docker pull "$VLLM_IMAGE"

# ---- Python venv ----
log "Creating venv at $VENV"
sudo apt-get install -y python3-venv python3-dev git >/dev/null
python3 -m venv "$VENV"
"$PY" -m pip install -U pip wheel >/dev/null
log "Installing torch from $JETSON_PIP_INDEX"
"$PY" -m pip install --index-url "$JETSON_PIP_INDEX" --extra-index-url https://pypi.org/simple torch "numpy<2"
"$PY" -m pip install -r "$ROOT/requirements.txt"
log "Installing bitsandbytes (needed for QLoRA)"
"$PY" -m pip install --index-url "$JETSON_PIP_INDEX" --extra-index-url https://pypi.org/simple bitsandbytes \
  || warn "bitsandbytes install failed: qlora runs will be skipped/fail; lora runs still work."

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
  warn "If the error mentions libcusparseLt.so, install cuSPARSELt for CUDA 12.6 (see README > Troubleshooting)."
  die "torch CUDA check failed"
fi
"$PY" -c "import bitsandbytes as b; print('bitsandbytes', b.__version__)" 2>/dev/null || warn "bitsandbytes not importable"

# ---- llama.cpp converters (pure Python; the binaries come from the container) ----
if [ ! -d "$ROOT/third_party/llama.cpp" ]; then
  log "Cloning llama.cpp ($LLAMA_CPP_REF) for convert_hf_to_gguf.py / convert_lora_to_gguf.py"
  git clone --depth 1 --branch "$LLAMA_CPP_REF" https://github.com/ggml-org/llama.cpp "$ROOT/third_party/llama.cpp"
fi
"$PY" -m pip install "$ROOT/third_party/llama.cpp/gguf-py"

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
log "Setup complete. Device info saved to results/device.json"
