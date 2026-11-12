# orin-lora-bench

Fine-tune a small LLM with **LoRA / QLoRA on a Jetson Orin Nano**, then serve it with **llama.cpp** and **vLLM**
and measure how **quantisation, context length and KV-cache type** trade off against speed, memory fit and
task accuracy.

Target: **Jetson Orin Nano 8 GB / Orin Nano Super, JetPack 6.2** (L4T r36.4.x, CUDA 12.6, Python 3.10).

```
 Qwen2.5-0.5B/1.5B-Instruct ──LoRA/QLoRA (on-device)──► adapter
                                                          │
                     merge (CPU) ◄────────────────────────┤
                         │                                │
        ┌────────────────┼──────────────────┐             │
   GGUF F16 → Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0  │       lora-F16.gguf
        │                                bf16 HF  (W4A16 optional)
   llama.cpp: llama-bench · perplexity ·   │
   fit matrix · serve · task eval        vLLM: fit matrix · serve · eval · runtime LoRA
        └──────────────► results/*.jsonl ──► results/report.md + plots
```

## What gets measured

| Stage | Measured |
|---|---|
| Training sweep | peak unified-memory use (system + torch), swap, s/step, tokens/s, eval loss/ppl, OOM or not — for 0.5B LoRA, 1.5B LoRA, 1.5B QLoRA at seq 512/1024 |
| GGUF quants | file size, `llama-bench` prompt + generation tok/s at KV depth 0 / 2k / 8k, perplexity on held-out task text |
| llama.cpp fit matrix | quant × KV type (f16/q8_0/q4_0) × context (2k…32k): loads or fails, memory after load, TTFT of a ~90%-of-context prompt |
| vLLM fit matrix | `--gpu-memory-utilization` × `--max-model-len`: starts or fails, KV-cache tokens, max concurrency, weight GiB, startup time |
| Serving | same OpenAI-API client for both runtimes: TTFT, TPOT, per-request and aggregate tok/s at concurrency 1/2/4/8, peak memory |
| Accuracy | text-to-SQL **execution match** in SQLite + exact match: base model vs fine-tuned, per quant, merged vs runtime-LoRA |

The task is text-to-SQL (`gretelai/synthetic_text_to_sql`): small models get visibly better with LoRA, and the
answers can be checked by running them, so "did quantisation hurt the fine-tune?" gets a real number.
Switch to `DATASET_PRESET=alpaca` for a generic instruction set (scored by token F1).

## Before you start

1. **Power mode**: `sudo bash scripts/power_max.sh` (it picks MAXN_SUPER or MAXN out of `/etc/nvpmodel.conf` and runs `jetson_clocks`).
   Every record stores the power mode; don't mix modes in one comparison.
2. **Storage**: put the repo on NVMe. You need ~25 GB (HF cache, merged model, GGUF files, containers).
3. **Memory**: 8 GB is shared by CPU and GPU. For the 1.5B training runs and the larger fit tests:
   - go headless (`sudo systemctl set-default multi-user.target && sudo reboot`, which frees about 1 GB; undo with `graphical.target`)
   - swap a zram for an NVMe swapfile (setup prints the commands). Swap keeps a run from being OOM-killed,
     but the `swap peak MB` column shows when a result was only reached by swapping, so treat that row as "doesn't really fit".
4. Docker with the NVIDIA runtime and your user in the `docker` group.

## Run

```bash
bash scripts/00_setup.sh          # venv + Jetson torch/bitsandbytes, pull containers, cache data/model
bash scripts/10_train.sh          # training sweep (TRAIN_MATRIX); `10_train.sh 3` runs just entry 3
bash scripts/20_export.sh         # merge SERVE_RUN, GGUF F16 + quants, LoRA GGUF  (add --w4a16 for vLLM GPTQ)
bash scripts/30_bench_llamacpp.sh # bench,ppl,fit,serve,eval,lora  (or pass a subset: fit,serve)
bash scripts/40_bench_vllm.sh     # fit,serve,eval
bash scripts/50_report.sh         # results/report.md + results/plots/*.png
```

Or run `bash scripts/run_all.sh` to do all of it. Rough timings on an Orin Nano Super: training 20–60 min per run,
export about 10 min, llama.cpp sweep 1–2 h, vLLM sweep 1.5–3 h (every config is a fresh container start).
Results are only ever appended, so you can re-run any stage; the report keeps the latest record for each config.

Change anything in `config.local.env` (same syntax as `config.env`, and it is sourced after it), for example:

```bash
BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
SERVE_RUN="Qwen2.5-0.5B-Instruct_lora_s512"
LCPP_CTX_LIST=(4096 8192 16384 32768)
```

## Layout

```
config.env                 all knobs (models, matrices, images, sweep ranges)
scripts/common.sh          env loading, power-mode check, llama.cpp tool runner (docker or native)
scripts/power_max.sh       MAXN/MAXN_SUPER + jetson_clocks
src/data.py                dataset presets + chat formatting (shared by train/eval/perplexity)
src/train_lora.py          LoRA/QLoRA trainer (answer-only loss, grad checkpointing, paged 8-bit AdamW)
src/merge_adapter.py       adapter -> merged bf16 checkpoint (on CPU)
src/quantize_w4a16.py      optional GPTQ W4A16 for vLLM (llm-compressor, separate venv)
src/memmon.py              unified-memory + swap + GR3D sampler; device metadata
src/servers.py             start/health/stop for llama-server and vLLM containers
src/bench_client.py        streaming OpenAI-API load generator (TTFT/TPOT/tok/s)
src/eval_task.py           SQL execution-match / exact-match / F1 eval against any server
src/sweep_llamacpp.py      llama.cpp parts: bench, ppl, fit, serve, eval, lora
src/sweep_vllm.py          vLLM parts: fit, serve, eval
src/report.py              tables + plots
```

## Software choices (JetPack 6.2)

- **PyTorch / bitsandbytes**: Jetson AI Lab index `https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/`. Wheels from plain PyPI install on aarch64 but don't include Orin `sm_87` CUDA kernels. Setup checks that it sees `sm_87` and can run a bf16 matmul.
- **llama.cpp**: `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin` for `llama-server/bench/quantize/perplexity`.
  The Python converters (`convert_hf_to_gguf.py`, `convert_lora_to_gguf.py`) come from a llama.cpp checkout in
  `third_party/`. Pin `LLAMA_CPP_REF` to a release tag close to the container's build so the converters match the runtime.
  To use a native CUDA build instead, set `LLAMA_CPP_BIN_DIR` to its `build/bin`.
- **vLLM**: `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin`, the NVIDIA-built image for Orin before JetPack 7.2.
  Upstream `vllm/vllm-openai` only applies to Orin from JetPack 7.2 / L4T r39. On Jetson, `--gpu-memory-utilization`
  is a fraction of the *whole* unified memory, so 0.6 on an 8 GB board is about 4.6 GB including weights.
- **Quantisation names**: llama.cpp uses GGUF (Q8_0…Q4_0). vLLM uses HF checkpoints (bf16, or W4A16 GPTQ via
  `--w4a16`). They are different formats, so compare them on measured speed, memory and accuracy rather than by name.
  Orin has no FP8, so FP8 isn't part of the sweep.
- `latest-*` tags are mutable. For a write-up, record the digests: `docker image inspect --format '{{index .RepoDigests 0}}' <image>`.

## How to read the results

- **Memory numbers** are system-wide `MemTotal − MemAvailable` deltas (in `memmon.py`). That is the right metric on unified
  memory, because CUDA allocations show up there. The fit tests run llama.cpp with `--no-mmap` so that weights
  count as resident memory instead of page cache.
- **llama.cpp allocates the whole KV cache at load**, so "loads" mostly means "fits". The probe prompt confirms that
  long prefill really works and shows how fast it is.
- **vLLM pre-allocates** `gpu_mem_util × total` and fills the remainder with KV blocks. The `KV tokens` and `max conc ×`
  columns show how many tokens of context it can actually hold at the same time.
- `llama-server -c` is split across the `-np` slots. The serving bench sets `-c = slot_ctx × LCPP_PARALLEL` for that reason.
- QLoRA adapters are merged into the bf16 base. The eval table shows whether that, or quantisation, cost accuracy.

## Troubleshooting

- **`libcusparseLt.so` not found** when importing torch: install cuSPARSELt for CUDA 12.6 (aarch64-jetson) from
  NVIDIA's cuSPARSELt downloads, or pick a torch wheel from the same index that doesn't need it.
- **bitsandbytes won't install or import**: QLoRA runs fail and get recorded as crashed. LoRA runs still work. Remove
  the qlora rows from `TRAIN_MATRIX` and set `SERVE_RUN` to a lora run.
- **Training is OOM-killed (`oom_killed`)**: go headless, add swap, or lower `BATCH` and raise `GRAD_ACCUM` (keep the
  product the same), shorten `max_seq_len`, or use qlora.
- **vLLM won't start**: check `outputs/logs/vllm-bench.log`. Usually the fix is a lower `--max-model-len`, a higher
  `--gpu-memory-utilization`, or `--enforce-eager` in `VLLM_EXTRA_ARGS`, which skips the CUDA-graph memory.
- **llama-server flags rejected**: `servers.llama_fa_args()` handles both the old and new `--flash-attn` syntax. If
  other flags have changed in your container build, check `docker run --rm --runtime nvidia <image> llama-server --help`.
