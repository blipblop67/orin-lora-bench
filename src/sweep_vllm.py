"""vLLM sweep on Orin (NVIDIA-AI-IOT Jetson image on JetPack 6).

Variants:
  merged-bf16   the merged fine-tuned checkpoint
  base+lora     base model with the adapter loaded at runtime (--enable-lora)
  w4a16         GPTQ W4A16 checkpoint, if outputs/w4a16 exists (scripts/20_export.sh --w4a16)

Parts:
  fit    max-model-len x gpu-memory-utilization -> starts? KV-cache tokens, max concurrency, MB used
  serve  throughput at BENCH_CONCURRENCY with (VLLM_BENCH_MAX_LEN, VLLM_BENCH_GPU_UTIL)
  eval   task accuracy at the same config (base+lora also evaluates the un-tuned base)
Records are appended to results/vllm_<part>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from bench_client import run_bench  # noqa: E402
from eval_task import run_eval  # noqa: E402
from memmon import device_meta, drop_caches, used_mb  # noqa: E402
from servers import vllm_server  # noqa: E402

E = os.environ
ROOT = E["ROOT"]
OUT = os.path.join(ROOT, "outputs")
RES = os.path.join(ROOT, "results")
PORT = int(E.get("VLLM_PORT", 8000))
TOK = os.path.join(OUT, "merged")
POWER = device_meta()["power_mode"]
ADAPTER = os.path.join(OUT, "train", E.get("SERVE_RUN", ""), "adapter")


def variants():
    v = {"merged-bf16": ([os.path.join(OUT, "merged"), "--served-model-name", "ft", "--dtype", "bfloat16"], ["ft"])}
    if os.path.isdir(ADAPTER):
        v["base+lora"] = ([E["BASE_MODEL"], "--served-model-name", "base", "--dtype", "bfloat16", "--enable-lora",
                           "--lora-modules", f"ft={ADAPTER}", "--max-lora-rank", E.get("LORA_RANK", "16")],
                          ["ft", "base"])
    if os.path.isdir(os.path.join(OUT, "w4a16")):
        v["w4a16"] = ([os.path.join(OUT, "w4a16"), "--served-model-name", "ft"], ["ft"])
    return v


def write(part, rec):
    rec = {"kind": f"vllm_{part}", "runtime": "vllm", "power_mode": POWER, **rec}
    with open(os.path.join(RES, f"vllm_{part}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def parse_log(txt):
    out = {}
    pats = {"kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
            "max_concurrency_x": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
            "weights_gib": r"Model loading took ([\d.]+) ?GiB",
            "gpu_blocks": r"# (?:GPU|cuda) blocks: (\d+)"}
    for k, p in pats.items():
        m = re.findall(p, txt)
        if m:
            out[k] = float(m[-1].replace(",", ""))
    return out


def launch(argv, max_len, gmu):
    drop_caches()
    time.sleep(1)
    base = used_mb()
    s = vllm_server([*argv, "--host", "127.0.0.1", "--port", str(PORT), "--max-model-len", str(max_len),
                     "--gpu-memory-utilization", str(gmu), *E.get("VLLM_EXTRA_ARGS", "").split()], PORT)
    s.start()
    ok = s.wait_ready(int(E.get("VLLM_STARTUP_TIMEOUT", 900)))
    logs = s.logs()
    rec = {"max_model_len": max_len, "gpu_mem_util": gmu, "baseline_mb": base, **parse_log(logs)}
    if ok:
        time.sleep(2)
        rec.update(status="ok", load_delta_mb=used_mb() - base, startup_s=s.startup_s)
    else:
        errs = [ln for ln in logs.splitlines() if re.search(r"error|memory|exceed|larger than", ln, re.I)]
        rec.update(status="fail", **s.exit_info(), log_tail=errs[-5:])
    return s, ok, base, rec


def part_fit():
    for name, (argv, _) in variants().items():
        for gmu in [float(x) for x in E.get("VLLM_GPU_UTILS_S", "0.6").split()]:
            for ml in [int(x) for x in E.get("VLLM_MAX_LENS_S", "4096").split()]:
                s, ok, _, rec = launch(argv, ml, gmu)
                s.stop()
                write("fit", {"variant": name, **rec})
                if not ok:
                    break


def part_serve_eval(do_bench, do_eval):
    ml, gmu = int(E.get("VLLM_BENCH_MAX_LEN", 4096)), float(E.get("VLLM_BENCH_GPU_UTIL", 0.6))
    for name, (argv, eval_models) in variants().items():
        s, ok, base, rec = launch(argv, ml, gmu)
        if not ok:
            write("serve", {"variant": name, **rec})
            s.stop()
            continue
        if do_bench:
            for c in [int(x) for x in E.get("BENCH_CONCURRENCY", "1,4").split(",")]:
                r = run_bench(s.url, "ft", TOK, int(E.get("BENCH_INPUT_LEN", 512)),
                              int(E.get("BENCH_OUTPUT_LEN", 128)), c, int(E.get("BENCH_NUM_PROMPTS", 16)),
                              baseline_mb=base)
                write("serve", {"variant": name, "max_model_len": ml, "gpu_mem_util": gmu,
                                "kv_cache_tokens": rec.get("kv_cache_tokens"), "status": "ok", **r})
        if do_eval:
            for m in eval_models:
                label = name if m == "ft" else "base-bf16 (no fine-tune)"
                write("eval", {"variant": label, **run_eval(s.url, m, E.get("DATASET_PRESET", "sql"),
                                                             int(E.get("EVAL_SAMPLES", 200)), concurrency=8)})
        s.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="fit,serve,eval")
    a = ap.parse_args()
    parts = a.parts.split(",")
    if not os.path.isdir(os.path.join(OUT, "merged")):
        sys.exit("outputs/merged missing; run scripts/20_export.sh")
    if "fit" in parts:
        part_fit()
    if "serve" in parts or "eval" in parts:
        part_serve_eval("serve" in parts, "eval" in parts)
