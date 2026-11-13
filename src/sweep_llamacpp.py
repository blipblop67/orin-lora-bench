"""llama.cpp sweep over GGUF quantisation x context length x KV-cache type.

Parts (--parts, comma separated; default all):
  bench  llama-bench pp/tg speed per quant (at several KV depths if supported)
  ppl    llama-perplexity on held-out task text per quant (quality cost of quantisation)
  fit    memory-fit matrix: quant x KV type x context -> loads? MB used, long-prompt TTFT
  serve  llama-server + bench_client at BENCH_CONCURRENCY per quant
  eval   task accuracy per quant, plus the un-tuned base model as a baseline
  lora   merged model vs base + runtime --lora adapter (speed and accuracy)
Records are appended to results/lcpp_<part>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from bench_client import run_bench  # noqa: E402
from eval_task import run_eval  # noqa: E402
from memmon import device_meta, drop_caches, image_digest, used_mb  # noqa: E402
from servers import llama_fa_args, llama_server  # noqa: E402

E = os.environ
ROOT = E["ROOT"]
G = os.path.join(ROOT, "outputs", "gguf")
TOK = os.path.join(ROOT, "outputs", "merged")
RES = os.path.join(ROOT, "results")
PORT = int(E.get("LCPP_PORT", 8080))
POWER = device_meta()["power_mode"]
IMAGE = None if E.get("LLAMA_CPP_BIN_DIR") else image_digest(E.get("LLAMA_CPP_IMAGE"))


def lst(name, default=""):
    return E.get(name, default).split()


def write(part, rec):
    rec = {"kind": f"lcpp_{part}", "runtime": "llama.cpp", "power_mode": POWER, "image": IMAGE, **rec}
    with open(os.path.join(RES, f"lcpp_{part}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def tool(name):
    if E.get("LLAMA_CPP_BIN_DIR"):
        return [os.path.join(E["LLAMA_CPP_BIN_DIR"], name)]
    return ["docker", "run", "--rm", "--runtime", "nvidia", "--network", "host",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{ROOT}:{ROOT}", "-w", ROOT, E["LLAMA_CPP_IMAGE"], name]


def models():
    quants = ["F16", *lst("GGUF_QUANTS_S")]
    return [(q, os.path.join(G, f"ft-{q}.gguf")) for q in quants if os.path.exists(os.path.join(G, f"ft-{q}.gguf"))]


def part_bench():
    help_txt = subprocess.run(tool("llama-bench") + ["--help"], capture_output=True, text=True)
    has_depth = "n-depth" in (help_txt.stdout + help_txt.stderr)
    for q, path in models():
        cmd = tool("llama-bench") + ["-m", path, "-p", "512", "-n", "128", "-ngl", "99", "-fa", "1",
                                     "-r", "3", "-o", "json"]
        if has_depth:
            cmd += ["-d", E.get("LCPP_BENCH_DEPTHS", "0,2048,8192")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            rows = json.loads(r.stdout)
        except json.JSONDecodeError:
            write("bench", {"quant": q, "status": "error", "error": r.stderr[-400:]})
            continue
        for row in rows:
            test = "pp" if row.get("n_prompt", 0) > 0 else "tg"
            write("bench", {"quant": q, "status": "ok", "test": test,
                            "n_prompt": row.get("n_prompt"), "n_gen": row.get("n_gen"),
                            "n_depth": row.get("n_depth", 0), "tok_s": round(row["avg_ts"], 2),
                            "tok_s_std": round(row.get("stddev_ts", 0), 2),
                            "model_size_mb": round(row.get("model_size", 0) / 2**20),
                            "build": row.get("build_commit")})


def part_ppl():
    text = os.path.join(RES, "data", "heldout.txt")
    for q, path in models():
        r = subprocess.run(tool("llama-perplexity") + ["-m", path, "-f", text, "-c", "512", "-ngl", "99",
                                                       "--chunks", E.get("PPL_CHUNKS", "64")],
                           capture_output=True, text=True)
        m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", r.stdout + r.stderr)
        write("ppl", {"quant": q, "ppl": float(m.group(1)) if m else None,
                      "ppl_err": float(m.group(2)) if m else None, "status": "ok" if m else "error"})


def start(argv, timeout=300):
    s = llama_server(["--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99", "--no-mmap",
                      *llama_fa_args(), *argv], PORT)
    s.start()
    return s, s.wait_ready(timeout)


def fail_info(s):
    tail = [ln for ln in s.logs(tail=40).splitlines() if re.search(r"error|fail|out of memory|alloc", ln, re.I)]
    return {**s.exit_info(), "log_tail": tail[-5:]}


def part_fit():
    ctxs = [int(c) for c in lst("LCPP_CTX_LIST_S", "2048 4096 8192")]
    probe_max = int(E.get("FIT_PROBE_MAX_CTX", 16384))
    for q, path in models():
        for kv in lst("LCPP_KV_TYPES_S", "f16"):
            for ctx in ctxs:
                drop_caches()
                time.sleep(1)
                base = used_mb()
                s, ok = start(["-m", path, "-c", str(ctx), "-np", "1", "-ctk", kv, "-ctv", kv])
                rec = {"quant": q, "kv_type": kv, "ctx": ctx, "baseline_mb": base}
                if ok:
                    time.sleep(2)
                    rec.update(status="ok", load_delta_mb=used_mb() - base, startup_s=s.startup_s)
                    if ctx <= probe_max:
                        p = run_bench(s.url, "ft", TOK, input_len=int(ctx * 0.9), output_len=32,
                                      concurrency=1, num_prompts=1, warmup=0, baseline_mb=base)
                        rec.update(probe_prompt_tokens=p["prompt_tokens_mean"], probe_ttft_ms=p["ttft_ms_mean"],
                                   probe_tpot_ms=p["tpot_ms_mean"], probe_peak_delta_mb=p.get("mem_peak_delta_mb"),
                                   probe_errors=p["errors"], swap_peak_used_mb=p.get("swap_peak_used_mb"))
                else:
                    rec.update(status="fail", **fail_info(s))
                s.stop()
                write("fit", rec)
                if not ok:
                    break  # larger contexts will not fit either


def slot_ctx():
    return max(2048, int(E.get("BENCH_INPUT_LEN", 512)) + int(E.get("BENCH_OUTPUT_LEN", 128)) + 256)


def serve_variant(label, argv, do_bench=True, do_eval=False, eval_models=("ft",)):
    np_ = int(E.get("LCPP_PARALLEL", 4))
    drop_caches()
    base = used_mb()
    s, ok = start([*argv, "-c", str(slot_ctx() * np_), "-np", str(np_)])
    if not ok:
        write("serve", {"variant": label, "status": "fail", **fail_info(s)})
        s.stop()
        return
    loaded = used_mb() - base
    if do_bench:
        for c in [int(x) for x in E.get("BENCH_CONCURRENCY", "1,4").split(",")]:
            r = run_bench(s.url, "ft", TOK, int(E.get("BENCH_INPUT_LEN", 512)), int(E.get("BENCH_OUTPUT_LEN", 128)),
                          c, int(E.get("BENCH_NUM_PROMPTS", 16)), baseline_mb=base)
            write("serve", {"variant": label, "status": "ok", "parallel": np_, "load_delta_mb": loaded, **r})
    if do_eval:
        for m in eval_models:
            write("eval", {"variant": label, **run_eval(s.url, m, E.get("DATASET_PRESET", "sql"),
                                                         int(E.get("EVAL_SAMPLES", 200)), concurrency=np_)})
    s.stop()


def part_serve():
    for q, path in models():
        serve_variant(f"ft-{q}", ["-m", path])


def part_eval():
    for q, path in models():
        serve_variant(f"ft-{q}", ["-m", path], do_bench=False, do_eval=True)
    base = os.path.join(G, "base-Q8_0.gguf")
    if os.path.exists(base):
        serve_variant("base-Q8_0 (no fine-tune)", ["-m", base], do_bench=False, do_eval=True)


def part_lora():
    base, lora, merged = (os.path.join(G, f) for f in ("base-Q4_K_M.gguf", "lora-F16.gguf", "ft-Q4_K_M.gguf"))
    if not all(map(os.path.exists, (base, lora, merged))):
        print("lora part skipped: need base-Q4_K_M.gguf, lora-F16.gguf, ft-Q4_K_M.gguf")
        return
    serve_variant("base-Q4_K_M + runtime LoRA", ["-m", base, "--lora", lora], do_eval=True)
    serve_variant("ft-Q4_K_M (merged)", ["-m", merged], do_eval=True)


PARTS = {"bench": part_bench, "ppl": part_ppl, "fit": part_fit, "serve": part_serve, "eval": part_eval,
         "lora": part_lora}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=",".join(PARTS))
    a = ap.parse_args()
    if not models():
        sys.exit("No GGUF models in outputs/gguf; run scripts/20_export.sh")
    for p in a.parts.split(","):
        print(f"=== llama.cpp part: {p} ===", flush=True)
        PARTS[p]()
