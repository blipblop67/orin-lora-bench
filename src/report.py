"""Aggregate results/*.jsonl into results/report.md and results/plots/*.png."""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

QUANT_ORDER = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0"]


def load(results):
    by = defaultdict(list)
    for p in glob.glob(os.path.join(results, "*.jsonl")):
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                by[r.get("kind", os.path.basename(p))].append(r)
    return by


def latest(rows, key):
    """Keep the last record per key (re-runs append)."""
    d = {}
    for r in rows:
        d[tuple(r.get(k) for k in key)] = r
    return list(d.values())


def table(rows, cols, title=None):
    if not rows:
        return ""
    head = [c if isinstance(c, str) else c[0] for c in cols]
    keys = [c if isinstance(c, str) else c[1] for c in cols]
    fmt = lambda v: "" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
    out = [f"\n### {title}\n" if title else "", "| " + " | ".join(head) + " |",
           "|" + "---|" * len(head)]
    out += ["| " + " | ".join(fmt(r.get(k)) for k in keys) + " |" for r in rows]
    return "\n".join(out) + "\n"


def qkey(q):
    return QUANT_ORDER.index(q) if q in QUANT_ORDER else 99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = load(a.results)
    P = os.path.join(a.results, "plots")
    os.makedirs(P, exist_ok=True)
    md = ["# Orin Nano LoRA fine-tune + serving benchmark\n"]
    dev = os.path.join(a.results, "device.json")
    if os.path.exists(dev):
        d = json.load(open(dev))
        md.append(f"Device: **{d.get('model')}** · {d.get('l4t')} · RAM {d.get('mem_total_mb')} MB · "
                  f"swap {d.get('swap_total_mb')} MB · power mode `{d.get('power_mode')}`\n")

    # ---- training ----
    tr = latest(R["train"], ["run"])
    md.append("## 1. Fine-tuning (LoRA vs QLoRA)\n")
    md.append(table(tr, ["run", "status", ("trainable params", "trainable_params"),
                         ("peak used MB (system)", "mem_peak_used_mb"), ("peak Δ MB", "mem_peak_delta_mb"),
                         ("torch peak alloc MB", "torch_peak_alloc_mb"), ("swap peak MB", "swap_peak_used_mb"),
                         ("s/step", "sec_per_optimizer_step_median"), ("train tok/s", "train_tokens_per_s"),
                         ("runtime s", "train_runtime_s"), ("eval loss", "eval_loss"), ("eval ppl", "eval_ppl")]))
    ok = [r for r in tr if r.get("status") == "ok"]
    if ok:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        names = [r["run"] for r in ok]
        ax[0].barh(names, [r.get("mem_peak_delta_mb", 0) for r in ok])
        ax[0].set_xlabel("peak memory Δ during training (MB)")
        for r in ok:
            if r.get("loss_curve"):
                s, l = zip(*r["loss_curve"])
                ax[1].plot(s, l, label=r["run"])
        ax[1].set_xlabel("step"); ax[1].set_ylabel("train loss"); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{P}/train.png", dpi=120); plt.close(fig)
        md.append("\n![training](plots/train.png)\n")

    sizes = {r["file"]: r["size_mb"] for r in R["model_size"]}
    if sizes:
        md.append("\n## 2. Exported model sizes\n")
        md.append(table([{"file": k, "MB": v} for k, v in sizes.items()], ["file", "MB"]))

    # ---- llama.cpp ----
    md.append("\n## 3. llama.cpp (GGUF)\n")
    lb = [r for r in latest(R["lcpp_bench"], ["quant", "test", "n_depth"]) if r.get("status") == "ok"]
    lb.sort(key=lambda r: (qkey(r["quant"]), r["test"], r.get("n_depth") or 0))
    md.append(table(lb, ["quant", "test", ("depth", "n_depth"), ("tok/s", "tok_s"), ("±", "tok_s_std"),
                         ("size MB", "model_size_mb")], "llama-bench throughput (pp512 / tg128)"))
    ppl = sorted(latest(R["lcpp_ppl"], ["quant"]), key=lambda r: qkey(r["quant"]))
    md.append(table(ppl, ["quant", "ppl", "ppl_err"], "Perplexity on held-out task text (lower is better)"))
    if lb:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for test, axi in zip(("pp", "tg"), ax):
            depths = sorted({r.get("n_depth") or 0 for r in lb})
            for dep in depths:
                rows = [r for r in lb if r["test"] == test and (r.get("n_depth") or 0) == dep]
                axi.plot([r["quant"] for r in rows], [r["tok_s"] for r in rows], marker="o", label=f"depth {dep}")
            axi.set_title("prompt processing" if test == "pp" else "generation"); axi.set_ylabel("tok/s")
            axi.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{P}/lcpp_speed.png", dpi=120); plt.close(fig)
        md.append("\n![llama.cpp speed](plots/lcpp_speed.png)\n")

    fit = latest(R["lcpp_fit"], ["quant", "kv_type", "ctx"])
    fit.sort(key=lambda r: (qkey(r["quant"]), r["kv_type"], r["ctx"]))
    md.append(table(fit, ["quant", ("KV", "kv_type"), "ctx", "status", ("load Δ MB", "load_delta_mb"),
                          ("probe prompt tok", "probe_prompt_tokens"), ("probe TTFT ms", "probe_ttft_ms"),
                          ("probe TPOT ms", "probe_tpot_ms"), ("probe peak Δ MB", "probe_peak_delta_mb"),
                          ("swap MB", "swap_peak_used_mb")], "Memory fit: quant × KV type × context"))
    if fit:
        kvs = sorted({r["kv_type"] for r in fit})
        fig, ax = plt.subplots(1, len(kvs), figsize=(5 * len(kvs), 4), squeeze=False)
        for axi, kv in zip(ax[0], kvs):
            for q in sorted({r["quant"] for r in fit}, key=qkey):
                rows = [r for r in fit if r["quant"] == q and r["kv_type"] == kv and r["status"] == "ok"]
                axi.plot([r["ctx"] for r in rows], [r["load_delta_mb"] for r in rows], marker="o", label=q)
            axi.set_xscale("log", base=2); axi.set_title(f"KV cache {kv}")
            axi.set_xlabel("context"); axi.set_ylabel("memory after load Δ (MB)"); axi.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{P}/lcpp_fit.png", dpi=120); plt.close(fig)
        md.append("\n![llama.cpp memory fit](plots/lcpp_fit.png)\n")

    # ---- vLLM ----
    md.append("\n## 4. vLLM\n")
    vf = latest(R["vllm_fit"], ["variant", "gpu_mem_util", "max_model_len"])
    vf.sort(key=lambda r: (r["variant"], r["gpu_mem_util"], r["max_model_len"]))
    md.append(table(vf, ["variant", ("gpu util", "gpu_mem_util"), ("max len", "max_model_len"), "status",
                         ("KV tokens", "kv_cache_tokens"), ("max conc ×", "max_concurrency_x"),
                         ("weights GiB", "weights_gib"), ("load Δ MB", "load_delta_mb"), ("startup s", "startup_s")],
                    "Memory fit: gpu-memory-utilization × max-model-len"))

    # ---- serving, both runtimes ----
    md.append("\n## 5. Serving throughput (same client, both runtimes)\n")
    serve = [r for r in latest(R["lcpp_serve"] + R["vllm_serve"], ["runtime", "variant", "concurrency"])
             if r.get("status") == "ok"]
    serve.sort(key=lambda r: (r["runtime"], r["variant"], r["concurrency"]))
    md.append(table(serve, ["runtime", "variant", ("conc", "concurrency"), ("TTFT p50 ms", "ttft_ms_p50"),
                            ("TPOT p50 ms", "tpot_ms_p50"), ("tok/s per req", "decode_tok_s_per_req"),
                            ("total out tok/s", "output_tok_s"), ("req/s", "req_s"),
                            ("peak used MB", "mem_peak_used_mb"), "errors"]))
    if serve:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for key in sorted({(r["runtime"], r["variant"]) for r in serve}):
            rows = [r for r in serve if (r["runtime"], r["variant"]) == key]
            lab = f"{key[0]}: {key[1]}"
            ax[0].plot([r["concurrency"] for r in rows], [r["output_tok_s"] for r in rows], marker="o", label=lab)
            ax[1].plot([r["concurrency"] for r in rows], [r["ttft_ms_p50"] for r in rows], marker="o", label=lab)
        ax[0].set_ylabel("aggregate output tok/s"); ax[1].set_ylabel("TTFT p50 (ms)")
        for axi in ax:
            axi.set_xlabel("concurrency"); axi.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(f"{P}/serve.png", dpi=120); plt.close(fig)
        md.append("\n![serving](plots/serve.png)\n")

    # ---- accuracy ----
    md.append("\n## 6. Task accuracy (did the fine-tune survive quantisation?)\n")
    ev = latest(R["lcpp_eval"] + R["vllm_eval"], ["runtime", "variant"])
    ev.sort(key=lambda r: (r["runtime"], r["variant"]))
    md.append(table(ev, ["runtime", "variant", ("exec match", "exec_match"), ("exact match", "exact_match"),
                         ("token F1", "token_f1"), ("n", "n"), ("exec scored", "exec_scored"), "errors"]))

    with open(os.path.join(a.results, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"wrote {a.results}/report.md and plots/")


if __name__ == "__main__":
    main()
