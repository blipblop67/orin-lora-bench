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
import matplotlib.ticker  # noqa: E402,F401  (PercentFormatter)

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


# Summary chart colours: slots 1-2 of a CVD-validated categorical palette, plus neutral chart chrome.
RUNTIME_COLOR = {"llama.cpp": "#2a78d6", "vllm": "#eb6834"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def summary_points(R, sizes):
    """One point per deployable variant: single-stream decode speed vs execution accuracy.

    Untuned baselines have no serve record and come back separately as a reference band.
    'ft-X (merged)' rows (llama.cpp lora part) re-measure the same file as 'ft-X', so they are dropped.
    """
    serve1 = {(r["runtime"], r["variant"]): r for r in latest(R["lcpp_serve"] + R["vllm_serve"],
                                                             ["runtime", "variant", "concurrency"])
              if r.get("status") == "ok" and r.get("concurrency") == 1}
    evals = latest(R["lcpp_eval"] + R["vllm_eval"], ["runtime", "variant"])
    names = {v for _, v in serve1}
    pts, base = [], []
    for e in evals:
        acc, n = e.get("exec_match"), e.get("exec_scored")
        if acc is None or not n:
            continue
        v = e["variant"]
        if "no fine-tune" in v:
            base.append(acc)
            continue
        if v.endswith(" (merged)") and v[: -len(" (merged)")] in names:
            continue
        s = serve1.get((e["runtime"], v))
        if not s or not s.get("decode_tok_s_per_req"):
            continue
        if v.startswith("ft-"):
            label, mb = v[3:], sizes.get(f"{v}.gguf")
        elif "runtime LoRA" in v:
            label, mb = "Q4_K_M + runtime LoRA", (sizes.get("base-Q4_K_M.gguf") or 0) + (sizes.get("lora-F16.gguf") or 0)
        elif v == "merged-bf16":
            label, mb = "bf16 merged", sizes.get("merged")
        elif v == "base+lora":
            label, mb = "bf16 + runtime LoRA", (sizes.get("merged") or 0) + (sizes.get("lora-F16.gguf") or 0)
        else:
            label, mb = v, None
        se = (acc * (1 - acc) / n) ** 0.5
        pts.append({"runtime": e["runtime"], "label": label, "mb": mb, "speed": s["decode_tok_s_per_req"],
                    "acc": acc, "ci": 1.96 * se, "n": n})
    return pts, base


def pareto(pts):
    """Points no other point beats on both speed and accuracy, ordered by speed."""
    front = [p for p in pts if not any(q["speed"] >= p["speed"] and q["acc"] >= p["acc"] and
                                       (q["speed"] > p["speed"] or q["acc"] > p["acc"]) for q in pts)]
    return sorted(front, key=lambda p: p["speed"])


def _direct_labels(fig, ax, pts, avoid_line=()):
    """Label every point, trying offsets around it until the text overlaps no other label, point or line."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # (x, y, weight): covering a point is far worse than crossing the line (which shows through the label box)
    marks = [(*ax.transData.transform((p["speed"], p["acc"])), 100) for p in pts]
    for a, b in zip(avoid_line, avoid_line[1:]):  # sample the polyline densely so labels steer clear of it
        marks += [(*ax.transData.transform((a[0] + (b[0] - a[0]) * t / 40, a[1] + (b[1] - a[1]) * t / 40)), 1)
                  for t in range(41)]
    taken = []  # display-space bboxes already used
    # (dx, dy) in points: right, left, then stepped above/below on both sides, nearest first
    cands = [(8, 0), (-8, 0)] + [(s * dx, k * 11) for k in (1, -1, 2, -2, 3, -3, 4, -4)
                                 for dx in (8, 24) for s in (1, -1)]

    def place(txt, p, dx, dy):
        return ax.annotate(txt, (p["speed"], p["acc"]), xytext=(dx, dy), textcoords="offset points",
                           fontsize=8.5, color=INK, va="center", ha="left" if dx > 0 else "right", zorder=4,
                           bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none", alpha=0.85),
                           arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6, shrinkA=0, shrinkB=5)
                           if dy or abs(dx) > 8 else None)

    for p in sorted(pts, key=lambda p: -p["acc"]):
        txt = p["label"] + (f" · {p['mb']:,.0f} MB" if p.get("mb") else "")
        best = None  # (cost, dx, dy): labels must never overlap; marks/line crossings are least-bad fallbacks
        for dx, dy in cands:
            t = place(txt, p, dx, dy)
            bb = t.get_window_extent(r).expanded(1.04, 1.15)
            t.remove()
            inside = ax.bbox.contains(bb.x0, bb.y0) and ax.bbox.contains(bb.x1, bb.y1)
            if not inside or any(bb.overlaps(o) for o in taken):
                continue
            cost = sum(w for x, y, w in marks if bb.contains(x, y))
            if best is None or cost < best[0]:
                best = (cost, dx, dy, bb)
            if cost == 0:
                break
        dx, dy, bb = (best[1], best[2], best[3]) if best else (8, 0, None)
        place(txt, p, dx, dy)
        if bb is not None:
            taken.append(bb)


def summary_chart(pts, base, path):
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    xs = [p["speed"] for p in pts]
    xlo, xhi = min(xs) - 22, max(xs) + 28  # margins for direct labels on either side
    if base:
        lo, hi = min(base), max(base)
        ax.axhspan(lo - 0.002, hi + 0.002, color=GRID, zorder=0, lw=0)
        ax.text(xlo + 1, hi + 0.004, f"untuned base model ({lo:.1%}–{hi:.1%})" if hi > lo
                else f"untuned base model ({lo:.1%})", color=INK2, fontsize=8.5, va="bottom")
    front = pareto(pts)
    ax.plot([p["speed"] for p in front], [p["acc"] for p in front], color=INK2, lw=1.2, ls=(0, (4, 3)),
            zorder=1, label="best trade-offs (nothing is both faster and more accurate)")
    for rt, col in RUNTIME_COLOR.items():
        sel = [p for p in pts if p["runtime"] == rt]
        if not sel:
            continue
        ax.errorbar([p["speed"] for p in sel], [p["acc"] for p in sel], yerr=[p["ci"] for p in sel], fmt="none",
                    ecolor=col, elinewidth=1, alpha=0.35, capsize=0, zorder=2)
        ax.scatter([p["speed"] for p in sel], [p["acc"] for p in sel], s=70, color=col, edgecolors=SURFACE,
                   linewidths=2, zorder=5, label="llama.cpp (GGUF)" if rt == "llama.cpp" else "vLLM (bf16)")
    ax.set_xlim(xlo, xhi)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _direct_labels(fig, ax, pts, [(p["speed"], p["acc"]) for p in front])
    ax.set_xlabel("generation speed, one request at a time (tokens/s)  →  faster", color=INK2)
    ax.set_ylabel("SQL answers returning the correct result  →  better", color=INK2)
    ax.set_title("Fine-tuned Qwen2.5-0.5B on Jetson Orin Nano 8 GB: accuracy vs speed",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    n = max(p["n"] for p in pts)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK2)
    fig.text(0.01, 0.01, f"Bars: 95% confidence interval on {n} executable test queries.\nSpeed: mean decode "
             "rate at concurrency 1, same OpenAI-compatible client for both runtimes.",
             fontsize=7.5, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


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
    head = len(md)

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

    # ---- summary: the one chart to look at, placed right under the training table ----
    pts, base = summary_points(R, sizes)
    if pts:
        summary_chart(pts, base, f"{P}/summary.png")
        best = {p["label"] for p in pareto(pts)}
        rows = [{"runtime": p["runtime"], "variant": p["label"], "MB": p["mb"], "tok/s": round(p["speed"], 1),
                 "exec": f"{p['acc']:.1%} ± {p['ci']:.1%}", "best trade-off": "✓" if p["label"] in best else ""}
                for p in sorted(pts, key=lambda p: -p["acc"])]
        summary = ["\n## Summary: accuracy vs speed\n", "![accuracy vs speed](plots/summary.png)\n",
                   table(rows, ["runtime", "variant", "MB", ("decode tok/s (1 req)", "tok/s"),
                                ("exec match (95% CI)", "exec"), "best trade-off"])]
        md[head:head] = summary  # top of the report, right after the device line
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
