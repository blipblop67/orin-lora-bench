"""Runtime-agnostic serving benchmark over the OpenAI /v1/completions streaming API.

The same client drives llama-server and vLLM so the numbers are apples-to-apples:
TTFT, TPOT (time per output token after the first), end-to-end latency, aggregate
output tokens/s, plus peak unified-memory use during the run.

  python bench_client.py --base-url http://127.0.0.1:8080 --model ft \
      --tokenizer outputs/merged --input-len 512 --output-len 128 --concurrency 1,4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(__file__))
from memmon import MemMonitor  # noqa: E402

WORDS = ("the of and to in is that for it as was with be by on not he this are or his from at which but "
         "have an they you were her she there been one all we their has would when if so no what up out "
         "more about into than them can only other time new some could these two may first then do any "
         "like my now over such our man me even most made after also did many off before must well back "
         "through years much where your way down should because each just those people how too little "
         "state good very make world still own see men work long here get both between life being under "
         "never day same another know while last might us great old year since against go came right used "
         "take three robot sensor camera edge model memory power").split()


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def make_prompts(tokenizer_path: str, n: int, input_len: int, seed: int = 0) -> list[str]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    rng = random.Random(seed)
    out = []
    for i in range(n):
        # unique prefix defeats prefix caching so every request pays full prefill
        text = f"[{i}-{rng.randrange(10**9)}] " + " ".join(rng.choice(WORDS) for _ in range(input_len * 2))
        ids = tok(text, add_special_tokens=False)["input_ids"][: max(1, input_len - 8)]
        out.append(tok.decode(ids) + "\nSummarise the text above.")
    return out


def one_request(url, model, prompt, output_len, timeout):
    body = {"model": model, "prompt": prompt, "max_tokens": output_len, "temperature": 0.0,
            "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft, n_chunks, usage, err = None, 0, None, None
    try:
        with requests.post(url + "/v1/completions", json=body, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    break
                c = json.loads(data)
                if c.get("usage"):
                    usage = c["usage"]
                ch = c.get("choices") or []
                if ch and ch[0].get("text"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_chunks += 1
    except requests.RequestException as ex:
        err = str(ex)[:200]
    e2e = time.perf_counter() - t0
    if err or ttft is None:
        return {"error": err or "no tokens"}
    n_out = (usage or {}).get("completion_tokens") or n_chunks
    return {"ttft": ttft, "e2e": e2e, "n_out": n_out,
            "n_in": (usage or {}).get("prompt_tokens"),
            "tpot": (e2e - ttft) / max(1, n_out - 1)}


def run_bench(base_url, model, tokenizer, input_len=512, output_len=128, concurrency=1,
              num_prompts=16, warmup=2, timeout=900, baseline_mb=None) -> dict:
    prompts = make_prompts(tokenizer, num_prompts + warmup, input_len, seed=input_len * 1000 + concurrency)
    for p in prompts[:warmup]:
        one_request(base_url, model, p, min(16, output_len), timeout)
    with MemMonitor(interval=0.25, baseline_mb=baseline_mb) as mon:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            res = list(ex.map(lambda p: one_request(base_url, model, p, output_len, timeout), prompts[warmup:]))
        wall = time.perf_counter() - t0
    ok = [r for r in res if "error" not in r]
    errs = [r["error"] for r in res if "error" in r]
    ms = lambda xs: [x * 1000 for x in xs]  # noqa: E731
    ttft, tpot, e2e = ms([r["ttft"] for r in ok]), ms([r["tpot"] for r in ok]), ms([r["e2e"] for r in ok])
    out_tok = sum(r["n_out"] for r in ok)
    n_in = [r["n_in"] for r in ok if r.get("n_in")]
    rnd = lambda x: None if x is None else round(x, 2)  # noqa: E731
    return {
        "input_len": input_len, "output_len": output_len, "concurrency": concurrency,
        "num_prompts": num_prompts, "completed": len(ok), "errors": len(errs), "first_error": errs[0] if errs else None,
        "prompt_tokens_mean": rnd(statistics.mean(n_in)) if n_in else None,
        "ttft_ms_mean": rnd(statistics.mean(ttft)) if ttft else None, "ttft_ms_p50": rnd(pct(ttft, 50)),
        "ttft_ms_p90": rnd(pct(ttft, 90)),
        "tpot_ms_mean": rnd(statistics.mean(tpot)) if tpot else None, "tpot_ms_p50": rnd(pct(tpot, 50)),
        "tpot_ms_p90": rnd(pct(tpot, 90)),
        "e2e_ms_p50": rnd(pct(e2e, 50)),
        "decode_tok_s_per_req": rnd(1000 / statistics.mean(tpot)) if tpot else None,
        "prefill_tok_s_est": rnd(input_len / (statistics.mean(ttft) / 1000)) if ttft and concurrency == 1 else None,
        "output_tok_s": rnd(out_tok / wall), "req_s": rnd(len(ok) / wall), "wall_s": rnd(wall),
        **mon.summary(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="ft")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--concurrency", default="1")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--tag", action="append", default=[], help="k=v added to each record")
    ap.add_argument("--out", help="append JSONL records here")
    a = ap.parse_args()
    tags = dict(t.split("=", 1) for t in a.tag)
    for c in [int(x) for x in a.concurrency.split(",")]:
        rec = {"kind": "serve", **tags, **run_bench(a.base_url, a.model, a.tokenizer, a.input_len,
                                                   a.output_len, c, a.num_prompts)}
        print(json.dumps(rec))
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
