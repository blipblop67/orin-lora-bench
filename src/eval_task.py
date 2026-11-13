"""Task accuracy of a served model (any OpenAI-compatible server).

For the sql preset each prediction is scored two ways:
  exact  - normalised string match with the reference query
  exec   - run schema+inserts in in-memory SQLite, compare result sets of reference vs prediction
           (only examples whose reference query executes in SQLite are counted)
For alpaca there is no checkable answer; we report a token-F1 against the reference.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(__file__))
from data import load_split, to_messages  # noqa: E402


def clean_sql(s: str) -> str:
    s = re.sub(r"^```(?:sql)?\s*|\s*```$", "", s.strip(), flags=re.I)
    return s.strip().rstrip(";").strip()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", clean_sql(s)).lower()


def run_sql(context: str, q: str):
    con = sqlite3.connect(":memory:")
    t0 = time.time()
    con.set_progress_handler(lambda: 1 if time.time() - t0 > 2 else 0, 10000)
    try:
        con.executescript(context)
        rows = con.execute(q).fetchall()
        return sorted(map(repr, rows))
    finally:
        con.close()


def exec_match(context, gold, pred):
    try:
        g = run_sql(context, clean_sql(gold))
    except Exception:
        return None  # reference not SQLite-compatible; skip
    try:
        return run_sql(context, clean_sql(pred)) == g
    except Exception:
        return False


def f1(a: str, b: str) -> float:
    ta, tb = a.lower().split(), b.lower().split()
    common = sum((Counter(ta) & Counter(tb)).values())
    if not common:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def ask(base_url, model, msgs, max_tokens=256):
    r = requests.post(base_url + "/v1/chat/completions", timeout=300,
                      json={"model": model, "messages": msgs, "temperature": 0.0, "max_tokens": max_tokens})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def run_eval(base_url, model, preset="sql", n=200, concurrency=4) -> dict:
    rows = list(load_split(preset, "test", n))

    def work(row):
        msgs, ref = to_messages(row, preset, with_answer=False)
        try:
            pred = ask(base_url, model, msgs)
        except Exception as ex:  # noqa: BLE001
            return {"error": str(ex)[:120]}
        if preset == "sql":
            return {"exact": norm(pred) == norm(ref), "exec": exec_match(row["sql_context"], ref, pred)}
        return {"f1": f1(pred, ref)}

    t0 = time.time()
    with ThreadPoolExecutor(concurrency) as ex:
        res = list(ex.map(work, rows))
    ok = [r for r in res if "error" not in r]
    out = {"preset": preset, "n": len(rows), "errors": len(res) - len(ok), "eval_wall_s": round(time.time() - t0, 1)}
    if preset == "sql":
        ex_scored = [r["exec"] for r in ok if r["exec"] is not None]
        out.update(exact_match=round(sum(r["exact"] for r in ok) / max(1, len(ok)), 4),
                   exec_match=round(sum(ex_scored) / max(1, len(ex_scored)), 4), exec_scored=len(ex_scored),
                   # Per-question outcomes in dataset order (None = request error / reference not executable),
                   # so two variants can be compared on the same questions with a paired test (McNemar).
                   per_item={"exact": [None if "error" in r else int(r["exact"]) for r in res],
                             "exec": [None if "error" in r or r["exec"] is None else int(r["exec"]) for r in res]})
    else:
        out["token_f1"] = round(sum(r["f1"] for r in ok) / max(1, len(ok)), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="ft")
    ap.add_argument("--preset", default=os.environ.get("DATASET_PRESET", "sql"))
    ap.add_argument("--n", type=int, default=int(os.environ.get("EVAL_SAMPLES", 200)))
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    print(json.dumps(run_eval(a.base_url, a.model, a.preset, a.n, a.concurrency), indent=2))


if __name__ == "__main__":
    main()
