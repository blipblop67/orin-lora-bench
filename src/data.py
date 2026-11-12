"""Dataset presets and prompt formatting shared by training, eval and perplexity.

Default task is text-to-SQL (gretelai/synthetic_text_to_sql): small models improve
visibly with LoRA, and answers can be checked by *executing* them in SQLite.
"""
from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset

SQL_SYSTEM = ("You are a text-to-SQL assistant. Given a database schema and a question, "
              "reply with a single SQL query and nothing else.")
ALPACA_SYSTEM = "You are a helpful assistant."

PRESETS = {
    "sql": {"hf": "gretelai/synthetic_text_to_sql"},
    "alpaca": {"hf": "yahma/alpaca-cleaned"},
}


def load_split(preset: str, split: str, n: int, seed: int = 42):
    """Return n rows of 'train' or 'test'. Alpaca has no test split, so we carve one out."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset}")
    ds = load_dataset(PRESETS[preset]["hf"])
    if "test" not in ds:
        ds = ds["train"].train_test_split(test_size=0.02, seed=seed)
    part = ds[split].shuffle(seed=seed)
    return part.select(range(min(n, len(part))))


def to_messages(row: dict, preset: str, with_answer: bool = True):
    """Returns (messages, reference_answer)."""
    if preset == "sql":
        system = SQL_SYSTEM
        user = f"Schema:\n{row['sql_context']}\n\nQuestion: {row['sql_prompt']}"
        answer = row["sql"]
    else:
        system = ALPACA_SYSTEM
        user = row["instruction"] + (f"\n\n{row['input']}" if row.get("input") else "")
        answer = row["output"]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if with_answer:
        msgs.append({"role": "assistant", "content": answer})
    return msgs, answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=os.environ.get("DATASET_PRESET", "sql"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dump-text", help="plain-text file for llama-perplexity")
    ap.add_argument("--dump-jsonl", help="messages as JSONL")
    a = ap.parse_args()
    rows = load_split(a.preset, a.split, a.n)
    if a.dump_text:
        os.makedirs(os.path.dirname(a.dump_text) or ".", exist_ok=True)
        with open(a.dump_text, "w") as f:
            for r in rows:
                msgs, _ = to_messages(r, a.preset)
                # ChatML-style rendering so perplexity reflects the fine-tuned format
                for m in msgs:
                    f.write(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
                f.write("\n")
    if a.dump_jsonl:
        with open(a.dump_jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps({"messages": to_messages(r, a.preset)[0]}) + "\n")
    print(f"{len(rows)} rows ({a.preset}/{a.split})")


if __name__ == "__main__":
    main()
