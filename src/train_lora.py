"""LoRA / QLoRA fine-tuning sized for Jetson Orin Nano (8 GB unified memory).

Memory levers used: bf16 base (lora) or NF4 4-bit base (qlora), gradient checkpointing,
micro-batch + gradient accumulation, loss masked to the assistant answer only,
paged 8-bit AdamW for qlora. Appends one record to results/train.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

sys.path.insert(0, os.path.dirname(__file__))
from data import load_split, to_messages  # noqa: E402
from memmon import MemMonitor, device_meta  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse():
    e = os.environ.get
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=e("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--mode", choices=["lora", "qlora"], default="qlora")
    ap.add_argument("--preset", default=e("DATASET_PRESET", "sql"))
    ap.add_argument("--train-samples", type=int, default=int(e("TRAIN_SAMPLES", 2000)))
    ap.add_argument("--eval-samples", type=int, default=int(e("EVAL_SAMPLES", 200)))
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--rank", type=int, default=int(e("LORA_RANK", 16)))
    ap.add_argument("--alpha", type=int, default=int(e("LORA_ALPHA", 32)))
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="all-linear")
    ap.add_argument("--lr", type=float, default=float(e("LR", 2e-4)))
    ap.add_argument("--batch", type=int, default=int(e("BATCH", 2)))
    ap.add_argument("--grad-accum", type=int, default=int(e("GRAD_ACCUM", 8)))
    ap.add_argument("--max-steps", type=int, default=300, help="<=0 means one epoch")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--out", help="default outputs/train/<model>_<mode>_s<seq>")
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "train.jsonl"))
    return ap.parse_args()


def run_name(a):
    return f"{a.model.split('/')[-1]}_{a.mode}_s{a.max_seq_len}"


def encode(tok, msgs, max_len):
    """Tokenise prompt and answer separately so the label mask is exact."""
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tokenize=False)
    assert full.startswith(prompt), "chat template prefix mismatch"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    if all(x == -100 for x in labels):
        return None  # prompt alone fills the window; no signal
    return {"input_ids": ids, "labels": labels}


def build(tok, preset, split, n, max_len):
    out, dropped = [], 0
    for r in load_split(preset, split, n):
        ex = encode(tok, to_messages(r, preset)[0], max_len)
        if ex is None:
            dropped += 1
        else:
            out.append(ex)
    return out, dropped


class Collator:
    def __init__(self, pad_id):
        self.pad = pad_id

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids, att, lab = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad] * (m - n))
            att.append([1] * n + [0] * (m - n))
            lab.append(b["labels"] + [-100] * (m - n))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att), "labels": torch.tensor(lab)}


class StepTimer(TrainerCallback):
    def __init__(self):
        self.t, self.times, self.losses = None, [], []

    def on_step_begin(self, *a, **k):
        torch.cuda.synchronize()
        self.t = time.perf_counter()

    def on_step_end(self, *a, **k):
        torch.cuda.synchronize()
        self.times.append(time.perf_counter() - self.t)

    def on_log(self, args, state, control, logs=None, **k):
        if logs and "loss" in logs:
            self.losses.append((state.global_step, logs["loss"]))


def load_model(a):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"  # renamed in 4.56
    kw = {dtype_key: torch.bfloat16, "device_map": {"": 0}, "attn_implementation": "sdpa"}
    if a.mode == "qlora":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    model.config.use_cache = False
    if a.mode == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=not a.no_grad_ckpt)
    elif not a.no_grad_ckpt:
        model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                     target_modules=a.target_modules, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    return model


def main():
    a = parse()
    name = run_name(a)
    out = a.out or os.path.join(ROOT, "outputs", "train", name)
    os.makedirs(out, exist_ok=True)
    rec = {"kind": "train", "run": name, "model": a.model, "mode": a.mode, "max_seq_len": a.max_seq_len,
           "rank": a.rank, "batch": a.batch, "grad_accum": a.grad_accum, "power_mode": device_meta()["power_mode"]}

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train, dropped = build(tok, a.preset, "train", a.train_samples, a.max_seq_len)
    evals, _ = build(tok, a.preset, "test", a.eval_samples, a.max_seq_len)
    if not train:
        sys.exit(f"All {dropped} training examples have prompts longer than --max-seq-len {a.max_seq_len}")
    avg_tokens = sum(len(x["input_ids"]) for x in train) / len(train)
    rec.update(train_examples=len(train), dropped_too_long=dropped, avg_tokens_per_example=round(avg_tokens, 1))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    timer = StepTimer()
    status = "ok"
    with MemMonitor(interval=0.5) as mon:
        try:
            t0 = time.perf_counter()
            model = load_model(a)
            rec["load_s"] = round(time.perf_counter() - t0, 1)
            trainable, total = model.get_nb_trainable_parameters()
            rec.update(trainable_params=trainable, total_params=total)

            eff_batch = a.batch * a.grad_accum
            steps = a.max_steps if a.max_steps > 0 else math.ceil(len(train) / eff_batch)
            args = TrainingArguments(
                output_dir=out, per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
                gradient_accumulation_steps=a.grad_accum, max_steps=steps, learning_rate=a.lr,
                lr_scheduler_type="cosine", warmup_steps=max(1, steps // 30), logging_steps=10,
                bf16=True, gradient_checkpointing=not a.no_grad_ckpt,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                optim="paged_adamw_8bit" if a.mode == "qlora" else "adamw_torch",
                save_strategy="no", report_to=[], remove_unused_columns=False,
                dataloader_num_workers=0, prediction_loss_only=True, seed=42)
            trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=evals,
                              data_collator=Collator(tok.pad_token_id), callbacks=[timer])
            res = trainer.train()
            ev = trainer.evaluate()
            # Adapter weights only: the tokenizer is the base model's (merge_adapter.py copies it from there),
            # and a transformers-5 tokenizer saved here would break vLLM's transformers 4.x loading the adapter.
            model.save_pretrained(os.path.join(out, "adapter"))
            m = res.metrics
            st = sorted(timer.times[3:]) or timer.times  # skip warm-up steps
            rec.update(
                steps=steps, train_runtime_s=round(m["train_runtime"], 1),
                train_samples_per_s=round(m["train_samples_per_second"], 3),
                train_tokens_per_s=round(m["train_samples_per_second"] * avg_tokens, 1),
                sec_per_optimizer_step_median=round(st[len(st) // 2], 3) if st else None,
                final_train_loss=timer.losses[-1][1] if timer.losses else m.get("train_loss"),
                eval_loss=round(ev["eval_loss"], 4), eval_ppl=round(math.exp(ev["eval_loss"]), 3),
                loss_curve=timer.losses)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as ex:
            # On Jetson an allocation failure often surfaces as "NVML_SUCCESS == r INTERNAL ASSERT FAILED"
            # (the allocator's OOM path queries NVML, which Tegra lacks) instead of a clean OutOfMemoryError.
            msg = str(ex).lower()
            if not (isinstance(ex, torch.cuda.OutOfMemoryError) or "out of memory" in msg or "nvml_success" in msg):
                raise
            status = "oom"
            rec["error"] = str(ex)[:400]
    rec.update(status=status,
               torch_peak_alloc_mb=round(torch.cuda.max_memory_allocated() / 2**20),
               torch_peak_reserved_mb=round(torch.cuda.max_memory_reserved() / 2**20),
               **mon.summary())
    os.makedirs(os.path.dirname(a.results), exist_ok=True)
    with open(a.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps({k: v for k, v in rec.items() if k != "loss_curve"}, indent=2))
    sys.exit(0 if status == "ok" else 3)


if __name__ == "__main__":
    main()
