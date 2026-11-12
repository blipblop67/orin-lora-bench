"""Optional: GPTQ W4A16 checkpoint of the merged model for vLLM (the preferred vLLM
quantisation on Orin). Calibrates on the fine-tuning task's own prompts.

Runs in its own venv (.venv-quant) because llm-compressor pins transformers/torch versions.
If it does not install on the Jetson, run this same script on any CUDA workstation and
copy outputs/w4a16/ back; the checkpoint format is hardware independent.
"""
import argparse
import os
import sys

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from data import load_split, to_messages  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--samples", type=int, default=128)
ap.add_argument("--seq", type=int, default=512)
a = ap.parse_args()

preset = os.environ.get("DATASET_PRESET", "sql")
tok = AutoTokenizer.from_pretrained(a.model)
texts = [tok.apply_chat_template(to_messages(r, preset)[0], tokenize=False)
         for r in load_split(preset, "train", a.samples, seed=7)]
ds = Dataset.from_dict({"text": texts})

oneshot(model=a.model, dataset=ds,
        recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
        max_seq_length=a.seq, num_calibration_samples=len(texts), output_dir=a.out)
tok.save_pretrained(a.out)
print(f"W4A16 -> {a.out}")
