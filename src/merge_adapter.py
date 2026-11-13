"""Merge a LoRA/QLoRA adapter into a bf16 copy of its base model (on CPU, to keep GPU/unified
memory free) and save a standalone HF checkpoint for vLLM and GGUF conversion.

QLoRA adapters were trained against an NF4 base; merging into the bf16 base is the standard
practice and the small mismatch is measured later by the task eval.
"""
import argparse
import json
import os
import shutil

import torch
import transformers
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Merging LoRA changes weight values only, so the base model's own config/tokenizer files describe the merged
# model exactly. Copying them keeps the checkpoint readable by the older transformers inside the vLLM container.
BASE_FILES = ["config.json", "generation_config.json",
              "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"]

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

cfg = json.load(open(os.path.join(a.adapter, "adapter_config.json")))
base = cfg["base_model_name_or_path"]
major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
dkey = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
print(f"base={base}")
model = AutoModelForCausalLM.from_pretrained(base, **{dkey: torch.bfloat16}, device_map={"": "cpu"})
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True)

# Overwrite what transformers 5 just wrote with the base model's originals. Its files break transformers 4.x:
#  - config.json moves rope_theta into "rope_parameters"; 4.x then falls back to Qwen2's default 10000 instead
#    of 1e6, which quietly halved vLLM task accuracy (37.6% vs 74.5% for the same weights via runtime LoRA)
#  - tokenizer_config.json stores extra_special_tokens as a list ("'list' object has no attribute 'keys'")
src = base if os.path.isdir(base) else snapshot_download(base, allow_patterns=BASE_FILES)
copied = [f for f in BASE_FILES if os.path.exists(os.path.join(src, f))]
for f in copied:
    shutil.copy(os.path.join(src, f), os.path.join(a.out, f))
if "tokenizer_config.json" not in copied:  # local base without tokenizer files: fall back to saving one
    AutoTokenizer.from_pretrained(base).save_pretrained(a.out)
with open(os.path.join(a.out, "merge_info.json"), "w") as f:
    json.dump({"base": base, "adapter": os.path.abspath(a.adapter)}, f, indent=2)
print(f"merged -> {a.out}")
