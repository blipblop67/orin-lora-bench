"""Merge a LoRA/QLoRA adapter into a bf16 copy of its base model (on CPU, to keep GPU/unified
memory free) and save a standalone HF checkpoint for vLLM and GGUF conversion.

QLoRA adapters were trained against an NF4 base; merging into the bf16 base is the standard
practice and the small mismatch is measured later by the task eval.
"""
import argparse
import json
import os

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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
AutoTokenizer.from_pretrained(a.adapter).save_pretrained(a.out)
with open(os.path.join(a.out, "merge_info.json"), "w") as f:
    json.dump({"base": base, "adapter": os.path.abspath(a.adapter)}, f, indent=2)
print(f"merged -> {a.out}")
