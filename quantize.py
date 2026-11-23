"""MXFP4 (W4A16, group 32) quantization of a Qwen3.8-27B checkpoint with llm-compressor.

method rtn:  QuantizationModifier, data-free round-to-nearest (CPU is fine).
method gptq: GPTQModifier with Swift-1.5's recipe (static act-order, 89 samples, 2048 tokens).
Same targets/ignore as slybase/Swift-1.5-Qwen3.8-27B-heretic-MXFP4-GPTQ.
"""
import argparse
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier

IGNORE = ["lm_head", "re:.*visual.*", "re:.*mtp.*"]
COPY = ["chat_template.jinja", "generation_config.json", "preprocessor_config.json",
        "processor_config.json", "video_preprocessor_config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "README.md"]

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--method", choices=["rtn", "gptq"], default="rtn")
args = ap.parse_args()

t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(args.src, dtype=torch.bfloat16, device_map="cpu")
processor = AutoProcessor.from_pretrained(args.src)
print(f"loaded in {time.time() - t0:.0f}s: {type(model).__name__}", flush=True)

if args.method == "rtn":
    recipe = QuantizationModifier(targets="Linear", scheme="MXFP4A16", ignore=IGNORE)
    oneshot(model=model, recipe=recipe)
else:
    raise SystemExit("gptq: not wired up yet (needs a calibration set); run rtn first")
print(f"quantized at {time.time() - t0:.0f}s", flush=True)

out = Path(args.out)
model.save_pretrained(out, save_compressed=True)
processor.save_pretrained(out)
for name in COPY:
    if (Path(args.src) / name).exists():
        shutil.copy2(Path(args.src) / name, out / name)
print(f"saved to {out} at {time.time() - t0:.0f}s", flush=True)
