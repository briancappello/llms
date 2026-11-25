"""MXFP4 (W4A16, group 32) GPTQ quantization of a Qwen3.8-27B-style checkpoint with llm-compressor,
Swift-1.5's recipe: GPTQModifier(targets=[Linear], scheme=MXFP4A16, actorder=static,
ignore=[lm_head, visual, mtp]), sequential pipeline, 89 samples, max_seq_length 2048.

Calibration text = the target's own answers (generate.py output over a prompt mix), rendered with the
model's chat template exactly as served: prompt + generation prompt + reasoning + </think> + answer.

Usage (inside the serving image, GPU visible, PYTHONPATH=gptq-site):
  quantize_gptq.py SRC OUT --calib calib_gen.jsonl --prompts calib.jsonl [--dry-run]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

IGNORE = ["lm_head", "re:.*visual.*", "re:.*mtp.*"]
COPY = ["chat_template.jinja", "generation_config.json", "preprocessor_config.json", "processor_config.json",
        "video_preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--calib", required=True); ap.add_argument("--prompts", required=True)
ap.add_argument("--samples", type=int, default=89); ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--scheme", default="MXFP4A16", help="llm-compressor preset; W4A16 = int4 sym group 128 (Radiance's q38-27b-w4a16.recipe)")
ap.add_argument("--ignore", action="append", default=[], help="extra ignore pattern; Radiance serves the delta net's a/b "
                "projections bf16: --ignore 're:.*linear_attn.in_proj_[ab]$'")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.src)
tmpl = (Path(args.src) / "chat_template.jinja").read_text()
prompts = {json.loads(line)["id"]: json.loads(line) for line in open(args.prompts)}
texts = []
for line in open(args.calib):
    g = json.loads(line)
    if "error" in g or not (g.get("content") or g.get("reasoning")):
        continue
    p = prompts[g["id"]]
    msgs = p.get("messages") or [{"role": "user", "content": p["prompt"]}]
    head = tok.apply_chat_template(msgs, chat_template=tmpl, add_generation_prompt=True, tokenize=False,
                                 reasoning_effort="medium")   # the serve passes --default-chat-template-kwargs reasoning_effort=medium
    body = ""
    if g.get("reasoning"):
        body = ("" if head.rstrip().endswith("<think>") else "<think>\n") + g["reasoning"].strip()
    if g.get("content"):   # answers cut inside the reasoning (finish=length) stay open, as served
        body += ("\n</think>\n\n" if g.get("reasoning") else "") + g["content"].strip() + "<|im_end|>\n"
    texts.append(head + body)
texts = texts[: args.samples]
enc = [tok(t, add_special_tokens=False, truncation=True, max_length=args.max_len) for t in texts]
lens = [len(e["input_ids"]) for e in enc]
print(f"{len(texts)} calibration samples, tokens min/mean/max {min(lens)}/{sum(lens) / len(lens):.0f}/{max(lens)}, "
      f"{sum(n == args.max_len for n in lens)} truncated", flush=True)
print("---- sample 0 head ----\n" + texts[0][:1200] + "\n...\n" + texts[0][-300:], flush=True)
if args.dry_run:
    raise SystemExit(0)

from llmcompressor import oneshot  # noqa: E402
from llmcompressor.modifiers.quantization import GPTQModifier  # noqa: E402

# the serving image ships no flash / mem-efficient SDPA kernels for gfx1201 (hipErrorInvalidImage): use the math path
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
ds = Dataset.from_dict({"input_ids": [e["input_ids"] for e in enc], "attention_mask": [e["attention_mask"] for e in enc]})
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(args.src, dtype=torch.bfloat16, device_map=None)
processor = AutoProcessor.from_pretrained(args.src)
print(f"loaded in {time.time() - t0:.0f}s: {type(model).__name__}", flush=True)
recipe = GPTQModifier(targets=["Linear"], scheme=args.scheme, actorder="static", ignore=IGNORE + args.ignore)
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=args.max_len, num_calibration_samples=len(enc),
        pipeline="sequential", layerwise_compression=True)   # pack each layer after use: bf16 copy freed, peak RAM ~model size
print(f"quantized at {time.time() - t0:.0f}s", flush=True)
out = Path(args.out)
model.save_pretrained(out, save_compressed=True, max_shard_size="2400MB")
processor.save_pretrained(out)
for name in COPY:
    if (Path(args.src) / name).exists():
        shutil.copy2(Path(args.src) / name, out / name)
print(f"saved to {out} at {time.time() - t0:.0f}s", flush=True)
