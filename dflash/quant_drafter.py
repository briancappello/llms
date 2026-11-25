"""Quantize a bf16 DFlash2 drafter to W4A16 compressed-tensors pack-quantized (group 128, symmetric,
no act-order) with GPTQ, in the exact layout of syvai/Qwen3.8-27B-DFlash2-W4A16.

Adapted from syv-ai/qwen38-27b-rtx3090 drafter/quant_dflash2.py (Apache-2.0). Hessians come from
train_drafter.py --hessians (the drafter's own query-path inputs on captured target traffic).

  quant_drafter.py SRC_BF16_DIR DST_DIR HESSIANS.pt [--fc-bits 4|16] [--rtn]

Quantized: layers.N.self_attn.{q,k,v,o}_proj, layers.N.mlp.{gate,up,down}_proj and fc (int4 GPTQ, g128).
bf16: grouped-conv kernel projections, candidate selector, norms (vLLM builds them unquantized).
"""
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gptq_utils import gptq_quantize, rtn_quantize, dequant  # noqa: E402

S = sys.argv[1].rstrip("/") + "/"; D = sys.argv[2].rstrip("/") + "/"; HP = sys.argv[3]
FC_BITS = int(sys.argv[sys.argv.index("--fc-bits") + 1]) if "--fc-bits" in sys.argv else 4
RTN = "--rtn" in sys.argv
BITS, GROUP = 4, 128
dev = "cuda" if torch.cuda.is_available() else "cpu"

cfg = json.load(open(S + "config.json"))
cfg.pop("quantization_config", None)
L = cfg["num_hidden_layers"]
LIN = {}
for i in range(L):
    for p in ("q_proj", "k_proj", "v_proj"):
        LIN[f"layers.{i}.self_attn.{p}"] = f"layers.{i}.self_attn.qkv_proj"
    LIN[f"layers.{i}.self_attn.o_proj"] = f"layers.{i}.self_attn.o_proj"
    for p in ("gate_proj", "up_proj"):
        LIN[f"layers.{i}.mlp.{p}"] = f"layers.{i}.mlp.gate_up_proj"
    LIN[f"layers.{i}.mlp.down_proj"] = f"layers.{i}.mlp.down_proj"
if FC_BITS < 16:
    LIN["fc"] = "fc"

HS = torch.load(HP, map_location="cpu") if not RTN else {}
with safe_open(S + "model.safetensors", "pt") as f:
    meta = f.metadata()
    tensors = {k: f.get_tensor(k) for k in f.keys()}
out = {}; stats = []; t0 = time.time()
for name, w in tensors.items():
    base = name[: -len(".weight")] if name.endswith(".weight") else None
    if base in LIN:
        W = w.to(dev)
        bits = FC_BITS if base == "fc" else BITS
        if RTN:
            q, s = rtn_quantize(W, bits, GROUP)
        else:
            if LIN[base] not in HS:
                raise SystemExit(f"no Hessian for {LIN[base]}")
            q, s = gptq_quantize(W, HS[LIN[base]]["H"].to(dev), bits=bits, group=GROUP, blocksize=GROUP)
        rel = ((dequant(q, s) - W.float()).norm() / W.float().norm()).item()
        stats.append((base, bits, rel))
        out[base + ".weight_packed"] = pack_to_int32(q.cpu(), bits, packed_dim=1).contiguous()
        out[base + ".weight_scale"] = s.to(torch.float16).cpu().contiguous()
        out[base + ".weight_shape"] = torch.tensor(list(W.shape), dtype=torch.int64)
        del W, q, s
    else:
        out[name] = w
for base, bits, rel in stats:
    if base.startswith("layers.0.") or not base.startswith("layers."):
        print(f"  int{bits} {base}: rel err {rel:.4f}")
print(f"quantized {len(stats)} matrices in {time.time() - t0:.0f}s; mean rel err {sum(r for _, _, r in stats) / len(stats):.4f}")

os.makedirs(D, exist_ok=True)
save_file(out, D + "model.safetensors", metadata=meta or {"format": "pt"})
for f in os.listdir(S):
    if f not in ("model.safetensors", "config.json") and not f.startswith(".") and not f.endswith(".pt") and os.path.isfile(S + f):
        shutil.copy(S + f, D + f)
ignore = ["re:.*kernel_projection$", "re:.*candidate_selector.*", "re:.*hidden_projection$"]
if FC_BITS >= 16:
    ignore.append("re:.*\\.fc$")
cfg["quantization_config"] = {
    "config_groups": {"group_0": {"format": "pack-quantized", "input_activations": None, "output_activations": None, "targets": ["Linear"],
                                  "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": GROUP, "num_bits": BITS,
                                              "observer": "memoryless_minmax", "observer_kwargs": {}, "scale_dtype": None, "strategy": "group",
                                              "symmetric": True, "type": "int", "zp_dtype": None}}},
    "format": "pack-quantized", "global_compression_ratio": None, "ignore": ignore, "kv_cache_scheme": None,
    "quant_method": "compressed-tensors", "quantization_status": "compressed", "sparsity_config": {}, "transform_config": {}, "version": "0.17.0"}
json.dump(cfg, open(D + "config.json", "w"), indent=2)
print(f"wrote {D} ({os.path.getsize(D + 'model.safetensors') / 2**30:.2f} GiB)")
