#!/usr/bin/env python3
"""
Rebuild a HF (bf16 safetensors) checkpoint for a Qwen3.5/3.6-MoE model (Qwen3_5MoeForConditionalGeneration, e.g.
Ornith-1.5-35B-A3B and its derivatives) from a llama.cpp GGUF, so it can be fed to exllamav3's convert.py.

A reference HF checkpoint of the same architecture supplies the layout: tensor names, shapes, dtypes, shard
assignment, config/tokenizer files, and optionally the tensors the GGUF does not carry well (vision tower,
MTP head). Every rebuilt tensor is compared against the reference (cosine, relative difference); a mapping or
transform error shows as cosine ~0, a genuine fine-tune difference as cosine slightly below 1.

Inverts llama.cpp conversion/qwen.py (Qwen3_5MoeTextModel = _LinearAttentionVReorderBase + Qwen3NextModel):
  - V-head reorder grouped->tiled on in_proj_qkv (V rows), in_proj_z, in_proj_a/b, A_log, dt_bias, conv1d (V
    channels), out_proj (columns)
  - ssm_a = -exp(A_log); conv1d squeezed; +1 on every *norm.weight except linear_attn.norm.weight
  - experts gate_up_proj [E, 2F, D] split into ffn_gate_exps / ffn_up_exps; down_proj -> ffn_down_exps

Usage:
  python gguf_to_hf_qwen35moe.py --gguf X.gguf --ref /path/to/hf_ref --out /path/to/out
      [--take-from-ref visual,mtp] [--check-only N]   (check-only: validate the first N layers, write nothing)
"""
import argparse, json, os, re, shutil, sys, time
sys.path.insert(0, os.path.expanduser("~/dev/llama.cpp/gguf-py"))
import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required = True)
ap.add_argument("--ref", required = True)
ap.add_argument("--out")
ap.add_argument("--take-from-ref", default = "visual", help = "comma list of prefixes: visual, mtp")
ap.add_argument("--check-only", type = int, default = -1)
ap.add_argument("--mtp-layer", type = int, default = None, help = "GGUF block index of the MTP layer (default: num_hidden_layers)")
args = ap.parse_args()

cfg = json.load(open(os.path.join(args.ref, "config.json")))
tc = cfg.get("text_config", cfg)
NK, NV = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
HK, HV = tc["linear_key_head_dim"], tc["linear_value_head_dim"]
NL = tc["num_hidden_layers"]
MTP_BLK = args.mtp_layer if args.mtp_layer is not None else NL
take_ref = [p for p in args.take_from_ref.split(",") if p]

def reorder_v(t, dim, nk, nvpk, hd):
    shape = list(t.shape); dim %= t.dim()
    t = t.reshape(*shape[:dim], nk, nvpk, hd, *shape[dim + 1:])
    perm = list(range(t.dim())); perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)

def unreorder_v(t, dim, hd):
    # inverse of reorder_v(t, dim, NK, NV // NK, hd): same op with the two group axes swapped
    return reorder_v(t, dim, NV // NK, NK, hd)

# ---- GGUF ----
t0 = time.time()
reader = GGUFReader(args.gguf)
gt = {t.name: t for t in reader.tensors}
print(f"GGUF: {len(gt)} tensors, types {sorted({t.tensor_type.name for t in reader.tensors})}", flush = True)

def G(name):
    t = gt[name]
    a = dequantize(t.data, t.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(a)).float()

def blk(n): return f"blk.{n}."

def build(key):
    """HF key -> fp32 tensor from the GGUF (None if the GGUF has no counterpart)"""
    m = re.match(r"(model\.language_model\.layers|mtp\.layers)\.(\d+)\.(.+)$", key)
    if key == "model.language_model.embed_tokens.weight": return G("token_embd.weight")
    if key == "lm_head.weight": return G("output.weight")
    if key == "model.language_model.norm.weight": return G("output_norm.weight") - 1
    if key.startswith("mtp.") and not m:
        sub = {"mtp.fc.weight": "nextn.eh_proj.weight", "mtp.pre_fc_norm_embedding.weight": "nextn.enorm.weight",
               "mtp.pre_fc_norm_hidden.weight": "nextn.hnorm.weight", "mtp.norm.weight": "nextn.shared_head_norm.weight"}[key]
        x = G(blk(MTP_BLK) + sub)
        return x - 1 if "norm" in sub else x
    if not m: return None
    n = int(m.group(2)) if m.group(1).startswith("model") else MTP_BLK + int(m.group(2))
    rest = m.group(3); b = blk(n)
    simple = {
        "input_layernorm.weight": ("attn_norm.weight", -1), "post_attention_layernorm.weight": ("post_attention_norm.weight", -1),
        "self_attn.q_proj.weight": ("attn_q.weight", 0), "self_attn.k_proj.weight": ("attn_k.weight", 0),
        "self_attn.v_proj.weight": ("attn_v.weight", 0), "self_attn.o_proj.weight": ("attn_output.weight", 0),
        "self_attn.q_norm.weight": ("attn_q_norm.weight", -1), "self_attn.k_norm.weight": ("attn_k_norm.weight", -1),
        "mlp.gate.weight": ("ffn_gate_inp.weight", 0), "mlp.shared_expert_gate.weight": ("ffn_gate_inp_shexp.weight", 0),
        "mlp.shared_expert.gate_proj.weight": ("ffn_gate_shexp.weight", 0),
        "mlp.shared_expert.up_proj.weight": ("ffn_up_shexp.weight", 0),
        "mlp.shared_expert.down_proj.weight": ("ffn_down_shexp.weight", 0),
        "linear_attn.norm.weight": ("ssm_norm.weight", 0),
    }
    if rest in simple:
        g, off = simple[rest]
        return G(b + g) + off
    if rest == "linear_attn.A_log":
        return unreorder_v(torch.log(-G(b + "ssm_a")).unsqueeze(-1), 0, 1).squeeze(-1)
    if rest == "linear_attn.dt_bias":
        return unreorder_v(G(b + "ssm_dt.bias").unsqueeze(-1), 0, 1).squeeze(-1)
    if rest == "linear_attn.conv1d.weight":
        x = G(b + "ssm_conv1d.weight"); qk = 2 * NK * HK
        return torch.cat([x[:qk], unreorder_v(x[qk:], 0, HV)]).unsqueeze(1)
    if rest == "linear_attn.in_proj_qkv.weight":
        x = G(b + "attn_qkv.weight"); qk = 2 * NK * HK
        return torch.cat([x[:qk], unreorder_v(x[qk:], 0, HV)])
    if rest == "linear_attn.in_proj_z.weight": return unreorder_v(G(b + "attn_gate.weight"), 0, HV)
    if rest == "linear_attn.in_proj_a.weight": return unreorder_v(G(b + "ssm_alpha.weight"), 0, 1)
    if rest == "linear_attn.in_proj_b.weight": return unreorder_v(G(b + "ssm_beta.weight"), 0, 1)
    if rest == "linear_attn.out_proj.weight": return unreorder_v(G(b + "ssm_out.weight"), 1, HV)
    if rest == "mlp.experts.gate_up_proj":
        return torch.cat([G(b + "ffn_gate_exps.weight"), G(b + "ffn_up_exps.weight")], dim = 1)
    if rest == "mlp.experts.down_proj": return G(b + "ffn_down_exps.weight")
    e = re.match(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight", rest)
    if e:
        cache_key = (n, e.group(2))
        if cache_key not in _exp_cache:
            _exp_cache.clear(); _exp_cache[cache_key] = G(b + f"ffn_{e.group(2)}_exps.weight")
        return _exp_cache[cache_key][int(e.group(1))]
    return None
_exp_cache = {}

# ---- reference layout ----
idx = json.load(open(os.path.join(args.ref, "model.safetensors.index.json")))
wmap = idx["weight_map"]
shards = {}
for k, f in wmap.items(): shards.setdefault(f, []).append(k)

def layer_of(k):
    m = re.match(r"model\.language_model\.layers\.(\d+)\.", k)
    return int(m.group(1)) if m else None

stats, worst = [], []
if args.out and args.check_only < 0:
    os.makedirs(args.out, exist_ok = True)

for si, (fname, keys) in enumerate(sorted(shards.items())):
    if args.check_only >= 0:
        keys = [k for k in keys if (layer_of(k) is not None and layer_of(k) < args.check_only)
                or k in ("lm_head.weight", "model.language_model.embed_tokens.weight", "model.language_model.norm.weight")
                or (k.startswith("mtp.") and not re.search(r"experts\.([1-9]\d*)\.", k))]
        if not keys: continue
    out = {}
    with safe_open(os.path.join(args.ref, fname), "pt") as f:
        for k in keys:
            ref = f.get_tensor(k)
            from_ref = any(k.startswith(p) or k.startswith("model." + p) for p in take_ref)
            x = None if from_ref else build(k)
            if x is None:
                if not from_ref:
                    print(f"  !! no GGUF counterpart, taking reference: {k}", flush = True)
                out[k] = ref
                continue
            if tuple(x.shape) != tuple(ref.shape) and x.numel() == ref.numel() and x.squeeze().shape == ref.squeeze().shape:
                x = x.reshape(ref.shape)   # llama.cpp drops unit dims (e.g. shared_expert_gate (1, D) -> (D,))
            assert tuple(x.shape) == tuple(ref.shape), f"{k}: rebuilt {tuple(x.shape)} vs ref {tuple(ref.shape)}"
            r = ref.float()
            xd, rd = x.double().flatten(), r.double().flatten()
            cos = (xd @ rd / (xd.norm() * rd.norm()).clamp_min(1e-30)).item()
            rel = ((x - r).norm() / r.norm().clamp_min(1e-12)).item()
            stats.append((k, cos, rel)); worst.append((cos, k, rel))
            out[k] = x.to(ref.dtype)
    if args.out and args.check_only < 0:
        save_file(out, os.path.join(args.out, fname), metadata = {"format": "pt"})
    print(f"[{si + 1}/{len(shards)}] {fname}: {len(keys)} tensors  ({time.time() - t0:.0f}s)", flush = True)

# ---- report ----
cos = np.array([s[1] for s in stats]); rel = np.array([s[2] for s in stats])
print(f"\nchecked {len(stats)} tensors: cosine min {cos.min():.5f} median {np.median(cos):.5f} | rel diff median {np.median(rel):.4f} max {rel.max():.4f}")
for c, k, r in sorted(worst)[:12]:
    print(f"   cos {c:.5f}  rel {r:.4f}  {k}")
groups = {}
for k, c, r in stats:
    g = re.sub(r"\.\d+\.", ".N.", k); groups.setdefault(g, []).append((c, r))
print("\nper tensor kind (min cos / median rel):")
for g, v in sorted(groups.items()):
    print(f"   {min(c for c, _ in v):.5f}  {np.median([r for _, r in v]):.4f}  {g}")

if args.out and args.check_only < 0:
    for f in os.listdir(args.ref):
        if not f.endswith(".safetensors") and os.path.isfile(os.path.join(args.ref, f)):
            shutil.copy2(os.path.join(args.ref, f), args.out)
    json.dump({"source_gguf": os.path.basename(args.gguf), "reference": args.ref, "taken_from_reference": take_ref,
               "min_cos": float(cos.min()), "median_rel": float(np.median(rel))},
              open(os.path.join(args.out, "gguf_rebuild_report.json"), "w"), indent = 2)
    print(f"wrote {args.out}")
