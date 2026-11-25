"""Compare two compressed-tensors checkpoints: tensor names/shapes/dtypes, quant config, ignore list.
Usage: check_layout.py REF_DIR NEW_DIR"""
import json, os, sys
from safetensors import safe_open
def load(d):
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    meta = {}
    for fn in sorted(set(idx.values())):
        with safe_open(os.path.join(d, fn), "pt") as f:
            for k in f.keys():
                s = f.get_slice(k); meta[k] = (tuple(s.get_shape()), s.get_dtype(), fn)
    cfg = json.load(open(os.path.join(d, "config.json")))
    return idx, meta, cfg
ri, rm, rc = load(sys.argv[1]); ni, nm, nc = load(sys.argv[2])
print("tensors ref/new:", len(rm), len(nm), "| index entries:", len(ri), len(ni))
print("files ref:", sorted(set(v[2] for v in rm.values()))); print("files new:", sorted(set(v[2] for v in nm.values())))
only_r = sorted(set(rm) - set(nm)); only_n = sorted(set(nm) - set(rm))
print("only in ref:", len(only_r), only_r[:10]); print("only in new:", len(only_n), only_n[:10])
diff = [(k, rm[k][:2], nm[k][:2]) for k in sorted(set(rm) & set(nm)) if rm[k][:2] != nm[k][:2]]
print("shape/dtype mismatches:", len(diff), diff[:10])
q1, q2 = rc.get("quantization_config", {}), nc.get("quantization_config", {})
for k in ("quant_method", "format", "ignore"):
    print(k, "ref:", q1.get(k), "| new:", q2.get(k))
g1 = {k: v for k, v in q1.get("config_groups", {}).get("group_0", {}).get("weights", {}).items()}
g2 = {k: v for k, v in q2.get("config_groups", {}).get("group_0", {}).get("weights", {}).items()}
print("weights ref:", g1); print("weights new:", g2)
