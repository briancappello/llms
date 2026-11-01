"""Write a GGUF's quantized token_embd.weight as the one-tensor BF16 GGUF Strata's --embd-gguf reads.

    embd_q_to_bf16.py <model.gguf> <out.gguf>

For Strata packs whose token embedding type has no GPU embedding dequantizer (e.g. Q6_K in Signal 3.8 AP-IQ4_XS):
dequantize with llama.cpp's gguf-py (STRATA_GGUF_PY or ~/dev/strata/third_party/llama.cpp/gguf-py) and round to
BF16 (nearest-even). Same layout as Strata's tools/embd_bf16_pack.py (arch "strata-embd", ne = [n_embd, n_vocab]).
"""
import os, struct, sys
import numpy as np

sys.path.insert(0, os.environ.get("STRATA_GGUF_PY", os.path.expanduser("~/dev/strata/third_party/llama.cpp/gguf-py")))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

GGUF_TYPE_STRING, GGML_TYPE_BF16, ALIGN = 8, 30, 32


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def kv(k, v):
    return s(k) + struct.pack("<I", GGUF_TYPE_STRING) + s(v)


def to_bf16(f32):
    u = f32.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)  # round to nearest even (no NaNs here)


src, out = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
t = next(t for t in r.tensors if t.name == "token_embd.weight")
dim, vocab = int(t.shape[0]), int(t.shape[1])
print(f"token_embd.weight {t.tensor_type.name} ne=[{dim}, {vocab}]", flush=True)
rows = np.asarray(t.data)  # [vocab, bytes per row]
kvs = [kv("general.architecture", "strata-embd"), kv("general.name", f"token embedding, {t.tensor_type.name}->BF16"),
       kv("strata.embd.source", os.path.basename(src))]
head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kvs)) + b"".join(kvs)
head += s("token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", dim, vocab)
head += struct.pack("<I", GGML_TYPE_BF16) + struct.pack("<Q", 0)
head += b"\0" * ((-len(head)) % ALIGN)
tmp = out + ".part"
maxerr = 0.0
with open(tmp, "wb") as w:
    w.write(head)
    for i in range(0, vocab, 8192):
        f = dequantize(rows[i:i + 8192], t.tensor_type).reshape(-1, dim).astype(np.float32)
        b = to_bf16(f)
        maxerr = max(maxerr, float(np.max(np.abs((b.astype(np.uint32) << 16).view(np.float32) - f))))
        w.write(b.tobytes())
assert os.path.getsize(tmp) == len(head) + dim * vocab * 2
os.replace(tmp, out)
print(f"wrote {out}: {dim * vocab * 2 / 2**30:.2f} GiB, max |bf16 rounding err| {maxerr:.3g}")
