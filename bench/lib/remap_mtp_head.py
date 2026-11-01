#!/usr/bin/env python3
"""Rename a qwen4exp MTP draft head's mixer tensors to the NextN layout.

Some qwen4exp draft-head-only GGUFs (e.g. agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF)
store the MTP block's own hyper-connection output mixer under the trunk's global
names, output_hc_{norm,down,up}. llama.cpp's qwen4exp MTP loaders (PRs #28097,
#28243) look for it at blk.<N>.nextn.hc_head_{norm,down,up} and abort with
"MTP block missing head mixer tensors".

The tensors are the MTP head's, not a copy of the trunk's: on the Signal AP-Q4_K_XL
pairing the F32 norm vectors differ from the main model's output_hc_norm
(corr ~0.51). So this is a pure rename. Every other tensor and every metadata
key is copied byte-for-byte; quantized data is passed through without requant.

  remap_mtp_head.py IN.gguf OUT.gguf
"""
import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

RENAMES = ("norm", "down", "up")


def main(src, dst):
    r = GGUFReader(src)
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    f = r.fields[f"{arch}.block_count"]
    nextn_blk = int(f.parts[f.data[0]][0]) - 1
    names = {t.name for t in r.tensors}
    for s in RENAMES:
        if f"output_hc_{s}.weight" not in names:
            sys.exit(f"error: output_hc_{s}.weight not in {src}; nothing to remap")
        if f"blk.{nextn_blk}.nextn.hc_head_{s}.weight" in names:
            sys.exit(f"error: {src} already has blk.{nextn_blk}.nextn.hc_head_{s}.weight")

    w = GGUFWriter(dst, arch, use_temp_file=False)
    for key, field in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue  # written by GGUFWriter itself
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            sub = field.types[-1]
            if sub == GGUFValueType.STRING:
                val = [bytes(field.parts[i]).decode("utf-8", errors="surrogateescape") for i in field.data]
            else:
                val = [field.parts[i][0].item() for i in field.data]
            w.add_key_value(key, val, vtype, sub_type=sub)
        elif vtype == GGUFValueType.STRING:
            w.add_key_value(key, bytes(field.parts[field.data[0]]).decode("utf-8", errors="surrogateescape"), vtype)
        else:
            w.add_key_value(key, field.parts[field.data[0]][0].item(), vtype)

    for t in r.tensors:
        name = t.name
        for s in RENAMES:
            if name == f"output_hc_{s}.weight":
                name = f"blk.{nextn_blk}.nextn.hc_head_{s}.weight"
                print(f"  {t.name} -> {name}")
        # Reader yields quantized tensors as their raw block bytes; with raw_dtype set,
        # the writer derives the logical shape from the byte shape itself.
        w.add_tensor(name, np.asarray(t.data), raw_dtype=t.tensor_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
