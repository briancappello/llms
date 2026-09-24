#!/usr/bin/env python3
"""Does the macOS process footprint track the memory a served model really uses?

bench/lib/mem.py reads ri_phys_footprint for the inference process on macOS
(design D10). Before any result relies on it, this checks it against what
the server itself says it allocated, at three context sizes, through the
production stack (llms render -> llama-swap -> engine):

  - llama.cpp logs its KV cache size and Metal buffer sizes at load; the
    footprint delta between contexts should track the KV delta;
  - weights are mmap'd from the GGUF, file-backed and clean, so they are
    expected to be ABSENT from phys_footprint: a capacity probe must add the
    weight size (or use a different measure) rather than trust footprint alone.

    bench/context/footprint_validation.py --model cyber-tiel --ctx 32768,65536,131072
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from common import PROVENANCE, TSV, Variant, arg_parser, log, mem, metal_budget, served_memory, stack_for  # noqa: E402

KV = re.compile(r"llama_kv_cache: size = +([\d.]+) MiB")
METAL_BUF = re.compile(r"(MTL\d|Metal)\w*[_ ]+(?:model|KV|compute)?\s*buffer size = +([\d.]+) MiB")
MODEL_BUF = re.compile(r"load_tensors:\s+(\S+) model buffer size = +([\d.]+) MiB")
COMPUTE_BUF = re.compile(r"sched_reserve:\s+(\S+) compute buffer size = +([\d.]+) MiB")


def logged(text):
    kv = [float(v) for v in KV.findall(text)]
    model = {dev: float(v) for dev, v in MODEL_BUF.findall(text)}
    compute = {dev: float(v) for dev, v in COMPUTE_BUF.findall(text)}
    return (kv[-1] if kv else None), model, compute


def main():
    parser = arg_parser(__doc__, default_out="results/context/mac-footprint-validation.tsv")
    parser.add_argument("--ctx", default="32768,65536,131072")
    args = parser.parse_args()
    columns = ["model", "ctx", "footprint_mib", "kv_logged_mib", "model_buf_mib", "compute_buf_mib",
               "weights_file_mib", "budget_mib", *PROVENANCE]
    out = TSV(args.out, columns)
    for model in args.model:
        for ctx in [int(c) for c in args.ctx.split(",")]:
            variant = Variant(f"ctx{ctx}", entries={model: {"ctx": ctx}})
            with stack_for(args, variant) as stack:
                seconds = stack.load(model)
                footprint, kind = served_memory(stack, model)
                text = stack.logs()
                kv, buffers, compute = logged(text)
                entry = stack.registry[model]
                weights = round(Path(entry["path"]).stat().st_size / 2**20) if entry.get("path") else None
                row = dict(model=model, ctx=ctx, footprint_mib=mem.cell(footprint), kv_logged_mib=mem.cell(kv),
                           model_buf_mib=mem.cell(round(sum(buffers.values())) if buffers else None),
                           compute_buf_mib=mem.cell(round(sum(compute.values())) if compute else None),
                           weights_file_mib=mem.cell(weights), budget_mib=mem.cell(metal_budget(stack, model)),
                           **stack.record(model))
                out.row(**row)
                log(f"{model} ctx={ctx}: loaded in {seconds:.0f}s  {kind}={footprint} MiB  kv(logged)={kv} MiB  "
                    f"model buffers={buffers}  compute={compute}")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
