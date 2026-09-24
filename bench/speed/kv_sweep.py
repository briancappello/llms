#!/usr/bin/env python3
"""KV-cache type/context arms, through the production stack (successor to kv-sweep.sh).

Each arm is a variant of the production entry, e.g. quantised V at the full
window versus f16 at a smaller window:

    bench/speed/kv_sweep.py --model fable-fusion \\
        --variant A-q8V='{"entries": {"fable-fusion": {"ctx": 262144, "common": false,
            "extra_args": ["-fa on --cache-type-k f16 --cache-type-v q8_0 --jinja --metrics"]}}}' \\
        --variant B-f16='{"entries": {"fable-fusion": {"ctx": 212992}}}' \\
        --depths 4096,16384,65536,131072

Adds per-row cache types (read back from the rendered command, so they are
what was served) and headroom = budget - measured memory. On macOS the
footprint excludes mmap'd weights, so headroom there also subtracts the
weight file size (see results/context/mac-footprint-validation.md).

SCOPE: throughput and memory only. Quantised KV can degrade long-context
recall, which PP/TG cannot see: gate a quantised arm on bench/quality/niah.py.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import depth_sweep  # noqa: E402
from common import mem  # noqa: E402

COLUMNS = depth_sweep.COLUMNS[:2] + ["ctx", "cache_k", "cache_v"] + depth_sweep.COLUMNS[2:] + ["headroom_mib"]


def flag(command, name):
    match = re.search(rf"(?:^|\s){re.escape(name)}\s+(\S+)", command)
    return match.group(1) if match else mem.NA


def weights_mib(stack, model):
    path = stack.registry[model].get("path")
    return round(Path(path).stat().st_size / 2**20) if path and Path(path).is_file() else 0


def extra(stack, model, memory, budget):
    command = stack.rendered_command(model)
    # the ${common} macro is expanded by llama-swap, so read it from the header too
    header = stack.settings.header.read_text()
    expanded = command.replace("${common}", " ".join(re.findall(r"common:\s*(.+)", header)[:1]))
    headroom = None
    if memory is not None and budget is not None:
        used = memory + (weights_mib(stack, model) if sys.platform == "darwin" else 0)
        headroom = budget - used
    return {"ctx": flag(command, "-c"), "cache_k": flag(expanded, "--cache-type-k"),
            "cache_v": flag(expanded, "--cache-type-v"), "headroom_mib": mem.cell(headroom)}


def main():
    args = depth_sweep.parser(__doc__, "results/speed/kv-sweep.tsv", "4096,16384,65536").parse_args()
    depth_sweep.run(args, columns=COLUMNS, extra=extra)


if __name__ == "__main__":
    main()
