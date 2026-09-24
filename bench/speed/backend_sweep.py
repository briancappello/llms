#!/usr/bin/env python3
"""Same production entry on different engines (successor to backend-sweep.sh).

Every --engine NAME becomes a variant that points the entry at that settings
engine; everything else (weights, ctx, flags, speculative decoding) is the
production entry, so differences are engine differences. Engines built with
bin/build-engine are RPATH-isolated and record their commit in
<prefix>/COMMIT, which is written into every row.

    bench/speed/backend_sweep.py --model cold-fusion --engine llama-vulkan --engine llama-hip
    # engines absent from settings.json can be declared inline, e.g. a new build:
    bench/speed/backend_sweep.py --model cyber-tiel --engine metal \\
        --engine metal-pr='{"server": "~/opt/llama.cpp-metal-pr/bin/llama-server", "args": ["-ngl", "999"]}'
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import depth_sweep  # noqa: E402
from common import Variant, mem  # noqa: E402

COLUMNS = depth_sweep.COLUMNS[:2] + ["engine", "commit"] + depth_sweep.COLUMNS[2:]


def engine_variants(model, specs):
    out = []
    for spec in specs:
        name, separator, body = spec.partition("=")
        engines = {name: json.loads(body)} if separator else {}
        if separator and "server" in engines[name]:
            engines[name]["server"] = str(Path(engines[name]["server"]).expanduser())
        out.append(Variant(name, entries={model: {"engine": name}}, engines=engines))
    return out


def commit(stack, model):
    engine = stack.settings.engines.get(stack.registry[model].get("engine") or "", {})
    server = Path(engine.get("server", ""))
    marker = server.parent.parent / "COMMIT"
    return marker.read_text().split()[0] if marker.is_file() else mem.NA


def main():
    parser = depth_sweep.parser(__doc__, "results/speed/backend-sweep.tsv", "4096,32768,65536")
    parser.add_argument("--engine", action="append", required=True, help="NAME or NAME=JSON engine spec")
    parser.set_defaults(reps=2, gen=128)
    args = parser.parse_args()
    if len(args.model) != 1:
        parser.error("backend_sweep compares engines for exactly one --model")
    args.variant = engine_variants(args.model[0], args.engine)
    depth_sweep.run(args, columns=COLUMNS, extra=lambda stack, model, memory, budget: {
        "engine": stack.registry[model].get("engine"), "commit": commit(stack, model)})


if __name__ == "__main__":
    main()
