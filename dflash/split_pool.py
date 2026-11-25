"""Split pool.jsonl into disjoint sets, round-robin over sources so every set covers the whole mix.

Usage: split_pool.py pool.jsonl OUTDIR calib=89 val=200 heldout=240 train=4800
Writes OUTDIR/<name>.jsonl for each name=count, in the order given (earlier sets pick first).
"""
import json
import random
import sys
from collections import defaultdict, Counter

pool, outdir = sys.argv[1], sys.argv[2]
sizes = [(a.split("=")[0], int(a.split("=")[1])) for a in sys.argv[3:]]
rows = [json.loads(line) for line in open(pool)]
by_src = defaultdict(list)
for r in rows:
    by_src[r["source"]].append(r)
rng = random.Random(1)
for v in by_src.values():
    rng.shuffle(v)
total = len(rows)
for name, n in sizes:
    # proportional quota per source (at least one each while it lasts), filled round-robin
    quota = {s: max(1, round(n * len(v) / total)) for s, v in by_src.items() if v}
    picked = []
    srcs = sorted(quota)
    while len(picked) < n and any(by_src[s] for s in srcs):
        for s in srcs:
            if len(picked) >= n:
                break
            if quota[s] > 0 and by_src[s]:
                picked.append(by_src[s].pop()); quota[s] -= 1
        if all(quota[s] == 0 or not by_src[s] for s in srcs):
            quota = {s: 1 for s in srcs if by_src[s]}; srcs = sorted(quota)
    rng.shuffle(picked)
    with open(f"{outdir}/{name}.jsonl", "w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    print(name, len(picked), dict(Counter(r["category"] for r in picked)))
