"""Held-out drafter measurement on the served target: prompts the drafter never trained on, single
stream, production sampling (server defaults, seeded per prompt), spec-decode counters read from the
server's /metrics before and after EVERY prompt, so acceptance is reported per category too.

Usage: eval_served.py HELDOUT.jsonl OUT.json --model NAME [--url http://127.0.0.1:18080/v1]
                      [--metrics http://127.0.0.1:18080/upstream/NAME/metrics] [--max-tokens 1024]
"""
import argparse
import json
import time
import urllib.request
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("prompts"); ap.add_argument("out")
ap.add_argument("--model", required=True); ap.add_argument("--url", default="http://127.0.0.1:18080/v1")
ap.add_argument("--metrics", default=""); ap.add_argument("--max-tokens", type=int, default=1024)
args = ap.parse_args()
metrics = args.metrics or f"{args.url.rsplit('/v1', 1)[0]}/upstream/{args.model}/metrics"


def counters():
    raw = urllib.request.urlopen(metrics, timeout=30).read().decode(); o = {"n": 0.0, "d": 0.0, "a": 0.0}
    for line in raw.splitlines():
        # vLLM's counters, or Radiance's: a drafting step is one that reached draft position 0
        for k, m in (("n", "vllm:spec_decode_num_drafts_total"), ("d", "vllm:spec_decode_num_draft_tokens_total"),
                     ("a", "vllm:spec_decode_num_accepted_tokens_total"),
                     ("n", 'radiance:draft_position_reached_total{'), ("d", "radiance:draft_tokens_total{"),
                     ("a", "radiance:draft_accepted_total{")):
            if m.startswith("radiance:draft_position_reached") and 'position="0"' not in line:
                continue
            if line.startswith(m):
                o[k] += float(line.rsplit(" ", 1)[1])
    return o


rows = [json.loads(line) for line in open(args.prompts)]
per = []; t0 = time.time()
c_prev = counters()
for i, r in enumerate(rows):
    req = {"model": args.model, "messages": r.get("messages") or [{"role": "user", "content": r["prompt"]}],
           "max_tokens": args.max_tokens, "seed": int(r["id"])}
    if r.get("reasoning_effort"):   # the prompt's own thinking level (build_prose_pool.py)
        req["reasoning_effort"] = r["reasoning_effort"]
    body = json.dumps(req).encode()
    a = time.time()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"{args.url}/chat/completions", body, {"Content-Type": "application/json"}), timeout=1800))
        ct = d["usage"]["completion_tokens"]
    except Exception as e:  # noqa: BLE001
        print("error", r["id"], repr(e)[:300], flush=True)
        raise SystemExit(1)
    dt = time.time() - a
    time.sleep(0.5)
    c = counters()
    per.append({"id": r["id"], "source": r["source"], "category": r.get("category"), "tokens": ct, "secs": round(dt, 2),
                "drafts": c["n"] - c_prev["n"], "draft_tokens": c["d"] - c_prev["d"], "accepted": c["a"] - c_prev["a"]})
    c_prev = c
    if (i + 1) % 20 == 0:
        n = sum(p["drafts"] for p in per); ac = sum(p["accepted"] for p in per)
        print(f"{i + 1}/{len(rows)} {sum(p['tokens'] for p in per)} tok, {sum(p['tokens'] for p in per) / sum(p['secs'] for p in per):.1f} tok/s, "
              f"tok/update {(ac + n) / max(n, 1):.3f}", flush=True)


def agg(ps):
    n = sum(p["drafts"] for p in ps); ac = sum(p["accepted"] for p in ps); tk = sum(p["tokens"] for p in ps); s = sum(p["secs"] for p in ps)
    return {"prompts": len(ps), "tokens": tk, "tok_s": round(tk / s, 2), "acc_per_draft": round(ac / max(n, 1), 4), "tok_per_update": round((ac + n) / max(n, 1), 4)}


cats = defaultdict(list)
for p in per:
    cats[p["category"]].append(p)
res = {"model": args.model, "all": agg(per), "by_category": {k: agg(v) for k, v in sorted(cats.items())}, "wall_s": round(time.time() - t0, 1), "per": per}
json.dump(res, open(args.out, "w"), indent=1)
print("RESULT", json.dumps(res["all"]), flush=True)
for k, v in res["by_category"].items():
    print(f"  {k:14s} {json.dumps(v)}", flush=True)
