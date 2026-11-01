#!/usr/bin/env python3
"""Generate design-task solutions from an OpenAI-compatible server.

    gen.py --label strata --url http://127.0.0.1:18080/v1 --model swift-1.5 [--samples 3] [--tasks a,b]

Writes results/design/<label>/<task>.s<N>.json (resumable: existing files are skipped). Sampling is sent explicitly
on every request (request params override server defaults on both llama.cpp and Strata), so the only variable
between labels is the serving engine. Thinking is fixed at reasoning_effort=medium.
"""
import argparse, json, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

DELIVERABLE = """

---
Deliverables (Python 3.12, standard library only):

1. `solution.py` -- the complete implementation of the interface above, production quality.
2. `test_solution.py` -- a pytest test suite for your implementation (`import solution`).

Start with a short design note (at most 10 lines: data structures, complexity, concurrency approach). Then give each
file under its own heading exactly like this:

### solution.py
```python
...
```

### test_solution.py
```python
...
```
"""

# Swift 1.5 model card (ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF): recommended sampling
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
# Thinking level, fixed for every engine. The Swift template knows low / medium / xhigh (default); both llama.cpp
# and Strata map a top-level reasoning_effort onto the template variable (checked with llama.cpp /apply-template).
REASONING_EFFORT = "medium"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--url", required=True, help="base URL ending in /v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--out", default=str(REPO / "results" / "design"))
    ap.add_argument("--stop-on-error", action="store_true", help="exit 2 on the first failed request (for a supervisor that restarts the server)")
    args = ap.parse_args()

    tasks = sorted(p.name for p in (HERE / "tasks").iterdir() if (p / "prompt.md").exists())
    if args.tasks:
        tasks = [t for t in tasks if t in args.tasks.split(",")]
    out = Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=True)

    for s in range(1, args.samples + 1):
        for task in tasks:
            dest = out / f"{task}.s{s}.json"
            if dest.exists():
                continue
            prompt = (HERE / "tasks" / task / "prompt.md").read_text() + DELIVERABLE
            body = {"model": args.model, "max_tokens": args.max_tokens, "seed": 1000 * s + tasks.index(task),
                    "messages": [{"role": "user", "content": prompt}], "reasoning_effort": REASONING_EFFORT,
                    **SAMPLING}
            t0 = time.time()
            try:
                req = urllib.request.Request(args.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=7200) as r:
                    resp = json.loads(r.read())
            except Exception as e:  # record and move on; a failed request is a data point, not a crash
                print(f"[{args.label}] {task} s{s}: request failed: {e}", flush=True)
                if args.stop_on_error:
                    return 2
                continue
            wall = time.time() - t0
            choice = resp["choices"][0]
            msg = choice["message"]
            rec = {"label": args.label, "task": task, "sample": s, "seed": body["seed"], "wall_s": round(wall, 1),
                   "finish_reason": choice.get("finish_reason"), "content": msg.get("content") or "",
                   "reasoning": msg.get("reasoning_content") or "", "usage": resp.get("usage"),
                   "timings": resp.get("timings"), "request": {k: v for k, v in body.items() if k != "messages"}}
            dest.write_text(json.dumps(rec, indent=1))
            u = resp.get("usage") or {}
            t = resp.get("timings") or {}
            print(f"[{args.label}] {task} s{s}: {u.get('completion_tokens')} tok, {wall:.0f}s, "
                  f"{t.get('predicted_per_second') or 0:.1f} tok/s, finish={choice.get('finish_reason')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
