#!/usr/bin/env python3
"""Deterministic PP/TG probe against llama-swap's currently-active model.

Isolates ONE server-side variable at a time (here: V cache type) by flipping
the registry between runs while holding ctx, sampling, and prompt fixed. Depth
prompts are built with lib/trim_to_tokens.py against the backend /tokenize,
discovered via /running -> /upstream/{model}/tokenize (llama-swap does not proxy
/tokenize itself). cache_prompt is forced OFF so PP reflects a full prefill on
every rep -- see depth-sweep.sh's method note: llama.cpp reports PP over tokens
actually evaluated, so this is the honest way to compare prefill cost at depth.

  llms use cold-fusion   # (served f16 or q8_0 V per registry)
  bench/speed/kv-probe.py --tag cf-f16 --depths 4096,65536 --reps 2 \
      --out results/speed/kv-probe-cf-f16.json
"""
import argparse, json, os, statistics as st, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.realpath(__file__))
TRIM = os.path.normpath(os.path.join(HERE, "..", "lib", "trim_to_tokens.py"))


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def running_model(swap_base):
    running = (get_json(swap_base.rstrip("/") + "/running") or {}).get("running", [])
    ready = [m for m in running if m.get("state") == "ready"] or running
    return ready[0]["model"] if ready else None


def resolve_tokenize(swap_base):
    m = running_model(swap_base)
    return f"{swap_base.rstrip('/')}/upstream/{m}/tokenize" if m else None


def post(url, payload, timeout=1800):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18080/v1")
    ap.add_argument("--swap-url", default="http://127.0.0.1:18080")
    ap.add_argument("--model", default="auto",
                    help="model id to request, or 'auto' for whatever is loaded")
    ap.add_argument("--corpus", default="/tmp/cf-corpus.txt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--depths", default="4096,65536")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--out", default="/tmp/kv-probe.json")
    args = ap.parse_args()

    if args.model == "auto":
        args.model = running_model(args.swap_url)
        if not args.model:
            print("error: no ready model in /running", file=sys.stderr); sys.exit(1)
    tok = resolve_tokenize(args.swap_url)
    if not tok:
        print("error: no ready backend in /running", file=sys.stderr); sys.exit(1)
    depths = [int(x) for x in args.depths.split(",")]
    print(f"== {args.tag} == tokenize={tok}  depths={depths} reps={args.reps} gen={args.gen}\n")

    recs = []
    for depth in depths:
        r = subprocess.run(["python3", TRIM, tok, args.corpus, str(depth), str(args.gen)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not r.stdout.strip():
            print(f"  depth {depth}: prompt build failed: {r.stderr[-200:]}"); continue
        payload = json.loads(r.stdout)
        payload["model"] = args.model
        payload["cache_prompt"] = False
        payload["temperature"] = 0.0
        for rep in range(1, args.reps + 1):
            t0 = time.time()
            try:
                d = post(args.url, payload)
            except Exception as e:
                print(f"  depth {depth} rep {rep}: request failed: {e}"); continue
            wall = time.time() - t0
            t = d.get("timings", {}) or {}
            u = d.get("usage", {}) or {}
            dn, da = t.get("draft_n"), t.get("draft_n_accepted")
            rec = {
                "depth": depth, "rep": rep,
                "prompt_tok": u.get("prompt_tokens"),
                "pp_tok_s": round(t.get("prompt_per_second", 0) or 0, 1),
                "tg_tok_s": round(t.get("predicted_per_second", 0) or 0, 2),
                "acc_pct": round(100.0 * da / dn, 1) if dn else None,
                "wall_s": round(wall, 1),
            }
            recs.append(rec)
            print(f"  depth {depth:6d} rep {rep}: prompt={rec['prompt_tok']:6d}  "
                  f"PP={rec['pp_tok_s']:8.1f} t/s  TG={rec['tg_tok_s']:6.2f} t/s  "
                  f"acc={rec['acc_pct']}%  {rec['wall_s']}s")

    # median per depth
    summ = {}
    for depth in depths:
        v = [r for r in recs if r["depth"] == depth]
        if not v:
            continue
        summ[str(depth)] = {
            "prompt_tok": v[0]["prompt_tok"], "n": len(v),
            "pp_med": round(st.median(r["pp_tok_s"] for r in v), 1),
            "tg_med": round(st.median(r["tg_tok_s"] for r in v), 2),
            "acc_med": round(st.median(r["acc_pct"] for r in v if r["acc_pct"] is not None), 1)
                        if any(r["acc_pct"] is not None for r in v) else None,
        }
    print("\n=== SUMMARY (median) ===")
    for dep, s in summ.items():
        print(f"  depth {dep:>6}: PP {s['pp_med']:8.1f} | TG {s['tg_med']:6.2f} | acc {s['acc_med']}%")
    json.dump({"tag": args.tag, "summary": summ, "records": recs}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
