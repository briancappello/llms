#!/usr/bin/env python3
"""Deterministic PP/TG probe of ONE served variable, through the production stack.

Isolates one server-side variable at a time (e.g. V cache type) as a declared
variant of the production entry, served by an isolated llms instance
(bench/lib/stack.py), while ctx, sampling, and prompt stay fixed. The
production registry is never edited. Depth prompts are trimmed against the
backend /tokenize via llama-swap's /upstream/{model}/tokenize. cache_prompt is
forced OFF so PP reflects a full prefill on every rep: llama.cpp reports PP
over tokens actually evaluated, so this is the honest way to compare prefill
cost at depth.

  bench/speed/kv-probe.py --model cold-fusion --tag cf-q8_0 \
      --variant 'q8V={"entries": {"cold-fusion": {"common": false, "extra_args": ["..."]}}}' \
      --depths 4096,65536 --reps 2 --out results/speed/kv-probe-cf-q8_0.json
"""
import json, os, statistics as st, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
from common import arg_parser, depth_prompt, load_corpus, stack_for, variants  # noqa: E402


def main():
    ap = arg_parser(__doc__, default_out="/tmp/kv-probe.json")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--depths", default="4096,65536")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--gen", type=int, default=128)
    args = ap.parse_args()
    if len(args.model) != 1 or len(variants(args)) != 1:
        ap.error("kv-probe measures exactly one --model and at most one --variant per run (use --tag)")
    model, variant = args.model[0], variants(args)[0]
    corpus = load_corpus(args.corpus) if args.corpus else load_corpus()
    depths = [int(x) for x in args.depths.split(",")]
    with stack_for(args, variant) as stack:
        stack.load(model)
        tok = stack.upstream(model, "/tokenize")
        provenance = stack.record(model)
        print(f"== {args.tag} == {model} variant={variant.name} depths={depths} reps={args.reps} gen={args.gen}\n")
        recs = run(stack, model, tok, corpus, depths, args)
    summarise(args, recs, depths, provenance)


def run(stack, model, tok, corpus, depths, args):
    recs = []
    for depth in depths:
        try:
            payload = depth_prompt(tok, corpus, depth, args.gen)
        except Exception as e:
            print(f"  depth {depth}: prompt build failed: {e}"); continue
        payload["model"] = model
        for rep in range(1, args.reps + 1):
            t0 = time.time()
            try:
                d = stack.chat(payload)
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
    return recs


def summarise(args, recs, depths, provenance):
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
    json.dump({"tag": args.tag, "provenance": provenance, "summary": summ, "records": recs},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
