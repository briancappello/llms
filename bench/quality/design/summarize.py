#!/usr/bin/env python3
"""Combine grade.json files and the blind review into one comparison.

    summarize.py [--labels llama,strata] [--cap-tokens 32768 --cap-label strata]

--cap-tokens/--cap-label: an equal-budget view that treats the capped label's samples longer than the cap as
truncated (no solution delivered), for comparing against a label that ran with that output limit.
"""
import argparse, json, random, statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / "results" / "design"
CATS = ("basic", "edge", "prop", "conc", "stress")
CRIT = ("correctness", "robustness", "concurrency", "performance", "design", "readability", "tests", "overall")


def hidden_rate(g, capped):
    h = g.get("hidden") or {}
    if capped or not g.get("has_solution"):
        return 0.0
    tot = (h.get("passed") or 0) + (h.get("failed") or 0)
    return (h.get("passed") or 0) / tot if tot else 0.0


def perm_test(a, b, n=20000, seed=1):
    """Two-sided permutation test on the difference of means of two independent samples."""
    obs = st.mean(a) - st.mean(b)
    pool, k, rng, hits = a + b, len(a), random.Random(seed), 0
    for _ in range(n):
        rng.shuffle(pool)
        hits += abs(st.mean(pool[:k]) - st.mean(pool[k:])) >= abs(obs) - 1e-12
    return obs, hits / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="llama,strata")
    ap.add_argument("--cap-tokens", type=int, default=0)
    ap.add_argument("--cap-label", default="")
    a = ap.parse_args()
    labels = a.labels.split(",")
    mapping = json.loads((ROOT / "_review" / "mapping.json").read_text())
    judge = defaultdict(lambda: defaultdict(list))  # sample key -> criterion -> [scores per pass]
    for f in sorted((ROOT / "_review").glob("*.p[0-9]*.json")):
        for anon, s in json.loads(f.read_text())["scores"].items():
            for c in CRIT:
                if isinstance(s.get(c), (int, float)):
                    judge[mapping[anon]][c].append(s[c])
            judge[mapping[anon]]["n_bugs"].append(len(s.get("bugs") or []))

    rows = []
    for lab in labels:
        for gf in sorted((ROOT / lab).glob("*.grade.json")):
            g = json.loads(gf.read_text())
            key = f"{lab}/{gf.name.replace('.grade.json', '.json')}"
            capped = bool(a.cap_tokens and lab == a.cap_label and (g.get("completion_tokens") or 0) > a.cap_tokens)
            j = {c: st.mean(v) for c, v in judge[key].items() if v}
            rows.append({"label": lab, "task": g["task"], "sample": g["sample"], "g": g, "capped": capped,
                         "hidden": hidden_rate(g, capped), "judge": j})

    def col(lab, fn):
        return [fn(r) for r in rows if r["label"] == lab and fn(r) is not None]

    out = []
    p = out.append
    p("| metric | " + " | ".join(labels) + " |")
    p("|---|" + "---:|" * len(labels))
    def line(name, vals): p(f"| {name} | " + " | ".join(vals) + " |")
    line("samples", [str(len(col(l, lambda r: 1))) for l in labels])
    line("finished on its own (not cut off by max_tokens)", [f"{sum(r['g']['finish_reason'] == 'stop' for r in rows if r['label'] == l)}" for l in labels])
    if a.cap_tokens:
        line(f"of which over {a.cap_tokens} tokens (counted as truncated in this view)", [str(sum(r['capped'] for r in rows if r['label'] == l)) for l in labels])
    line("delivered a solution", [str(sum(1 for r in rows if r['label'] == l and r['g'].get('has_solution') and not r['capped'])) for l in labels])
    line("hidden tests passed, mean per sample", [f"{100 * st.mean(col(l, lambda r: r['hidden'])):.1f}%" for l in labels])
    line("samples passing 100% of hidden tests", [str(sum(1 for r in rows if r['label'] == l and r['hidden'] == 1.0)) for l in labels])
    line("samples passing >= 95%", [str(sum(1 for r in rows if r['label'] == l and r['hidden'] >= 0.95)) for l in labels])
    for c in CATS:
        def cat(l, c=c):
            p_, t_ = 0, 0
            for r in rows:
                if r["label"] != l or r["capped"]:
                    continue
                bc = ((r["g"].get("hidden") or {}).get("by_cat") or {}).get(c)
                if bc:
                    p_, t_ = p_ + bc[0], t_ + bc[1]
            return f"{100 * p_ / t_:.1f}% of {t_}" if t_ else "-"
        line(f"  hidden '{c}' tests passed (pooled, delivered solutions)", [cat(l) for l in labels])
    line("perf vs reference, median (bench ran and validated)", [
        (lambda v: f"{st.median(v):.2f}x (n={len(v)})" if v else "-")(col(l, lambda r: None if r['capped'] else r['g'].get('perf_ratio'))) for l in labels])
    line("own tests: all pass", [f"{sum(1 for r in rows if r['label'] == l and not r['capped'] and (r['g'].get('own_tests') or {}).get('passed') and not (r['g'].get('own_tests') or {}).get('failed'))}" for l in labels])
    line("ruff serious findings (F/B/PLE), median", [f"{st.median(col(l, lambda r: None if r['capped'] else r['g'].get('ruff_serious')) or [0]):.0f}" for l in labels])
    line("mypy errors, median", [f"{st.median(col(l, lambda r: None if r['capped'] else r['g'].get('mypy_errors')) or [0]):.0f}" for l in labels])
    line("completion tokens, median", [f"{st.median(col(l, lambda r: r['g'].get('completion_tokens'))):.0f}" for l in labels])
    line("generation wall time, total", [f"{sum(col(l, lambda r: r['g']['wall_s'])) / 3600:.2f} h" for l in labels])
    for c in CRIT:
        line(f"judge {c} (mean{', 1-10' if c == 'overall' else ', 1-5'})", [
            (lambda v: f"{st.mean(v):.2f}" if v else "-")(col(l, lambda r, c=c: 1.0 if (r['capped'] and c != 'concurrency') else r['judge'].get(c))) for l in labels])
    line("judge confident bugs per sample, mean", [f"{st.mean(col(l, lambda r: r['judge'].get('n_bugs'))):.2f}" for l in labels])

    p("\nSignificance (two-sided permutation test, %s - %s, samples independent):" % (labels[1], labels[0]))
    for name, fn in (("hidden pass rate", lambda r: r["hidden"]),
                     ("judge overall", lambda r: 1.0 if r["capped"] else r["judge"].get("overall"))):
        d, pv = perm_test(col(labels[1], fn), col(labels[0], fn))
        p(f"- {name}: difference {d:+.3f}, p = {pv:.2f}")

    p("\nPer task (hidden pass rate mean / judge overall mean):\n")
    p("| task | " + " | ".join(labels) + " |")
    p("|---|" + "---|" * len(labels))
    for t in sorted({r["task"] for r in rows}):
        cells = []
        for l in labels:
            rs = [r for r in rows if r["label"] == l and r["task"] == t]
            hs = "/".join(f"{100 * r['hidden']:.0f}" for r in sorted(rs, key=lambda r: r["sample"]))
            jo = [1.0 if r["capped"] else r["judge"].get("overall") for r in rs]
            jo = [x for x in jo if x is not None]
            cells.append(f"{hs}% / {st.mean(jo):.1f}" if jo else hs + "%")
        p(f"| {t} | " + " | ".join(cells) + " |")

    # judge consistency: pass-1 vs pass-2 overall
    pairs = [(v["overall"][0], v["overall"][1]) for v in judge.values() if len(v.get("overall", [])) == 2]
    if pairs:
        diffs = [abs(x - y) for x, y in pairs]
        p(f"\nJudge self-consistency: overall differs between the two blind passes by {st.mean(diffs):.2f} on average "
          f"(max {max(diffs)}), over {len(pairs)} samples.")
    hj = [(r["hidden"], r["judge"]["overall"]) for r in rows if "overall" in r["judge"] and not r["capped"]]
    if len(hj) > 2:
        xs, ys = zip(*hj)
        mx, my = st.mean(xs), st.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in hj)
        r_ = cov / ((sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5)
        p(f"Judge overall vs hidden pass rate: Pearson r = {r_:.2f} (n={len(hj)}).")
    print("\n".join(out))


if __name__ == "__main__":
    main()
