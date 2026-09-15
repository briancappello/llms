#!/usr/bin/env python3
"""Needle-in-a-haystack retrieval across context depth.

A model advertising a 262144 trained window does not necessarily *use* it.
Recall commonly collapses well before the nominal limit, and it usually
degrades worst in the middle of the context ("lost in the middle").

This buries a unique fact at several relative positions within a filler
corpus of N tokens and asks for it back. Depth is measured with the server's
own /tokenize endpoint, so the numbers are real tokens, not estimates.

Usage:
  niah.py --url http://127.0.0.1:8099/v1 --tokenize http://127.0.0.1:8099/tokenize \
          --corpus /tmp/code-corpus.txt --depths 4096,16384,65536,131072,262000
"""
import argparse, json, sys, time, urllib.request

NEEDLE = ("\n// NOTE: The Meridian project access code is 7429-XKQ. "
          "Remember this exact code.\n")
ANSWER = "7429-XKQ"
QUESTION = ("\n\nQUESTION: Somewhere in the text above, an access code for the "
            "Meridian project was stated. Reply with ONLY that access code, "
            "nothing else.\n")


def tokenize(url, text, timeout=300):
    body = json.dumps({"content": text, "add_special": False}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return len(json.loads(r.read()).get("tokens", []))


def trim_to_tokens(tok_url, corpus, target):
    """Binary-search a byte prefix of corpus that is ~target tokens."""
    lo, hi, best, bestd = 0, len(corpus), "", 10 ** 9
    for _ in range(18):
        mid = (lo + hi) // 2
        chunk = corpus[:mid]
        n = tokenize(tok_url, chunk)
        d = abs(n - target)
        if d < bestd:
            bestd, best = d, chunk
        if n < target:
            lo = mid + 1
        elif n > target:
            hi = mid - 1
        else:
            break
    return best


def post(url, payload, api_key=None, timeout=3600):
    hdrs = {"Content-Type": "application/json"}
    if api_key:
        hdrs["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(), headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tokenize", required=True)
    ap.add_argument("--corpus", default="/tmp/code-corpus.txt")
    ap.add_argument("--model", default="local")
    ap.add_argument("--depths", default="4096,16384,65536,131072")
    ap.add_argument("--positions", default="0.1,0.5,0.9")
    ap.add_argument("--out", default="/tmp/niah-results.json")
    ap.add_argument("--api-key", default=None)
    # Reasoning models spend their budget in reasoning_content before emitting
    # any content at all. Too small a value here measures nothing but the
    # truncation point -- 64 tokens produced a 0/12 false negative.
    ap.add_argument("--max-tokens", type=int, default=768)
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    positions = [float(x) for x in args.positions.split(",") if x.strip()]
    corpus = open(args.corpus, encoding="utf-8", errors="replace").read()

    results, t0 = [], time.time()
    print(f"{'depth':>8} {'pos':>6} {'tokens':>8} {'result':>8}  latency")
    for depth in depths:
        base = trim_to_tokens(args.tokenize, corpus, depth)
        if not base:
            print(f"{depth:>8} -- corpus too small, skipping")
            continue
        for pos in positions:
            cut = int(len(base) * pos)
            haystack = base[:cut] + NEEDLE + base[cut:] + QUESTION
            rec = {"depth": depth, "position": pos}
            try:
                s = time.time()
                d = post(args.url, {
                    "model": args.model,
                    "messages": [{"role": "user", "content": haystack}],
                    "max_tokens": args.max_tokens, "temperature": 0.0,
                }, api_key=args.api_key)
                lat = time.time() - s
                msg = d["choices"][0]["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                ptok = (d.get("timings") or {}).get("prompt_n")
                fin = d["choices"][0].get("finish_reason")
                # Credit a hit in either channel, but record which one so a
                # model that only "finds" it mid-reasoning is visible.
                in_content = ANSWER.lower() in content.lower()
                in_reason = ANSWER.lower() in reasoning.lower()
                ok = in_content or in_reason
                rec.update({"found": ok, "in_content": in_content,
                            "in_reasoning_only": (in_reason and not in_content),
                            "finish_reason": fin, "prompt_tokens": ptok,
                            "answer": (content.strip() or reasoning.strip())[:120],
                            "latency_s": round(lat, 1)})
                print(f"{depth:>8} {pos:>6} {str(ptok):>8} {'FOUND' if ok else 'MISS':>8}  {lat:.1f}s")
            except Exception as e:
                rec.update({"found": False, "error": str(e)[:160]})
                print(f"{depth:>8} {pos:>6} {'-':>8} {'ERROR':>8}  {str(e)[:60]}")
            results.append(rec)

    found = sum(1 for r in results if r.get("found"))
    by_depth = {}
    for r in results:
        d = r["depth"]
        by_depth.setdefault(d, []).append(bool(r.get("found")))
    summary = {
        "model": args.model,
        "total": len(results),
        "found": found,
        "recall_pct": round(100.0 * found / max(len(results), 1), 1),
        "recall_by_depth": {str(k): f"{sum(v)}/{len(v)}" for k, v in sorted(by_depth.items())},
        "elapsed_s": round(time.time() - t0, 1),
    }
    print("\n=== NIAH SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:18s} {v}")
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
