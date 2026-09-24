#!/usr/bin/env python3
"""Agentic-coding profile: thinking-token cost, throughput, MTP acceptance.

Purpose. DavidAU's Cold-Fusion card makes three measurable claims that matter
for an agent loop: (1) it emits "1/5 to 1/2 the thinking tokens" of stock
Qwen3.8, (2) it is faster (partly via MTP), (3) MTP token-acceptance is high.
This harness measures all three on a fixed set of coding prompts and, crucially,
drives every model with IDENTICAL request-level sampling so the only variable is
the weights -- see MODELS.md, "the one unavoidable duplication": request params
override --temp/--top-p, and top_k/min_p must be sent explicitly or they leak
from whatever the server was launched with.

For each prompt it records, from llama-server's OpenAI response:
  - usage.prompt_tokens / completion_tokens
  - reasoning_tok / answer_tok  (reasoning_content vs content, tokenised via
    /tokenize; falls back to parsing <think>..</think> out of content)
  - timings.prompt_per_second / predicted_per_second   (PP / TG tok/s)
  - timings.draft_n / draft_n_accepted -> MTP acceptance %
  - wall seconds, finish_reason

Each model is served as production serves it, by an isolated llms instance
(bench/lib/stack.py); an optional --variant is a declared change to that
entry. The production registry is never edited:

  bench/quality/coding-profile.py --model cold-fusion --tag cold-fusion \
      --out results/quality/cp-cold-fusion.json
"""
import json, os, re, statistics as st, sys, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
from common import arg_parser, stack_for, variants  # noqa: E402

# Bounded agentic-coding tasks: each induces some reasoning but has a short,
# well-defined answer, so runtime stays sane while thinking-length differences
# are exposed. Prompt 2 contains a real off-by-one (lo=mid -> infinite loop).
PROMPTS = [
    ("merge_intervals",
     "Write a Python function `merge_intervals(intervals)` that merges "
     "overlapping intervals given as a list of [start, end] pairs. Return only "
     "the function code."),
    ("fix_binsearch",
     "This function sometimes loops forever. Identify the bug and give a "
     "corrected version:\n\n"
     "def binary_search(a, t):\n"
     "    lo, hi = 0, len(a)\n"
     "    while lo < hi:\n"
     "        mid = (lo + hi) // 2\n"
     "        if a[mid] == t: return mid\n"
     "        elif a[mid] < t: lo = mid\n"
     "        else: hi = mid\n"
     "    return -1"),
    ("lru_cache",
     "Implement an LRU cache class in Python supporting O(1) get(key) and "
     "put(key, value) using a dict plus a doubly linked list. Return the code."),
    ("async_refactor",
     "Refactor this callback-based Node.js function to async/await, preserving "
     "error handling:\n\n"
     "function getUser(id, cb) {\n"
     "  db.query('SELECT ...', id, (e, u) => {\n"
     "    if (e) return cb(e);\n"
     "    loadPosts(u.id, (e2, p) => {\n"
     "      if (e2) return cb(e2);\n"
     "      cb(null, { ...u, posts: p });\n"
     "    });\n"
     "  });\n"
     "}"),
    ("sql_second_salary",
     "Write a SQL query returning the second-highest distinct salary from an "
     "Employees(id, name, salary) table, returning NULL if it does not exist. "
     "One or two sentences of explanation, then the query."),
]


def post(url, payload, timeout=1200):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ntokens(tokenize_url, text):
    """Token count via llama-server /tokenize; 0 for empty text."""
    if not text:
        return 0
    try:
        body = json.dumps({"content": text}).encode()
        req = urllib.request.Request(tokenize_url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return len((json.loads(r.read()) or {}).get("tokens", []))
    except Exception:
        return 0


THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def split_reasoning(msg):
    """(reasoning_text, answer_text) whether the server separated them or not."""
    rc = msg.get("reasoning_content")
    content = msg.get("content") or ""
    if rc:
        return rc, content
    m = THINK_RE.search(content)
    if m:
        return m.group(1), THINK_RE.sub("", content)
    return "", content


def main():
    ap = arg_parser(__doc__, default_out="/tmp/coding-profile.json")
    ap.add_argument("--tag", required=True, help="label for this model/run")
    # Qwen3.8 "precise coding" preset; identical for every model under test.
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=3000)
    args = ap.parse_args()
    if len(args.model) != 1 or len(variants(args)) != 1:
        ap.error("coding-profile runs exactly one --model and at most one --variant (use --tag)")
    with stack_for(args, variants(args)[0]) as stack:
        args.model = args.model[0]
        stack.load(args.model)
        args.url = stack.base + "/v1"
        args.tokenize = stack.upstream(args.model, "/tokenize")
        profile(args, stack.record(args.model))


def profile(args, provenance):

    print(f"== {args.tag} == sampling temp={args.temperature} top_p={args.top_p} "
          f"top_k={args.top_k} min_p={args.min_p} max_tokens={args.max_tokens}\n")
    recs = []
    for pid, prompt in PROMPTS:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "min_p": args.min_p,
            "max_tokens": args.max_tokens,
        }
        t0 = time.time()
        try:
            d = post(args.url, payload)
        except Exception as e:
            print(f"  {pid:18s} ERROR {e}")
            recs.append({"id": pid, "error": str(e)[:200]})
            continue
        wall = time.time() - t0
        ch = (d.get("choices") or [{}])[0]
        msg = ch.get("message", {}) or {}
        reasoning, answer = split_reasoning(msg)
        t = d.get("timings", {}) or {}
        u = d.get("usage", {}) or {}
        dn, da = t.get("draft_n"), t.get("draft_n_accepted")
        acc = (100.0 * da / dn) if (dn) else None
        rtok = ntokens(args.tokenize, reasoning)
        atok = ntokens(args.tokenize, answer)
        rec = {
            "id": pid, "finish": ch.get("finish_reason"),
            "prompt_tok": u.get("prompt_tokens"),
            "completion_tok": u.get("completion_tokens"),
            "reasoning_tok": rtok, "answer_tok": atok,
            "think_ratio": round(rtok / max(rtok + atok, 1), 3),
            "pp_tok_s": round(t.get("prompt_per_second", 0) or 0, 1),
            "tg_tok_s": round(t.get("predicted_per_second", 0) or 0, 2),
            "draft_n": dn, "draft_acc": da,
            "acc_pct": round(acc, 1) if acc is not None else None,
            "wall_s": round(wall, 1),
        }
        recs.append(rec)
        print(f"  {pid:18s} think={rtok:5d} ans={atok:4d} "
              f"({rec['think_ratio']*100:4.0f}% think)  TG={rec['tg_tok_s']:6.2f} t/s  "
              f"acc={rec['acc_pct']}%  {rec['finish']}  {rec['wall_s']}s")

    ok = [r for r in recs if "error" not in r]
    def med(k):
        vals = [r[k] for r in ok if r.get(k) is not None]
        return round(st.median(vals), 2) if vals else None
    summary = {
        "tag": args.tag, "n": len(ok),
        "reasoning_tok_total": sum(r["reasoning_tok"] for r in ok),
        "reasoning_tok_median": med("reasoning_tok"),
        "answer_tok_total": sum(r["answer_tok"] for r in ok),
        "completion_tok_total": sum((r["completion_tok"] or 0) for r in ok),
        "think_ratio_median": med("think_ratio"),
        "tg_tok_s_median": med("tg_tok_s"),
        "pp_tok_s_median": med("pp_tok_s"),
        "acc_pct_median": med("acc_pct"),
        "wall_s_total": round(sum(r["wall_s"] for r in ok), 1),
        "hit_cap": sum(1 for r in ok if r["finish"] == "length"),
    }
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    with open(args.out, "w") as f:
        json.dump({"provenance": provenance, "summary": summary, "records": recs}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
