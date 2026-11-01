"""Deep-context benchmark: prefill and decode speed at depth, with a needle as a sanity check.

Each depth gets a prompt with a unique nonce at the start (no prefix-cache reuse within a server run), a code word at
50% depth, and a question asking for the code word on the first line followed by a ~250-word technical summary, so
decode speed at depth is measured too. One request per depth.

Servers:
  * llama-server / Strata: reads the llama.cpp-style `timings` object.
  * TabbyAPI (exllamav3): reads usage.prompt_tokens_per_sec / completion_tokens_per_sec / prompt_time /
    completion_time and completion_tokens_details.accepted/rejected_prediction_tokens (MTP), and counts reasoning
    tokens (reasoning_content) and answer tokens (content) separately via /v1/token/encode.
Needs Strata's tools/needle_bench.py for the haystack text (STRATA_TOOLS overrides its path).

Old behaviour (thinking off, greedy, max_tokens 600, time-based nonce) is the default:
  BENCH_URL=http://127.0.0.1:18080 BENCH_MODEL=swift-1.5 python depth_bench.py 8000,32000,128000

Settings (environment):
  BENCH_URL, BENCH_MODEL, BENCH_OUT     endpoint, model name, output JSON path
  BENCH_LABEL                           free-form label stored in every row (e.g. "mtp" / "nomtp")
  BENCH_TABBY=1                         size prompts to exact token targets with Tabby's /v1/apply-template and
                                        /v1/token/encode, and count reasoning/answer tokens
  BENCH_CTX=262144                      max_tokens = BENCH_CTX - prompt tokens (needs BENCH_TABBY=1); otherwise
  BENCH_MAX_TOKENS=600                  a fixed max_tokens
  BENCH_REQ='{"temperature":0,...}'     JSON merged into every request (sampling, reasoning_effort, seed, ...).
                                        Default: thinking off, greedy (the original settings).
  BENCH_NONCE=<tag>                     deterministic nonce "<tag>-<target>" (same prompt across server restarts,
                                        e.g. MTP on vs off); default: time-based
  BENCH_PROMPT_DIR=<dir>                cache built prompts there and reuse them (identical prompts across runs)
  BENCH_MEMLOG=<tsv>                    memory sampler log (memsample.sh); peak VRAM/RAM per request read from it
  BENCH_STREAM=1                        stream (progress line on stderr every 30 s); default on with BENCH_TABBY
"""
import json, os, random, sys, time, urllib.request
sys.path.insert(0, os.environ.get("STRATA_TOOLS", os.path.expanduser("~/dev/strata/tools")))
from needle_bench import haystack, WORDS

BASE = os.environ.get("BENCH_URL", "http://127.0.0.1:18080")
URL = BASE + "/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "swift-1.5")
LABEL = os.environ.get("BENCH_LABEL", "")
OUT = os.environ.get("BENCH_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/context",
                                               f"depth-{MODEL}-{time.strftime('%Y%m%d-%H%M%S')}.json"))
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
LENGTHS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "8000,32000,64000,128000,200000,255000").split(",")]
TABBY = os.environ.get("BENCH_TABBY") == "1"
CTX = int(os.environ["BENCH_CTX"]) if os.environ.get("BENCH_CTX") else None
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "600"))
REQ = json.loads(os.environ.get("BENCH_REQ") or json.dumps(
    {"temperature": 0, "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}))
NONCE = os.environ.get("BENCH_NONCE")
PROMPT_DIR = os.environ.get("BENCH_PROMPT_DIR")
MEMLOG = os.environ.get("BENCH_MEMLOG")
STREAM = os.environ.get("BENCH_STREAM", "1" if TABBY else "0") == "1"
CPT = 3.2
Q = ("\n\n---\nFirst line of your answer: the secret code word hidden in the text above, exactly. "
     "Then write a detailed 250-word technical summary of what the documents above describe.")
if CTX and not TABBY:
    sys.exit("BENCH_CTX needs BENCH_TABBY=1 (exact prompt token counts)")


def post(path, body, timeout=600):
    r = urllib.request.urlopen(urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=timeout)
    return json.loads(r.read())


def count_tokens(text):
    if not text:
        return 0
    return post("/v1/token/encode", {"text": text, "add_bos_token": False})["length"]


def prompt_tokens(content):
    """Exact prompt length as the server will see it: the request's own template rendering, then tokenized."""
    req = {k: v for k, v in REQ.items() if k not in ("max_tokens", "stream", "stream_options")}
    prompt = post("/v1/apply-template", {**req, "model": MODEL, "messages": [{"role": "user", "content": content}]})["prompt"]
    return post("/v1/token/encode", {"text": prompt, "add_bos_token": False})["length"]


def build(n, word):
    nonce = f"{NONCE}-{n}" if NONCE else str(time.time_ns())
    needle = f"\n\nIMPORTANT: the secret code word is {word}. Remember it.\n\n"

    def make(chars):
        text = haystack(chars)[:chars]
        cut = len(text) // 2
        return f"[run {nonce}]\n" + text[:cut] + needle + text[cut:] + Q

    chars = int(n * CPT * 0.98)
    content = make(chars)
    if TABBY:  # rescale to the target token count (deterministic haystack, so this converges in 2-3 steps)
        for _ in range(4):
            got = prompt_tokens(content)
            if abs(got - n) <= max(16, n // 1000):
                break
            chars = int(chars * n / got)
            content = make(chars)
    return content


def mem_peak(t0, t1):
    """Peak VRAM / RAM from the memsample.sh log over [t0, t1] (samples every 30 s, so +-30 s at the edges)."""
    if not MEMLOG or not os.path.exists(MEMLOG):
        return {}
    rows = []
    for line in open(MEMLOG):
        p = line.split("\t")
        if p[0] == "epoch":
            continue
        try:
            if t0 - 1 <= float(p[0]) <= t1 + 30:
                rows.append([float(x) for x in p[1:6]] if len(p) >= 7 else [float(x) for x in p[1:5]] + [0.0])
        except (ValueError, IndexError):
            pass
    if not rows:
        return {"mem_samples": 0}
    return {"mem_samples": len(rows), "vram_peak_gib": max(r[0] for r in rows), "ram_used_peak_gib": max(r[1] for r in rows),
            "tabby_pss_peak_gib": max(r[2] for r in rows), "swap_used_peak_gib": max(r[3] for r in rows),
            "tabby_swap_peak_gib": max(r[4] for r in rows)}


def run_stream(req, t0):
    req = {**req, "stream": True, "stream_options": {"include_usage": True}}
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(req).encode(), {"Content-Type": "application/json"}),
                               timeout=36000)
    reasoning, content, usage, timings, finish, first = [], [], None, {}, None, None
    nd, last_print = 0, time.time()
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        d = json.loads(data)
        if "error" in d:
            raise RuntimeError(json.dumps(d["error"]))
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"]); nd += 1
            if delta.get("content"):
                content.append(delta["content"]); nd += 1
            if first is None and (delta.get("reasoning_content") or delta.get("content")):
                first = time.time()
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
        if d.get("usage"):
            usage = d["usage"]
        if d.get("timings"):
            timings = d["timings"]
        if time.time() - last_print >= 30:
            last_print = time.time()
            phase = "prefill" if first is None else ("answer" if content else "reasoning")
            tail = ("".join(content) or "".join(reasoning))[-100:].replace("\n", " ")
            print(f"  .. {time.time() - t0:7.0f}s {phase:9s} deltas={nd} reasoning_chars={sum(map(len, reasoning))} "
                  f"answer_chars={sum(map(len, content))} | ...{tail}", file=sys.stderr, flush=True)
    return "".join(reasoning), "".join(content), usage or {}, timings, finish, first


results = []
for n in LENGTHS:
    rng = random.Random(n)
    word = f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randint(100, 999)}"
    cache = os.path.join(PROMPT_DIR, f"{MODEL}-{NONCE}-{n}.txt") if PROMPT_DIR and NONCE else None
    if cache and os.path.exists(cache):
        body = open(cache, encoding="utf-8").read()
    else:
        body = build(n, word)
        if cache:
            os.makedirs(PROMPT_DIR, exist_ok=True)
            open(cache, "w", encoding="utf-8").write(body)
    req = {"model": MODEL, "max_tokens": MAX_TOKENS, **REQ, "messages": [{"role": "user", "content": body}]}
    est = None
    if TABBY:
        est = prompt_tokens(body)
        if CTX:
            req["max_tokens"] = CTX - est
    print(f"-> target {n}: prompt {est} tokens, max_tokens {req['max_tokens']}, word {word}", file=sys.stderr, flush=True)
    t0 = time.time()
    if STREAM:
        reasoning, content, u, t, finish, first = run_stream(req, t0)
    else:
        r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(req).encode(), {"Content-Type": "application/json"}),
                                   timeout=36000)
        d = json.loads(r.read()); msg = d["choices"][0]["message"]
        reasoning, content, u, t = msg.get("reasoning_content") or "", msg.get("content") or "", d.get("usage", {}), d.get("timings", {})
        finish, first = d["choices"][0].get("finish_reason"), None
    t1 = time.time(); wall = t1 - t0
    ctd = u.get("completion_tokens_details") or {}
    acc, rej = ctd.get("accepted_prediction_tokens"), ctd.get("rejected_prediction_tokens")
    if acc is None and t.get("draft_n") is not None:  # llama-server
        acc, rej = t.get("draft_n_accepted"), (t.get("draft_n") or 0) - (t.get("draft_n_accepted") or 0)
    row = {
        "label": LABEL, "target": n, "word": word,
        "prompt_tokens": u.get("prompt_tokens"), "prompt_tokens_est": est,
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", t.get("cache_n")),
        "prefill_tps": u.get("prompt_tokens_per_sec") or t.get("prompt_per_second"),
        "prefill_s": u.get("prompt_time") if u.get("prompt_time") is not None else round((t.get("prompt_ms") or 0) / 1000, 2),
        "gen_tokens": u.get("completion_tokens") or t.get("predicted_n"),
        "reasoning_tokens": count_tokens(reasoning) if TABBY else None,
        "answer_tokens": count_tokens(content) if TABBY else None,
        "decode_tps": u.get("completion_tokens_per_sec") or t.get("predicted_per_second"),
        "decode_s": u.get("completion_time") if u.get("completion_time") is not None else round((t.get("predicted_ms") or 0) / 1000, 2),
        "mtp_accepted": acc, "mtp_rejected": rej,
        "mtp_accept_pct": round(100 * acc / (acc + rej), 1) if acc is not None and rej is not None and acc + rej > 0 else None,
        "max_tokens": req["max_tokens"], "finish_reason": finish,
        "ttft_s": round(first - t0, 2) if first else None, "wall_s": round(wall, 1),
        "found": word in content, "found_in_reasoning": word in reasoning,
        "first_line": content.strip().splitlines()[0][:120] if content.strip() else "",
        "answer_words": len(content.split()),
        **mem_peak(t0, t1),
        "t_start": round(t0, 1), "t_end": round(t1, 1),
        "usage": u, "timings": t,
        "reasoning_text": reasoning, "answer_text": content,
    }
    results.append(row)
    print(json.dumps({k: v for k, v in row.items() if k not in ("usage", "timings", "reasoning_text", "answer_text")}), flush=True)
    json.dump(results, open(OUT, "w"), indent=1)  # after every depth, so a crash keeps the finished points
