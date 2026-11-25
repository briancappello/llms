"""Diverse prompt pool for DFlash2 drafter self-distillation (and GPTQ calibration).

Adapted from vllm-sly-radiance paroquant/drafter/build_pool.py, widened with long documents,
more file-edit and agentic-coding (SWE-bench issue) prompts. Only prompts come from datasets; the
answers are generated later by the served target. GSM8K *test* is never used (it is the quality eval).

Usage: build_pool.py OUT.jsonl  -> records {id, source, category, prompt, messages?}
"""
import json
import random
import sys
from collections import Counter

from datasets import load_dataset

out = sys.argv[1]
rng = random.Random(97)
rows = []
MAXC = 7000            # normal prompts (~1.7k tokens)
LONG_MIN, LONG_MAX = 14000, 30000   # long documents (~3.5k-7.5k tokens)


def stream(path, split, **kw):
    try:
        return load_dataset(path, split=split, streaming=True, **kw)
    except Exception as e:  # noqa: BLE001
        print(f"skip {path}: {e!r}"[:200], flush=True)
        return []


def take(src, cat, it, n, fn):
    c = 0
    try:
        for ex in it:
            r = fn(ex)
            if r:
                if isinstance(r, str):
                    r = {"prompt": r}
                r["source"], r["category"] = src, cat
                rows.append(r)
                c += 1
            if c >= n:
                break
    except Exception as e:  # noqa: BLE001
        print(f"{src}: stopped after {c}: {e!r}"[:200], flush=True)
    print(src, c, flush=True)


# ---------------------------------------------------------------- chat
take("ultrachat", "chat", stream("HuggingFaceH4/ultrachat_200k", "train_sft"), 1500,
     lambda ex: ex["messages"][0]["content"] if ex["messages"][0]["role"] == "user" and 30 <= len(ex["messages"][0]["content"]) <= 3000 else None)


def multiturn(ex):
    m = ex["messages"]
    if len(m) >= 3 and m[0]["role"] == "user" and m[2]["role"] == "user" and sum(len(x["content"]) for x in m[:3]) <= 6000:
        return {"prompt": m[2]["content"], "messages": [{"role": x["role"], "content": x["content"]} for x in m[:3]]}


take("chat_multiturn", "chat", stream("HuggingFaceH4/ultrachat_200k", "train_gen"), 600, multiturn)
take("no_robots", "chat", stream("HuggingFaceH4/no_robots", "train"), 800,
     lambda ex: ex["messages"][0]["content"] if ex["messages"][0]["role"] == "user" and 30 <= len(ex["messages"][0]["content"]) <= 4000 else None)
take("oasst", "chat", stream("OpenAssistant/oasst2", "train"), 600,
     lambda ex: ex["text"] if ex.get("role") == "prompter" and ex.get("lang") == "en" and 30 <= len(ex["text"]) <= 3000 else None)

# ---------------------------------------------------------------- code
take("magicoder", "code", stream("ise-uiuc/Magicoder-OSS-Instruct-75K", "train"), 1200,
     lambda ex: ex["problem"] if 80 <= len(ex.get("problem") or "") <= 4000 else None)
take("self_oss", "code", stream("bigcode/self-oss-instruct-sc2-exec-filter-50k", "train"), 600,
     lambda ex: ex["instruction"] if 60 <= len(ex.get("instruction") or "") <= 3000 else None)
take("codealpaca", "code", stream("sahil2801/CodeAlpaca-20k", "train"), 400,
     lambda ex: (ex["instruction"] + ("\n\n" + ex["input"] if ex.get("input") else "")) if 20 <= len(ex["instruction"]) <= 2000 else None)
take("code_feedback", "code", stream("m-a-p/CodeFeedback-Filtered-Instruction", "train"), 800,
     lambda ex: ex["query"] if 60 <= len(ex.get("query") or "") <= 4000 else None)


def swe(ex):
    ps = ex.get("problem_statement") or ""
    if 200 <= len(ps) <= MAXC - 600:
        return (f"You are working in the `{ex['repo']}` repository. A user filed this issue:\n\n{ps}\n\n"
                "Locate the likely cause, explain it, and propose a patch as a unified diff. Mention which tests you would add.")


take("swe_issue", "code", stream("princeton-nlp/SWE-bench_Verified", "test"), 500, swe)

# ---------------------------------------------------------------- file edit
def mk_edit(old, msg, lang, fn):
    return (f"Here is the current content of `{fn}`:\n\n```{lang}\n{old}\n```\n\nMake this change: {msg}\n\n"
            "Return the complete updated file in a single code block, followed by a one-line note on what changed.")


def editpack(ex):
    old, msg = ex.get("old_contents") or "", ex.get("message") or ex.get("subject") or ""
    if 400 <= len(old) <= MAXC - 400 and 10 <= len(msg) <= 400:
        return mk_edit(old, msg, (ex.get("lang") or "python").lower(), ex.get("old_file") or "file")


take("file_edit", "file_edit", stream("nuprl/EditPackFT", "train"), 1500, editpack)

# ---------------------------------------------------------------- synthetic polyglot code
TASKS = ["Write a {L} program that reads a CSV file path from the command line, groups rows by column {n}, and prints per-group counts and the average of column {m}. Include error handling.",
         "Implement an LRU cache in {L} with get/put in O(1), capacity {cap}, and unit tests.",
         "In {L}, write a small HTTP server exposing GET /health and POST /{res} (JSON body, in-memory store) with proper status codes and validation.",
         "Write a {L} function that parses ISO-8601 timestamps with offsets, converts them to UTC, and returns them sorted; add tests for {n} edge cases.",
         "Implement a concurrent worker pool in {L} with at most {cap} workers and graceful shutdown on interrupt.",
         "Write a {L} CLI that walks a directory tree, computes SHA-256 of every file larger than {n} KB, and reports duplicates.",
         "Implement a token-bucket rate limiter in {L} ({cap} req/s, burst {m}) for middleware, with tests.",
         "Here is a {L} function with a bug: it should return the {n} most frequent words in a text but sometimes returns fewer. Write the corrected version, an explanation, and a test that would have caught it.",
         "Refactor a {L} module that mixes database access, business rules and HTTP handling for {res} into three layers; show the code and interfaces.",
         "Implement retry with exponential backoff and jitter in {L} for an unreliable {res} API client; make it cancellable.",
         "Implement a pub/sub event bus in {L} with typed topics, backpressure and at-least-once delivery.",
         "Implement in {L} a state machine for a {res} order with states created, paid, shipped, delivered, cancelled; illegal transitions must error.",
         "Implement a thread-safe in-memory key-value store in {L} with TTL expiry and a snapshot-to-disk command.",
         "Write unit tests in {L} for a {res} parser module: valid input, malformed input, and {n} boundary cases.",
         "Write a {L} program that implements a tiny expression evaluator (+ - * / parentheses, variables) with a REPL and tests."]
SQL = ["Write PostgreSQL SQL to find the top {n} {res} by revenue per month for the last year, handling months with no sales; note index choices.",
       "Write a recursive CTE listing a {res} hierarchy with depth and path, then the {n} largest subtrees.",
       "Write a window-function query computing a {cap}-row moving average of daily {res} counts and flag days more than 2 standard deviations off.",
       "Design 3NF tables for a {res} system, write DDL with constraints, and {n} dashboard queries."]
RES = ["orders", "invoices", "sensors", "tickets", "users", "shipments", "products", "sessions", "events", "payments", "documents", "jobs", "devices", "accounts"]
for L, key, n in (("Go", "go", 200), ("Rust", "rust", 150), ("TypeScript", "typescript", 200), ("Python", "python", 200), ("Kotlin", "kotlin", 60),
                  ("SQL", "sql", 100), ("JavaScript", "javascript", 120), ("Bash", "shell", 100), ("C++", "cpp", 100), ("C", "c", 60), ("Java", "java", 80)):
    pool = SQL if key == "sql" else TASKS
    for i in range(n):
        t = pool[i % len(pool)]
        rows.append({"source": f"code_{key}", "category": "code", "prompt": t.format(L=L, n=rng.choice([3, 5, 7, 10, 12, 20]), m=rng.choice([2, 4, 5, 8, 15]),
                                                                                     cap=rng.choice([8, 16, 32, 64, 100, 256]), res=rng.choice(RES))})
    print(f"code_{key}", n)

# ---------------------------------------------------------------- summarization / long documents
take("summ_cnn", "summarization", stream("abisee/cnn_dailymail", "train", name="3.0.0"), 900,
     lambda ex: f"Summarize the following article in 3-5 sentences.\n\n{ex['article']}" if 2500 <= len(ex["article"]) <= MAXC - 200 else None)
take("summ_xsum", "summarization", stream("EdinburghNLP/xsum", "train"), 300,
     lambda ex: f"Summarize this article in one sentence, then in one paragraph.\n\n{ex['document']}" if 1500 <= len(ex["document"]) <= MAXC else None)
LONG_ASKS = ["Write an executive summary of this report, then list its five key findings with the section each comes from.",
             "Summarize this document for a busy decision maker: purpose, main evidence, conclusions, open questions. Then give three quotes that support the conclusions.",
             "Produce a structured outline of this document (headings and bullet points), then a 150-word abstract.",
             "Answer: what problem does this document address, what does it recommend, and what are the weaknesses of its argument? Cite passages."]
take("long_govreport", "summarization", stream("ccdv/govreport-summarization", "train"), 250,
     lambda ex: f"{rng.choice(LONG_ASKS)}\n\n{ex['report'][:LONG_MAX]}" if len(ex["report"]) >= LONG_MIN else None)
take("long_arxiv", "summarization", stream("ccdv/arxiv-summarization", "train"), 200,
     lambda ex: f"{rng.choice(LONG_ASKS)}\n\n{ex['article'][:LONG_MAX]}" if len(ex["article"]) >= LONG_MIN else None)
take("summ_arxiv", "summarization", stream("ccdv/arxiv-summarization", "validation"), 200,
     lambda ex: f"Produce an abstract of at most 150 words for this paper excerpt, then explain the main contribution to a non-expert.\n\n{ex['article'][:MAXC-300]}" if len(ex["article"]) >= 4000 else None)

# ---------------------------------------------------------------- reasoning / math / science
take("reason_openr1", "reasoning", stream("open-r1/OpenR1-Math-220k", "train", name="default"), 700,
     lambda ex: ex["problem"] if 60 <= len(ex.get("problem") or "") <= 2500 else None)
take("reason_mmlupro", "reasoning", stream("TIGER-Lab/MMLU-Pro", "test"), 500,
     lambda ex: (ex["question"] + "\n\n" + "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(ex["options"])) + "\n\nThink it through, then state the answer letter.") if ex.get("options") else None)
take("reason_arc", "reasoning", stream("allenai/ai2_arc", "train", name="ARC-Challenge"), 300,
     lambda ex: ex["question"] + "\n\n" + "\n".join(f"{lbl}. {t}" for lbl, t in zip(ex["choices"]["label"], ex["choices"]["text"])) + "\n\nExplain briefly and give the answer.")
take("gsm8k_train", "math", stream("openai/gsm8k", "train", name="main"), 700, lambda ex: ex["question"])
take("math_hendrycks", "math", stream("EleutherAI/hendrycks_math", "train", name="algebra"), 300,
     lambda ex: ex["problem"] if 40 <= len(ex.get("problem") or "") <= 2000 else None)
take("science_qa", "reasoning", stream("allenai/sciq", "train"), 200, lambda ex: ex["question"] + " Explain the reasoning.")

# ---------------------------------------------------------------- structured output
news = [r["prompt"].split("\n\n", 1)[1] for r in rows if r["source"] == "summ_cnn"][:500]
schemas = ["Extract every person, organization and location mentioned into JSON with keys people, organizations, locations (arrays of strings). Output only JSON.",
           "Return a JSON object with fields title, summary (<=40 words), topics (3-6 strings), sentiment (positive/neutral/negative). Output only JSON.",
           "Produce a JSON array of the 5 most important facts in the text; each item has fact (string) and confidence (0-1). Output only JSON.",
           "Convert the article into a JSON timeline: an array of {date, event} objects in chronological order. Output only JSON."]
for i, a in enumerate(news):
    rows.append({"source": "json_extract", "category": "json", "prompt": f"{schemas[i % len(schemas)]}\n\nText:\n{a[:5000]}"})
print("json_extract", len(news))
things = ["library management system", "weather station", "recipe", "flight booking", "smart thermostat", "e-commerce order", "chess game state", "employee record",
          "movie review site", "bus schedule", "hospital appointment", "podcast episode", "git repository", "solar installation", "game character", "restaurant menu",
          "shipping container", "IoT sensor network", "conference schedule", "bank transaction", "CI pipeline", "kubernetes deployment", "invoice", "GraphQL API", "music playlist"]
gen = ["Design a JSON schema for a {x} and give two example documents that validate against it.",
       "Return a JSON object describing a {x} with at least 8 fields, realistic values and one nested array. Output only JSON.",
       "Write a YAML configuration for a {x} with comments, then the equivalent JSON.",
       "Produce a JSON API response for listing {x}s with pagination metadata and 3 example items. Output only JSON."]
for i in range(400):
    rows.append({"source": "json_gen", "category": "json", "prompt": gen[i % 4].format(x=things[i % len(things)])})
print("json_gen", 400)

# ---------------------------------------------------------------- prose
take("prose_wp", "prose", stream("euclaise/writingprompts", "train"), 900,
     lambda ex: f"{ex['prompt']}\n\nWrite a short story (400-700 words) based on this prompt." if 30 <= len(ex.get("prompt") or "") <= 600 else None)


# ---------------------------------------------------------------- tools
def tool(ex):
    sysm, chat = ex.get("system") or "", ex.get("chat") or ""
    if "USER:" in chat and len(sysm) < 4000:
        user = chat.split("USER:", 1)[1].split("ASSISTANT:", 1)[0].strip()
        if 10 <= len(user) <= 2000:
            return {"prompt": user, "messages": [{"role": "system", "content": sysm.replace("SYSTEM: ", "", 1)}, {"role": "user", "content": user}]}


take("tool_calling", "chat", stream("glaiveai/glaive-function-calling-v2", "train"), 600, tool)

rng.shuffle(rows)
with open(out, "w") as f:
    for i, r in enumerate(rows):
        r["id"] = 100000 + i
        if not r["source"].startswith("long_"):
            r["prompt"] = r["prompt"][:MAXC]
        f.write(json.dumps(r) + "\n")
print("POOL", len(rows), dict(Counter(r["source"] for r in rows)))
print("BY CATEGORY", dict(Counter(r["category"] for r in rows)))
