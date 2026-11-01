import json, sys, time, glob, os, urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:8091")
TARGET = int(os.environ.get("TARGET", "131072"))
MODE = sys.argv[1] if len(sys.argv) > 1 else "build"
SRC = os.path.join(os.environ.get("LLAMA_CPP_SRC", os.path.expanduser("~/dev/llama.cpp")), "src")
OUT = os.environ.get("OUT", "/tmp/opencode/flint_bench")
os.makedirs(OUT, exist_ok=True)


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=None) as r:
        return json.loads(r.read())


def ntok(text):
    return len(post("/tokenize", {"content": text})["tokens"])


TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from the repository", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search the repository with a regex", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
]
SYSTEM = "You are a coding agent working in the llama.cpp repository. Use the tools to inspect and edit code. Be precise and cite file paths and function names."
TASK = "\n\nTask: Using the source files above, explain how the KV cache handles sliding-window attention layers differently from full-attention layers (name the relevant classes/functions and files), then identify one place where a context shift or sequence removal could interact badly with SWA, and use the tools to inspect it."


def build():
    files = sorted(glob.glob(SRC + "/*.cpp") + glob.glob(SRC + "/*.h"), key=os.path.getsize)
    parts, total = [], 0
    for f in files:
        txt = open(f, errors="replace").read()
        block = f"=== FILE: src/{os.path.basename(f)} ===\n{txt}\n"
        n = ntok(block)
        if total + n > TARGET - 1500:
            continue
        parts.append(block)
        total += n
    ctx = "".join(parts)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ctx + TASK}]
    full = post("/apply-template", {"messages": msgs, "tools": TOOLS})["prompt"]
    print(f"files={len(parts)} ctx_tokens~{total} templated_prompt_tokens={ntok(full)}")
    json.dump(msgs, open(OUT + "/turn1_msgs.json", "w"))


def chat(msgs, tag):
    t0 = time.time()
    r = post("/v1/chat/completions", {"messages": msgs, "tools": TOOLS, "temperature": float(os.environ.get("TEMP", "1.0")), "top_p": 0.95, "top_k": 20, "timings_per_token": False, **({"seed": int(os.environ["SEED"])} if os.environ.get("SEED") else {})})
    wall = time.time() - t0
    json.dump(r, open(f"{OUT}/{tag}_resp.json", "w"))
    t = r["timings"]
    m = r["choices"][0]["message"]
    print(f"[{tag}] wall={wall:.1f}s cache_n={t.get('cache_n')} prompt_n={t['prompt_n']} pp={t['prompt_per_second']:.1f} t/s ({t['prompt_ms']/1000:.1f}s) | gen_n={t['predicted_n']} tg={t['predicted_per_second']:.1f} t/s | finish={r['choices'][0]['finish_reason']} tool_calls={[c['function']['name'] + ' ' + c['function']['arguments'][:120] for c in (m.get('tool_calls') or [])]} reasoning_len={len(m.get('reasoning_content') or '')} content_len={len(m.get('content') or '')}")
    return r


def run():
    msgs = json.load(open(OUT + "/turn1_msgs.json"))
    r1 = chat(msgs, "turn1")
    m = r1["choices"][0]["message"]
    asst = {"role": "assistant", "content": m.get("content") or ""}
    if m.get("reasoning_content"):
        asst["reasoning_content"] = m["reasoning_content"]
    calls = m.get("tool_calls") or []
    tool_out = open(SRC + "/llama-kv-cache.cpp", errors="replace").read()[:8000]
    if calls:
        asst["tool_calls"] = calls[:1]
        nxt = {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "content": tool_out}
    else:
        nxt = {"role": "user", "content": "Here is the start of src/llama-kv-cache.cpp:\n" + tool_out + "\nContinue the analysis."}
    msgs2 = msgs + [asst, nxt]
    chat(msgs2, "turn2")


build() if MODE == "build" else run()
