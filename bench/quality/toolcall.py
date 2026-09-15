#!/usr/bin/env python3
"""Tool-calling reliability harness.

This is the single most predictive test for "will this model work as an
opencode agent". A model can write beautiful code and still be useless in an
agent loop if it emits malformed tool calls, picks the wrong tool, or calls a
tool when it should just answer.

Measured dimensions:
  - call_when_needed   : did it call a tool when one was required?
  - correct_tool       : did it pick the right tool from a crowded list?
  - args_correct       : were the arguments extracted correctly?
  - valid_json         : were arguments parseable JSON matching the schema?
  - no_false_positive  : did it correctly NOT call a tool when none applied?
  - multiturn          : can it consume a tool result and answer?

Usage:
  toolcall.py --url http://127.0.0.1:8099/v1 --model x [--out results.json]
"""
import argparse, json, sys, time, urllib.request, urllib.error

# ---------------------------------------------------------------- tool defs
def fn(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}

TOOLS = [
    fn("read_file", "Read the contents of a file at a given path",
       {"path": {"type": "string", "description": "File path"}}, ["path"]),
    fn("write_file", "Write content to a file",
       {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    fn("list_directory", "List files in a directory",
       {"path": {"type": "string"}}, ["path"]),
    fn("run_command", "Execute a shell command",
       {"command": {"type": "string"}, "timeout": {"type": "integer"}}, ["command"]),
    fn("search_code", "Search for a regex pattern across the repository",
       {"pattern": {"type": "string"}, "file_glob": {"type": "string"}}, ["pattern"]),
    fn("get_weather", "Get current weather for a city",
       {"city": {"type": "string"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    fn("create_issue", "Open a bug tracker issue",
       {"title": {"type": "string"}, "body": {"type": "string"},
        "labels": {"type": "array", "items": {"type": "string"}}}, ["title"]),
]

# scenario: (id, prompt, expected_tool_or_None, arg_checker)
def eq(k, v):
    return lambda a: str(a.get(k, "")).strip().lower() == v.lower()
def contains(k, v):
    return lambda a: v.lower() in str(a.get(k, "")).lower()
def anyof(k, vals):
    return lambda a: any(v.lower() in str(a.get(k, "")).lower() for v in vals)

SCENARIOS = [
    # --- basic single-tool selection -------------------------------------
    ("read_simple", "Show me what's inside /etc/hostname", "read_file", contains("path", "/etc/hostname")),
    ("list_simple", "What files are in the /var/log directory?", "list_directory", contains("path", "/var/log")),
    ("weather_simple", "What's the weather in Tokyo right now?", "get_weather", contains("city", "Tokyo")),
    ("run_simple", "Run 'git status' for me", "run_command", contains("command", "git status")),
    ("search_simple", "Find every place we call malloc in the C sources",
     "search_code", contains("pattern", "malloc")),

    # --- correct tool among confusable alternatives -----------------------
    ("read_not_list", "I need to see the contents of src/main.rs", "read_file", contains("path", "main.rs")),
    ("list_not_read", "Which files live under src/handlers?", "list_directory", contains("path", "handlers")),
    ("search_not_run", "Where in the codebase is the term TODO_REFACTOR used?",
     "search_code", contains("pattern", "TODO_REFACTOR")),
    ("run_not_search", "Execute the test suite with 'cargo test --all'",
     "run_command", contains("command", "cargo test")),
    ("write_not_read", "Create a file at /tmp/notes.txt containing the text 'hello world'",
     "write_file", lambda a: "notes.txt" in str(a.get("path", "")) and "hello" in str(a.get("content", "")).lower()),

    # --- multi-argument extraction ---------------------------------------
    ("weather_units", "Give me the temperature in Berlin in fahrenheit",
     "get_weather", lambda a: "berlin" in str(a.get("city", "")).lower()
                              and "fahren" in str(a.get("units", "")).lower()),
    ("run_timeout", "Run 'sleep 5' but give up after 10 seconds",
     "run_command", lambda a: "sleep" in str(a.get("command", "")) and str(a.get("timeout", "")) in ("10", "10.0")),
    ("search_glob", "Search for 'unwrap()' but only in .rs files",
     "search_code", lambda a: "unwrap" in str(a.get("pattern", "")) and "rs" in str(a.get("file_glob", ""))),
    ("issue_labels", "Open an issue titled 'Crash on startup' and tag it with the labels bug and urgent",
     "create_issue", lambda a: "crash" in str(a.get("title", "")).lower()),
    ("write_multiline", "Write a Python hello world script to /tmp/hi.py",
     "write_file", lambda a: "hi.py" in str(a.get("path", "")) and "print" in str(a.get("content", "")).lower()),

    # --- must NOT call a tool (false-positive detection) -------------------
    ("no_tool_greeting", "Hello! How are you doing today?", None, None),
    ("no_tool_concept", "Explain the difference between a mutex and a semaphore in two sentences.", None, None),
    ("no_tool_math", "What is 17 multiplied by 23? Just answer.", None, None),
    ("no_tool_opinion", "In your view, is Rust a good choice for systems programming? Answer briefly.", None, None),
    ("no_tool_defn", "What does the acronym RAII stand for?", None, None),
]

MULTITURN = [
    ("multiturn_read",
     "Read /tmp/config.json and tell me the value of the 'port' field.",
     "read_file",
     '{"port": 8080, "host": "localhost"}',
     ["8080"]),
    ("multiturn_weather",
     "What's the weather in Oslo?",
     "get_weather",
     '{"city":"Oslo","temp_c":-3,"condition":"snow"}',
     ["-3", "snow"]),
]


def post(url, payload, timeout=300, api_key=None):
    body = json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json"}
    if api_key:
        hdrs["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def extract_calls(msg):
    """Return list of (name, args_dict, raw_args, json_ok)."""
    out = []
    for tc in (msg.get("tool_calls") or []):
        f = tc.get("function", {}) or {}
        raw = f.get("arguments", "")
        if isinstance(raw, dict):
            out.append((f.get("name"), raw, json.dumps(raw), True))
            continue
        try:
            out.append((f.get("name"), json.loads(raw or "{}"), raw, True))
        except Exception:
            out.append((f.get("name"), {}, raw, False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="local")
    ap.add_argument("--out", default="/tmp/toolcall-results.json")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    results, t_start = [], time.time()

    for sid, prompt, want_tool, checker in SCENARIOS:
        rec = {"id": sid, "expected": want_tool, "status": "?"}
        try:
            d = post(args.url, {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "tools": TOOLS, "tool_choice": "auto",
                "temperature": args.temperature, "max_tokens": 512,
            }, api_key=args.api_key)
            msg = d["choices"][0]["message"]
            calls = extract_calls(msg)
            rec["n_calls"] = len(calls)
            rec["called"] = [c[0] for c in calls]
            rec["json_ok"] = all(c[3] for c in calls) if calls else True

            if want_tool is None:
                # should NOT have called anything
                rec["status"] = "pass" if not calls else "false_positive"
            elif not calls:
                rec["status"] = "no_call"
            elif not rec["json_ok"]:
                rec["status"] = "bad_json"
                rec["raw_args"] = calls[0][2][:200]
            elif calls[0][0] != want_tool:
                rec["status"] = "wrong_tool"
            elif checker and not checker(calls[0][1]):
                rec["status"] = "bad_args"
                rec["got_args"] = calls[0][1]
            else:
                rec["status"] = "pass"
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e)[:200]
        results.append(rec)
        print(f"  {rec['status']:16s} {sid}", flush=True)

    # ---- multi-turn: model calls tool, we feed result back, it must answer
    for sid, prompt, want_tool, tool_result, must_contain in MULTITURN:
        rec = {"id": sid, "expected": want_tool, "status": "?"}
        try:
            msgs = [{"role": "user", "content": prompt}]
            d = post(args.url, {"model": args.model, "messages": msgs, "tools": TOOLS,
                                "tool_choice": "auto", "temperature": args.temperature,
                                "max_tokens": 512}, api_key=args.api_key)
            msg = d["choices"][0]["message"]
            calls = extract_calls(msg)
            if not calls or calls[0][0] != want_tool:
                rec["status"] = "no_call" if not calls else "wrong_tool"
                rec["called"] = [c[0] for c in calls]
            else:
                tc_id = (msg.get("tool_calls") or [{}])[0].get("id", "call_0")
                msgs.append({"role": "assistant", "tool_calls": msg["tool_calls"],
                             "content": msg.get("content") or ""})
                msgs.append({"role": "tool", "tool_call_id": tc_id,
                             "name": want_tool, "content": tool_result})
                d2 = post(args.url, {"model": args.model, "messages": msgs, "tools": TOOLS,
                                     "temperature": args.temperature, "max_tokens": 512},
                          api_key=args.api_key)
                answer = (d2["choices"][0]["message"].get("content") or "")
                rec["answer"] = answer[:200]
                rec["status"] = "pass" if all(s.lower() in answer.lower() for s in must_contain) else "bad_answer"
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e)[:200]
        results.append(rec)
        print(f"  {rec['status']:16s} {sid}", flush=True)

    # ------------------------------------------------------------ summary
    total = len(results)
    npass = sum(1 for r in results if r["status"] == "pass")
    by = {}
    for r in results:
        by[r["status"]] = by.get(r["status"], 0) + 1

    pos = [r for r in results if r["expected"] is not None]
    neg = [r for r in results if r["expected"] is None]
    summary = {
        "model": args.model,
        "total": total,
        "passed": npass,
        "pass_rate": round(100.0 * npass / total, 1),
        "positive_pass_rate": round(100.0 * sum(1 for r in pos if r["status"] == "pass") / max(len(pos), 1), 1),
        "false_positive_rate": round(100.0 * sum(1 for r in neg if r["status"] == "false_positive") / max(len(neg), 1), 1),
        "malformed_json": by.get("bad_json", 0),
        "breakdown": by,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    print("\n=== TOOL-CALLING SUMMARY ===")
    for k, v in summary.items():
        if k != "breakdown":
            print(f"  {k:22s} {v}")
    print(f"  breakdown             {by}")

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
