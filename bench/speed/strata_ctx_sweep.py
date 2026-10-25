#!/usr/bin/env python3
"""Context ceiling and long-context speed of a Strata-served model, through the production stack.

Strata's server has no /tokenize and ignores cache_prompt, so depth_sweep.py cannot drive it. This driver
sizes prompts with Strata's /v1/messages/count_tokens (the server's own tokenizer and template; the model
does not run) and keeps every cold request cold by starting it with a unique line, so no checkpoint or
parked conversation of an earlier request can match its prefix. `cache_n` is recorded to prove it.

Each --ctx NAME=CONFIG is one variant: the registry entry's extra_args point at that Strata run config.

  probe  load each variant, read the engine's /metrics (expert slots, KV residency, RAM), run one ~4K request
  sweep  on each variant: cold requests at each depth, each followed by an agentic follow-up turn (the answer
         plus a new instruction appended), which reads only the new tail from the conversation cache

    bench/speed/strata_ctx_sweep.py probe --model qwen38-coder-strata \\
        --ctx 32k=~/.config/llms/strata/coder-iq1_m.json --ctx 256k=~/.config/llms/strata/coder-iq1_m-256k.json
    bench/speed/strata_ctx_sweep.py sweep --model qwen38-coder-strata \\
        --ctx 256k=~/.config/llms/strata/coder-iq1_m-256k.json --depths 4096,65536,250000 --reps 2

With --agent-request CAPTURE.json (one chat request recorded from the agent you use), every request carries that
agent's system prompt and tools and no sampling fields, so the server applies the model card's defaults as it does
for the agent. The context is a session of read tool calls answered with corpus files, and the follow-up answers
whatever the model did (tool results for its tool calls, else a new user turn). The capture holds the agent's whole
system prompt, personal instructions included: keep it out of the repository.
"""

import json
from pathlib import Path
import re
import sys
import threading
import time
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from common import (PROVENANCE, QUESTION, TSV, StackError, Variant, arg_parser, chat_with_peak,  # noqa: E402
                    load_corpus, log, mem, stack_for, timings)

COLUMNS = ["model", "variant", "max_ctx", "phase", "depth", "rep", "prompt_tok", "cache_n", "prompt_n", "compl_tok",
           "pp_tok_s", "tg_tok_s", "ttft_s", "wall_s", "draft_n", "draft_acc", "acc_pct", "expert_slots",
           "expert_cache_mib", "kv_resident", "arena_mib", "ram_used_gib", "min_avail_gib", "vram_mib", "load_s",
           "tool_call", "finish", "reasoning_chars", "status", *PROVENANCE]
FOLLOWUP = ("Now write the fix for the problem you named as a short unified diff against that file. "
            "Only the diff, no explanation.")


def post_json(url, body, timeout=600):
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def mem_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return None


class AvailSampler:
    """Lowest MemAvailable while a request runs: the host-RAM headroom the run left."""

    def __init__(self, interval=0.5):
        self.low, self.interval, self._stop = None, interval, threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            value = mem_available_gib()
            if value is not None and (self.low is None or value < self.low):
                self.low = value
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def variant_for(model, spec):
    name, _, path = spec.partition("=")
    if not path:
        raise SystemExit(f"--ctx wants NAME=CONFIG, got {spec!r}")
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    max_ctx = int(config["args"][config["args"].index("--max-context") + 1])
    return Variant(name, entries={model: {"extra_args": [f"--config {path}"], "ctx": max_ctx}}), max_ctx


def engine_state(stack, model):
    try:
        metrics = get_json(stack.upstream(model, "/metrics"))
    except (OSError, ValueError):
        return {}
    engine, hardware = metrics.get("engine") or {}, metrics.get("hardware") or {}
    used = hardware.get("ram_used")
    return {"expert_slots": engine.get("expert_slots", mem.NA), "expert_cache_mib": engine.get("expert_cache_mib", mem.NA),
            "kv_resident": engine.get("kv_resident", mem.NA), "arena_mib": engine.get("arena_mib", mem.NA),
            "ram_used_gib": round(used / 2**30, 1) if used else mem.NA}


def sized_text(count, corpus, target):
    """The longest corpus prefix whose prompt is about target tokens (bisection on count_tokens)."""
    lo, hi, best, best_diff = 0, len(corpus), "", 10**9
    for _ in range(24):
        mid = (lo + hi) // 2
        n = count(corpus[:mid])
        if abs(n - target) < best_diff:
            best, best_diff = corpus[:mid], abs(n - target)
        if n < target:
            lo = mid + 1
        elif n > target:
            hi = mid - 1
        if best_diff <= max(64, target // 500) or lo > hi:
            break
    return best


def request(stack, model, payload):
    """(row fields, response) for one chat request, with timings, wall time, VRAM peak and RAM headroom."""
    started = time.monotonic()
    with AvailSampler() as avail:
        response, vram, _ = chat_with_peak(stack, model, payload, timeout=3600)
    wall = time.monotonic() - started
    t = response.get("timings") or {}
    row = {**timings(response), "cache_n": t.get("cache_n", mem.NA), "prompt_n": t.get("prompt_n", mem.NA),
           "ttft_s": round(t["prompt_ms"] / 1000, 2) if t.get("prompt_ms") is not None else mem.NA,
           "wall_s": round(wall, 1), "vram_mib": mem.cell(vram), "tool_call": made_tool_call(response),
           "finish": (response.get("choices") or [{}])[0].get("finish_reason", mem.NA),
           "reasoning_chars": len(((response.get("choices") or [{}])[0].get("message") or {}).get("reasoning_content") or ""),
           "min_avail_gib": round(avail.low, 1) if avail.low is not None else mem.NA}
    return row, response


def cold_payload(model, text, gen):
    nonce = f"// session {uuid.uuid4().hex}\n"
    return {"model": model, "max_tokens": gen, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": nonce + text + QUESTION}]}


def followup_payload(payload, response, gen, tool_output):
    """The next agentic turn: the answer, then a tool result (a file from elsewhere in the corpus) and an instruction."""
    message = (response.get("choices") or [{}])[0].get("message") or {}
    answer = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("reasoning_content"):
        answer["reasoning_content"] = message["reasoning_content"]
    turn = f"Here is another file you asked to see:\n\n{tool_output}\n\n{FOLLOWUP}"
    return {**payload, "max_tokens": gen, "messages": [*payload["messages"], answer, {"role": "user", "content": turn}]}


FILE_MARK = re.compile(r"^// ===== (.+?) =====$", re.M)
AGENT_TASK = ("Find the most error-prone function in the allocator and backend scheduling code, read whatever you need, "
              "then fix it.")


def corpus_files(corpus, max_lines=2000):
    """(path, numbered text) per corpus file, as the agent's read tool returns it (line numbers, capped)."""
    marks = list(FILE_MARK.finditer(corpus))
    files = []
    for i, mark in enumerate(marks):
        body = corpus[mark.end() + 1:marks[i + 1].start() if i + 1 < len(marks) else len(corpus)]
        lines = body.splitlines()[:max_lines]
        if lines:
            files.append((mark.group(1), "\n".join(f"{n}: {line}" for n, line in enumerate(lines, 1))))
    return files


class Agent:
    """Requests shaped like a captured agent request: its system prompt and tools, no sampling fields (the
    server applies the model card's), and a history of read tool calls answered with corpus files."""

    def __init__(self, path, files, count):
        captured = json.loads(Path(path).expanduser().read_text())
        self.system = [m for m in captured["messages"] if m["role"] == "system"]
        self.tools = captured.get("tools") or []
        self.files, self.count = files, count
        self.base = count("\n".join(m["content"] for m in self.system) + json.dumps(self.tools))
        self._sizes = {}

    def size(self, i):
        if i not in self._sizes:
            self._sizes[i] = self.count(self.files[i][1]) + 40
        return self._sizes[i]

    def history(self, depth):
        """The task, then read calls and their results until about depth tokens (the system prompt included)."""
        messages = [{"role": "user", "content": f"[session {uuid.uuid4().hex}] {AGENT_TASK}"}]
        total, i = self.base + 60, 0
        while i < len(self.files) and total < depth - 256:
            call = f"call_{i}"
            path, text = self.files[i]
            arguments = {"path": path}
            size = self.size(i)
            if total + size > depth:                     # the last read takes only the lines that fit (read's limit)
                lines = text.splitlines()
                keep = max(1, int(len(lines) * (depth - total) / size))
                text, arguments["limit"] = "\n".join(lines[:keep]), keep
                size = self.count(text) + 40
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": call, "type": "function", "function": {"name": "read", "arguments": json.dumps(arguments)}}]})
            messages.append({"role": "tool", "tool_call_id": call, "content": text})
            total, i = total + size, i + 1
        return messages

    def payload(self, model, messages, gen):
        return {"model": model, "stream": False, "max_tokens": gen, "messages": [*self.system, *messages],
                "tools": self.tools}

    def followup(self, payload, response, tool_output):
        """Answer what the model did: its tool calls get results (the first one a file), a text answer gets a new
        user turn with a file. Only this tail is new to the conversation cache."""
        message = (response.get("choices") or [{}])[0].get("message") or {}
        answer = {"role": "assistant", "content": message.get("content") or ""}
        for key in ("reasoning_content", "tool_calls"):
            if message.get(key):
                answer[key] = message[key]
        tail = [answer]
        if message.get("tool_calls"):
            for n, call in enumerate(message["tool_calls"]):
                tail.append({"role": "tool", "tool_call_id": call.get("id"),
                             "content": tool_output if n == 0 else "(no output)"})
        else:
            tail.append({"role": "user", "content": f"Here is another file you should look at:\n\n{tool_output}\n\n"
                                                    "Continue with the fix."})
        return {**payload, "messages": [*payload["messages"], *tail]}


def made_tool_call(response):
    message = (response.get("choices") or [{}])[0].get("message") or {}
    return "yes" if message.get("tool_calls") else "no"


def run_variant(args, out, corpus, variant, max_ctx):
    model = args.model[0]
    base = {"model": model, "max_ctx": max_ctx}   # stack.record() carries the variant name
    log(f"############ {model} / {variant.name} (max context {max_ctx})")
    with stack_for(args, variant) as stack:
        record = stack.record(model)
        try:
            load_s = round(stack.load(model, timeout=1800), 1)
        except StackError as error:
            log(f"  LOAD FAILED: {str(error)[:2000]}")
            out.row(**base, phase="load", status="LOAD_FAIL", **record)
            return
        state = engine_state(stack, model)
        log(f"  loaded in {load_s}s: {state}")
        count_url = stack.upstream(model, "/v1/messages/count_tokens")

        def count(text):
            body = {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": text}]}
            return post_json(count_url, body)["input_tokens"]

        tool_output = corpus[-args.tool_chars:]
        if args.phase == "sweep" and not args.no_followup:
            log(f"  follow-up tool output: {count(tool_output)} tokens ({args.tool_chars} chars from the corpus end)")
        depths = [4096] if args.phase == "probe" else [d for d in args.depths if d + args.gen * 2 + 512 <= max_ctx]
        skipped = [d for d in args.depths if d not in depths] if args.phase == "sweep" else []
        if skipped:
            log(f"  depths over this context, skipped: {skipped}")
        reps = 1 if args.phase == "probe" else args.reps
        agent = Agent(args.agent_request, corpus_files(corpus), count) if args.agent_request else None
        if agent:
            log(f"  agent request: {len(agent.system)} system message(s), {len(agent.tools)} tools, ~{agent.base} tokens "
                "before the conversation; no sampling fields (the server's model-card defaults)")
        for depth in depths:
            text = None if agent else sized_text(count, corpus, depth - len(QUESTION) // 3)
            for rep in range(1, reps + 1):
                payload = agent.payload(model, agent.history(depth), args.gen) if agent else cold_payload(model, text, args.gen)
                try:
                    row, response = request(stack, model, payload)
                except (OSError, ValueError) as error:
                    log(f"  depth {depth} rep {rep}: cold request failed: {error}")
                    out.row(**base, phase="cold", depth=depth, rep=rep, status="REQUEST_FAIL", load_s=load_s, **record)
                    break
                out.row(**base, phase=args.phase if args.phase == "probe" else "cold", depth=depth, rep=rep, **row,
                        **engine_state(stack, model), load_s=load_s, status="OK", **record)
                log(f"  depth {depth} rep {rep} cold: {row['prompt_tok']} tok (cache {row['cache_n']}) PP {row['pp_tok_s']} "
                    f"TTFT {row['ttft_s']}s | TG {row['tg_tok_s']} | acc {row['acc_pct']}% | avail {row['min_avail_gib']} GiB"
                    f" | tool call {row['tool_call']}, finish {row['finish']}, reasoning {row['reasoning_chars']} chars")
                if args.phase == "probe" or args.no_followup:
                    continue
                try:
                    next_payload = (agent.followup(payload, response, tool_output) if agent
                                    else followup_payload(payload, response, args.gen, tool_output))
                    frow, _ = request(stack, model, next_payload)
                except (OSError, ValueError) as error:
                    log(f"  depth {depth} rep {rep}: follow-up failed: {error}")
                    out.row(**base, phase="followup", depth=depth, rep=rep, status="REQUEST_FAIL", load_s=load_s, **record)
                    continue
                out.row(**base, phase="followup", depth=depth, rep=rep, **frow, **engine_state(stack, model),
                        load_s=load_s, status="OK", **record)
                log(f"  depth {depth} rep {rep} follow-up: {frow['prompt_tok']} tok, read {frow['prompt_n']} "
                    f"(cache {frow['cache_n']}) TTFT {frow['ttft_s']}s | TG {frow['tg_tok_s']} | acc {frow['acc_pct']}%"
                    f" | tool call {frow['tool_call']}, finish {frow['finish']}")


def main():
    parser = arg_parser(__doc__, default_out="results/speed/strata-ctx-sweep.tsv")
    parser.add_argument("phase", choices=["probe", "sweep"])
    parser.add_argument("--ctx", action="append", required=True, help="NAME=STRATA_RUN_CONFIG (repeatable)")
    parser.add_argument("--depths", default="4096,16384,32768,65536,131072,196608,253952",
                        type=lambda s: [int(d) for d in s.split(",")])
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--gen", type=int, default=192)
    parser.add_argument("--no-followup", action="store_true")
    parser.add_argument("--tool-chars", type=int, default=7000, help="size of the follow-up turn's tool output")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--agent-request", default=None,
                        help="a captured agent chat request (JSON): send its system prompt and tools, no sampling "
                             "fields, and fill the context with read tool calls instead of one user message")
    args = parser.parse_args()
    if len(args.model) != 1:
        parser.error("one --model per run")
    corpus = load_corpus(args.corpus) if args.corpus else load_corpus()
    out = TSV(args.out, COLUMNS)
    for spec in args.ctx:
        variant, max_ctx = variant_for(args.model[0], spec)
        run_variant(args, out, corpus, variant, max_ctx)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
