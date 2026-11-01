#!/usr/bin/env python3
"""Several agentic sessions on one served model at once: per-session speed, waiting, and what a long read costs.

Client-timed over llama-swap (the production stack, one --variant per arm), streamed, so it works for any engine:

  ttft_s        request sent -> first streamed token (includes waiting for a slot and reading the prompt)
  decode_tok_s  (completion tokens - 1) / (first token -> last token)
  max_gap_s     the longest pause between two streamed chunks after the first

Scenarios (each request starts with a unique line, so no prompt cache or parked conversation can match it):

  solo    one session: a cold --depth prompt, then a follow-up turn on the same conversation
  concN   N such sessions started together (N from --sessions), each running its two turns on its own
  join    one session decodes a long answer from a short prompt; --join-delay s later a cold --join-depth
          prompt arrives: the decoding stream's longest stop and rate while it is read, and the joiner's TTFT

Sampling is the agent's (temperature 1.0, top_p 0.95, top_k 20, a fixed seed per session). Peak device VRAM, GTT
(amdgpu) and used RAM are sampled every 0.5 s during each scenario.

    bench/speed/concurrent_sessions.py --model coder-3.8fn \\
        --variant 'p3={"entries": {"coder-3.8fn": {"extra_args": ["--config /path/strata-p3.json"]}}}' \\
        --sessions 2,3 --depth 65536 --join-depth 131072
"""

import json
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from common import PROVENANCE, TSV, StackError, arg_parser, load_corpus, log, mem, stack_for, variants  # noqa: E402

COLUMNS = ["model", "variant", "scenario", "session", "turn", "prompt_tok", "cached_tok", "compl_tok", "ttft_s",
           "decode_tok_s", "max_gap_s", "wall_s", "t_start_s", "peak_vram_mib", "peak_gtt_mib", "peak_ram_mib",
           "join_read_s", "join_stream_chunks_s", "join_stream_stop_s", "finish", "status", *PROVENANCE]

SYSTEM = ("You are a coding agent working in the llama.cpp repository. Use what you are given to answer precisely, "
          "citing file paths and function names.")
TASKS = ["Explain how the KV cache is laid out and allocated in the code above, then propose one refactor.",
         "Find a function above that could overflow or index out of bounds, explain why, and show a fix.",
         "Summarise how the sampling code above chains its samplers, then write a new top-n-sigma sampler.",
         "Pick the most complex function above, explain it step by step, and suggest how to test it."]
FOLLOWUP = "Good. Now write a focused unit test for the change you proposed, as a single code block."
STREAM_TASK = "Write a complete, well-commented Python implementation of a B-tree with insert, delete and search."


def post_json(url, body, timeout=600):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def stream_chat(url, payload, timeout=3600):
    """Stream one chat request; returns the timing record and the assistant text."""
    body = dict(payload, stream=True, stream_options={"include_usage": True})
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    sent = time.monotonic()
    first = last = None
    chunks, gaps, usage, finish, text, times = 0, [], {}, None, [], []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                if not piece:
                    continue
                now = time.monotonic()
                if first is None:
                    first = now
                else:
                    gaps.append(now - last)
                last = now
                times.append(now)
                chunks += 1
                text.append(delta.get("content") or "")
    done = time.monotonic()
    tokens = usage.get("completion_tokens") or chunks
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tok": usage.get("prompt_tokens", mem.NA), "cached_tok": details.get("cached_tokens", mem.NA),
        "compl_tok": tokens, "ttft_s": round(first - sent, 2) if first else mem.NA,
        "decode_tok_s": round((tokens - 1) / (last - first), 2) if first and last > first and tokens > 1 else mem.NA,
        "max_gap_s": round(max(gaps), 2) if gaps else mem.NA, "wall_s": round(done - sent, 2),
        "finish": finish or mem.NA, "_sent": sent, "_first": first, "_times": times,
    }, "".join(text)


class Sampler:
    """Peak device VRAM, amdgpu GTT and used RAM (MiB) over a scenario."""

    def __init__(self, interval=0.5):
        self.interval, self.peak = interval, {"vram": None, "gtt": None, "ram": None}
        self._stop = threading.Event()

    @staticmethod
    def _gtt():
        best = None
        for path in Path("/sys/class/drm").glob("card[0-9]*/device/mem_info_gtt_used"):
            try:
                best = max(best or 0, int(path.read_text()) // 1048576)
            except (OSError, ValueError):
                pass
        return best

    @staticmethod
    def _ram():
        try:
            info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            return (int(info["MemTotal"].split()[0]) - int(info["MemAvailable"].split()[0])) // 1024
        except (OSError, KeyError, ValueError):
            return None

    def _loop(self):
        while not self._stop.is_set():
            for key, value in (("vram", mem.vram_used_mib()), ("gtt", self._gtt()), ("ram", self._ram())):
                if value is not None:
                    self.peak[key] = max(self.peak[key] or 0, value)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def sized_text(count, corpus, target):
    """The longest corpus prefix whose prompt is about target tokens (bisection on the server's count)."""
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


def session_messages(context, index):
    nonce = f"[session {uuid.uuid4().hex}]\n"
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": nonce + context + "\n\n" + TASKS[index % len(TASKS)]}]


def run(args):
    corpus = load_corpus(args.corpus) if args.corpus else load_corpus()
    counts = [int(n) for n in args.sessions.split(",") if n]
    out = TSV(args.out, COLUMNS)
    for model in args.model:
        for variant in variants(args):
            log(f"############ {model} / {variant.name}")
            with stack_for(args, variant) as stack:
                record = stack.record(model)
                try:
                    load_s = round(stack.load(model), 1)
                except StackError as error:
                    log(f"  LOAD FAILED: {error}")
                    out.row(model=model, status="LOAD_FAIL", **record)
                    continue
                log(f"  loaded in {load_s}s")
                url = stack.base + "/v1/chat/completions"
                count_url = stack.upstream(model, "/v1/messages/count_tokens")

                def count(text):
                    body = {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": text}]}
                    return post_json(count_url, body)["input_tokens"]

                context = sized_text(count, corpus, args.depth)
                # a tool result for the follow-up turn, as an agent's read returns it (~2K tokens)
                tool_output = "\n\nHere is the file you asked for:\n" + corpus[-8000:]
                join_context = sized_text(count, corpus, args.join_depth) if args.join_depth else ""

                def payload(messages, seed, max_tokens):
                    return {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 1.0,
                            "top_p": 0.95, "top_k": 20, "seed": seed, "reasoning_effort": args.reasoning}

                def emit(scenario, session, turn, stats, sampler, t0, **extra):
                    clean = {k: v for k, v in stats.items() if not k.startswith("_")}
                    out.row(model=model, scenario=scenario, session=session, turn=turn,
                            t_start_s=round(stats["_sent"] - t0, 2), peak_vram_mib=mem.cell(sampler.peak["vram"]),
                            peak_gtt_mib=mem.cell(sampler.peak["gtt"]), peak_ram_mib=mem.cell(sampler.peak["ram"]),
                            status="OK", **clean, **extra, **record)
                    log(f"  {scenario} s{session} t{turn}: prompt {clean['prompt_tok']} (cached {clean['cached_tok']})"
                        f" ttft {clean['ttft_s']}s | {clean['compl_tok']} tok at {clean['decode_tok_s']} t/s | "
                        f"max gap {clean['max_gap_s']}s")

                def sessions(n, scenario):
                    results, errors = {}, []

                    def one(i):
                        try:
                            messages = session_messages(context, i)
                            first, answer = stream_chat(url, payload(messages, 1000 + i, args.gen))
                            messages = messages + [{"role": "assistant", "content": answer},
                                                   {"role": "user", "content": FOLLOWUP + tool_output}]
                            second, _ = stream_chat(url, payload(messages, 2000 + i, args.gen))
                            results[i] = (first, second)
                        except (OSError, ValueError) as error:
                            errors.append(f"session {i}: {error}")

                    t0 = time.monotonic()
                    with Sampler() as sampler:
                        threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
                        for thread in threads:
                            thread.start()
                        for thread in threads:
                            thread.join()
                    for i in sorted(results):
                        for turn, stats in enumerate(results[i], 1):
                            emit(scenario, i, turn, stats, sampler, t0)
                    total = sum(int(s["compl_tok"]) for pair in results.values() for s in pair)
                    span = time.monotonic() - t0
                    log(f"  {scenario}: {total} tokens in {span:.1f}s = {total / span:.1f} tok/s aggregate"
                        f" | peak vram {sampler.peak['vram']} gtt {sampler.peak['gtt']} ram {sampler.peak['ram']} MiB")
                    if errors:
                        raise StackError("; ".join(errors))

                try:
                    sessions(1, "solo")
                    for n in counts:
                        sessions(n, f"conc{n}")
                    if join_context:
                        stats, errors = {}, []

                        def stream():
                            try:
                                stats["stream"] = stream_chat(url, payload(
                                    [{"role": "user", "content": f"[{uuid.uuid4().hex}]\n{STREAM_TASK}"}],
                                    3000, args.stream_gen))[0]
                            except (OSError, ValueError) as error:
                                errors.append(f"stream: {error}")

                        def joiner():
                            time.sleep(args.join_delay)
                            try:
                                stats["join"] = stream_chat(url, payload(session_messages(join_context, 0), 3001,
                                                                         args.gen))[0]
                            except (OSError, ValueError) as error:
                                errors.append(f"joiner: {error}")

                        t0 = time.monotonic()
                        with Sampler() as sampler:
                            threads = [threading.Thread(target=stream), threading.Thread(target=joiner)]
                            for thread in threads:
                                thread.start()
                            for thread in threads:
                                thread.join()
                        if errors:
                            raise StackError("; ".join(errors))
                        s, j = stats["stream"], stats["join"]
                        emit("join", 0, 1, s, sampler, t0)
                        if s["_first"] and j["_first"]:
                            lo, hi = j["_sent"], j["_first"]
                            inside = [t for t in s["_times"] if lo <= t <= hi]
                            edges = [lo, *inside, min(hi, s["_times"][-1])]
                            stop = max(b - a for a, b in zip(edges, edges[1:])) if len(edges) > 1 else mem.NA
                            ended = "" if s["_times"][-1] >= hi else " (the stream ended during the read)"
                            log(f"  join: while the {j['prompt_tok']}-token prompt was read ({hi - lo:.1f}s) the "
                                f"stream got {len(inside)} chunks ({len(inside) / (hi - lo):.1f}/s), longest stop "
                                f"{stop:.2f}s{ended}")
                            emit("join", 1, 1, j, sampler, t0, join_read_s=round(hi - lo, 2),
                                 join_stream_chunks_s=round(len(inside) / (hi - lo), 2),
                                 join_stream_stop_s=round(stop, 2))
                except StackError as error:
                    log(f"  STOPPED: {error}")
                    out.row(model=model, status="REQUEST_FAIL", **record)
    log(f"wrote {args.out}")


def main():
    parser = arg_parser(__doc__, default_out="results/speed/concurrent-sessions.tsv")
    parser.add_argument("--sessions", default="2,3", help="concurrent session counts after the solo run")
    parser.add_argument("--depth", type=int, default=65536, help="each session's cold prompt, tokens")
    parser.add_argument("--join-depth", type=int, default=131072, help="the joining prompt (0 skips the join)")
    parser.add_argument("--join-delay", type=float, default=5.0)
    parser.add_argument("--gen", type=int, default=512, help="max tokens per session turn")
    parser.add_argument("--stream-gen", type=int, default=3000, help="max tokens of the join scenario's stream")
    parser.add_argument("--reasoning", default="medium")
    parser.add_argument("--corpus", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
