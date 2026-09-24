#!/usr/bin/env python3
"""Which serving of a model is faster here: client-timed, same prompts, same stack.

Different engines report `timings` differently (or not at all), so this
measures every model the same way from the client, over llama-swap:

  prefill tok/s = prompt tokens / time to first streamed token
  decode  tok/s = (completion tokens - 1) / (time from first to last token)

Prompts are the code corpus trimmed to each depth ONCE, then sent verbatim
to every model: with --tokenizer (a snapshot dir or tokenizer.json; needs
`uv run --with tokenizers`), or else with the first model's /tokenize. The
first way works when no model runs on llama.cpp (oMLX has no /tokenize).
cache_prompt is off, and a unique random nonce leads every request, so no
server can reuse a cached prefix. oMLX ignores cache_prompt and caches
prefixes itself, so check its log for "reused" after a run.

    uv run --with tokenizers python bench/speed/engine_compare.py \\
        --model ornith-9b --model cyber-tiel --tokenizer <snapshot dir>
"""

import json
from pathlib import Path
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from common import (PROVENANCE, TSV, LocalTokenizer, StackError, arg_parser, depth_prompt,  # noqa: E402
                    load_corpus, log, median_cells, mem, read_tsv, served_memory, stack_for, tokenize, variants)

COLUMNS = ["model", "variant", "depth", "rep", "prompt_tok", "compl_tok", "ttft_s", "prefill_tok_s", "decode_tok_s",
           "server_pp_tok_s", "server_tg_tok_s", "acc_pct", "mem_mib", "mem_kind", "status", *PROVENANCE]


def stream(url, payload, timeout=1800):
    """(ttft, total, completion_tokens, usage, timings) from an SSE chat stream."""
    body = dict(payload, stream=True, stream_options={"include_usage": True})
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.monotonic()
    first = last = None
    pieces, usage, timings = 0, {}, {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            timings = chunk.get("timings") or timings
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if any(delta.get(k) for k in ("content", "reasoning_content", "reasoning", "tool_calls")):
                    now = time.monotonic()
                    first = first or now
                    last = now
                    pieces += 1
    if first is None:
        raise ValueError("stream produced no tokens")
    return first - start, last - first, usage.get("completion_tokens") or pieces, usage, timings


def main():
    parser = arg_parser(__doc__, default_out="results/speed/engine-compare.tsv")
    parser.add_argument("--depths", default="4096,16384")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--gen", type=int, default=256)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--tokenizer", help="snapshot dir or tokenizer.json used to build prompts locally")
    args = parser.parse_args()
    local = LocalTokenizer(args.tokenizer) if args.tokenizer else None
    corpus = load_corpus(args.corpus) if args.corpus else load_corpus()
    depths = [int(d) for d in args.depths.split(",")]
    out = TSV(args.out, COLUMNS)
    prompts, done = {}, set()
    for variant in variants(args):
        for model in args.model:
            if model in done and model not in variant.entries:
                continue  # this variant does not change the model; measured already
            done.add(model)
            log(f"############ {model} / {variant.name}")
            with stack_for(args, variant) as stack:
                record = stack.record(model)
                try:
                    stack.load(model)
                except StackError as error:
                    log(f"  LOAD FAILED: {str(error).splitlines()[0][:160]}")
                    out.row(model=model, status="LOAD_FAIL", **record)
                    continue
                for depth in depths:
                    if depth not in prompts:
                        url = local or stack.upstream(model, "/tokenize")
                        built = depth_prompt(url, corpus, depth, args.gen)
                        prompts[depth] = (built, len(tokenize(url, built["messages"][0]["content"])))
                    for rep in range(1, args.reps + 1):
                        payload = json.loads(json.dumps(prompts[depth][0]))
                        payload["model"] = model
                        # Unique per request: oMLX keeps its own prefix cache and ignores
                        # cache_prompt, so a nonce shared across depths let it reuse the
                        # shallower prompt (seen in its log as "reused 4096").
                        nonce = f"[run {uuid.uuid4().hex} {variant.name}/{model}/d{depth}/r{rep}]"
                        payload["messages"][0]["content"] = nonce + "\n" + payload["messages"][0]["content"]
                        try:
                            ttft, decode_s, tokens, usage, timings = stream(stack.base + "/v1/chat/completions", payload)
                        except (OSError, ValueError) as error:
                            log(f"  depth {depth} rep {rep}: FAILED {error}")
                            out.row(model=model, depth=depth, rep=rep, status="REQUEST_FAIL", **record)
                            continue
                        # servers that do not report usage while streaming: the build-time count
                        prompt_tok = usage.get("prompt_tokens") or prompts[depth][1]
                        memory, kind = served_memory(stack, model)
                        draft_n, accepted = timings.get("draft_n"), timings.get("draft_n_accepted")
                        row = dict(
                            prompt_tok=mem.cell(prompt_tok), compl_tok=tokens, ttft_s=round(ttft, 2),
                            prefill_tok_s=round(prompt_tok / ttft, 1) if prompt_tok else mem.NA,
                            decode_tok_s=round((tokens - 1) / decode_s, 1) if decode_s > 0 and tokens > 1 else mem.NA,
                            server_pp_tok_s=round(timings["prompt_per_second"], 1) if timings.get("prompt_per_second") else mem.NA,
                            server_tg_tok_s=round(timings["predicted_per_second"], 1) if timings.get("predicted_per_second") else mem.NA,
                            acc_pct=round(100 * accepted / draft_n, 1) if draft_n else mem.NA,
                            mem_mib=mem.cell(memory), mem_kind=kind)
                        out.row(model=model, depth=depth, rep=rep, status="OK", **row, **record)
                        log(f"  depth {depth} rep {rep}: {prompt_tok} tok  TTFT {row['ttft_s']}s  "
                            f"prefill {row['prefill_tok_s']} tok/s  decode {row['decode_tok_s']} tok/s  "
                            f"(server PP {row['server_pp_tok_s']} TG {row['server_tg_tok_s']}, acc {row['acc_pct']}%)")
    rows = [r for r in read_tsv(args.out) if r["status"] == "OK"]
    summary = median_cells(rows, ("model", "variant", "depth"), ("prefill_tok_s", "decode_tok_s", "ttft_s"))
    log("=== median ===")
    for (model, variant, depth), cell in sorted(summary.items(), key=lambda kv: (int(kv[0][2]), kv[0][0], kv[0][1])):
        log(f"  depth {depth:>6}  {model:16s} {variant:10s} prefill {cell['prefill_tok_s_med']:>7} tok/s  "
            f"decode {cell['decode_tok_s_med']:>6} tok/s  TTFT {cell['ttft_s_med']}s  (n={cell['n']})")


if __name__ == "__main__":
    main()
