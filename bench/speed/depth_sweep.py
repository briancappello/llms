#!/usr/bin/env python3
"""PP/TG and speculative acceptance at real depths, through the production stack.

Successor to bench/legacy/speed/depth-sweep.sh. Each variant is the
production registry entry and engine plus one declared merge-patch, served by
llama-swap exactly as `llms render` renders it; nothing here writes a server
command line. Prompts are the code corpus trimmed to each depth through the
backend's own /tokenize (via llama-swap's /upstream), with cache_prompt off
so every rep is a full prefill at that depth: llama.cpp reports PP over the
tokens it actually evaluated, so the rate stays honest.

Columns keep the legacy names (model, spec->variant, depth, rep, prompt_tok,
compl_tok, pp_tok_s, tg_tok_s, draft_n, draft_acc, acc_pct) and add memory and
provenance. Memory is the peak sampled during each request, never an idle
reading. Unmeasurable values are NA, never 0.

    bench/speed/depth_sweep.py --model cyber-tiel \\
        --variant no-mtp='{"entries": {"cyber-tiel": {"mtp": false}}}' \\
        --variant mtp-n3='{"engines": {"metal": {"mtp_args": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]}}}' \\
        --variant mtp-n1='{"engines": {"metal": {"mtp_args": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"]}}}' \\
        --depths 4096,16384,65536 --reps 3 --out results/speed/mac-depth-sweep.tsv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from common import (PROVENANCE, TSV, StackError, arg_parser, depth_prompt, load_corpus, log,  # noqa: E402
                    median_cells, mem, metal_budget, read_tsv, chat_with_peak, stack_for, timings, variants)

COLUMNS = ["model", "variant", "depth", "rep", "prompt_tok", "compl_tok", "pp_tok_s", "tg_tok_s",
           "draft_n", "draft_acc", "acc_pct", "mem_mib", "mem_kind", "budget_mib", "load_s", "status", *PROVENANCE]


def write_summary(path, out):
    rows = [r for r in read_tsv(path) if r["status"] == "OK"]
    cells = median_cells(rows, ("model", "variant", "depth"), ("pp_tok_s", "tg_tok_s", "acc_pct"))
    summary = TSV(out, ["model", "variant", "depth", "n", "pp_med", "pp_min", "pp_max",
                        "tg_med", "tg_min", "tg_max", "acc_med", "mem_mib"])
    memory = {(r["model"], r["variant"], r["depth"]): r["mem_mib"] for r in rows}
    for key in sorted(cells, key=lambda k: (k[0], k[1], int(k[2]))):
        c = cells[key]
        summary.row(model=key[0], variant=key[1], depth=key[2], n=c["n"],
                    pp_med=c["pp_tok_s_med"], pp_min=c["pp_tok_s_min"], pp_max=c["pp_tok_s_max"],
                    tg_med=c["tg_tok_s_med"], tg_min=c["tg_tok_s_min"], tg_max=c["tg_tok_s_max"],
                    acc_med=c["acc_pct_med"], mem_mib=memory.get(key, mem.NA))
    return out


def run(args, *, columns=COLUMNS, extra=lambda stack, model, memory, budget: {}, default_depths=None):
    """The shared depth loop; wrappers add columns via `extra`."""
    corpus = load_corpus(args.corpus) if args.corpus else load_corpus()
    depths = [int(d) for d in args.depths.split(",")]
    out = TSV(args.out, columns)
    for model in args.model:
        for variant in variants(args):
            log(f"############ {model} / {variant.name}")
            with stack_for(args, variant) as stack:
                record = stack.record(model)
                try:
                    load_s = round(stack.load(model), 1)
                except StackError as error:
                    log(f"  LOAD FAILED: {error}")
                    out.row(model=model, status="LOAD_FAIL", **extra(stack, model, None, None), **record)
                    continue
                budget = metal_budget(stack, model)
                log(f"  loaded in {load_s}s")
                tokenize = stack.upstream(model, "/tokenize")
                for depth in depths:
                    try:
                        payload = depth_prompt(tokenize, corpus, depth, args.gen)
                    except (OSError, ValueError) as error:
                        log(f"  depth {depth}: prompt build failed: {error}")
                        continue
                    payload["model"] = model
                    for rep in range(1, args.reps + 1):
                        try:
                            response, memory, kind = chat_with_peak(stack, model, payload)
                        except (OSError, ValueError) as error:
                            log(f"  depth {depth} rep {rep}: request failed: {error}")
                            out.row(model=model, depth=depth, rep=rep, status="REQUEST_FAIL",
                                    **extra(stack, model, None, budget), **record)
                            continue
                        stats = timings(response)
                        out.row(model=model, depth=depth, rep=rep, **stats,
                                mem_mib=mem.cell(memory), mem_kind=kind, budget_mib=mem.cell(budget),
                                load_s=load_s, status="OK", **extra(stack, model, memory, budget), **record)
                        log(f"  depth {depth} rep {rep}: {stats['prompt_tok']} tok -> PP {stats['pp_tok_s']} | "
                            f"TG {stats['tg_tok_s']} | acc {stats['acc_pct']}% | {kind} {memory} MiB")
    summary = write_summary(args.out, str(Path(args.out).with_suffix("")) + "-summary.tsv")
    log(f"wrote {args.out} and {summary}")


def parser(doc=__doc__, default_out="results/speed/depth-sweep.tsv", depths="4096,16384,65536"):
    parser = arg_parser(doc, default_out=default_out)
    parser.add_argument("--depths", default=depths)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--gen", type=int, default=192)
    parser.add_argument("--corpus", default=None)
    return parser


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
