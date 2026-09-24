#!/usr/bin/env python3
"""Is the headroom at a context actually usable once the window is full?

Successor to soak-ctx.sh. For each candidate context (largest first), serve
the production entry at that ctx through the stack, then fire a ladder of
prompts at increasing fractions of the window, ending with repeats at the
deepest point to probe fragmentation. A 1 Hz sampler records PEAK served
memory across the whole ladder, because the interesting number is the
transient during prefill, not the steady state after load. The first context
whose whole ladder succeeds is reported STABLE and the ladder stops.

    bench/context/soak_ctx.py --model cyber-tiel --ctx 262144,229376,196608
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from common import (PROVENANCE, TSV, StackError, Variant, arg_parser, depth_prompt, load_corpus, log,  # noqa: E402
                    mem, metal_budget, stack_for, timings)

FRACTIONS = (0.25, 0.50, 0.75, 0.90, 0.94, 0.94, 0.94)
COLUMNS = ["model", "ctx", "phase", "depth_tok", "status", "peak_mem_mib", "headroom_mib", "pp_tok_s", "tg_tok_s",
           "acc_pct", "budget_mib", *PROVENANCE]


def main():
    parser = arg_parser(__doc__, default_out="results/context/soak-ctx.tsv")
    parser.add_argument("--ctx", required=True, help="comma-separated candidate contexts, largest first")
    parser.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS))
    parser.add_argument("--gen", type=int, default=256)
    parser.add_argument("--budget-mib", type=int, help="required on Linux; macOS reads the Metal budget")
    parser.add_argument("--corpus", default=None)
    args = parser.parse_args()
    corpus = load_corpus(args.corpus) if args.corpus else load_corpus()
    fractions = [float(f) for f in args.fractions.split(",")]
    out = TSV(args.out, COLUMNS)
    for model in args.model:
        for ctx in [int(c) for c in args.ctx.split(",")]:
            log(f"########## {model} ctx={ctx}")
            with stack_for(args, Variant(f"ctx{ctx}", entries={model: {"ctx": ctx}})) as stack:
                record = stack.record(model)
                try:
                    stack.load(model)
                except StackError as error:
                    log(f"  LOAD FAILED: {str(error).splitlines()[0][:160]}")
                    out.row(model=model, ctx=ctx, phase="load", status="LOAD_FAIL", **record)
                    continue
                budget = args.budget_mib or metal_budget(stack, model)
                weights = 0
                if sys.platform == "darwin" and stack.registry[model].get("path"):
                    weights = round(Path(stack.registry[model]["path"]).stat().st_size / 2**20)
                tokenize = stack.upstream(model, "/tokenize")
                failed = False
                with mem.PeakSampler(lambda: stack.upstream_pids(model)) as sampler:
                    for index, fraction in enumerate(fractions):
                        depth = int(ctx * fraction)
                        try:
                            payload = depth_prompt(tokenize, corpus, depth, args.gen)
                            payload["model"] = model
                            stats = timings(stack.chat(payload, timeout=3600))
                        except (OSError, ValueError) as error:
                            log(f"  depth {depth} ({fraction}): FAILED {error}")
                            failed = True
                            break
                        peak = None if sampler.peak is None else sampler.peak + weights
                        headroom = None if None in (peak, budget) else budget - peak
                        out.row(model=model, ctx=ctx, phase=f"req{index}", depth_tok=stats["prompt_tok"],
                                status="OK", peak_mem_mib=mem.cell(peak), headroom_mib=mem.cell(headroom),
                                pp_tok_s=stats["pp_tok_s"], tg_tok_s=stats["tg_tok_s"], acc_pct=stats["acc_pct"],
                                budget_mib=mem.cell(budget), **record)
                        log(f"  depth {depth} ({fraction}): ok ptok={stats['prompt_tok']} PP={stats['pp_tok_s']} "
                            f"TG={stats['tg_tok_s']} peak={peak} MiB")
                peak = None if sampler.peak is None else sampler.peak + weights
                verdict = "UNSTABLE" if failed else "STABLE"
                out.row(model=model, ctx=ctx, phase="verdict", status=verdict, peak_mem_mib=mem.cell(peak),
                        headroom_mib=mem.cell(None if None in (peak, budget) else budget - peak),
                        budget_mib=mem.cell(budget), **record)
                log(f"  ==> ctx={ctx} {verdict} (peak {peak} MiB, budget {budget} MiB)")
            if not failed:
                log("  first stable context found; stopping ladder")
                break
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
