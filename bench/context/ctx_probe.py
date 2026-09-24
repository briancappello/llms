#!/usr/bin/env python3
"""Context ceiling of a served model, through the production stack.

Successor to ctx-probe.sh and ctx-gap.sh. Method (unchanged from the Linux
probe, which mandates measuring rather than deriving KV cost from metadata):

  1. load the production entry at CTX_LO and CTX_HI; record served memory
  2. marginal cost = (M_hi - M_lo) / (CTX_HI - CTX_LO)          [KiB/token]
  3. base          = M_lo - CTX_LO * cost                       [weights + buffers]
  4. ceiling       = (BUDGET - base) / cost, clamped to trained_ctx
  5. VERIFY: load at the largest step <= ceiling and generate a token
  6. STRETCH: one step above, to bracket the true ceiling
  7. --test CTX,...: load each listed context directly (what ctx-gap.sh did
     for a bracket the stretch step left open)

Memory is sampled while the server generates, never idle (the amdgpu Vulkan
driver evicts an idle model and VRAM then reads ~0; results/quality/turbo-735.md).
It is device VRAM on Linux. On macOS it is weights (mmap'd, absent from
the footprint) + the served process footprint, against the Metal budget; see
results/context/mac-footprint-validation.md. The model is probed in its
production shape (parallel, kv_unified, cache types, MTP on if the entry has
it), because the output of this probe is a production ctx value.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from common import (PROVENANCE, TSV, StackError, Variant, arg_parser, chat_with_peak, log, mem,  # noqa: E402
                    metal_budget, stack_for)

STEPS = [262144, 229376, 196608, 180224, 163840, 131072, 98304, 65536, 32768]
COLUMNS = ["model", "phase", "ctx", "status", "mem_mib", "kib_per_tok", "base_mib", "derived_ceiling",
           "prod_ctx", "budget_mib", *PROVENANCE]
TRAINED = {}
GEN_PROBE = {"messages": [{"role": "user", "content": "say ok"}], "max_tokens": 300, "temperature": 0}


def with_weights(stack, model, memory):
    """On macOS the footprint excludes the mmap'd weights; add the file size."""
    if memory is None:
        return None
    if sys.platform == "darwin":
        path = stack.registry[model].get("path")
        memory += round(Path(path).stat().st_size / 2**20) if path else 0
    return memory


def probe(args, model, ctx):
    """(status, memory MiB | None, budget MiB | None, record); record carries trained_ctx."""
    with stack_for(args, Variant(f"ctx{ctx}", entries={model: {"ctx": ctx}})) as stack:
        record = stack.record(model)
        TRAINED[model] = int(stack.registry[model].get("trained_ctx") or STEPS[0])
        try:
            stack.load(model)
        except StackError as error:
            log(f"  ctx={ctx}: LOAD FAILED ({str(error).splitlines()[0][:160]})")
            return "FAIL", None, None, record
        budget = args.budget_mib or metal_budget(stack, model)
        try:
            # sampled while generating: idle VRAM reads are not trustworthy (see chat_with_peak)
            response, memory, _ = chat_with_peak(stack, model, {"model": model, **GEN_PROBE}, timeout=300)
            message = response["choices"][0]["message"]
            generated = any(isinstance(message.get(k), str) for k in ("content", "reasoning_content", "reasoning"))
        except (OSError, ValueError, KeyError, IndexError):
            response, memory, generated = None, None, False
        memory = with_weights(stack, model, memory)
        status = "OK" if generated else "NOGEN"
        log(f"  ctx={ctx}: {status}  memory={memory} MiB  budget={budget} MiB")
        return status, memory, budget, record


def main():
    parser = arg_parser(__doc__, default_out="results/context/ctx-probe.tsv")
    parser.add_argument("--ctx-lo", type=int, default=32768)
    parser.add_argument("--ctx-hi", type=int, default=131072)
    parser.add_argument("--budget-mib", type=int, help="memory budget (required on Linux; macOS reads Metal's)")
    parser.add_argument("--reserve-mib", type=int, default=0, help="subtract from the budget before deriving")
    parser.add_argument("--test", default="", help="comma-separated contexts to load directly")
    parser.add_argument("--no-derive", action="store_true", help="only run --test contexts")
    args = parser.parse_args()
    out = TSV(args.out, COLUMNS)
    for model in args.model:
        log(f"############ {model}")
        if not args.no_derive:
            status_lo, m_lo, budget, record = probe(args, model, args.ctx_lo)
            out.row(model=model, phase="anchor_lo", ctx=args.ctx_lo, status=status_lo, mem_mib=mem.cell(m_lo),
                    budget_mib=mem.cell(budget), **record)
            status_hi, m_hi, budget_hi, record = probe(args, model, args.ctx_hi)
            budget = budget or budget_hi
            if status_lo != "OK" or status_hi != "OK" or None in (m_lo, m_hi, budget):
                out.row(model=model, phase="anchor_hi", ctx=args.ctx_hi, status=status_hi,
                        mem_mib=mem.cell(m_hi), budget_mib=mem.cell(budget), **record)
                log("  cannot derive: an anchor failed or memory/budget is unavailable")
            else:
                cost = (m_hi - m_lo) * 1024.0 / (args.ctx_hi - args.ctx_lo)
                base = m_lo - args.ctx_lo * cost / 1024.0
                trained = TRAINED.get(model, STEPS[0])
                ceiling = min(int((budget - args.reserve_mib - base) * 1024.0 / cost) if cost > 0 else trained,
                              trained)
                # never recommend above the derived ceiling: below the smallest
                # step, round the ceiling itself down to a 4096 multiple
                prod = next((s for s in STEPS if s <= ceiling), max(4096, ceiling // 4096 * 4096))
                derived = dict(kib_per_tok=round(cost, 2), base_mib=round(base), derived_ceiling=ceiling,
                               prod_ctx=prod, budget_mib=budget)
                out.row(model=model, phase="anchor_hi", ctx=args.ctx_hi, status=status_hi, mem_mib=m_hi,
                        **derived, **record)
                log(f"  derived {cost:.2f} KiB/token, base {base:.0f} MiB, ceiling {ceiling} -> step {prod}")
                status, memory, _, record = probe(args, model, prod)
                out.row(model=model, phase="verify", ctx=prod, status=status, mem_mib=mem.cell(memory),
                        **derived, **record)
                higher = [s for s in STEPS if s > prod and s <= trained]
                if higher:
                    stretch = min(higher)
                    status, memory, _, record = probe(args, model, stretch)
                    out.row(model=model, phase="stretch", ctx=stretch, status=status, mem_mib=mem.cell(memory),
                            **derived, **record)
                    log(f"  stretch ctx={stretch}: {status}" + ("  <-- ceiling is higher than derived"
                                                                 if status == "OK" else "  (brackets the ceiling)"))
        for ctx in [int(c) for c in args.test.split(",") if c]:
            status, memory, budget, record = probe(args, model, ctx)
            out.row(model=model, phase="test", ctx=ctx, status=status, mem_mib=mem.cell(memory),
                    budget_mib=mem.cell(budget), **record)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
