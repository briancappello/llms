#!/usr/bin/env python3
"""Capability gates for a candidate model, through the production stack.

Successor to gates.sh and mtp-engage.sh. Gates, not measurements: each can
disqualify a model before an hour of context probes and quality runs.
Cheapest first, all against the model as production would serve it:

  G1 load         llama-swap brings the entry up and answers (seconds, memory)
  G5 reasoning    REASONS (reasoning_content/reasoning) | RAW_THINK | PLAIN
  G4 toolcall     bench/quality/toolcall.py's 22 scenarios; the gate that matters
  G3 spec_engage  speculative decoding really drafts: draft_n > 0, with
                  acceptance and TG for a copy-heavy and a prose prompt, and
                  the same prompts with it disabled as the control
                  (mtp-engage.sh's method; MTP presence itself is a GGUF
                  fact, so no log grep decides anything)

G3 runs only for entries that enable speculation (mtp, or a --spec-type in
extra_args, e.g. draft-dflash / draft-dspark). Its control variant is the
production entry with speculation removed, which is the one declared change.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from common import (PROVENANCE, TSV, StackError, Variant, arg_parser, chat_with_peak, log, mem,  # noqa: E402
                    stack_for, timings)

HERE = Path(__file__).resolve().parent
TOOLCALL = HERE.parent / "quality" / "toolcall.py"
COPY_PROMPT = ('Repeat the following JSON back to me character for character, with no commentary:\n'
               '{"name":"textstat_core","version":"0.1.0","edition":"2021","deps":{"pyo3":{"version":"0.22",'
               '"features":["extension-module"]}},"lib":{"name":"textstat_core","crate-type":["cdylib"]}}')
PROSE_PROMPT = ("Explain, in one paragraph of original prose, why speculative decoding helps more when a model "
                "is bandwidth-bound than when it is dequantisation-bound.")
COLUMNS = ["model", "gate", "variant", "workload", "result", "tg_tok_s", "draft_n", "draft_acc", "acc_pct",
           "mem_mib", "load_s", *PROVENANCE[1:]]


def speculative(entry):
    return bool(entry.get("mtp")) or any("--spec-type" in a or "-md " in a for a in entry.get("extra_args", []))


def without_speculation(entry):
    args = [a for a in entry.get("extra_args", []) if "--spec-type" not in a and not a.startswith("-md ")]
    return {"mtp": False, "extra_args": args}


def reasoning(response):
    message = (response.get("choices") or [{}])[0].get("message") or {}
    if (message.get("reasoning_content") or message.get("reasoning") or "").strip():
        return "REASONS"
    if "<think>" in (message.get("content") or ""):
        return "RAW_THINK"
    return "PLAIN"


def generate(stack, model, prompt):
    return timings(stack.chat({"model": model, "messages": [{"role": "user", "content": prompt}],
                               "max_tokens": 400, "temperature": 0}, timeout=600))


def main():
    parser = arg_parser(__doc__, default_out="results/gates/gates.tsv")
    parser.add_argument("--skip-toolcall", action="store_true")
    args = parser.parse_args()
    out = TSV(args.out, COLUMNS)
    for model in args.model:
        log(f"############ {model}")
        with stack_for(args, Variant()) as stack:
            record = {k: v for k, v in stack.record(model).items() if k != "variant"}
            entry = stack.registry[model]
            try:
                load_s = round(stack.load(model), 1)
            except StackError as error:
                log(f"  G1 load FAIL: {str(error).splitlines()[0][:160]}")
                out.row(model=model, gate="G1_load", variant="production", result="FAIL", **record)
                continue
            _, memory, kind = chat_with_peak(stack, model, {"model": model, "max_tokens": 64, "temperature": 0,
                                                            "messages": [{"role": "user", "content": "Count to 20."}]})
            out.row(model=model, gate="G1_load", variant="production", result="PASS", mem_mib=mem.cell(memory),
                    load_s=load_s, **record)
            log(f"  G1 load        PASS ({load_s}s, {kind} {memory} MiB)")
            try:
                g5 = reasoning(stack.chat({"model": model, "max_tokens": 256, "temperature": 0,
                                           "messages": [{"role": "user", "content": "Reply with exactly: OK"}]}))
            except (OSError, ValueError):
                g5 = "ERROR"
            out.row(model=model, gate="G5_reasoning", variant="production", result=g5, **record)
            log(f"  G5 reasoning   {g5}")
            if not args.skip_toolcall:
                with tempfile.NamedTemporaryFile(suffix=".json") as results:
                    run = subprocess.run([sys.executable, str(TOOLCALL), "--url", stack.base + "/v1",
                                          "--model", model, "--out", results.name], capture_output=True, text=True)
                    try:
                        summary = json.loads(Path(results.name).read_text())["summary"]
                        g4 = f"{summary['passed']}/{summary['total']}"
                    except (OSError, ValueError, KeyError):
                        g4 = "ERROR"
                        log(run.stderr[-400:])
                out.row(model=model, gate="G4_toolcall", variant="production", result=g4, **record)
                log(f"  G4 toolcall    {g4}")
            if speculative(entry):
                for workload, prompt in (("copy", COPY_PROMPT), ("prose", PROSE_PROMPT)):
                    stats = generate(stack, model, prompt)
                    result = "NO_DRAFT" if stats["draft_n"] in (0, mem.NA) else f"{stats['acc_pct']}%acc"
                    out.row(model=model, gate="G3_spec_engage", variant="production", workload=workload,
                            result=result, tg_tok_s=stats["tg_tok_s"], draft_n=stats["draft_n"],
                            draft_acc=stats["draft_acc"], acc_pct=stats["acc_pct"], **record)
                    log(f"  G3 {workload:5s}      {result}  TG {stats['tg_tok_s']}")
        if speculative(entry):
            control = Variant("no-spec", entries={model: without_speculation(entry)})
            with stack_for(args, control) as stack:
                record = {k: v for k, v in stack.record(model).items() if k != "variant"}
                stack.load(model)
                for workload, prompt in (("copy", COPY_PROMPT), ("prose", PROSE_PROMPT)):
                    stats = generate(stack, model, prompt)
                    out.row(model=model, gate="G3_spec_engage", variant="no-spec", workload=workload,
                            result="control", tg_tok_s=stats["tg_tok_s"], **record)
                    log(f"  G3 {workload:5s} ctrl TG {stats['tg_tok_s']}")
        else:
            out.row(model=model, gate="G3_spec_engage", variant="production", result="NA",
                    **{k: v for k, v in record.items()})
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
