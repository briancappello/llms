"""Shared pieces for stack benchmark drivers: prompts, timings, TSV, CLI."""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stack import Stack, StackError, Variant  # noqa: E402
import mem  # noqa: E402

__all__ = ["Stack", "StackError", "Variant", "mem", "log", "tokenize", "LocalTokenizer", "depth_prompt", "load_corpus", "timings",
           "TSV", "median_cells", "read_tsv", "arg_parser", "variants", "stack_for", "served_memory", "chat_with_peak",
           "metal_budget", "PROVENANCE"]

DEFAULT_CORPUS = Path(os.environ.get("LLMS_BENCH_CORPUS", "/tmp/code-corpus.txt"))

QUESTION = (
    "\n\n// =============================================================\n"
    "// QUESTION:\n"
    "// You have just read a substantial slice of the llama.cpp codebase.\n"
    "// Identify what you believe is the single most error-prone area in\n"
    "// the code above, name a specific file and function, and explain in\n"
    "// 3-5 sentences why it's brittle. Be concrete and brief.\n"
)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class LocalTokenizer:
    """Tokenize with a model's own tokenizer.json, for servers without /tokenize.

    Needs the `tokenizers` package: `uv run --with tokenizers python bench/...`.
    """

    def __init__(self, path):
        from tokenizers import Tokenizer

        path = Path(path).expanduser()
        self.path = path / "tokenizer.json" if path.is_dir() else path
        self._tokenizer = Tokenizer.from_file(str(self.path))

    def __call__(self, text):
        return self._tokenizer.encode(text, add_special_tokens=False).ids


def tokenize(url, text):
    """Token ids for text: url is a llama.cpp-style /tokenize URL or a LocalTokenizer."""
    if callable(url):
        return url(text)
    body = json.dumps({"content": text, "add_special": False}).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read()).get("tokens", [])


def depth_prompt(tokenize_url, corpus, target, max_tokens):
    """Chat payload whose corpus portion is ~target tokens (bench/lib/trim_to_tokens.py method).

    cache_prompt is forced off: llama.cpp reports PP over tokens actually
    evaluated, so every rep is a full prefill at this depth.
    """
    lo, hi, best, best_diff = 0, len(corpus), "", 10**9
    for _ in range(20):
        mid = (lo + hi) // 2
        chunk = corpus[:mid]
        count = len(tokenize(tokenize_url, chunk))
        diff = abs(count - target)
        if diff < best_diff:
            best, best_diff = chunk, diff
        if count < target:
            lo = mid + 1
        elif count > target:
            hi = mid - 1
        else:
            break
    return {"messages": [{"role": "user", "content": best + QUESTION}], "max_tokens": max_tokens,
            "stream": False, "temperature": 0.0, "cache_prompt": False}


def load_corpus(path=DEFAULT_CORPUS):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"missing corpus {path}: build it with `make corpus`")
    return path.read_text(encoding="utf-8", errors="replace")


def timings(response):
    """Throughput and speculative acceptance from an OpenAI-shaped response.

    llama.cpp reports `timings`; servers that do not (e.g. oMLX) yield NA
    rather than invented numbers.
    """
    t = response.get("timings") or {}
    u = response.get("usage") or {}
    draft_n, accepted = t.get("draft_n"), t.get("draft_n_accepted")
    return {
        "prompt_tok": u.get("prompt_tokens", mem.NA),
        "compl_tok": u.get("completion_tokens", mem.NA),
        "pp_tok_s": round(t["prompt_per_second"], 2) if t.get("prompt_per_second") else mem.NA,
        "tg_tok_s": round(t["predicted_per_second"], 2) if t.get("predicted_per_second") else mem.NA,
        "draft_n": draft_n if draft_n is not None else mem.NA,
        "draft_acc": accepted if accepted is not None else mem.NA,
        "acc_pct": round(100.0 * accepted / draft_n, 1) if draft_n else mem.NA,
    }


class TSV:
    """Append-as-you-go TSV so an interrupted run keeps every finished row."""

    def __init__(self, path, columns):
        self.path, self.columns = Path(path), list(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as stream:
            csv.writer(stream, delimiter="\t").writerow(self.columns)

    def row(self, **values):
        unknown = values.keys() - set(self.columns)
        if unknown:
            raise KeyError(f"unknown TSV columns: {sorted(unknown)}")
        with self.path.open("a", newline="") as stream:
            csv.writer(stream, delimiter="\t").writerow([values.get(c, mem.NA) for c in self.columns])


def numeric(values):
    return [float(v) for v in values if v not in (None, "", mem.NA, "-")]


def median_cells(rows, keys, fields):
    """{key tuple: {field_med/min/max, n}} over numeric fields."""
    cells = {}
    for row in rows:
        cells.setdefault(tuple(row[k] for k in keys), []).append(row)
    summary = {}
    for key, group in cells.items():
        out = {"n": len(group)}
        for f in fields:
            values = numeric(r.get(f) for r in group)
            out[f + "_med"] = round(statistics.median(values), 2) if values else mem.NA
            out[f + "_min"] = round(min(values), 2) if values else mem.NA
            out[f + "_max"] = round(max(values), 2) if values else mem.NA
        summary[key] = out
    return summary


def read_tsv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


PROVENANCE = ["variant", "patch", "rendered_cmd", "production"]


def arg_parser(description, *, default_out):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, action="append",
                        help="registry model ID (repeatable)")
    parser.add_argument("--variant", action="append", type=Variant.parse, default=[],
                        help="NAME or NAME='{\"entries\":{...},\"engines\":{...}}' merge-patch "
                             "against production; repeatable (default: production only)")
    parser.add_argument("--port", type=int, default=18190, help="isolated llama-swap port")
    parser.add_argument("--config-dir", help="production llms config dir (default: the host's)")
    parser.add_argument("--no-manage-production", action="store_true",
                        help="do not stop/start the production service around the run")
    parser.add_argument("--out", default=default_out, help="results TSV")
    return parser


def variants(args):
    return args.variant or [Variant()]


def stack_for(args, variant):
    return Stack(variant, port=args.port, production_dir=args.config_dir,
                 manage_production=not args.no_manage_production, log=log)


def served_memory(stack, model=None):
    """(MiB | None, kind) for the process serving model, sampled now."""
    return mem.measure(stack.upstream_pids(model))


def chat_with_peak(stack, model, payload, *, timeout=1800):
    """(response, peak MiB | None, kind): memory sampled DURING the request.

    Idle readings are not trustworthy everywhere: the amdgpu Vulkan driver
    evicts an idle model to GTT and VRAM reads ~0 (results/quality/turbo-735.md),
    so every recorded number comes from active serving. The post-request
    reading is a floor for requests shorter than the sampling interval.
    """
    pids = stack.upstream_pids(model)
    with mem.PeakSampler(lambda: pids, interval=0.5) as sampler:
        response = stack.chat(payload, timeout=timeout)
    after, kind = mem.measure(pids)
    values = [v for v in (sampler.peak, after) if v is not None]
    return response, (max(values) if values else None), kind


def metal_budget(stack, model=None):
    """Metal budget: from the served log if present, else the model's llama.cpp
    engine's --list-devices (the same recommendedMaxWorkingSetSize), else NA."""
    server = None
    entry = stack.registry.get(model, {}) if model else {}
    engine = stack.settings.engines.get(entry.get("engine") or "", {})
    if engine and engine.get("kind", "llama.cpp") == "llama.cpp":
        server = engine["server"]
    elif not entry.get("engine"):
        server = stack.settings.server
    return mem.metal_budget_mib(stack.logs(), llama_server=server)
