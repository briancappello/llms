"""Promote a measured benchmark variant to the production configuration.

A benchmark conclusion is only finished when production serves the winner.
Promotion applies the variant patch recorded in a results file, exactly as the
benchmark applied it, after proving that production has not drifted since the
measurement. See openspec/specs/benchmark-promotion.
"""

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stack import Variant  # noqa: E402


class PromoteError(RuntimeError):
    pass


@dataclass
class Measured:
    """One variant of one model, as a results file recorded it."""

    results: Path
    model: str
    variant: Variant
    base_digest: str
    rendered_cmd: str
    rows: int


def _variant(patch, results):
    try:
        data = json.loads(patch)
        return Variant(name=data["name"], entries=data.get("entries", {}), engines=data.get("engines", {}))
    except (ValueError, KeyError, TypeError) as error:
        raise PromoteError(f"{results}: unreadable variant patch: {error}") from error


def _rows(path):
    """Uniform rows: TSV rows as written, or one row per kv-probe record."""
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        provenance = data.get("provenance") or {}
        if provenance.get("mode") == "standalone":
            raise PromoteError(f"{path} is a standalone measurement of {provenance.get('server')}; "
                               "re-measure the change as a stack --variant and promote that")
        base = {**provenance, "model": data.get("model", ""), "status": "OK"}
        variants = [provenance.get("variant")] if provenance.get("variant") else []
        return [dict(base) for _ in data.get("records", [])], variants
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return rows, list(dict.fromkeys(r.get("variant", "") for r in rows))


def load_measured(results, variant, model=None):
    """The single, consistent measurement of variant (and model) in results."""
    results = Path(results)
    rows, present = _rows(results)
    if variant not in present:
        raise PromoteError(f"{results}: no variant {variant!r}; it has: {', '.join(present) or 'none'}")
    rows = [r for r in rows if r.get("variant") == variant]
    models = sorted({r.get("model", "") for r in rows})
    if model is None and len(models) > 1:
        raise PromoteError(f"{results}: variant {variant} covers several models; pass --model ({', '.join(models)})")
    model = model if model is not None else (models[0] if models else "")
    ok = [r for r in rows if r.get("model") == model and r.get("status") == "OK"]
    if not ok:
        raise PromoteError(f"{results}: no OK rows for variant {variant}" + (f" of model {model}" if model else ""))
    if not all(r.get("base_digest") for r in ok):
        raise PromoteError(f"{results}: no base digest (recorded before promotion existed); re-run the benchmark")
    for field in ("patch", "base_digest", "rendered_cmd"):
        if len({r.get(field) for r in ok}) != 1:
            raise PromoteError(f"{results}: rows for variant {variant} of {model} disagree on {field}")
    first = ok[0]
    return Measured(results=results, model=model, variant=_variant(first["patch"], results),
                    base_digest=first["base_digest"], rendered_cmd=first["rendered_cmd"], rows=len(ok))
