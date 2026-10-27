"""Promote a measured benchmark variant to the production configuration.

A benchmark conclusion is only finished when production serves the winner.
Promotion applies the variant patch recorded in a results file, exactly as the
benchmark applied it, after proving that production has not drifted since the
measurement. See openspec/specs/benchmark-promotion.
"""

import copy
import csv
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stack import ModelManager, Variant, base_digest, merge_patch  # noqa: E402
from llms import ManagerError, render_config  # noqa: E402


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


@dataclass
class Plan:
    """Production as it will be after promotion; nothing is written yet."""

    measured: Measured
    registry: dict
    settings: dict
    affected: list = field(default_factory=list)
    changes: bool = True


def _rendered_models(registry, header, settings):
    import yaml

    try:
        return yaml.safe_load(render_config(registry, header, settings))["models"]
    except ManagerError as error:
        raise PromoteError(f"render failed: {error}") from error


def plan(measured, production, *, accept_shared_engine=False):
    """Check that production still matches the measurement, and plan the patch (design D3)."""
    model, variant = measured.model, measured.variant
    registry = ModelManager(production).load_registry()
    if model not in registry:
        raise PromoteError(f"{model} is not in the production registry")
    others = sorted(set(variant.entries) - {model})
    if others:
        raise PromoteError(f"variant {variant.name} patches registry entries other than {model}: {', '.join(others)}")
    current = base_digest(production, registry, model, variant)
    if current != measured.base_digest:
        raise PromoteError(f"base digest differs: production changed since {measured.results} was recorded "
                           f"(recorded {measured.base_digest[:12]}, now {current[:12]}); re-run the benchmark")

    settings_path = production.config_dir / "settings.json"
    raw_settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    new_settings = copy.deepcopy(raw_settings)
    engines = new_settings.setdefault("engines", {})
    for name, patch in variant.engines.items():
        engines[name] = merge_patch(engines.get(name, {}), patch)
    new_registry = copy.deepcopy(registry)
    for name, patch in variant.entries.items():
        new_registry[name] = merge_patch(new_registry[name], patch)

    header = production.header.read_text()
    patched_settings = replace(production, engines=merge_engines(production.engines, variant.engines))
    before = _rendered_models(registry, header, production)
    after = _rendered_models(new_registry, header, patched_settings)
    if after.get(model, {}).get("cmd") != measured.rendered_cmd:
        raise PromoteError(f"production would render a different command for {model} than the one measured:\n"
                           f"  measured: {measured.rendered_cmd}\n  promoted: {after.get(model, {}).get('cmd')}")
    affected = sorted(name for name in set(before) | set(after) if name != model and before.get(name) != after.get(name))
    if affected and not accept_shared_engine:
        raise PromoteError(f"variant {variant.name} also changes {', '.join(affected)} (shared engine); "
                           "re-run with --accept-shared-engine to promote it for them too")
    return Plan(measured=measured, registry=new_registry, settings=new_settings, affected=affected,
                changes=new_registry != registry or new_settings != raw_settings)


def merge_engines(engines, patches):
    merged = copy.deepcopy(dict(engines))
    for name, patch in patches.items():
        merged[name] = merge_patch(merged.get(name, {}), patch)
    return merged
