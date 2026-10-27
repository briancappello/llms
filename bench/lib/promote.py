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
    production: object = None  # Settings with the engine patch applied, for rendering


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
                changes=new_registry != registry or new_settings != raw_settings, production=patched_settings)


def merge_engines(engines, patches):
    merged = copy.deepcopy(dict(engines))
    for name, patch in patches.items():
        merged[name] = merge_patch(merged.get(name, {}), patch)
    return merged


def _snapshot(paths):
    return {path: (path.read_bytes() if path.is_file() else None) for path in paths}


def _restore(snapshot):
    from llms.manager import atomic_write

    for path, data in snapshot.items():
        if data is None:
            path.unlink(missing_ok=True)
        else:
            atomic_write(path, data.decode("utf-8"))


def apply(planned, production, *, manager=None):
    """Write the plan, restart, warm the model; restore everything on any failure (design D4).

    Returns the warm-up outcome: "ok", "skipped" (service was not active) or
    "unchanged" (nothing to write, e.g. the production variant).
    """
    from llms.manager import atomic_write

    if not planned.changes:
        return "unchanged"
    manager = manager or ModelManager(production)
    active = manager._verify_service()
    snapshot = _snapshot([production.config_dir / "settings.json", production.registry, production.output])
    step = "write"
    try:
        atomic_write(production.registry, json.dumps(planned.registry, indent=2, sort_keys=True) + "\n")
        atomic_write(production.config_dir / "settings.json", json.dumps(planned.settings, indent=2) + "\n")
        step = "render"
        # Not manager.render(): the manager holds the Settings from before the engine patch.
        atomic_write(production.output, render_config(planned.registry, production.header.read_text(),
                                                      planned.production))
        if not active:
            return "skipped"
        step = "restart"
        manager.service("restart")
        manager.wait_ready()
        step = "warm-up"
        manager.use(planned.measured.model)
        return "ok"
    except BaseException as error:
        _restore(snapshot)
        message = f"{step} failed: {error}; production files restored"
        if active:
            try:
                manager.service("restart")
                manager.wait_ready()
            except Exception as again:  # report both; the files are already back
                message += f"; restart after restoring also failed: {again} (run `llms restart`)"
        raise PromoteError(message) from error


REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "results" / "promotions.tsv"
LOG_COLUMNS = ["utc", "host", "model", "variant", "results", "patch", "base_digest", "rendered_cmd", "warmup"]


def log_promotion(planned, warmup, path=LOG):
    """Append one row; the header only when the file is new (design D5)."""
    import socket
    import time

    measured = planned.measured
    results = Path(measured.results).resolve()
    try:
        results = results.relative_to(REPO)
    except ValueError:
        pass
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        if new:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), socket.gethostname(), measured.model,
                         measured.variant.name, str(results),
                         json.dumps(measured.variant.to_dict(), sort_keys=True), measured.base_digest,
                         measured.rendered_cmd, warmup])
    return path


def promote(results, variant, *, model=None, production, accept_shared_engine=False, manager=None, log_path=LOG):
    """Load, check, apply, then log. Nothing is logged unless every step succeeded."""
    measured = load_measured(results, variant, model=model)
    planned = plan(measured, production, accept_shared_engine=accept_shared_engine)
    warmup = apply(planned, production, manager=manager)
    log_promotion(planned, warmup, log_path)
    return warmup
