"""Importable model management with no import-time I/O or service actions."""

from .settings import Settings
from .manager import (
    ManagerError, ModelManager, collapse_shards, default_model, detect_quant,
    pick_model_file, probe_gguf, render_config, resolve_source, shard_group, suggest_ctx,
)

__all__ = [
    "Settings", "ModelManager", "ManagerError", "render_config", "resolve_source",
    "shard_group", "collapse_shards", "pick_model_file", "detect_quant",
    "default_model", "probe_gguf", "suggest_ctx",
]
