"""Explicit configuration; only Settings.from_env reads the process environment."""

from dataclasses import asdict, dataclass, field, fields
import json
import os
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    config_dir: Path
    registry: Path | None = None
    header: Path | None = None
    output: Path | None = None
    hf_cache: Path | None = None
    client_path: Path | None = None
    service_dir: Path | None = None
    server: str = "llama-server"
    swap_binary: str = "llama-swap"
    systemctl: str = "systemctl"
    swap_url: str = "http://127.0.0.1:18080"
    client_url: str = "http://127.0.0.1:18080/v1"
    listen: str = "127.0.0.1:18080"
    unit: str = "llama-swap.service"
    preload: bool = False
    gpu_memory_mib: int | None = None
    hf_endpoint: str = "https://huggingface.co"
    hf_token: str | None = field(default=None, repr=False)

    def __post_init__(self):
        root = Path(self.config_dir).expanduser().absolute()
        object.__setattr__(self, "config_dir", root)
        for key, fallback in {
            "registry": root / "registry.json",
            "header": root / "config.header.yaml",
            "output": root / "llama-swap.yaml",
            "hf_cache": root / "cache" / "huggingface" / "hub",
            "client_path": root / "clients" / "pi-models.json",
            "service_dir": root / "systemd",
        }.items():
            path = Path(getattr(self, key) or fallback).expanduser()
            object.__setattr__(self, key, path if path.is_absolute() else root / path)
        for key in ("server", "swap_binary", "systemctl", "listen", "unit"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value or any(c in value for c in "\x00\r\n"):
                raise ValueError(f"invalid {key}")
            if key in ("server", "swap_binary", "systemctl") and ("/" in value or value.startswith("~")):
                path = Path(value).expanduser()
                object.__setattr__(self, key, str(path if path.is_absolute() else root / path))
        if "/" in self.unit or not self.unit.endswith(".service") or self.unit.startswith("-"):
            raise ValueError("unit must be a .service basename")
        managed_paths = [root / "settings.json", self.registry, self.header, self.output,
                         self.client_path, self.service_dir / self.unit]
        if len({p.resolve() for p in managed_paths}) != len(managed_paths):
            raise ValueError("managed settings, registry, header, output, client, and service unit paths must be distinct")
        if not isinstance(self.preload, bool):
            raise ValueError("preload must be a boolean")
        for key in ("swap_url", "client_url", "hf_endpoint"):
            url = urlsplit(getattr(self, key))
            if url.scheme not in ("http", "https") or not url.netloc:
                raise ValueError(f"{key} must be an HTTP(S) URL")
        if self.gpu_memory_mib is not None and (
            isinstance(self.gpu_memory_mib, bool) or not isinstance(self.gpu_memory_mib, int)
            or self.gpu_memory_mib <= 0
        ):
            raise ValueError("gpu_memory_mib must be a positive integer or null")

    @classmethod
    def from_env(cls, config_dir=None, *, environ: Mapping[str, str] | None = None):
        """Defaults < settings.json < LLM_* overrides; config_dir wins over env.

        Direct Settings(config_dir=...) is isolated and does not read environment
        variables or settings.json. Relative file paths are config-dir relative.
        """
        env = os.environ if environ is None else environ
        home = Path(env.get("HOME", str(Path.home())))
        xdg = Path(env.get("XDG_CONFIG_HOME", str(home / ".config")))
        cache = Path(env.get("XDG_CACHE_HOME", str(home / ".cache")))
        explicit = config_dir or env.get("LLM_CONFIG_DIR")
        root = Path(explicit or xdg / "llama-swap-manager").expanduser().absolute()
        values = {
            "hf_cache": env.get("HF_HUB_CACHE", str(Path(env.get("HF_HOME", str(cache / "huggingface"))) / "hub")),
            "hf_endpoint": env.get("HF_ENDPOINT", "https://huggingface.co"),
            "hf_token": env.get("HF_TOKEN"),
            "output": root / "llama-swap.yaml" if explicit else xdg / "llama-swap" / "config.yaml",
            "client_path": home / ".pi" / "agent" / "models.json",
            "service_dir": xdg / "systemd" / "user",
        }
        settings_file = root / "settings.json"
        if settings_file.exists():
            data = json.loads(settings_file.read_text())
            allowed = {f.name for f in fields(cls)} - {"config_dir", "hf_token"}
            if not isinstance(data, dict) or data.keys() - allowed:
                raise ValueError("settings.json must be an object with known Settings fields (no hf_token)")
            values.update(data)
        for f in fields(cls):
            if f.name in ("config_dir", "hf_token"):
                continue
            key = "LLM_" + f.name.upper()
            if key in env:
                if f.name == "preload":
                    if env[key].lower() not in ("true", "false", "1", "0"):
                        raise ValueError("LLM_PRELOAD must be true, false, 1, or 0")
                    values[f.name] = env[key].lower() in ("true", "1")
                else:
                    values[f.name] = int(env[key]) if f.name == "gpu_memory_mib" else env[key]
        if "LLAMASWAP_URL" in env and "LLM_SWAP_URL" not in env:
            values["swap_url"] = env["LLAMASWAP_URL"]
        if "client_url" not in values:
            values["client_url"] = values.get("swap_url", cls.swap_url).rstrip("/") + "/v1"
        return cls(config_dir=root, **values)

    def to_dict(self):
        """Serializable settings without the authentication token or root."""
        return {key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self).items() if key not in ("config_dir", "hf_token")}
