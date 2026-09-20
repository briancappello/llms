"""Explicit configuration; only Settings.from_env reads the process environment."""

from dataclasses import asdict, dataclass, field, fields
import json
import os
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import urlsplit

# llama-swap requires ENV_NAME=value, uppercase name (see its config schema).
ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
ENGINE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ENGINE_KEYS = {"kind", "server", "cwd", "env", "args", "host",
               "check_endpoint", "use_model_name", "unload_timeout"}
ENGINE_KINDS = ("llama.cpp", "custom")


def _validate_engines(engines):
    """Engines name a local server process: binary, environment, fixed args.

    Host-specific by nature, so they live in settings.json and never in the
    registry. A registry entry refers to one by logical name, which is what
    keeps the same registry usable on a CUDA box and on an AMD one.
    """
    if not isinstance(engines, dict):
        raise ValueError("engines must be a mapping of name to engine")
    for name, spec in engines.items():
        if not isinstance(name, str) or not ENGINE_NAME.match(name):
            raise ValueError(f"invalid engine name: {name!r}")
        if not isinstance(spec, dict) or spec.keys() - ENGINE_KEYS:
            raise ValueError(f"engine {name} must be an object with known keys: {sorted(ENGINE_KEYS)}")
        if spec.get("kind", "llama.cpp") not in ENGINE_KINDS:
            raise ValueError(f"engine {name} kind must be one of {ENGINE_KINDS}")
        for key in ("server", "cwd", "host", "check_endpoint", "use_model_name"):
            value = spec.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or not value or any(c in value for c in "\x00\r\n"):
                raise ValueError(f"engine {name} {key} must be a non-empty single-line string")
        if not spec.get("server"):
            raise ValueError(f"engine {name} requires a server command")
        if (cwd := spec.get("cwd")) and not Path(cwd).expanduser().is_absolute():
            raise ValueError(f"engine {name} cwd must be an absolute path")
        if (check := spec.get("check_endpoint")) and not (check.startswith("/") or check == "none"):
            raise ValueError(f"engine {name} check_endpoint must start with / or be 'none'")
        args = spec.get("args", [])
        if not isinstance(args, list) or not all(
                isinstance(a, str) and not any(c in a for c in "\x00\r\n") for a in args):
            raise ValueError(f"engine {name} args must be a list of single-line strings")
        env = spec.get("env", {})
        if not isinstance(env, dict):
            raise ValueError(f"engine {name} env must be a mapping")
        for key, value in env.items():
            if not isinstance(key, str) or not ENV_NAME.match(key):
                raise ValueError(f"engine {name} env name {key!r} must match {ENV_NAME.pattern}")
            if not isinstance(value, str) or any(c in value for c in "\x00\r\n"):
                raise ValueError(f"engine {name} env value for {key} must be a single-line string")
        timeout = spec.get("unload_timeout")
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 0):
            raise ValueError(f"engine {name} unload_timeout must be a non-negative integer")


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
    engines: dict = field(default_factory=dict)

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
        _validate_engines(self.engines)

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
                elif f.name == "engines":
                    try:
                        values[f.name] = json.loads(env[key])
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"LLM_ENGINES must be a JSON object: {exc}") from exc
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
