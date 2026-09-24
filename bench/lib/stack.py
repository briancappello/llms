"""Serve models through the production stack, in an isolated llms instance.

Every benchmark goes through here so that what is measured is what is served:
llama-swap in front of the engine the registry names, launched with the
command llms renders from this host's settings, engines, header, and registry.
A variant is the production configuration plus one declared merge-patch; the
patch and the resulting rendered command are recorded with every result.

    with Stack(variant=Variant("mtp-n1", engines={"metal": {"mtp_args": [...]}})) as stack:
        stack.load("cyber-tiel")
        response = stack.chat({"model": "cyber-tiel", "messages": [...]})

Isolation (design D9):
  - settings/registry/header are copied into a temporary config dir; the
    production files are checksummed on entry and must be byte-identical on
    exit, however the run ends;
  - the isolated llama-swap listens on its own port, in its own process
    group, and is torn down (with every upstream it spawned) in `finally`;
  - the production service is stopped through llms (so the GPU/unified
    memory is free) and started again through llms, including on Ctrl-C,
    SIGTERM, and SIGHUP (unless SIGHUP is ignored, as under nohup).

Stopping a backgrounded run: send SIGTERM. Shells start background jobs with
SIGINT ignored, so `kill -INT` does nothing to them; SIGTERM always tears down
cleanly. llama-swap puts each upstream in its own process group, so upstream
PIDs are tracked and killed explicitly rather than via the group.
"""

import copy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llms import ManagerError, ModelManager, Settings  # noqa: E402

DEFAULT_PORT = 18190


def merge_patch(target, patch):
    """RFC 7386 JSON merge patch: dicts merge recursively, None deletes."""
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    result = copy.deepcopy(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = merge_patch(result.get(key), value)
    return result


@dataclass
class Variant:
    """One declared change to production: registry entries and/or engines.

    entries maps registry ID -> merge-patch; engines maps engine name ->
    merge-patch (an engine absent from production is created). An empty
    variant is the production configuration itself.
    """

    name: str = "production"
    entries: dict = field(default_factory=dict)
    engines: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, text):
        """`NAME` or `NAME=JSON` where JSON has optional entries/engines keys."""
        name, separator, body = text.partition("=")
        if not separator:
            return cls(name=name)
        data = json.loads(body)
        if not isinstance(data, dict) or data.keys() - {"entries", "engines"}:
            raise ValueError(f"variant {name}: JSON must be an object with entries and/or engines")
        return cls(name=name, entries=data.get("entries", {}), engines=data.get("engines", {}))

    def to_dict(self):
        return {"name": self.name, "entries": self.entries, "engines": self.engines}


def _digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _isolated_environ(environ):
    # LLMS_* overrides are already folded into the production Settings copied
    # below; leaking them again would re-point the isolated instance.
    return {k: v for k, v in environ.items() if not k.startswith("LLMS_")}


class StackError(RuntimeError):
    pass


class Stack:
    def __init__(self, variant=None, *, port=DEFAULT_PORT, production_dir=None, environ=None,
                 manage_production=True, ready_timeout=60, keep_dir=False, log=print,
                 production_manager=ModelManager):
        self.variant = variant or Variant()
        self.port = port
        self.environ = dict(os.environ if environ is None else environ)
        self.production = Settings.from_env(production_dir, environ=self.environ)
        self.manage_production = manage_production
        self.ready_timeout = ready_timeout
        self.keep_dir = keep_dir
        self.log = log
        self.production_manager = production_manager
        self.production_state = None
        self._restore = False
        self._process = None
        self._tmp = None
        self._upstreams = set()
        self._checksums = {}
        self._signals = {}

    # -- setup ------------------------------------------------------------------
    def _production_files(self):
        p = self.production
        return [p.config_dir / "settings.json", p.registry, p.header, p.output]

    def _write_isolated_config(self):
        root = Path(self._tmp.name)
        prod = self.production
        values = prod.to_dict()
        values.update({
            "registry": str(root / "registry.json"),
            "header": str(root / "config.header.yaml"),
            "output": str(root / "llama-swap.yaml"),
            "client_path": str(root / "clients.json"),
            "service_dir": str(root / "services"),
            "unit": "llms-bench",
            "listen": f"127.0.0.1:{self.port}",
            "swap_url": f"http://127.0.0.1:{self.port}",
            "client_url": f"http://127.0.0.1:{self.port}/v1",
            "preload": False,
            "companions": {},
        })
        engines = copy.deepcopy(values.get("engines", {}))
        for name, patch in self.variant.engines.items():
            engines[name] = merge_patch(engines.get(name, {}), patch)
        values["engines"] = engines
        registry = ModelManager(prod).load_registry()
        for model, patch in self.variant.entries.items():
            if model not in registry:
                raise StackError(f"variant {self.variant.name} patches unknown registry entry {model!r}")
            registry[model] = merge_patch(registry[model], patch)
        (root / "settings.json").write_text(json.dumps(values, indent=2) + "\n")
        (root / "registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
        shutil.copyfile(prod.header, root / "config.header.yaml")
        self.settings = Settings.from_env(root, environ=_isolated_environ(self.environ))
        self.manager = ModelManager(self.settings)
        self.manager.render()
        self.registry = registry

    def _stop_production(self):
        if not self.manage_production:
            self.production_state = "unmanaged"
            return
        manager = self.production_manager(self.production)
        definition = manager.backend.definition_path(self.production.unit)
        if not definition.exists():
            self.production_state = "not-installed"
            return
        active = manager._verify_service()  # fail closed before touching anything
        self.production_state = "active" if active else "inactive"
        if active:
            self.log(f"[stack] stopping production {self.production.unit} for the run")
            manager.service("stop")
            self._restore = True

    def _start_swap(self):
        binary = shutil.which(self.settings.swap_binary)
        if binary is None:
            raise StackError(f"llama-swap not found: {self.settings.swap_binary}")
        self.swap_log = Path(self._tmp.name) / "llama-swap.log"
        self._log_stream = open(self.swap_log, "w")
        self._process = subprocess.Popen(
            [binary, "-config", str(self.settings.output), "-listen", self.settings.listen],
            stdout=self._log_stream, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + self.ready_timeout
        while True:
            if self._process.poll() is not None:
                raise StackError(f"isolated llama-swap exited ({self._process.returncode}); see {self.swap_log}")
            try:
                self.get("/running", timeout=2)
                return
            except (OSError, ValueError):
                if time.monotonic() > deadline:
                    raise StackError(f"isolated llama-swap not ready after {self.ready_timeout}s")
                time.sleep(0.2)

    def _on_signal(self, signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    def __enter__(self):
        for path in self._production_files():
            self._checksums[str(path)] = _digest(path)
        self._tmp = tempfile.TemporaryDirectory(prefix="llms-bench-")
        try:
            self._write_isolated_config()
            self._stop_production()
            for signum in (signal.SIGTERM, signal.SIGHUP):
                # An inherited SIG_IGN is deliberate (nohup ignores SIGHUP); keep it.
                if signal.getsignal(signum) is not signal.SIG_IGN:
                    self._signals[signum] = signal.signal(signum, self._on_signal)
            self._start_swap()
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    # -- teardown ---------------------------------------------------------------
    def _stop_swap(self):
        if self._process is None:
            return
        if self._process.poll() is None:
            try:
                self.post("/api/models/unload", None, timeout=30, allow_empty=True)
            except (OSError, ValueError):
                pass
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self._process.pid, signal.SIGKILL)
                self._process.wait()
        for pid in self._upstreams:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self._log_stream.close()

    def __exit__(self, exc_type, exc, tb):
        errors = []
        try:
            self._stop_swap()
        except Exception as error:  # keep going: production must come back
            errors.append(f"teardown: {error}")
        for signum, handler in self._signals.items():
            signal.signal(signum, handler)
        if self._restore:
            try:
                self.log(f"[stack] restoring production {self.production.unit}")
                manager = self.production_manager(self.production)
                manager.service("start")
                manager.wait_ready(timeout=120)
            except (ManagerError, OSError) as error:
                errors.append(f"production restore failed: {error}")
        changed = [path for path, digest in self._checksums.items() if _digest(path) != digest]
        if changed:
            errors.append("production files changed during the run: " + ", ".join(changed))
        if self._tmp is not None and not self.keep_dir:
            self._tmp.cleanup()
        if errors and exc_type is None:
            raise StackError("; ".join(errors))
        for error in errors:
            self.log(f"[stack] {error}")
        return False

    # -- HTTP ---------------------------------------------------------------------
    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def _request(self, path, payload=None, *, method=None, timeout=10, allow_empty=False):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method or ("POST" if data else "GET"),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
        if not body and allow_empty:
            return {}
        return json.loads(body)

    def get(self, path, timeout=10):
        return self._request(path, timeout=timeout)

    def post(self, path, payload, timeout=1800, allow_empty=False):
        return self._request(path, payload, method="POST", timeout=timeout, allow_empty=allow_empty)

    def text(self, path, timeout=10):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")

    # -- model helpers -------------------------------------------------------------
    def rendered_command(self, model):
        import yaml

        rendered = yaml.safe_load(self.settings.output.read_text())
        entry = rendered["models"].get(model)
        if entry is None:
            raise StackError(f"{model} is not a rendered chat model")
        return entry["cmd"]

    def record(self, model):
        """Provenance attached to every result row."""
        return {"variant": self.variant.name, "patch": json.dumps(self.variant.to_dict(), sort_keys=True),
                "rendered_cmd": self.rendered_command(model), "production": self.production_state}

    def running(self):
        return self.get("/running").get("running", [])

    def load(self, model, *, timeout=900):
        """Warm the model through llama-swap; returns seconds, or raises StackError."""
        started = time.monotonic()
        try:
            self.chat({"model": model, "max_tokens": 1, "temperature": 0,
                       "messages": [{"role": "user", "content": "ok"}]}, timeout=timeout)
        except (OSError, ValueError) as error:
            tail = "\n".join(self.logs().splitlines()[-40:])
            raise StackError(f"load failed for {model}: {error}; tail of log:\n{tail}") from error
        self._track_upstreams()
        return time.monotonic() - started

    def unload(self):
        self.post("/api/models/unload", None, timeout=60, allow_empty=True)

    def chat(self, payload, *, timeout=1800):
        return self.post("/v1/chat/completions", payload, timeout=timeout)

    def upstream(self, model, path):
        return f"{self.base}/upstream/{model}{path}"

    def upstream_pids(self, model=None):
        """PIDs serving model (or all upstreams): by /running proxy port, else llama-swap's children."""
        from urllib.parse import urlsplit

        from mem import child_pids, pid_listening

        if model is not None:
            for row in self.running():
                if row.get("model") == model and row.get("proxy"):
                    pid = pid_listening(urlsplit(row["proxy"]).port)
                    if pid:
                        return [pid]
        return child_pids(self._process.pid) if self._process else []

    def _track_upstreams(self):
        self._upstreams.update(self.upstream_pids())

    def logs(self):
        """llama-swap's buffered proxy+upstream log (/logs), else its own stdout file."""
        try:
            return self.text("/logs", timeout=10)
        except (OSError, ValueError):
            try:
                return self.swap_log.read_text(errors="replace")
            except OSError:
                return ""

    def log_tail(self, lines=40):
        try:
            return "\n".join(self.swap_log.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""
