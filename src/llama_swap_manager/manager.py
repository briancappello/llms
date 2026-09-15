"""Registry operations and pure model planning/rendering helpers.

Registry command overrides and header macros are trusted executable input.
Generated path arguments are quoted, not interpolated as shell fragments.
"""

import copy
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import quote, urlsplit

from .settings import Settings


class ManagerError(RuntimeError):
    """An operation failed without a successful service/configuration outcome."""


SHARD_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d{4,5})-of-(?P<tot>\d{4,5})\.gguf$", re.I)
QUANT_RE = re.compile(r"(UD-)?(I?Q\d[_A-Z0-9]*|BF16|F16|F32|MXFP4[_A-Z0-9]*)", re.I)
QUANT_PREFERENCE = (
    "UD-Q4_K_XL", "Q4_K_XL", "UD-Q4_K_M", "Q4_K_M", "IQ4_NL", "IQ4_XS",
    "UD-Q5_K_XL", "Q5_K_XL", "UD-Q5_K_M", "Q5_K_M", "UD-Q5_K_L",
    "UD-Q6_K_XL", "Q6_K", "Q8_0",
)


def atomic_write(path, text, *, overwrite=True):
    """Write in the destination directory; never expose a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # link is atomic and fails if a concurrent init created the file.
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def default_model(registry):
    chat = {k: v for k, v in registry.items() if v.get("capability") == "chat"}
    return next((k for k, v in chat.items() if v.get("default")), min(chat, default=None))


def detect_quant(filename):
    match = QUANT_RE.search(Path(filename).name)
    return match.group(0) if match else ""


def shard_group(filename, files):
    """Return the complete ordered group, rejecting missing/conflicting shards."""
    files = list(files)
    if filename not in files:
        raise ManagerError(f"file not present: {filename}")
    match = SHARD_RE.match(filename)
    if not match:
        return [filename]
    total = int(match["tot"])
    group = []
    for candidate in files:
        part = SHARD_RE.match(candidate)
        if part and part["base"] == match["base"]:
            if int(part["tot"]) != total:
                raise ManagerError(f"conflicting shard totals: {filename}")
            group.append((int(part["idx"]), candidate))
    if total < 1 or sorted(i for i, _ in group) != list(range(1, total + 1)):
        raise ManagerError(f"incomplete or duplicate shard group: {filename}")
    return [f for _, f in sorted(group)]


def collapse_shards(files):
    files = list(files)
    return sorted({shard_group(f, files)[0] for f in files})


def pick_model_file(files, quant=None):
    """Deterministic selection; never select projector/draft weights as targets."""
    # MTP-capable targets may contain MTP in their name; exclude draft prefixes
    # and directories rather than that feature anywhere in a target filename.
    candidates = collapse_shards([
        f for f in files if f.lower().endswith(".gguf")
        and not any(word in f.lower() for word in ("mmproj", "draft", "dflash"))
        and not Path(f).name.lower().startswith(("mtp-", "mtp_"))
        and "mtp" not in [p.lower() for p in PurePosixPath(f).parts[:-1]]
    ])
    if quant:
        candidates = [f for f in candidates if quant.lower() in Path(f).name.lower()]
    if len(candidates) == 1:
        return candidates[0]
    for preferred in QUANT_PREFERENCE:
        matches = [f for f in candidates if detect_quant(f).upper() == preferred]
        if len(matches) == 1:
            return matches[0]
    raise ManagerError("no unambiguous model selection; use --file or --quant: " + ", ".join(candidates))


def pick_mmproj(files):
    candidates = sorted(f for f in files if "mmproj" in Path(f).name.lower() and f.lower().endswith(".gguf"))
    for token in ("bf16", "-f16", "f32", "q8_0", "q8"):
        match = next((f for f in candidates if token in f.lower()), None)
        if match:
            return match
    return next(iter(candidates), None)


def resolve_source(source, files, *, quant=None, file=None, no_mmproj=False):
    """Pure HF plan from an authoritative file listing; no network or cache I/O.

    Returns repo, model_file, files (all target shards), and mmproj_file.
    The manager attaches the immutable revision and downloads those exact files.
    """
    source = source.removeprefix("hf://").strip()
    parts = source.split("/")
    if len(parts) < 2 or not all(re.fullmatch(r"[\w.-]+", p) and p not in (".", "..") for p in parts[:1]):
        raise ManagerError("source must be owner/repo[:quant] or hf://owner/repo/file.gguf")
    repo_name, sep, source_quant = parts[1].partition(":")
    if not re.fullmatch(r"[\w.-]+", repo_name) or repo_name in (".", ".."):
        raise ManagerError("invalid Hugging Face repository")
    repo = parts[0] + "/" + repo_name
    files = list(files)
    for candidate in files:
        path = PurePosixPath(candidate)
        if path.is_absolute() or ".." in path.parts or "\\" in candidate:
            raise ManagerError(f"unsafe repository filename: {candidate}")
    explicit = file or ("/".join(parts[2:]) if len(parts) > 2 else None)
    selected = explicit or pick_model_file(files, quant or (source_quant if sep else None))
    if not selected.lower().endswith(".gguf"):
        raise ManagerError("model must be a GGUF file")
    group = shard_group(selected, files)
    return {"repo": repo, "model_file": group[0], "files": group,
            "mmproj_file": None if no_mmproj else pick_mmproj(files)}


def suggest_ctx(meta, size_bytes, gpu_memory_mib=None):
    """Conservative f16 KV estimate, never above a known trained window.

    No configured GPU budget means a modest CPU-compatible context, not a
    hardware guess. Hybrid/MLA models still need measured tuning.
    """
    arch = meta.get("general.architecture", "")
    trained = int(meta.get(f"{arch}.context_length", 0) or 0)
    limit = trained if trained > 0 else 4096
    if gpu_memory_mib is None:
        return min(limit, 4096)
    layers = int(meta.get(f"{arch}.block_count", 0) or 0)
    kv_heads = int(meta.get(f"{arch}.attention.head_count_kv", 0) or 0)
    heads = int(meta.get(f"{arch}.attention.head_count", 0) or 0)
    embedding = int(meta.get(f"{arch}.embedding_length", 0) or 0)
    dimension = embedding // heads if heads > 0 else 128
    key = int(meta.get(f"{arch}.attention.key_length", dimension) or dimension)
    value = int(meta.get(f"{arch}.attention.value_length", dimension) or dimension)
    free = max(0, gpu_memory_mib * 2**20 - size_bytes)
    if free == 0:
        # A model can use CPU or host-memory offload when weights exceed VRAM.
        # GPU capacity alone cannot estimate that hybrid context budget.
        return min(limit, 4096)
    if layers <= 0 or kv_heads <= 0:
        return min(limit, 4096 if free > 0 else 512)
    per_token = max(1, layers * kv_heads * (key + value) * 2)
    maximum = min(limit, max(1, int(free / per_token)))
    return min(limit, 2 ** int(math.log2(maximum)))


def probe_gguf(path):
    """Read metadata locally using the optional gguf extra; no subprocesses."""
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise ManagerError("GGUF probing requires llama-swap-manager[gguf]") from exc
    try:
        reader = GGUFReader(str(path))
        metadata = {}
        for name, field in reader.fields.items():
            if name.startswith(("general.", "tokenizer.chat_template")) or any(
                key in name for key in ("context_length", "block_count", "head_count", "embedding_length", "key_length", "value_length", "nextn")
            ):
                value = field.contents()
                metadata[name] = value.item() if hasattr(value, "item") else value
        metadata["_has_mtp"] = any("nextn" in t.name or ".mtp" in t.name.lower() for t in reader.tensors) or any(
            int(v or 0) > 0 for k, v in metadata.items() if k.endswith("nextn_predict_layers")
        )
        metadata["_reasoning_preserve"] = "preserve" in str(metadata.get("tokenizer.chat_template", ""))
        return metadata
    except Exception as exc:
        raise ManagerError(f"cannot probe GGUF {path}: {exc}") from exc


def _quoted(value):
    value = str(value)
    if any(c in value for c in ("\x00", "\n", "\r")) or "${" in value:
        raise ManagerError("generated command arguments cannot contain newlines, NUL, or macro substitutions")
    return shlex.quote(value)


def render_config(registry, header, settings):
    """Pure safe YAML rendering, preserving header comments and trusted macros.

    extra_args retains the imported schema: a list of argument fragments,
    parsed with shlex then re-quoted. cmd is a trusted string or line list.
    Explicit header preload lists take precedence (including an empty list).
    Otherwise a default-model startup hook is generated only with preload=True.
    """
    import yaml

    try:
        base = yaml.safe_load(header)
    except yaml.YAMLError as exc:
        raise ManagerError(f"invalid YAML header: {exc}") from exc
    base = {} if base is None else base
    if not isinstance(base, dict) or "models" in base:
        raise ManagerError("header must be a mapping without a models section")
    hooks = copy.deepcopy(base.get("hooks", {}))
    if not isinstance(hooks, dict) or not isinstance(hooks.get("on_startup", {}), dict):
        raise ManagerError("header hooks/on_startup must be mappings")
    models = {}
    for name, entry in sorted(registry.items()):
        if entry.get("capability") != "chat":
            continue
        model = {"name": entry.get("display", name), "description": entry.get("description", ""), "ttl": 0}
        if entry.get("cmd"):
            cmd = entry["cmd"]
            model["cmd"] = "\n".join(cmd) if isinstance(cmd, list) else cmd
            if not isinstance(model["cmd"], str):
                raise ManagerError(f"invalid custom command for {name}")
        else:
            macros = base.get("macros", {})
            server = "${server}" if "server" in macros else _quoted(settings.server) + " --host 127.0.0.1 --port ${PORT}"
            command = [server, "-m", _quoted(entry["path"])]
            if entry.get("mmproj"):
                command += ["--mmproj", _quoted(entry["mmproj"])]
            if "common" in macros:
                command += ["${common}"]
            ctx, parallel = int(entry["ctx"]), int(entry.get("parallel", 4))
            if ctx < 1 or parallel < 1:
                raise ManagerError(f"context and parallel must be positive for {name}")
            # Preserve explicitly tuned split-slot registries. The trained bound
            # applies per slot when KV is partitioned, to the pool otherwise.
            trained = int(entry.get("trained_ctx") or 0)
            if trained > 0:
                ctx = min(ctx, trained * (parallel if entry.get("kv_unified") is False else 1))
            command += ["-c", str(ctx), "-np", str(parallel), "--no-kv-unified" if entry.get("kv_unified") is False else "--kv-unified"]
            fragments = entry.get("extra_args", [])
            if not isinstance(fragments, list) or not all(isinstance(f, str) for f in fragments):
                raise ManagerError(f"extra_args must be a list of strings for {name}")
            for fragment in fragments:
                command.extend(_quoted(token) for token in shlex.split(fragment))
            model["cmd"] = " ".join(command)
        for key in ("proxy", "checkEndpoint", "useModelName", "unloadTimeout", "aliases"):
            if key in entry:
                model[key] = entry[key]
        model["metadata"] = {key: entry[key] for key in ("capability", "quant", "arch", "trained_ctx", "params") if entry.get(key) is not None}
        models[name] = model
    startup = hooks.get("on_startup", {})
    if "preload" in startup:
        if not isinstance(startup["preload"], list) or not all(isinstance(n, str) and n for n in startup["preload"]):
            raise ManagerError("header startup preload must be a list of model IDs")
    elif settings.preload and (default := default_model(registry)):
        hooks.setdefault("on_startup", {})["preload"] = [default]
    generated = {"models": models}
    if hooks:
        generated["hooks"] = hooks
    # A header with hooks needs merging rather than a duplicate YAML key.
    if "hooks" in base:
        del base["hooks"]
        header = yaml.safe_dump(base, sort_keys=False)
    text = header.rstrip() + "\n\n# Generated by llama-swap-manager.\n" + yaml.safe_dump(generated, sort_keys=False, allow_unicode=True)
    try:
        yaml.safe_load(text)
    except yaml.YAMLError:
        # Flow-style mappings and explicit YAML document endings cannot be
        # extended by appending block keys. Re-serialize those headers instead.
        text = yaml.safe_dump({**base, **generated}, sort_keys=False, allow_unicode=True)
    return text


class ModelManager:
    """Per-instance state. Library mutations default to no service/client I/O.

    Writers must be serialized by the caller. Individual files are atomically
    replaced; registry and generated YAML are not a cross-file transaction.
    """

    def __init__(self, settings: Settings, *, probe=None, opener=None, runner=None, hub=None):
        self.settings = settings
        self.probe = probe or probe_gguf
        self.opener = opener or urllib.request.urlopen
        self.runner = runner or subprocess.run
        self.hub = hub

    def load_registry(self):
        if not self.settings.registry.exists():
            return {}
        registry = json.loads(self.settings.registry.read_text())
        if not isinstance(registry, dict) or not all(isinstance(k, str) and k and isinstance(v, dict) for k, v in registry.items()):
            raise ManagerError("registry must map non-empty model IDs to objects")
        return registry

    def init(self):
        """Create only missing settings/header/registry; never render or start."""
        import yaml

        files = {
            self.settings.config_dir / "settings.json": json.dumps(self.settings.to_dict(), indent=2) + "\n",
            self.settings.registry: "{}\n",
            self.settings.header: yaml.safe_dump({
                "healthCheckTimeout": 500, "logLevel": "info", "startPort": 10001, "globalTTL": 0,
                "macros": {"common": "--jinja --metrics"},
            }, sort_keys=False),
        }
        created = []
        for path, text in files.items():
            try:
                atomic_write(path, text, overwrite=False)
                created.append(str(path))
            except FileExistsError:
                pass
        return created

    def render(self, registry=None):
        registry = self.load_registry() if registry is None else registry
        text = render_config(registry, self.settings.header.read_text(), self.settings)
        atomic_write(self.settings.output, text)
        return self.settings.output

    def _commit(self, registry):
        text = render_config(registry, self.settings.header.read_text(), self.settings)
        encoded = json.dumps(registry, indent=2, sort_keys=True) + "\n"
        atomic_write(self.settings.registry, encoded)
        atomic_write(self.settings.output, text)

    def _hub(self):
        if self.hub is None:
            from huggingface_hub import HfApi
            self.hub = HfApi(endpoint=self.settings.hf_endpoint, token=self.settings.hf_token or False)
        return self.hub

    def plan(self, source, *, quant=None, file=None, no_mmproj=False, revision="main"):
        """Resolve local/cached files or pin an HF listing to an immutable SHA."""
        path = Path(source).expanduser()
        if path.is_file() or ("/" not in source and source.lower().endswith(".gguf")):
            if not path.is_file():
                hits = sorted(p for p in self.settings.hf_cache.glob("models--*/snapshots/**/*.gguf") if p.name == source and p.is_file())
                if len(hits) != 1:
                    raise ManagerError(f"expected one cached {source}, found {len(hits)}; use an explicit path")
                path = hits[0]
            path = path.absolute()
            if path.suffix.lower() != ".gguf":
                raise ManagerError("local model must be a GGUF file")
            files = [str(p) for p in path.parent.iterdir() if p.is_file() and p.suffix.lower() == ".gguf"]
            group = shard_group(str(path), files)
            match = re.search(r"models--(.+?)--(.+?)/snapshots/([^/]+)/", str(path))
            return {"local": True, "repo": f"{match[1]}/{match[2]}" if match else None,
                    "sha": match[3] if match else None, "model_file": Path(group[0]).name,
                    "model_path": group[0], "files": group,
                    "mmproj_file": None if no_mmproj else pick_mmproj(files)}
        stripped = source.removeprefix("hf://")
        repo = "/".join(stripped.split("/")[:2]).split(":", 1)[0]
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or any(p in (".", "..") for p in repo.split("/")):
            raise ManagerError("invalid Hugging Face repository")
        try:
            info = self._hub().model_info(repo, revision=revision)
        except Exception as exc:
            raise ManagerError(f"could not list HF repository {repo}: {exc}") from exc
        sha = info.sha
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", sha):
            raise ManagerError("HF did not return an immutable commit SHA")
        plan = resolve_source(source, [s.rfilename for s in info.siblings], quant=quant, file=file, no_mmproj=no_mmproj)
        return {**plan, "sha": sha, "local": False}

    def add(self, source, *, name=None, quant=None, file=None, no_mmproj=False,
            revision="main", restart=False, load=False, sync_clients=False):
        if restart:
            self._verify_service(check_api=load)
        if self.probe is probe_gguf and importlib.util.find_spec("gguf") is None:
            raise ManagerError("GGUF probing requires llama-swap-manager[gguf]; nothing downloaded")
        plan = self.plan(source, quant=quant, file=file, no_mmproj=no_mmproj, revision=revision)
        if plan["local"]:
            paths = [Path(p) for p in plan["files"]]
            mmproj = plan["mmproj_file"]
        else:
            hub = self._hub()
            def download(filename):
                try:
                    return hub.hf_hub_download(repo_id=plan["repo"], filename=filename,
                                               revision=plan["sha"], cache_dir=str(self.settings.hf_cache))
                except Exception as exc:
                    raise ManagerError(f"could not download {filename} at {plan['sha']}: {exc}") from exc
            paths = [Path(download(f)) for f in plan["files"]]
            mmproj = download(plan["mmproj_file"]) if plan["mmproj_file"] else None
        if not all(p.is_file() for p in paths):
            raise ManagerError("downloaded shard group is incomplete")
        size = sum(p.stat().st_size for p in paths)
        metadata = self.probe(paths[0])
        arch = str(metadata.get("general.architecture", "unknown"))
        filename = paths[0].name
        lower = filename.lower()
        capability = "chat"
        if any(word in lower or word in arch.lower() for word in ("flux", "stable-diffusion", "qwen-image", "sd35")):
            capability = "diffusion"
        elif "mmproj" in lower:
            capability = "projector"
        elif any(word in lower for word in ("embed", "bge", "e5-")):
            capability = "embedding"
        registry = self.load_registry()
        if name is None:
            base = re.sub(r"[-_. ]?gguf$", "", (plan["repo"] or filename).split("/")[-1], flags=re.I)
            base = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-") or "model"
            name, index = base, 2
            while name in registry:
                name, index = f"{base}-{index}", index + 1
        if not name or name in registry:
            raise ManagerError(f"model ID already exists or is empty: {name!r}")
        extra = []
        if metadata.get("_has_mtp"):
            extra.append("--spec-type draft-mtp")
        if metadata.get("_reasoning_preserve"):
            extra.append("--reasoning-preserve")
        entry = {
            "path": str(paths[0]), "repo": plan["repo"], "sha": plan["sha"], "file": filename,
            "display": str(metadata.get("general.name", name)),
            "description": "Scaffolded from GGUF metadata; verify context and tune sampling.",
            "arch": arch, "capability": capability, "trained_ctx": metadata.get(f"{arch}.context_length"),
            "ctx": suggest_ctx(metadata, size + (Path(mmproj).stat().st_size if mmproj else 0), self.settings.gpu_memory_mib),
            "parallel": 1, "size_bytes": size, "quant": quant or detect_quant(filename),
            "extra_args": extra,
        }
        if mmproj and capability == "chat":
            entry["mmproj"] = str(mmproj)
        registry[name] = entry
        self._commit(registry)
        if sync_clients:
            self.ensure_clients()
        if restart:
            self.service("restart")
        if load and capability == "chat":
            if restart:
                self.wait_ready()
                if not self._verify_service(check_api=True):
                    raise ManagerError("explicit warm-up refused: managed service is not active")
            self.use(name)
        return name, entry

    def api(self, path, *, method="GET", payload=None, timeout=10, allow_empty=False):
        request = urllib.request.Request(self.settings.swap_url.rstrip("/") + path,
                                         data=json.dumps(payload).encode() if payload is not None else None,
                                         method=method, headers={"Content-Type": "application/json"})
        try:
            with self.opener(request, timeout=timeout) as response:
                body = response.read()
                if not body and allow_empty:
                    return {}
                data = json.loads(body)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            raise ManagerError(f"API {method} {path} failed: {exc}") from exc
        if not isinstance(data, dict) or data.get("error") or data.get("_error") or data.get("success") is False:
            raise ManagerError(f"API {method} {path} returned an error or malformed response: {data!r}")
        return data

    def running(self):
        data = self.api("/running")
        rows = data.get("running")
        if not isinstance(rows, list) or not all(isinstance(r, dict) and isinstance(r.get("model"), str) and r["model"] and isinstance(r.get("state"), str) and r["state"] for r in rows):
            raise ManagerError("malformed /running response")
        return rows

    def status(self):
        running = self.running()
        data = self.api("/v1/models").get("data")
        if not isinstance(data, list) or not all(isinstance(r, dict) and isinstance(r.get("id"), str) and r["id"] for r in data):
            raise ManagerError("malformed /v1/models response")
        return {"endpoint": self.settings.swap_url, "running": running, "models": data}

    def wait_ready(self, *, timeout=30):
        """Wait for a restarted swap API, never start a process ourselves."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.status()
            except ManagerError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.5)

    def use(self, name, *, timeout=240):
        entry = self.load_registry().get(name)
        if not entry or entry.get("capability") != "chat":
            raise ManagerError(f"unknown or non-chat model: {name}")
        data = self.api("/v1/chat/completions", method="POST", timeout=timeout,
                        payload={"model": name, "max_tokens": 1, "temperature": 0,
                                 "messages": [{"role": "user", "content": "ok"}]})
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not all(
            isinstance(c, dict) and isinstance(c.get("message"), dict)
            and c["message"].get("role") == "assistant"
            and (isinstance(c["message"].get("content"), str)
                 or isinstance(c["message"].get("reasoning_content"), str)
                 or (isinstance(c["message"].get("tool_calls"), list) and c["message"]["tool_calls"]
                     and all(isinstance(t, dict) and t.get("type") == "function"
                             and isinstance(t.get("function"), dict)
                             and isinstance(t["function"].get("name"), str)
                             and isinstance(t["function"].get("arguments"), str)
                             for t in c["message"]["tool_calls"]))) for c in choices
        ):
            raise ManagerError("malformed chat completion; model did not return a valid answer")
        return data

    def remove(self, name, *, purge=False, restart=False, sync_clients=False):
        if purge:
            raise ManagerError("purge is disabled: exclusive ownership of a shared HF cache cannot be proven across headers, registries, and other consumers; no changes made")
        active = self._verify_service(check_api=True) if restart else False
        registry = self.load_registry()
        if name not in registry:
            raise ManagerError(f"unknown model: {name}")
        if active:
            if name in {row["model"] for row in self.running()}:
                self.api("/api/models/unload/" + quote(name, safe=""), method="POST", allow_empty=True)
                if name in {row["model"] for row in self.running()}:
                    raise ManagerError(f"unload did not stop {name}; registry and weights left unchanged")
        del registry[name]
        self._commit(registry)
        if restart:
            self.service("restart")
        if sync_clients:
            self.ensure_clients()

    def ensure_clients(self):
        """Explicit pi provider update; preserve other providers and credentials."""
        path = self.settings.client_path
        data = json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(data, dict) or not isinstance(data.get("providers", {}), dict):
            raise ManagerError("client config must contain an object of providers")
        providers = data.setdefault("providers", {})
        block = providers.setdefault("llama-swap", {})
        if not isinstance(block, dict):
            raise ManagerError("client llama-swap provider must be an object")
        block.setdefault("baseUrl", self.settings.client_url)
        block.setdefault("api", "openai-completions")
        block.setdefault("apiKey", "none")
        block["compat"] = {"supportsStore": False, "supportsDeveloperRole": False,
                           "supportsReasoningEffort": False, "supportsStrictMode": False,
                           "maxTokensField": "max_tokens", "thinkingFormat": "deepseek",
                           "requiresReasoningContentOnAssistantMessages": True}
        block["models"] = []
        for name, entry in sorted(self.load_registry().items()):
            if entry.get("capability") != "chat":
                continue
            context = int(entry.get("ctx") or entry.get("trained_ctx") or 4096)
            if entry.get("kv_unified") is False:
                context //= int(entry.get("parallel", 4))
            context = min(context, int(entry.get("trained_ctx") or context))
            block["models"].append({"id": name, "name": entry.get("display", name), "reasoning": True,
                                    "input": ["text", "image"] if entry.get("mmproj") else ["text"],
                                    "contextWindow": context, "maxTokens": min(context, 65536),
                                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}})
        atomic_write(path, json.dumps(data, indent=2) + "\n")
        return path

    def service(self, action):
        if action not in ("start", "stop", "restart"):
            raise ManagerError(f"unsupported service action: {action}")
        self._verify_service()
        result = self.runner([self.settings.systemctl, "--user", action, self.settings.unit], capture_output=True, text=True)
        if result.returncode:
            raise ManagerError(f"{action} failed: {result.stderr.strip()}")

    def _service_definition(self):
        executable = shutil.which(self.settings.swap_binary)
        if executable is None:
            raise ManagerError(f"binary not executable: {self.settings.swap_binary}")
        def systemd_quote(value):
            if any(c in str(value) for c in "\x00\r\n"):
                raise ManagerError("service arguments cannot contain control characters")
            return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'
        args = [str(Path(executable).absolute()), "-config", str(self.settings.output), "-listen", self.settings.listen]
        text = "[Unit]\nDescription=llama-swap model server\n\n[Service]\nType=simple\nExecStart=" + " ".join(systemd_quote(a) for a in args) + "\nRestart=on-failure\n\n[Install]\nWantedBy=default.target\n"
        return args, text

    def _verify_service(self, *, check_api=False):
        """Fail closed unless source, loaded unit, and live process match.

        v0.1 manages only its unmodified generated unit, without drop-ins. The
        systemctl ExecStart display loses argv boundaries, so source equality
        and the live NUL-separated argv are also checked. No daemon reload or
        link is performed here. Operators must serialize unit/config changes.
        """
        args, text = self._service_definition()
        path = self.settings.service_dir / self.settings.unit
        link = shlex.join([self.settings.systemctl, "--user", "link", str(path)])
        if check_api:
            api, listen = urlsplit(self.settings.swap_url), urlsplit("http://" + self.settings.listen)
            host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(listen.hostname, listen.hostname)
            if (api.scheme != "http" or not host or api.hostname != host
                    or api.port != listen.port or not listen.port
                    or api.path not in ("", "/") or api.query or api.fragment or api.username or api.password):
                raise ManagerError("managed API operation refused: swap_url must address the configured listen endpoint directly over HTTP")
        properties = ("Id", "LoadState", "FragmentPath", "ExecStart", "ActiveState", "MainPID",
                      "ControlPID", "NeedDaemonReload", "DropInPaths", "Transient", "Type")
        try:
            if path.read_text() != text:
                raise ManagerError(f"service refused: {path} is not the unmodified unit generated for these settings")
            result = self.runner([self.settings.systemctl, "--user", "show", self.settings.unit,
                                  "--no-pager", "--all", "--property=" + ",".join(properties)],
                                 capture_output=True, text=True, timeout=10)
            if result.returncode:
                raise ManagerError(f"cannot inspect user service: {result.stderr.strip()}")
            loaded = {}
            for line in result.stdout.splitlines():
                key, separator, value = line.partition("=")
                if not separator or key in loaded or key not in properties:
                    raise ManagerError("service refused: malformed systemctl show response")
                loaded[key] = value
            if loaded.keys() != set(properties):
                raise ManagerError("service refused: incomplete systemctl show response")
            if loaded["LoadState"] != "loaded" or not loaded["FragmentPath"]:
                raise ManagerError(f"unit is not discoverable by the user manager; for an external unit run {link}, then systemctl --user daemon-reload")
            if (loaded["Id"] != self.settings.unit or not Path(loaded["FragmentPath"]).is_absolute()
                    or Path(loaded["FragmentPath"]).resolve(strict=True) != path.resolve(strict=True)):
                raise ManagerError("service refused: loaded FragmentPath/Id belongs to a different unit source")
            if (loaded["NeedDaemonReload"] != "no" or loaded["DropInPaths"]
                    or loaded["Transient"] != "no" or loaded["Type"] != "simple"):
                raise ManagerError("service refused: stale, overridden, transient, or unsupported unit; reconcile it explicitly")
            prefix = f"{{ path={args[0]} ; argv[]={' '.join(args)} ; ignore_errors=no ; "
            metadata = r"start_time=\[[^\]\n]*\] ; stop_time=\[[^\]\n]*\] ; pid=\d+ ; code=[^;{}\n]* ; status=[^;{}\n]* }"
            if not re.fullmatch(re.escape(prefix) + metadata, loaded["ExecStart"]):
                raise ManagerError("service refused: loaded ExecStart does not match swap binary, output, and listen")
            state, pid = loaded["ActiveState"], loaded["MainPID"]
            if state not in ("active", "inactive", "failed") or not pid.isdecimal() or loaded["ControlPID"] != "0":
                raise ManagerError("service refused: unstable or unknown process state")
            if state == "active":
                if int(pid) <= 0:
                    raise ManagerError("service refused: active unit has no verifiable main process")
                process = Path("/proc") / pid
                command = (process / "cmdline").read_bytes()
                if (not command.endswith(b"\0") or command[:-1].split(b"\0") != [os.fsencode(a) for a in args]
                        or (process / "exe").resolve(strict=True) != Path(args[0]).resolve(strict=True)):
                    raise ManagerError("service refused: live process binding differs from configured binary, output, or listen")
            elif pid != "0":
                raise ManagerError("service refused: inactive unit still has a main process")
            return state == "active"
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ManagerError(f"cannot verify installed/live service binding: {exc}; install the unit first and link external units with {link}") from exc

    def install_service(self):
        """Write a user unit only; external paths require an explicit user link."""
        _, text = self._service_definition()
        path = self.settings.service_dir / self.settings.unit
        atomic_write(path, text, overwrite=False)
        return path

    def doctor(self, *, live=False):
        """Return structured checks; no network unless live=True."""
        checks = []
        for module in ("yaml", "huggingface_hub", "gguf"):
            checks.append({"check": module, "ok": importlib.util.find_spec(module) is not None,
                           "required": module != "gguf", "detail": "gguf extra required for add" if module == "gguf" else "Python dependency"})
        for key in ("server", "swap_binary", "systemctl"):
            value = getattr(self.settings, key)
            checks.append({"check": key, "ok": shutil.which(value) is not None, "required": key != "systemctl", "detail": value})
        for key in ("registry", "header", "output"):
            path = getattr(self.settings, key)
            checks.append({"check": key, "ok": path.is_file(), "required": key != "output", "detail": str(path)})
        try:
            registry = self.load_registry()
            render_config(registry, self.settings.header.read_text(), self.settings)
            for name, entry in registry.items():
                if not entry.get("cmd") and entry.get("path"):
                    path = Path(entry["path"])
                    if not path.is_file():
                        raise ManagerError(f"missing model path: {name}: {path}")
                    shard_group(str(path), [str(p) for p in path.parent.iterdir() if p.is_file()])
                if entry.get("mmproj") and not Path(entry["mmproj"]).is_file():
                    raise ManagerError(f"missing mmproj: {name}")
            checks.append({"check": "config", "ok": True, "required": True, "detail": "registry and render valid"})
        except (OSError, ValueError, KeyError, TypeError, ManagerError) as exc:
            checks.append({"check": "config", "ok": False, "required": True, "detail": str(exc)})
        if live:
            try:
                self.status()
                checks.append({"check": "live", "ok": True, "required": True, "detail": self.settings.swap_url})
            except ManagerError as exc:
                checks.append({"check": "live", "ok": False, "required": True, "detail": str(exc)})
        return checks
