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
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import quote, urlsplit

from .errors import ManagerError
from .services import backend_for
from .settings import Settings


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
        raise ManagerError("GGUF probing requires llms[gguf]") from exc
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


def embedded_sampling(metadata):
    """general.sampling.* keys a GGUF ships; llama.cpp applies them by default."""
    prefix = "general.sampling."
    # GGUF stores these as float32; 0.6 would otherwise read back as 0.6000000238.
    return {key[len(prefix):]: float(f"{value:.6g}") if isinstance(value, float) else value
            for key, value in sorted(metadata.items())
            if key.startswith(prefix) and isinstance(value, (int, float, str, bool))}


def _quoted(value):
    value = str(value)
    if any(c in value for c in ("\x00", "\n", "\r")) or "${" in value:
        raise ManagerError("generated command arguments cannot contain newlines, NUL, or macro substitutions")
    return shlex.quote(value)


def _extra_args(entry, name):
    fragments = entry.get("extra_args", [])
    if not isinstance(fragments, list) or not all(isinstance(f, str) for f in fragments):
        raise ManagerError(f"extra_args must be a list of strings for {name}")
    return [_quoted(token) for fragment in fragments for token in shlex.split(fragment)]


# Rendered for `"mtp": true` when the engine sets no mtp_args of its own. The
# right speculative arguments are host-specific (a draft length that pays on a
# ROCm dGPU can lose on Apple unified memory), so engines may override them.
DEFAULT_MTP_ARGS = ("--spec-type", "draft-mtp")


def _mtp_flag(entry, name):
    value = entry.get("mtp", False)
    if not isinstance(value, bool):
        raise ManagerError(f"mtp must be true or false for {name}")
    return value


def _custom_ctx(entry, name):
    """A custom server (e.g. an MLX server) may enforce no context limit of its
    own, so the measured ctx a client is told is the only safeguard: required."""
    ctx = entry.get("ctx")
    if isinstance(ctx, bool) or not isinstance(ctx, int) or ctx < 1:
        raise ManagerError(f"{name} uses a custom engine and must declare a measured positive integer ctx; "
                           "the client context window is the only limit such servers get")
    return ctx


def _llama_model_args(entry, name, macros, mtp_args=DEFAULT_MTP_ARGS):
    """The llama.cpp-shaped part of a command: weights, context, slots, tuning.

    Shared by the header-macro path and the engine path so both stay identical.
    A model whose cache types or offload split contradict the shared macro sets
    "common": false and carries the full set itself.
    """
    if not entry.get("path"):
        raise ManagerError(f"{name} needs a path, or an engine of kind 'custom'")
    args = ["-m", _quoted(entry["path"])]
    if entry.get("mmproj"):
        args += ["--mmproj", _quoted(entry["mmproj"])]
    if entry.get("common", True) and "common" in macros:
        args += ["${common}"]
    ctx, parallel = int(entry["ctx"]), int(entry.get("parallel", 4))
    if ctx < 1 or parallel < 1:
        raise ManagerError(f"context and parallel must be positive for {name}")
    # Preserve explicitly tuned split-slot registries. The trained bound
    # applies per slot when KV is partitioned, to the pool otherwise.
    trained = int(entry.get("trained_ctx") or 0)
    if trained > 0:
        ctx = min(ctx, trained * (parallel if entry.get("kv_unified") is False else 1))
    args += ["-c", str(ctx), "-np", str(parallel),
             "--no-kv-unified" if entry.get("kv_unified") is False else "--kv-unified"]
    if _mtp_flag(entry, name):
        args += [_quoted(a) for a in mtp_args]
    return args + _extra_args(entry, name)


def _engine_command(entry, name, settings, macros):
    """Build a command from a named engine in settings.

    Only the engine knows the binary, its device pin, and its working
    directory, so the registry entry stays portable between hosts.
    """
    engine_name = entry["engine"]
    engine = settings.engines.get(engine_name)
    if engine is None:
        known = ", ".join(sorted(settings.engines)) or "none defined"
        raise ManagerError(f"{name} refers to unknown engine {engine_name!r} (known: {known})")
    command = []
    if cwd := engine.get("cwd"):
        # env(1) rather than a shell: no nested quoting, no extra process
        # between llama-swap and the server it signals. `-C DIR` is the one
        # spelling both BSD (macOS) and GNU (coreutils >= 8.28) env accept;
        # GNU-only `--chdir=` is rejected by macOS env.
        command += ["/usr/bin/env", "-C", _quoted(cwd)]
    command.append(_quoted(engine["server"]))
    command += [_quoted(a) for a in engine.get("args", [])]
    command += ["--host", _quoted(engine.get("host", "127.0.0.1")), "--port", "${PORT}"]
    if engine.get("kind", "llama.cpp") == "custom":
        _custom_ctx(entry, name)
        if _mtp_flag(entry, name):
            raise ManagerError(f"{name} sets mtp on custom engine {engine_name!r}; put that server's own speculative flags in extra_args")
        command += _extra_args(entry, name)
    else:
        command += _llama_model_args(entry, name, macros, engine.get("mtp_args", DEFAULT_MTP_ARGS))
    extras = {}
    for source, target in (("check_endpoint", "checkEndpoint"),
                           ("use_model_name", "useModelName"),
                           ("unload_timeout", "unloadTimeout")):
        if source in engine:
            extras[target] = engine[source]
    if env := engine.get("env"):
        extras["env"] = [f"{key}={value}" for key, value in sorted(env.items())]
    return " ".join(command), extras


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
        macros = base.get("macros", {})
        engine_extras = {}
        if entry.get("cmd"):
            if entry.get("engine"):
                raise ManagerError(f"{name} sets both cmd and engine; pick one")
            if _mtp_flag(entry, name):
                raise ManagerError(f"{name} sets both cmd and mtp; a verbatim cmd carries its own flags")
            cmd = entry["cmd"]
            model["cmd"] = "\n".join(cmd) if isinstance(cmd, list) else cmd
            if not isinstance(model["cmd"], str):
                raise ManagerError(f"invalid custom command for {name}")
        elif entry.get("engine"):
            model["cmd"], engine_extras = _engine_command(entry, name, settings, macros)
        else:
            server = "${server}" if "server" in macros else _quoted(settings.server) + " --host 127.0.0.1 --port ${PORT}"
            model["cmd"] = " ".join([server] + _llama_model_args(entry, name, macros))
        model.update(engine_extras)
        # An explicit registry value always beats the engine default.
        for key in ("proxy", "checkEndpoint", "useModelName", "unloadTimeout", "aliases", "env"):
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
    text = header.rstrip() + "\n\n# Generated by llms.\n" + yaml.safe_dump(generated, sort_keys=False, allow_unicode=True)
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
        self._backend = None

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
            raise ManagerError("GGUF probing requires llms[gguf]; nothing downloaded")
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
        if metadata.get("_has_mtp"):
            # Intent, not flags: each host's engine renders its own mtp_args.
            entry["mtp"] = True
        sampling = embedded_sampling(metadata)
        if sampling:
            # Recorded, never rendered: llama.cpp applies these silently unless a
            # flag overrides them, so the operator needs to see them.
            entry["embedded_sampling"] = sampling
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

    def unregistered(self):
        """Complete cached models that no registry entry references; offline, read-only.

        GGUF: target files (projectors/drafts excluded, shard groups collapsed,
        incomplete groups skipped). MLX: snapshot dirs with config.json and
        safetensors, and every shard an index names. Partial downloads never show.
        """
        registry = self.load_registry()
        referenced = set()
        for entry in registry.values():
            for key in ("path", "mmproj", "repo"):
                if entry.get(key):
                    referenced.add(str(entry[key]))
            for fragment in entry.get("extra_args", []) if isinstance(entry.get("extra_args"), list) else []:
                referenced.update(shlex.split(fragment))
        resolved = {str(Path(r).resolve()) for r in referenced if r.startswith("/")}
        found = []
        for snapshot in sorted(self.settings.hf_cache.glob("models--*/snapshots/*")):
            if not snapshot.is_dir():
                continue
            match = re.fullmatch(r"models--(.+?)--(.+)", snapshot.parent.parent.name)
            repo = f"{match[1]}/{match[2]}" if match else None
            ggufs = [str(p.relative_to(snapshot)) for p in snapshot.rglob("*.gguf") if p.is_file()]
            targets = [f for f in ggufs if not any(w in f.lower() for w in ("mmproj", "draft", "dflash"))
                       and not Path(f).name.lower().startswith(("mtp-", "mtp_"))]
            for first in sorted({f for f in targets}):
                try:
                    group = shard_group(first, targets)
                except ManagerError:
                    continue  # incomplete or conflicting shard group
                if group[0] != first:
                    continue
                path = snapshot / first
                if str(path) in referenced or str(path.resolve()) in resolved:
                    continue
                size = sum((snapshot / f).stat().st_size for f in group)
                found.append({"kind": "gguf", "repo": repo, "path": str(path), "files": len(group), "size_bytes": size})
            shards = [p for p in snapshot.glob("*.safetensors") if p.is_file()]
            if (snapshot / "config.json").is_file() and shards:
                index = snapshot / "model.safetensors.index.json"
                if index.is_file():
                    try:
                        needed = set(json.loads(index.read_text()).get("weight_map", {}).values())
                    except (OSError, ValueError):
                        continue
                    if not all((snapshot / name).is_file() for name in needed):
                        continue
                if repo in referenced or str(snapshot) in referenced or str(snapshot.resolve()) in resolved:
                    continue
                found.append({"kind": "mlx", "repo": repo, "path": str(snapshot), "files": len(shards),
                              "size_bytes": sum(p.stat().st_size for p in shards)})
        return found

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
                 or isinstance(c["message"].get("reasoning"), str)  # MLX servers
                 or (isinstance(c["message"].get("tool_calls"), list) and c["message"]["tool_calls"]
                     and all(isinstance(t, dict) and t.get("type") == "function"
                             and isinstance(t.get("function"), dict)
                             and isinstance(t["function"].get("name"), str)
                             and isinstance(t["function"].get("arguments"), str)
                             for t in c["message"]["tool_calls"]))) for c in choices
        ):
            raise ManagerError("malformed chat completion; model did not return a valid answer")
        return data

    def unload(self, name=None):
        """Unload one registered model, or every running model; llama-swap stays up."""
        if name is None:
            self.api("/api/models/unload", method="POST", allow_empty=True)
            return
        if name not in self.load_registry():
            raise ManagerError(f"unknown model: {name}")
        self.api("/api/models/unload/" + quote(name, safe=""), method="POST", allow_empty=True)

    @property
    def pi_settings_path(self):
        """pi's settings.json sits beside its models.json (the client_path)."""
        return self.settings.client_path.parent / "settings.json"

    def set_client_default(self, name):
        """Point pi's default provider/model at name; other keys are kept."""
        path = self.pi_settings_path
        data = json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(data, dict):
            raise ManagerError(f"{path} must contain a JSON object")
        data["defaultProvider"] = "llama-swap"
        data["defaultModel"] = name
        atomic_write(path, json.dumps(data, indent=2) + "\n")
        return path

    def use_and_sync(self, name, *, timeout=240):
        """Warm name; only after a valid answer, sync pi's models and default."""
        data = self.use(name, timeout=timeout)
        self.ensure_clients()
        self.set_client_default(name)
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
            engine = self.settings.engines.get(entry.get("engine") or "", {})
            if engine.get("kind") == "custom" and not entry.get("cmd"):
                context = _custom_ctx(entry, name)
            else:
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

    @property
    def backend(self):
        """The host's user-service backend; built lazily so pure use never needs one."""
        if self._backend is None:
            self._backend = backend_for(self.settings, self.runner)
        return self._backend

    def service(self, action):
        if action not in ("start", "stop", "restart"):
            raise ManagerError(f"unsupported service action: {action}")
        active = self._verify_service()
        if action in ("start", "restart") and not active:
            self._require_free_listen_address()
        self.backend.act(action, [self.settings.unit])

    def _require_free_listen_address(self):
        """Refuse to start when something else already answers on our listen address."""
        listen = urlsplit("http://" + self.settings.listen)
        host = {"0.0.0.0": "127.0.0.1", "::": "::1", "": "127.0.0.1"}.get(listen.hostname or "", listen.hostname)
        try:
            with socket.create_connection((host, listen.port), timeout=1):
                pass
        except OSError:
            return
        raise ManagerError(f"listen address {self.settings.listen} is in use by a process other than the managed "
                           f"service; stop it or change listen before starting")

    def _service_definition(self):
        executable = shutil.which(self.settings.swap_binary)
        if executable is None:
            raise ManagerError(f"binary not executable: {self.settings.swap_binary}")
        args = [str(Path(executable).absolute()), "-config", str(self.settings.output), "-listen", self.settings.listen]
        return args, self.backend.render(self.settings.unit, "llama-swap model server", args)

    def _companion_unit(self, name):
        return self.settings.companions[name].get("unit", f"llms-{name}.service")

    def _companion_definition(self, name):
        spec = self.settings.companions[name]
        command = list(spec["command"])
        executable = shutil.which(command[0])
        if executable is None:
            raise ManagerError(f"companion {name} binary not executable: {command[0]}")
        args = [str(Path(executable).absolute()), *command[1:]]
        unit = self._companion_unit(name)
        description = spec.get("description", f"llms companion: {name}")
        return unit, args, self.backend.render(unit, description, args)

    def _verify_service(self, *, check_api=False):
        """Fail closed unless source, loaded definition, and live process match.

        Only the unmodified generated definition is managed. Service-manager
        displays lose argv boundaries, so source equality and the live exact
        argv are also checked. No reload, link, or load is performed here.
        Operators must serialize unit/config changes.
        """
        args, text = self._service_definition()
        if check_api:
            api, listen = urlsplit(self.settings.swap_url), urlsplit("http://" + self.settings.listen)
            host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(listen.hostname, listen.hostname)
            if (api.scheme != "http" or not host or api.hostname != host
                    or api.port != listen.port or not listen.port
                    or api.path not in ("", "/") or api.query or api.fragment or api.username or api.password):
                raise ManagerError("managed API operation refused: swap_url must address the configured listen endpoint directly over HTTP")
        return self.backend.verify(self.settings.unit, args, text)

    def install_service(self):
        """Write a service definition only; never load, enable, or start it."""
        _, text = self._service_definition()
        path = self.backend.definition_path(self.settings.unit)
        atomic_write(path, text, overwrite=False)
        return path

    def install_companions(self):
        definitions = []
        for name in sorted(self.settings.companions):
            unit, _, text = self._companion_definition(name)
            definitions.append((self.backend.definition_path(unit), text))
        if not definitions:
            raise ManagerError("no companion services configured")
        existing = [str(path) for path, _ in definitions if path.exists()]
        if existing:
            raise FileExistsError("companion units already exist: " + ", ".join(existing))
        for path, text in definitions:
            atomic_write(path, text, overwrite=False)
        return [path for path, _ in definitions]

    def companion_service(self, action):
        if action not in ("start", "stop", "restart"):
            raise ManagerError(f"unsupported companion action: {action}")
        names = sorted(self.settings.companions, reverse=action == "stop")
        if not names:
            raise ManagerError("no companion services configured")
        units = []
        for name in names:
            unit, args, text = self._companion_definition(name)
            self.backend.verify(unit, args, text, verify_process=False)
            units.append(unit)
        self.backend.act(action, units, prefix="companion ")

    def companion_status(self):
        rows = []
        for name in sorted(self.settings.companions):
            spec = self.settings.companions[name]
            unit, args, text = self._companion_definition(name)
            active = self.backend.verify(unit, args, text, verify_process=False)
            healthy = False
            error = None
            if active:
                request = urllib.request.Request(spec["url"].rstrip("/") + spec.get("check_endpoint", "/health"))
                try:
                    with self.opener(request, timeout=10) as response:
                        healthy = 200 <= getattr(response, "status", 200) < 300
                except (OSError, urllib.error.URLError) as exc:
                    error = str(exc)
            rows.append({"name": name, "unit": self.backend.unit_name(unit), "url": spec["url"],
                         "active": active, "healthy": healthy, "error": error})
        return rows

    def doctor(self, *, live=False):
        """Return structured checks; no network unless live=True."""
        checks = []
        for module in ("yaml", "huggingface_hub", "gguf"):
            checks.append({"check": module, "ok": importlib.util.find_spec(module) is not None,
                           "required": module != "gguf", "detail": "gguf extra required for add" if module == "gguf" else "Python dependency"})
        # The top-level server is only rendered for chat entries with neither an
        # engine nor a cmd; a host whose models all name engines does not need it.
        try:
            uses_server = any(e.get("capability") == "chat" and not e.get("engine") and not e.get("cmd")
                              for e in self.load_registry().values())
        except (OSError, ValueError, ManagerError):
            uses_server = True
        for key in ("server", "swap_binary"):
            value = getattr(self.settings, key)
            required = key == "swap_binary" or uses_server
            detail = value if required else f"{value} (unused: every chat entry names an engine or cmd)"
            checks.append({"check": key, "ok": shutil.which(value) is not None, "required": required, "detail": detail})
        manager = self.settings.resolved_service_manager
        tool = self.settings.systemctl if manager == "systemd" else self.settings.launchctl
        checks.append({"check": "service_manager", "ok": shutil.which(tool) is not None, "required": False,
                       "detail": f"{manager} via {tool} (setting: {self.settings.service_manager})"})
        if manager == "launchd" and shutil.which(tool):
            try:
                available = self.backend.domain_available()
            except (OSError, subprocess.TimeoutExpired):
                available = False
            checks.append({"check": "launchd-domain", "ok": available, "required": False,
                           "detail": f"{self.backend.domain()} " + ("is available" if available else
                                     "is missing; agents need a console (GUI) login session, not SSH-only")})
        if manager == "launchd":
            # launchd agents get only service_path; a bare name that resolves in
            # this shell may not resolve for llama-swap under launchd.
            search = self.backend.service_path()
            bare = {"server": self.settings.server} if uses_server and "/" not in self.settings.server else {}
            bare.update({f"engine:{name}": spec["server"] for name, spec in self.settings.engines.items()
                         if "/" not in spec["server"] and not spec.get("cwd")})
            for check, name in sorted(bare.items()):
                found = shutil.which(name, path=search)
                checks.append({"check": f"service-path:{check}", "ok": found is not None, "required": True,
                               "detail": f"{name} -> {found}" if found else f"{name} not found on launchd PATH {search}"})
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
        # An engine binary is host state and moves independently of the
        # registry, so report each one rather than failing the whole render.
        for engine_name, engine in sorted(self.settings.engines.items()):
            server = engine["server"]
            cwd = engine.get("cwd")
            candidate = Path(cwd) / server if cwd and server.startswith(".") else Path(server)
            found = candidate.is_file() and os.access(candidate, os.X_OK) if (
                "/" in server) else shutil.which(server) is not None
            checks.append({"check": f"engine:{engine_name}", "ok": bool(found), "required": False,
                           "detail": f"{server}{'' if not cwd else f' (cwd {cwd})'}"})
        if any(engine.get("cwd") for engine in self.settings.engines.values()):
            # Rendered cwd commands rely on `env -C`; GNU coreutils < 8.28 lacks it.
            try:
                result = self.runner(["/usr/bin/env", "-C", "/", "true"], capture_output=True, text=True, timeout=5)
                ok, detail = result.returncode == 0, (result.stderr or "").strip() or "/usr/bin/env -C supported"
            except (OSError, subprocess.TimeoutExpired) as exc:
                ok, detail = False, str(exc)
            checks.append({"check": "env-chdir", "ok": ok, "required": True, "detail": detail})
        for name, companion in sorted(self.settings.companions.items()):
            executable = companion["command"][0]
            found = Path(executable).is_file() and os.access(executable, os.X_OK) if (
                "/" in executable) else shutil.which(executable) is not None
            checks.append({"check": f"companion:{name}", "ok": bool(found), "required": True,
                           "detail": f"{executable} -> {companion['url']}"})
        if live:
            try:
                self.status()
                checks.append({"check": "live", "ok": True, "required": True, "detail": self.settings.swap_url})
            except ManagerError as exc:
                checks.append({"check": "live", "ok": False, "required": True, "detail": str(exc)})
            try:
                for row in self.companion_status():
                    checks.append({"check": f"companion-live:{row['name']}",
                                   "ok": row["active"] and row["healthy"], "required": True,
                                   "detail": row["error"] or row["url"]})
            except ManagerError as exc:
                checks.append({"check": "companions-live", "ok": False, "required": True, "detail": str(exc)})
        return checks
