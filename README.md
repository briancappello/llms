# llms

`llms` is a Python library and command-line utility for GGUF model
registries. It downloads or adopts models, reads GGUF metadata, writes a
`llama-swap` configuration, and controls one verified user service: a
systemd user unit on Linux, a launchd agent on macOS. Windows is not
supported.

The package does not start a service during installation. It does not change
client files unless you use `ensure-clients` or `--sync-clients`.

## Requirements

- Python 3.10 or newer, and `uv`
- `llama-swap` and a compatible `llama-server` (or another engine, see below)
- The `gguf` package extra for `llms add`
- Linux: systemd user services; GNU coreutils 8.28 or newer (for `env -C`)
- macOS (Apple Silicon): launchd, which is always present. To build engines:
  `xcode-select --install` and `brew install cmake ninja`.

## Install

Install the package from a checkout:

```bash
make install            # uv tool install --force --with gguf .
```

Install `llama-swap` on either platform from its
[releases](https://github.com/mostlygeek/llama-swap/releases) and put it on
`PATH`, e.g. `~/.local/bin`, which is on the launchd agent's default
`service_path`. Upstream also documents a Homebrew tap
(`mostlygeek/llama-swap`); it is not a core formula.

Create the configuration. The command does not overwrite files.

```bash
llms init --server /path/to/llama-server
llms render
llms doctor
```

### Service on Linux (systemd)

```bash
llms install-service                # writes the unit; never enables or starts it
systemctl --user daemon-reload
llms start
llms doctor --live
```

### Service on macOS (launchd)

```bash
llms install-service                # writes <config>/launchd/llms.llama-swap.plist; loads nothing
llms start                          # launchctl bootstrap gui/<uid> <plist>
llms doctor --live
```

`llms stop` unloads the job (`launchctl bootout`), so it stays down.
`llms restart` replaces the process (`launchctl kickstart -k`). launchd
restarts it after a crash, but not after a clean exit. Installing never starts
anything at login. To start at login, copy the plist into
`~/Library/LaunchAgents/`, as `install-service` prints.

launchd agents get no login-shell `PATH`, so the plist carries a fixed one
from `service_path` in `settings.json`. `llms doctor` checks every engine
binary given by bare name against it.

Output goes to `~/Library/Logs/llms/<name>.log`. launchd does not rotate it.
To rotate it with the system's `newsyslog`, add a file to `/etc/newsyslog.d/`,
for example:

```text
# /etc/newsyslog.d/llms.conf
/Users/you/Library/Logs/llms/*.log  you:staff  644  5  10240  *  NJ
```

### Both platforms

```bash
llms logs [-f] [-n N] [--companion NAME]   # journal on systemd, log file on launchd
llms status
```

Every start, stop, and restart first verifies three things. The definition on
disk must be the unmodified one `llms` generated. The loaded unit or job must
come from that file. The live process's exact argv and executable must match
the generated command: `/proc` is read on Linux, `KERN_PROCARGS2` and
`proc_pidpath` on macOS. If anything differs, or cannot be read, the command
refuses and changes nothing. A crash-looping launchd job in `spawn scheduled`
also counts as an unstable state; recover it with
`launchctl bootout gui/$(id -u)/llms.llama-swap`.

## Model Commands

Download and register a model:

```bash
llms add owner/repository:Q4_K_M
```

Register a GGUF file that is already on disk:

```bash
llms add short-name /path/to/model.gguf
```

Use `--no-restart` to change the registry without a service restart. Use
`--no-load` to omit the explicit warm-up request after a restart.

```bash
llms ls
llms ls --unregistered        # complete cached GGUF/MLX models no entry references
llms status
llms use short-name
llms use short-name --sync-clients   # also make it pi's default model
llms unload [short-name]      # free memory; llama-swap stays up
llms rm short-name
```

`llms add` records two things from the GGUF metadata. If the GGUF carries MTP
layers, the entry gets `"mtp": true`: an intent, not a flag. Each host's engine
renders it (see `mtp_args` below). Any embedded `general.sampling.*` values are
reported and stored as `embedded_sampling`. llama.cpp applies those unless a
flag overrides them, so `add` shows them, but it never turns them into flags.

The package does not delete model weights. The `--purge` option fails before it
changes files because a shared Hugging Face cache can have unknown consumers.

## Configuration

The default configuration directory is
`$XDG_CONFIG_HOME/llms`. The fallback is
`~/.config/llms`.

The directory contains these files:

```text
settings.json       Host paths, engines, endpoint, service, and GPU budget
registry.json       Model IDs, files, contexts, and model arguments
config.header.yaml  Shared llama-swap configuration and macros
```

None of these are in this repository. They hold absolute paths into one
machine's model cache and one machine's binaries. `config/` carries
`settings.example.json`, `registry.example.json` and
`config.header.example.yaml`, which document every field.

## Engines

An engine names a local server process: a binary, its environment, its
working directory, and its fixed arguments. Engines are defined in
`settings.json`; a registry entry selects one by logical name.

```json
"engines": {
  "llama-hip": {
    "server": "/home/you/opt/llama.cpp-hip/bin/llama-server",
    "env": { "HIP_VISIBLE_DEVICES": "0" },
    "args": ["-ngl", "999"]
  }
}
```

```json
"my-model": { "engine": "llama-hip", "path": "...", "ctx": 65536 }
```

This is the split that makes one registry usable on several machines: the
model tuning is portable and only `settings.json` changes per host. It is
also where device pins belong, because they are backend-specific --
`MESA_VK_DEVICE_SELECT` does nothing for a HIP build, and
`HIP_VISIBLE_DEVICES` does nothing for a Vulkan one. A pin set on the
systemd unit applies to every backend indiscriminately.

`kind: "custom"` marks an engine that is not llama.cpp, so no
`-m`/`-c`/`-np` are composed for it; it receives its own arguments plus the
entry's `extra_args`. A custom-engine entry must declare a measured `ctx`.
Such servers may enforce no context limit of their own, so the
context window clients are told is the only limit. `cwd` is rendered with
`env -C`, so no shell sits between llama-swap and the server it signals.

`mtp_args` (llama.cpp engines) is what an entry's `"mtp": true` renders to on
this host. The default is `--spec-type draft-mtp`. The best draft length
differs by hardware, so it belongs to the engine:

```json
"metal": {
  "server": "/Users/you/opt/llama.cpp-metal/bin/llama-server",
  "args": ["-ngl", "999"],
  "mtp_args": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
}
```

MLX models are served by [oMLX](https://github.com/jundot/omlx) as a custom
engine. Install the app from its `.dmg`, which ships precompiled Qwen3.5
kernels; source and Homebrew builds need full Xcode for them. On an M3 Max,
oMLX matched llama.cpp on decode and was about 1.4x faster at 16k prefill (cold), while
`mlx_vlm` serving the same weights was 3-5x slower
(`results/speed/mac-engine-compare.md`). `mlx_lm` mis-loads vision-language
checkpoints and emits garbage for them.

oMLX gets a private data directory, so its per-model settings are separate
from the menu-bar app's `~/.omlx`:

```json
"omlx": {"kind": "custom", "server": "/Applications/oMLX.app/Contents/MacOS/omlx-cli",
         "check_endpoint": "/health",
         "args": ["serve", "--base-path", "/Users/you/.config/llms/omlx",
                  "--model-dir", "/Users/you/.config/llms/omlx/models", "--no-hf-cache",
                  "--paged-ssd-cache-max-size", "10GB", "--max-concurrent-requests", "1"]}
```

```json
"my-mlx-model": {"engine": "omlx", "capability": "chat", "ctx": 131072, "useModelName": "my-mlx-model"}
```

oMLX names models after the subdirectories of `--model-dir`. Make each one a
symlink to its HF snapshot, e.g. `~/.config/llms/omlx/models/my-mlx-model ->
~/.cache/huggingface/hub/models--o--r/snapshots/<sha>`. `useModelName` sends
that name upstream. Per-model settings live in
`<base-path>/model_settings.json`:
`{"version": 1, "models": {"my-mlx-model": {"mtp_enabled": false, "max_context_window": 131072}}}`.
Keep `max_context_window` equal to the entry's `ctx`.

A model is rendered by `cmd` if present, otherwise by `engine`, otherwise
from the `${server}` macro in the header. Setting both `cmd` and `engine` is
an error rather than a silent precedence rule.

## Building engines

`bin/build-engine` builds a llama.cpp variant and installs it under
`~/opt/llama.cpp-<name>`. Backends are `metal` and `cpu` on macOS, and
`vulkan`, `hip`, `hip-wmma`, `cuda` and `cpu` on Linux. An unsupported one
fails before any source is touched. The script runs under macOS's stock
`/bin/bash` 3.2.

```bash
make engine NAME=metal  BACKEND=metal
make engine NAME=vulkan BACKEND=vulkan
make engine NAME=hip    BACKEND=hip
make engine NAME=mtp    BACKEND=hip REF=pr/28097
make engine NAME=bonsai BACKEND=hip SRC=~/dev/bonsai-llama.cpp
make engine NAME=cuda   BACKEND=cuda ARCH=90
```

Every build is installed with an rpath relative to the binary
(`$ORIGIN/../lib` on Linux, `@loader_path/../lib` on macOS) and then checked.
Linux uses `ldd`. macOS resolves each image's load commands against its
`LC_RPATH`s with `otool`. If `llama-server` or any library in the prefix
resolves a `libggml`/`libllama` outside the prefix, the build fails.
`build-engine NAME BACKEND --check-only` re-runs just that check. Without that, variants silently share one library, which
invalidates backend comparisons and, for a fork with its own quantization
types, produces a binary that either refuses the weights or misreads them.

### vLLM on ROCm (Linux only)

The vLLM build and its reranker/embedding servers are ROCm-only. On other
hosts they exit immediately, except `bin/build-vllm --print-config` and
`--dry-run`. The vLLM build is independently pinned in `config/vllm-build.env`. It clones
the selected revision to `~/dev/vllm`, builds a private OpenMPI 4 compatibility
runtime, and compiles vLLM for the configured AMD GPU architecture:

```bash
make vllm
make vllm-reranker
make vllm-embeddings
```

The second command serves `BAAI/bge-reranker-v2-m3` at
`http://127.0.0.1:8010`. Edit the pin file to upgrade the source revision or
ROCm dependency set; the resolved build is recorded in `~/opt/vllm/BUILD-INFO`.

Persistent reranking and embedding servers can be configured as `companions`
in `settings.json` (see `config/settings.example.json`). Companions are plain
commands, so on macOS they can be, for example, `llama-server --embeddings`.
They are managed through the same backend as llama-swap:

```bash
llms install-companions            # prints the systemd reload / launchd load steps
llms companions-start
llms companions-status
# Linux, start at login:
systemctl --user enable llms-reranker.service llms-embeddings.service
# macOS, start at login: copy the printed plists into ~/Library/LaunchAgents/
```

On the 32 GiB R9700, the verified resident stack is Bonsai 2 27B chat through
llama-swap, BGE reranking on `:8010`, and Voyage embeddings on `:8011`. The two
pooling models use eager execution to preserve VRAM for the chat context.

```bash
curl http://127.0.0.1:8010/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"BAAI/bge-reranker-v2-m3","query":"capital of France","documents":["Paris is in France.","Berlin is in Germany."]}'
```

The generated file is `$XDG_CONFIG_HOME/llama-swap/config.yaml`.
The default local endpoint is `http://127.0.0.1:18080`.

Use `llms --config-dir DIR ...` for an isolated instance. `LLMS_*` environment
variables override fields from `settings.json`. `HF_TOKEN` is read from the
environment and is never stored by this package.

Startup preload is off by default. Set `preload` to `true` in `settings.json`
to load the default chat model during service startup. An explicit preload list
in `config.header.yaml` has priority.

## Python API

The library has no import-time file, network, or service operations.

```python
from pathlib import Path

from llms import ModelManager, Settings

settings = Settings(
    config_dir=Path("/srv/llms"),
    server="/opt/llama.cpp/bin/llama-server",
    gpu_memory_mib=24_576,
)
manager = ModelManager(settings)
manager.init()
manager.add("/models/example.gguf", name="example")
manager.render()
```

Library mutations do not restart services or change client files by default.
Callers must serialize registry writers.

## Authentication Proxy

`proxy/` contains an optional Go reverse proxy for LAN access. It accepts a
Bearer token or `x-api-key`, then removes credentials before it forwards the
request to a keyless loopback service.

```bash
make proxy
LSA_LISTEN=0.0.0.0:4096 ~/.local/bin/llama-swap-auth
```

The proxy binds to `127.0.0.1:4096` by default. Keys come from
`$XDG_CONFIG_HOME/llama-swap/api-keys`, falling back to
`~/.config/llama-swap/api-keys` on both Linux and macOS, with one key on each
line. `LSA_KEYS_FILE` overrides the path.

## Reference Data

This repository started as the operational toolkit on a host named `taichi`.
The files in `config/`, `MODELS.md`, `bench/`, and `docs/` preserve that model
research and measurement harness. The installed package does not read the
repository configuration by default.

The reference registry contains absolute paths and host-specific backend
commands. Do not deploy it on another host without an explicit migration.

## Benchmarks

Benchmarks measure the model as it is served. `bench/lib/stack.py` runs an
isolated `llms` instance built from this host's real settings, engines, header
and registry. It applies one declared `--variant` merge-patch and runs
llama-swap on its own port. Everything is measured over HTTP: completion
`timings`, `/upstream/<model>/tokenize`, and the served process's memory.

- Memory is device VRAM on Linux. On macOS it is the process footprint, which
  excludes the mmap'd weights; see `results/context/mac-footprint-validation.md`.
- Memory is always sampled while the server generates.
- The production service is stopped and restored around each run. Production
  files are checksummed and must be unchanged afterwards.
- `llama-bench` and `llama-fit-params` are not result sources.

See `MODELS.md`, "Running the benchmarks". Stop a backgrounded run with
`kill -TERM`, because shells start background jobs with SIGINT ignored.

## Development

CI runs the checks on Ubuntu and macOS. Locally:

```bash
make test
make build
```

Build artifacts go to `dist/`.

## License

This project uses the MIT License. See `LICENSE`.
