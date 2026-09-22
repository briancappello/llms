# llms

`llms` is a Python library and command-line utility for GGUF model
registries. It downloads or adopts models, reads GGUF metadata, writes a
`llama-swap` configuration, and controls one verified user service.

The package does not start a service during installation. It does not change
client files unless you use `ensure-clients` or `--sync-clients`.

## Requirements

- Python 3.10 or newer
- `llama-swap`
- A compatible `llama-server`
- systemd user services for the service commands
- The `gguf` package extra for `llms add`

## Install

Install the package from a checkout:

```bash
uv tool install --with gguf .
```

Create the configuration. The command does not overwrite files.

```bash
llms init --server /path/to/llama-server
llms render
llms doctor
```

Install the user service after you inspect the generated configuration:

```bash
llms install-service
systemctl --user daemon-reload
llms start
llms doctor --live
```

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
llms status
llms use short-name
llms rm short-name
```

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
entry's `extra_args`. `cwd` is rendered with `env --chdir`, so no shell sits
between llama-swap and the server it signals.

A model is rendered by `cmd` if present, otherwise by `engine`, otherwise
from the `${server}` macro in the header. Setting both `cmd` and `engine` is
an error rather than a silent precedence rule.

## Building engines

`bin/build-engine` builds a llama.cpp variant and installs it under
`~/opt/llama.cpp-<name>`:

```bash
make engine NAME=vulkan BACKEND=vulkan
make engine NAME=hip    BACKEND=hip
make engine NAME=mtp    BACKEND=hip REF=pr/28097
make engine NAME=bonsai BACKEND=hip SRC=~/dev/bonsai-llama.cpp
make engine NAME=cuda   BACKEND=cuda ARCH=90
```

Every build is installed with `RPATH=$ORIGIN/../lib` and then checked: if
`llama-server` resolves any `libggml`/`libllama` outside its own prefix the
build fails. Without that, variants silently share one library, which
invalidates backend comparisons and, for a fork with its own quantization
types, produces a binary that either refuses the weights or misreads them.

### vLLM on ROCm

The vLLM build is independently pinned in `config/vllm-build.env`. It clones
the selected revision to `~/dev/vllm`, builds a private OpenMPI 4 compatibility
runtime, and compiles vLLM for the configured AMD GPU architecture:

```bash
make vllm
make vllm-reranker
```

The second command serves `BAAI/bge-reranker-v2-m3` at
`http://127.0.0.1:8010`. Edit the pin file to upgrade the source revision or
ROCm dependency set; the resolved build is recorded in `~/opt/vllm/BUILD-INFO`.

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
`$XDG_CONFIG_HOME/llama-swap/api-keys`, with one key on each line.

## Reference Data

This repository started as the operational toolkit on a host named `taichi`.
The files in `config/`, `MODELS.md`, `bench/`, and `docs/` preserve that model
research and measurement harness. The installed package does not read the
repository configuration by default.

The reference registry contains absolute paths and host-specific backend
commands. Do not deploy it on another host without an explicit migration.

## Development

Run the Python and Go checks:

```bash
make test
make build
```

Build artifacts go to `dist/`.

## License

This project uses the MIT License. See `LICENSE`.
