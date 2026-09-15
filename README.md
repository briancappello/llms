# llama-swap-manager

`llama-swap-manager` is a Python library and command-line utility for GGUF model
registries. It downloads or adopts models, reads GGUF metadata, writes a
`llama-swap` configuration, and controls one verified user service.

The package does not start a service during installation. It does not change
client files unless you use `ensure-clients` or `--sync-clients`.

## Requirements

- Python 3.10 or newer
- `llama-swap`
- A compatible `llama-server`
- systemd user services for the service commands
- The `gguf` package extra for `llm add`

## Install

Install the package from a checkout:

```bash
uv tool install --with gguf .
```

Create the configuration. The command does not overwrite files.

```bash
llm init --server /path/to/llama-server
llm render
llm doctor
```

Install the user service after you inspect the generated configuration:

```bash
llm install-service
systemctl --user daemon-reload
llm start
llm doctor --live
```

## Model Commands

Download and register a model:

```bash
llm add owner/repository:Q4_K_M
```

Register a GGUF file that is already on disk:

```bash
llm add short-name /path/to/model.gguf
```

Use `--no-restart` to change the registry without a service restart. Use
`--no-load` to omit the explicit warm-up request after a restart.

```bash
llm ls
llm status
llm use short-name
llm rm short-name
```

The package does not delete model weights. The `--purge` option fails before it
changes files because a shared Hugging Face cache can have unknown consumers.

## Configuration

The default configuration directory is
`$XDG_CONFIG_HOME/llama-swap-manager`. The fallback is
`~/.config/llama-swap-manager`.

The directory contains these files:

```text
settings.json       Host paths, endpoint, service, and GPU budget
registry.json       Model IDs, files, contexts, and model arguments
config.header.yaml  Shared llama-swap configuration and macros
```

The generated file is `$XDG_CONFIG_HOME/llama-swap/config.yaml`.
The default local endpoint is `http://127.0.0.1:18080`.

Use `llm --config-dir DIR ...` for an isolated instance. `LLM_*` environment
variables override fields from `settings.json`. `HF_TOKEN` is read from the
environment and is never stored by this package.

Startup preload is off by default. Set `preload` to `true` in `settings.json`
to load the default chat model during service startup. An explicit preload list
in `config.header.yaml` has priority.

## Python API

The library has no import-time file, network, or service operations.

```python
from pathlib import Path

from llama_swap_manager import ModelManager, Settings

settings = Settings(
    config_dir=Path("/srv/llama-swap-manager"),
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
