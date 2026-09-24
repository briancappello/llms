# Proposal

## Why

`llms` began on one Linux ROCm host. Its service layer supports only systemd,
several scripts rely on GNU and bash-4 behavior, and the benchmark harness has
hard-coded AMD tooling. It does not run on the Apple Silicon Mac that
`../llms.bak` served from. Linux and macOS must both work as first-class hosts
from one registry. Windows is out of scope. The unit tests pass on macOS today
only because systemd and `/proc` are mocked, so a green test run proves nothing
about the Mac host.

## What Changes

- **Service management becomes pluggable**:
  - The existing systemd backend is kept.
  - A launchd backend is added for macOS, with the same fail-closed checks of
    the definition on disk, the loaded job and the live process. The OS
    selects the backend unless settings override it.
  - Live process verification no longer depends on `/proc`.
  - Services get log files and `llms logs`.
- **Engine command rendering becomes portable**: the working-directory prefix
  changes from `env --chdir=DIR` (GNU-only) to `env -C DIR`, which works on
  both platforms.
- **Engine builds**:
  - `bin/build-engine` works on macOS: Metal backend, `@loader_path` rpath,
    `otool` library-isolation check, CPU count without `nproc`, portable dates,
    and safety under stock bash 3.2.
  - `cpu` builds disable Metal explicitly.
  - The vLLM ROCm scripts fail early and clearly on non-Linux hosts.
- **Auth proxy**: the default key file follows `$XDG_CONFIG_HOME` or
  `~/.config` on both platforms. It no longer uses Go's `os.UserConfigDir`,
  which points to `~/Library/Application Support` on macOS.
- **Benchmarks go through the whole serving stack**:
  - Scripts stop launching `llama-server` with copied flags.
  - A bench helper runs an isolated `llms` instance built from the real
    registry, plus one declared change per variant, behind llama-swap. All
    measurements are made over HTTP.
  - Server memory is read per platform: dGPU VRAM on Linux, process footprint
    on macOS unified memory.
  - Service hand-off goes through `llms stop/start`.
  - `llama-bench` and `llama-fit-params` are not measurement sources.
- **Model registration** (items taken from `../llms.bak`):
  - MTP scaffolding records intent (`"mtp": true`). Each engine renders it with
    its own speculative arguments, so the draft length can differ per host.
  - The GGUF's embedded `general.sampling.*` values are reported, because
    llama.cpp applies them silently.
  - Complete but unregistered GGUF and MLX models in the HF cache can be listed.
- **Serving controls and clients** (taken from `../llms.bak`):
  - `llms unload [name]`.
  - A port-in-use check before start.
  - `use --sync-clients` sets pi's default model.
  - `use` accepts the MLX `reasoning` field.
  - Custom engines (for example oMLX) must declare a measured `ctx`,
    because they have no server-side context limit.
- **Docs and CI**: README requirements for both platforms, example Metal and MLX
  engines, and tests run on Linux and macOS.

**BREAKING**:
- Rendered custom-`cwd` commands change from `/usr/bin/env --chdir=DIR` to
  `/usr/bin/env -C DIR`. This needs GNU coreutils 8.28 or newer on Linux.
- New `llms add` MTP entries use the `mtp` key instead of a literal
  `--spec-type draft-mtp` in `extra_args`. Existing entries keep working
  unchanged.
- Registry entries on a `custom` engine must now declare a positive integer
  `ctx`. Rendering and `ensure-clients` refuse entries without one; before,
  clients silently got `trained_ctx` or 4096. On the Linux host, the
  `ds4-hip` entry (or any other custom-engine entry) needs a `ctx` before
  upgrading.

## Capabilities

### New Capabilities
- `service-management`: installing, verifying, starting, stopping, restarting,
  and logging the llama-swap service and companion services under systemd
  (Linux) or launchd (macOS).
- `engine-rendering`: how registry entries and settings engines become
  llama-swap commands on both platforms, including the working directory, MTP
  intent, and custom (non-llama.cpp) engines such as MLX.
- `engine-builds`: building isolated llama.cpp engine variants on Linux and
  macOS, and scoping the vLLM ROCm build to Linux.
- `auth-proxy`: key file location and behavior of the LAN authentication proxy.
- `stack-benchmarks`: benchmarking models through the production serving stack
  in an isolated instance, with portable memory measurement.
- `model-registration`: `llms add` scaffolding (MTP intent, reporting sampling
  metadata) and discovering unregistered cached models.
- `serving-controls`: runtime controls and client sync: unload, port preflight,
  warm-up answer validation, and the pi default model.

### Modified Capabilities
- None. `openspec/specs/` is empty; this change sets the first baseline.

## Impact

- **Code**:
  - `src/llms/manager.py`: service methods are extracted; rendering, `add`,
    `use`, `doctor` and `ensure_clients` change.
  - `src/llms/settings.py`: new `service_manager` and `launchctl` fields;
    `unit` handling; engine `mtp_args`.
  - `src/llms/cli.py`: new commands and platform-specific hints.
  - New modules: `src/llms/services/`, `src/llms/_proc.py`.
- **Scripts**: `bin/build-engine`, `bin/build-vllm`, `bin/run-vllm-*`, every
  script in `bench/**`, and a new bench helper in `bench/lib/`.
- **Other**: `proxy/main.go`, `Makefile`, `README.md`, `config/*.example.*`,
  `tests/`, and new CI configuration.
- **Dependencies**: no new Python runtime dependency. The macOS process checks
  use `ctypes` against libc/libproc. Mac builds need `cmake` and `ninja` (via
  Homebrew). llama-swap is installed with `brew install llama-swap`.
- **Hosts**: the Linux host needs no settings changes. Existing systemd units
  stay valid, but llama-swap config is re-rendered because of the `env -C`
  change.
