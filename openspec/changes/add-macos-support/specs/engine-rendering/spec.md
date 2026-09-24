# Spec Delta

## Purpose

Turn portable registry entries and host-specific settings engines into
llama-swap model commands that run the same way on Linux and macOS. This covers
llama.cpp engines, MTP intent, and custom non-llama.cpp servers such as MLX.

## ADDED Requirements

### Requirement: Portable working directory
When an engine sets `cwd`, the rendered command SHALL change into that
directory with `/usr/bin/env -C <dir>` and then exec the server directly. No
shell is involved. This form SHALL work with macOS `env` and with GNU coreutils
`env` 8.28 or newer.

#### Scenario: Custom engine with cwd on macOS
- **WHEN** an engine has `"cwd": "/srv/ds4"` and the config is rendered on macOS
- **THEN** the command starts with `/usr/bin/env -C /srv/ds4`, and running it starts the server in `/srv/ds4`

#### Scenario: Same render on Linux
- **WHEN** the same registry and engine are rendered on Linux
- **THEN** the command is byte-identical to the macOS render

### Requirement: MTP intent rendered by the engine
A registry entry SHALL be able to declare `"mtp": true`. For a llama.cpp
engine, the rendered command SHALL include that engine's `mtp_args`. If the
engine does not set `mtp_args`, the default SHALL be
`["--spec-type", "draft-mtp"]`. An entry with `mtp` and a `cmd` override, or
with `mtp` on a `custom` engine, SHALL be a render error. Entries that put
speculative flags directly in `extra_args` SHALL render unchanged.

#### Scenario: Per-host draft length
- **WHEN** the macOS engine sets `"mtp_args": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"]`, the Linux engine leaves `mtp_args` unset, and both hosts share one registry entry with `"mtp": true`
- **THEN** each host renders its own speculative arguments, and the registry entry is identical on both hosts

#### Scenario: Legacy extra_args
- **WHEN** an entry has `"extra_args": ["--spec-type draft-mtp"]` and no `mtp` key
- **THEN** the rendered command matches the render from before this change

### Requirement: Custom engines declare a context window
A registry entry that uses a `custom` engine SHALL declare a positive integer
`ctx`. Custom servers (for example MLX servers) may not enforce a context limit,
so the client context window is the only safeguard. Rendering and
`ensure-clients` SHALL fail when `ctx` is missing, rather than falling back to
`trained_ctx` or 4096.

#### Scenario: MLX entry without ctx
- **WHEN** a registry entry uses an engine with `"kind": "custom"` and has no `ctx`
- **THEN** `llms render` fails with an error naming the entry and asking for a measured `ctx`

#### Scenario: MLX entry with ctx
- **WHEN** the entry declares `"ctx": 65536`
- **THEN** `ensure-clients` writes `contextWindow` 65536 for that model
