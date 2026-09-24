# Design

## Context

See proposal.md (Why). The current state that shapes this design:

- **Service code.** Service handling lives in `ModelManager`:
  `_service_definition`, `_companion_definition`, `_verify_unit`,
  `_verify_service`, `service`, `companion_service`, `install_*`.
  - It writes systemd text by hand.
  - It parses `systemctl --user show`.
  - It checks the live process through `/proc/<pid>/cmdline` and `exe`.
  - The `runner` and `opener` are injected, so tests mock every external call.
    This is why the suite passes on macOS today.
- **Settings.** `Settings` hard-codes `systemctl`, requires `unit` to end in
  `.service`, and defaults `service_dir` to `$XDG_CONFIG_HOME/systemd/user`.
- **Engine working directory.** Engine `cwd` renders as
  `/usr/bin/env --chdir=DIR`. macOS `env` rejects this (checked on this
  machine). Both BSD and GNU `env` accept `-C DIR`.
- **Stock bash.** macOS `/bin/bash` is 3.2. Under `set -u`, `"${arr[@]}"` on an
  empty array aborts (checked). `printf '%(...)T'` needs bash 4.2 or newer.
  `date -Is`, `nproc`, `ldd`, `stat -c`, `sha256sum` and `xargs -r` are
  GNU-only or missing.
- **Benchmarks.**
  - `kv-probe.py` and `coding-profile.py` already go through `llms use`,
    `/running` and `/upstream/<m>/tokenize`.
  - The shell drivers do not: `backend-sweep`, `ctx-probe`, `ctx-gap`,
    `soak-ctx`, `gates`, `mtp-engage`, `depth-sweep` and `kv-sweep` start
    `llama-server` themselves, with hand-copied flags, `~/opt/...` paths, HF
    cache globs, `rocm-smi` and `systemctl`.
- **MTP measurements.** The Linux host's results show MTP is a TG win there.
  The `llms.bak` measurement on the M3 Max, taken through a live
  `llama-server`, showed `draft-mtp` at the default `n-max` 3 about 40% slower
  on a 256-expert MoE, and break-even at `n-max` 1. The right speculative
  arguments therefore depend on the host.

## Goals / Non-Goals

**Goals:**
- One registry works on both hosts. Only `settings.json` differs.
- The macOS service path is exactly as strict as the systemd path: the
  definition on disk, the loaded job and the live argv are all checked, and any
  doubt refuses the action.
- The existing Linux host keeps working. Its only visible change is a
  re-render.
- Tests for both service backends run on either OS.

**Non-Goals:**
- Windows, Intel Macs, or Linux distributions without systemd.
- System-wide services (LaunchDaemons, `systemctl --system`).
- Porting vLLM/ROCm to macOS. On the Mac, companions are plain commands, for
  example `llama-server --embeddings` or MLX.
- Re-running the existing benchmark results. They stay valid as Linux
  measurements.
- Log rotation beyond what the OS provides.

## Decisions

### D1. Service backends behind one interface, chosen once in Settings
- **Layout.** Add `src/llms/services/` with a small interface: `definition`,
  `install`, `verify(verify_process)`, `act(action, names)`, `logs`, and
  `activation_hint`. Two implementations: `systemd.py` and `launchd.py`.
- **Construction.** `ModelManager` builds the backend from
  `settings.service_manager` after resolving it, and keeps the injected
  `runner`. The platform is injectable, so the launchd tests run on Linux CI
  and the systemd tests run on macOS.
- **Order of work.** First extract the systemd code with no behavior change,
  so that the existing 77 tests still pass. Then add launchd.
- **Alternative rejected:** `if sys.platform` branches inside `ModelManager`.
  That doubles the complexity of the most safety-critical code and can't be
  tested on the other OS.

### D2. launchd lifecycle: loading means running; login start means copying the plist
- **Starting.** A plist with `KeepAlive = {SuccessfulExit: false}` implies
  `RunAtLoad`: loading the job starts it. So:
  - `start` = `launchctl bootstrap gui/<uid> <plist>`
  - `stop` = `launchctl bootout gui/<uid>/<label>`
  - `restart` = `launchctl kickstart -k gui/<uid>/<label>`
- **Stopping.** `bootout` does not respawn the job. launchd also kills the
  job's process group, which catches any orphaned `llama-server` child.
- **Install location.** A plist in `~/Library/LaunchAgents` would start at
  every login. To keep "install never enables", the default launchd
  `service_dir` is `<config_dir>/launchd/`. Starting at login is the explicit,
  printed step "copy the plist into `~/Library/LaunchAgents`". This mirrors
  today's systemd split between `link` and `enable`.
- **Verification.** The loaded `path` must be one of those two locations, and
  the file there must be byte-identical to the generated plist.
- **Naming.** The label is `llms.<name>`, where `<name>` is `unit` with any
  `.service` suffix removed. Companions follow the same rule.
- **Alternative rejected:** `RunAtLoad=false` and a plain `KeepAlive`. The
  plain form respawns after `launchctl kill`, so `stop` would not stop.

### D3. Plist generated with `plistlib`; environment fixed by settings
- **Generation.** `plistlib.dumps` writes the plist, so no quoting code is
  needed. The keys are `Label`, `ProgramArguments`, `KeepAlive`,
  `StandardOutPath` and `StandardErrorPath`
  (`~/Library/Logs/llms/<name>.log`), `EnvironmentVariables.PATH`, and
  `ProcessType = Interactive`.
- **PATH.** `PATH` comes from a new `service_path` setting. Its default is
  fixed: `~/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin`.
  It is not copied from the caller's environment, because a PATH that changes
  from shell to shell would make the regenerated plist differ and every
  verification fail. `doctor` checks bare engine and companion names against
  this PATH.
- **ProcessType.** `Interactive` keeps macOS from throttling a background GPU
  process. Whether this matters is verified in the benchmark phase.
- **systemd.** The unit text is unchanged.

### D4. Reading launchd state: `launchctl print`, parsed strictly
- **Source.** The backend runs `launchctl print gui/<uid>/<label>` and reads
  only `path`, `state`, `pid`, `program` and the `arguments = { ... }` block.
- **Failure handling.** Apple documents this output as unstable. The parser
  therefore fails closed: a missing key, a duplicate key or an unexpected
  structure raises "cannot verify", and nothing is done.
- **Not found.** "Could not find service" means the job is not loaded
  (inactive). For `start`, that is a valid state, and it then loads the
  verified plist.
- **Alternative considered:** `launchctl list <label>`, which gives a
  dictionary with PID and ProgramArguments. It is equally unofficial and lacks
  the plist path, which the check against the on-disk definition needs. It is
  kept as a possible fallback if `print` changes.

### D5. Live process check without `/proc`
- **Module.** A new `src/llms/_proc.py` exposes `argv(pid)` and `exe(pid)`.
  - Linux keeps the current `/proc` reads.
  - macOS uses `ctypes`: `sysctl({CTL_KERN, KERN_PROCARGS2, pid})` returns
    argc, the exec path and NUL-separated argv, so argument boundaries are
    exact. `libproc.proc_pidpath` returns the executable path.
  - Any error raises `OSError`, and the existing handling turns that into
    "cannot verify".
- **Alternatives rejected:**
  - `psutil`: a new compiled dependency, for about 50 lines of code.
  - `ps -o command=`: loses argument boundaries, which is the thing the check
    exists to compare.

### D6. `env -C` for engine `cwd`
- **Change.** The render emits `/usr/bin/env -C <dir> <server> ...`. Output is
  identical on both platforms and needs GNU coreutils 8.28 or newer, which
  every current distribution ships.
- **Alternative rejected:** a `sh -c 'cd ... && exec ...'` wrapper. It puts a
  shell back between llama-swap and the server, which the current design
  deliberately avoids.

### D7. MTP intent and per-engine speculative arguments
- **Registry.** Entries carry `"mtp": true`.
- **Engines.** An engine can set `mtp_args`. The default is
  `["--spec-type", "draft-mtp"]`, so the Linux host renders exactly what it
  does today. The macOS engine can set `--spec-draft-n-max 1`, or whatever the
  stack benchmark finds best.
- **Header-macro path.** It uses the default.
- **Validation.** `mtp` together with `cmd`, or `mtp` on a `custom` engine, is
  a render error.
- **Existing entries.** Entries with literal flags in `extra_args` are not
  rewritten. Migrating them is an optional per-host edit.
- **Alternative rejected:** changing `add`'s hard-coded flag to
  `--spec-draft-n-max 1` everywhere. That would slow down the Linux host,
  where MTP is a measured win.

### D8. Custom engines (MLX) must declare `ctx`
An MLX server may grow its KV cache on demand with no context limit, so the
client window is the only safeguard. Rendering fails for a custom-engine entry
without `ctx`. `ensure_clients` uses `ctx` as-is for these entries. This needs
no new engine kind: `kind: "custom"` with an MLX server (oMLX, see below) already
receives `--host` and `--port ${PORT}`, plus `extra_args` such as
`--model <repo>`.

### D9. Stack benchmarks: an isolated `llms` instance, variants as patches
- **Library.** A new `bench/lib/stack.py` is a context manager. It:
  1. copies the host's `settings.json`, `registry.json` and header into a
     temporary config directory;
  2. applies a declared variant patch: a JSON merge patch to one registry entry
     and/or one engine;
  3. sets `listen`/`swap_url` to a bench port and renders with
     `llms --config-dir`;
  4. runs `llms stop`;
  5. starts `llama-swap` in the foreground in its own process group, and waits
     for `/running`.
- **Teardown.** Teardown kills the process group and runs `llms start`. It
  happens in `finally` and in SIGINT/SIGTERM handlers.
- **Results.** Each result row records the variant patch and the full rendered
  command.
- **Alternative rejected:** flipping the production registry between runs, as
  `kv-probe.py` does today. It risks leaving production changed after a crash.

### D10. Portable memory measurement
- **Finding the process.** `bench/lib/mem.py` finds the inference process by
  the upstream port llama-swap assigned to the model. It uses `lsof -t` (on
  both systems), with `ss -ltnp` as a Linux fallback.
- **Linux.** Device VRAM in use, from `rocm-smi` or `nvidia-smi`. This matches
  how the existing results were taken.
- **macOS.** The process's `ri_phys_footprint` from `proc_pid_rusage`
  (`ctypes`), which counts Metal allocations on unified memory. The budget is
  the `recommendedMaxWorkingSetSize` that llama.cpp's Metal backend logs at
  load; it is read from llama-swap's `/logs/stream/<model>` history.
- **Unavailable values** are recorded as unavailable.
- **Fallback.** If validation (task 7.3) shows that the footprint does not
  track KV growth, a small Swift helper reporting
  `MTLDevice.currentAllocatedSize` replaces it. The spec does not change.

### D11. Bench drivers become Python; the old shell drivers are kept as legacy
- **New drivers.** The shell drivers become Python drivers built on
  `stack.py`/`mem.py`, with the existing TSV/JSON columns kept where they mean
  the same thing, so old and new results stay comparable.
- **Old drivers.** The old scripts move to `bench/legacy/` with a README
  stating they are Linux/ROCm-only and are the provenance of the existing
  `results/`. They are not deleted, because `MODELS.md` and `results/`
  reference them.
- **Candidates for retirement.** `ctx-gap.sh` was a one-off that closed a gap
  in a previous sweep. Its logic becomes a parameter of the context probe.
- **Alternative rejected:** fixing each bash script in place for 3.2 and BSD
  tools. The code copies flags instead of rendering them from the registry,
  which is the methodology problem, so fixing only the portability would
  preserve the wrong design.

### D12. `build-engine` on macOS
- **Backends.** Add `metal` (`-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON`,
  so the shader library ships inside the binary). `cpu` sets
  `-DGGML_METAL=OFF`. The script branches on `uname -s` to accept or reject
  each backend before touching sources.
- **Rpath.** `@loader_path/../lib` on Darwin, `$ORIGIN/../lib` on Linux.
- **Isolation check on Darwin.**
  1. `otool -L` lists `@rpath/libggml*` and `@rpath/libllama*`.
  2. `otool -l` lists `LC_RPATH`.
  3. Each library is resolved against those rpaths, with `@loader_path`
     replaced by the binary's directory.
  4. The build fails unless each library resolves to a file under the prefix,
     and none is an absolute path outside it.
- **Portability.**
  - CPU count: `getconf _NPROCESSORS_ONLN`.
  - Dates: `date -u +%Y-%m-%dT%H:%M:%SZ`.
  - Arrays: `${arr[@]+"${arr[@]}"}`.
  - Missing tools (`cmake`, `ninja`, `git`) are checked up front.
- **vLLM scripts.** `build-vllm`'s `log` stops using `printf %()T`. The script
  exits with an error on non-Linux, except under `--print-config`/`--dry-run`.

### D13. Proxy key path
`defaultKeysFile` uses `$XDG_CONFIG_HOME`, else `$HOME/.config`, and no longer
calls `os.UserConfigDir`. Covered by a Go test that sets `HOME`.

### D14. CI
A GitHub Actions matrix (`ubuntu-latest`, `macos-latest`) runs:
- the unittest suite, `go test` and `go vet`;
- `bin/build-vllm --print-config`;
- `/bin/bash -n` over all shell entry points;
- on macOS only, a smoke test that renders a registry with an engine that has
  `cwd` and executes the resulting `env -C` prefix.

Real launchd and systemd round trips stay as manual verification tasks. CI
runners can't host a user service session reliably.

## Risks / Trade-offs

- [`launchctl print` format changes in a future macOS] → The strict parser
  fails closed, so we refuse rather than act wrongly. D4 names the fallback.
  Tests are pinned to captured output from the current macOS.
- [No `gui/<uid>` domain in an SSH-only session on a headless Mac] → `doctor`
  detects it and explains; a console login is required. The `user/<uid>`
  domain is not supported in this change.
- [The `KERN_PROCARGS2` layout is undocumented] → It has been stable for many
  years and is what `ps` uses. Any parse error gives "cannot verify", never
  "match".
- [The footprint may not reflect all Metal memory] → Task 7.3 checks it
  against a context sweep before any results rely on it. D10 names the
  fallback.
- [Moving the bench drivers to Python changes the harness] → Output columns
  are kept, and a re-run of one existing Linux result is compared against the
  recorded result before the legacy scripts are retired.
- [launchd log files grow without bound] → Documented, with a `newsyslog`
  example in the README. `llms logs` reads only the tail.
- [Old `env` on unusual Linux hosts] → `doctor` checks that `env -C` works when
  any engine sets `cwd`.

## Migration Plan

1. **Linux host.** Upgrade `llms`, run `llms render`: the only diff is
   `--chdir=` becoming `-C`. Then `llms doctor` and `llms restart`. The unit
   files do not change, and `settings.json` needs no edits.
2. **macOS host.**
   - Install prerequisites: `brew install cmake ninja`; the llama-swap release
     binary into `~/.local/bin` (no core Homebrew formula exists); and
     `make install` for the package.
   - Run `build-engine metal metal`, or point an engine at a Homebrew
     `llama-server`.
   - Run `llms init`, and copy the shared `registry.json`.
   - Write a Mac `settings.json` with the Metal engine, `mtp_args` and
     `service_path`.
   - Run `llms install-service`, then `llms start`, then
     `llms doctor --live`.
3. **MTP.** Optionally, on either host, replace a literal `--spec-type` in an
   entry's `extra_args` with `"mtp": true`. The Linux render is unchanged
   either way.
4. **Rollback.** Reinstall the previous version and run `llms render`. launchd
   jobs are removed with `launchctl bootout` and by deleting the plist. Nothing
   in the registry needs reverting, because older versions ignore `mtp`.
   Entries relying on it would lose MTP, and are listed by `doctor` before
   upgrading.

## Resolved During Implementation

- **"dspark"** is llama.cpp's `--spec-type draft-dspark`, a drafter-model mode
  alongside `draft-dflash` and `draft-mtp`. It is configured through an
  entry's `extra_args` like DFlash. `--spec-type` takes a comma-separated
  list, so an entry combining MTP with a drafter needs one combined flag, not
  `mtp_args` plus a second `--spec-type`.
- **MLX serving: oMLX, not mlx_vlm or mlx_lm.** The first MLX engine tried was
  `mlx_vlm.server`; it was retired and uninstalled once oMLX was measured.
  Historical notes on it follow. Current
  Qwen3.5-family MLX checkpoints are vision-language models. `mlx_lm` loads
  them and emits garbage (documented on the Cyber-Tiel card). `mlx_vlm.server`
  fits `kind: custom` unchanged: it takes `--host`/`--port`/`--model`, serves
  `/health`, and returns llama.cpp-shaped `timings` including `draft_n`.
  Its MTP path (`--draft-kind mtp`) needs a split drafter. mlx_vlm 0.6.12's
  splitter does not recognise the `language_model.mtp.*` key layout, and oMLX
  (the card's MTP runtime) is not installed, so the Mac MLX entry runs without
  MTP for now. MLX entries also need `useModelName` set to the `--model` path:
  mlx_vlm loads whatever the request's `model` field names.
- **oMLX also fits `kind: custom`.** The engine is
  `omlx-cli serve --base-path <config>/omlx --model-dir <config>/omlx/models --no-hf-cache`.
  Model IDs are symlinks in that model dir. `mtp_enabled` and
  `max_context_window` live in its `model_settings.json`, and `useModelName`
  selects the ID. On this Mac, oMLX matches llama.cpp on decode and is 1.85x
  faster at 16k prefill, while mlx_vlm is 3-5x slower; see
  `results/speed/mac-engine-compare.md`.
- **The macOS footprint excludes mmap'd weights.** It tracks KV growth exactly
  (22.0 KiB/token measured vs 22.0 analytic for Cyber-Tiel; see
  `results/context/mac-footprint-validation.md`), so D10's fallback is not
  needed. Capacity figures on macOS are weights file + footprint.
- **Memory is sampled during generation, never idle.** This is carried over from
  `results/quality/turbo-735.md`: the amdgpu Vulkan driver evicts an idle
  model, and VRAM then reads ~0.
- **llama-swap puts each upstream in its own process group,** so the stack
  kills tracked upstream PIDs explicitly; killing llama-swap's group is not
  enough.
- **Background benchmark jobs ignore SIGINT** (POSIX shells without job
  control). SIGTERM is the documented way to stop one. The stack respects an
  inherited SIG_IGN for SIGHUP, so `nohup` still works.
- **The `n-max` 3 default is not a Mac verdict yet.** The one sweep attempted on
  this Mac was discarded: a 16 GB VM and other resident apps had the host
  swapping (21.6 of 22.5 GB swap used), and PP/TG fell 2-3x across reps.
  Stack benchmarks on a 36 GB machine need that memory free. A memory-pressure
  guard in `stack.py` is a proposed follow-up, not part of this change.

## Open Questions

- **Mac MTP draft length.** Task 7.4 decides it, once it can run on a host
  that isn't swapping.
- **Mac engine source.** `build-engine metal` (now built and used here) versus
  Homebrew's `llama-server`. Either works as an engine.
