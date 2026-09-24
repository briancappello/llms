# Tasks

## 1. Quick portability fixes

- [x] 1.1 Render engine `cwd` as `/usr/bin/env -C <dir>` in `_engine_command` (D6) and update `test_custom_engine_skips_llama_arguments_and_needs_no_path`. Verify: the unit tests pass, and on macOS the rendered prefix runs (`/usr/bin/env -C /tmp pwd` prints `/private/tmp`).
- [x] 1.2 Add a `doctor` check that `env -C` works whenever any engine sets `cwd`. Verify: a new test in which a mocked `env` fails turns the check red.
- [x] 1.3 Make `proxy` `defaultKeysFile` use `$XDG_CONFIG_HOME`, else `$HOME/.config` (D13), and fix the doc comment. Verify: a new Go test with `HOME` set and `XDG_CONFIG_HOME` unset expects `$HOME/.config/llama-swap/api-keys`; `go test ./... && go vet ./...` pass on macOS.
- [x] 1.4 Replace `printf '%(...)T'` in `bin/build-vllm` `log()` with a `date +%H:%M:%S` call, and make `build-vllm`, `run-vllm-reranker` and `run-vllm-embeddings` exit on non-Linux (except with `--print-config` or `--dry-run`). Verify: `/bin/bash bin/build-vllm --dry-run` runs on macOS without a bash error, `bin/run-vllm-reranker` exits non-zero with "Linux-only", and `tests/test_build_vllm.py` passes.
- [x] 1.5 Guard the `vllm`, `vllm-reranker` and `vllm-embeddings` Makefile targets on `uname -s`. Verify: `make vllm` on macOS fails immediately with the Linux-only message.

## 2. Service layer extraction (no behavior change)

- [x] 2.1 Create `src/llms/services/` with the backend interface (D1), and move the systemd unit generation, `systemctl show` parsing and lifecycle actions into `services/systemd.py`. Verify: all existing `ServiceTests`/`CompanionTests` pass unchanged, and a manual diff of the generated unit text shows it is identical.
- [x] 2.2 Make `ModelManager` delegate `install_service`, `install_companions`, `service`, `companion_service`, `companion_status` and `_verify_service` to the backend, keeping `runner` injection. Verify: the full unittest suite passes (77+ tests).
- [x] 2.3 Move `/proc` argv/exe reading into `src/llms/_proc.py` with a Linux implementation, and patch tests at the new seam. Verify: the live-process tests (`test_live_*`, `test_matching_live_process_allows_actions`) pass.

## 3. macOS process introspection

- [x] 3.1 Implement `_proc.argv(pid)` and `_proc.exe(pid)` for Darwin with `ctypes` (`KERN_PROCARGS2`, `proc_pidpath`), raising `OSError` on any failure (D5). Verify: a macOS-only test starts `sleep 30` with arguments that include a space and an empty string, and gets the exact argv and `/bin/sleep`.
- [x] 3.2 Add parser tests for `KERN_PROCARGS2` buffers (argc, exec path, padding, env tail), built from bytes. Verify: they pass on Linux and macOS, and truncated or malformed buffers raise `OSError`.

## 4. launchd backend and settings

- [x] 4.1 Add Settings fields `service_manager` (`auto|systemd|launchd`), `launchctl` and `service_path`, with `auto` resolved by platform. Accept `unit` with or without `.service`. The launchd `service_dir` defaults to `<config_dir>/launchd`. Verify: new Settings tests cover resolution on mocked Darwin/Linux/other, legacy `llama-swap.service`, and the checks that managed paths are distinct.
- [x] 4.2 Implement launchd plist generation with `plistlib` (D2, D3): label `llms.<name>`, `ProgramArguments`, `KeepAlive.SuccessfulExit=false`, log paths, `EnvironmentVariables.PATH` from `service_path`, and `ProcessType=Interactive`. Verify: golden tests for the llama-swap and companion plists; `plutil -lint` passes on the output on macOS.
- [x] 4.3 Implement install for launchd: write without overwriting, print the load and login-start instructions, and run no `launchctl` command. Verify: tests assert that the file is written, an existing file is refused, and `runner` is never called.
- [x] 4.4 Implement a strict `launchctl print` parser (D4) using captured fixtures (running, not running, unknown service, malformed or duplicate keys). Verify: the fixtures parse as expected, and every malformed variant raises a "cannot verify" error.
- [x] 4.5 Implement launchd verification: the plist at the loaded `path` (config dir or `~/Library/LaunchAgents`) is byte-identical to the generated plist; `program` and `arguments` match; the state and PID are consistent; and the live argv and exe match through `_proc`. Verify: tests mirror the systemd refusal cases (modified definition, other source path, argv drift, exe drift, unreadable process state).
- [x] 4.6 Implement the lifecycle: `start` = bootstrap, `stop` = bootout, `restart` = kickstart -k, and the companion equivalents (D2). Verify: tests assert the exact `launchctl` argv for each action after a successful verification, and assert that nothing runs after a failed one.
- [x] 4.7 Manual round trip on this Mac: `llms install-service`, then `llms start`, `llms status`, `llms restart`, `kill -9 <pid>` (the process respawns), and `llms stop` (it stays stopped; no stray `llama-server`). Verify: record the observed outcomes in the PR description.

## 5. CLI and doctor

- [x] 5.1 Make the `install-service` and `install-companions` hints come from the backend (systemd link/daemon-reload, launchd bootstrap/copy to LaunchAgents). Verify: the CLI tests for both backends assert the printed commands, and the existing systemd hint tests still pass.
- [x] 5.2 Add `llms logs [--follow] [--companion NAME]`, using `journalctl --user -u` on systemd and a tail of the log file on launchd. Verify: tests assert the journalctl argv, and read or follow a temporary log file under launchd.
- [x] 5.3 Extend `doctor`: the chosen service manager and its tool, the presence of the `gui/<uid>` domain on launchd, and bare engine/companion names that don't resolve on `service_path`. Verify: tests for each check, both passing and failing.

## 6. Engine builds

- [x] 6.1 Make `bin/build-engine` safe for bash 3.2 and BSD tools: `${arr[@]+"${arr[@]}"}`, `getconf _NPROCESSORS_ONLN`, a portable UTC date, and an up-front check for `cmake`, `ninja` and `git`. Verify: `/bin/bash -n bin/build-engine` passes, and on macOS with ninja removed from PATH the script exits naming `ninja` before cloning.
- [x] 6.2 Add per-platform backend validation and a `metal` backend (`-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON`); make `cpu` set `-DGGML_METAL=OFF` (D12). Verify: `build-engine hip hip` on macOS exits "unsupported on Darwin" before touching `~/src/llama.cpp`.
- [x] 6.3 Use `@loader_path/../lib` as the rpath on Darwin, and implement the `otool` isolation check. Verify: a `build-engine metal metal` on this Mac installs to `~/opt/llama.cpp-metal` and reports isolation OK; pointing `DYLD`/rpath at a second prefix makes the check fail.
- [x] 6.4 Run a real `build-engine cpu cpu` under `/bin/bash` on macOS. Verify: it completes, and `otool -L` shows no Metal framework linked.

## 7. Stack benchmark harness

- [x] 7.1 Implement `bench/lib/stack.py` (D9): copy the host config into a temporary directory, apply a variant merge-patch, render with the bench port, stop and start production through `llms`, run llama-swap in its own process group, and tear down in `finally` and on signals. Verify: a test with a stub `llama-swap` script checks that the production files are byte-identical after both a normal exit and a SIGINT, and that `llms start` is always called.
- [x] 7.2 Implement `bench/lib/mem.py` (D10): find the upstream PID by port (`lsof -t`, with `ss` as a fallback); Linux VRAM via `rocm-smi`/`nvidia-smi`; macOS `ri_phys_footprint` via `proc_pid_rusage`; parse the Metal `recommendedMaxWorkingSetSize` from `/logs`; return "unavailable" on any failure. Verify: unit tests with captured tool and log output; on macOS it reads a nonzero footprint for a running `llama-server`.
- [x] 7.3 Check that the macOS footprint tracks served KV memory: load one registered model through `stack.py` at three `ctx` values, and confirm the footprint grows roughly in line with the KV size the server logs. Verify: write the findings to `results/context/mac-footprint-validation.md`. If it does not track, implement the D10 fallback helper and repeat.
- [ ] 7.4 Port `depth-sweep` to a Python driver on `stack.py`, keeping its TSV columns and adding the variant and rendered-command columns. Run it on the Mac for one MTP model with `mtp_args` draft length 1 versus 3 (engine patch variants). Verify: the results are committed under `results/speed/`, and the chosen Mac `mtp_args` are recorded in `config/settings.example.json`'s Mac engine example.
- [x] 7.5 Port `ctx-probe` (absorbing `ctx-gap`), `kv-sweep`, `soak-ctx`, `gates`/`mtp-engage` and `backend-sweep` to Python drivers on `stack.py`/`mem.py`. Verify: each driver runs to completion against a registered model on the Mac, and no driver contains a hand-written server command line, a HF glob or `rocm-smi`.
- [x] 7.6 Update `kv-probe.py` and `coding-profile.py` to use `stack.py` variants instead of editing the production registry. Verify: a run leaves the production `registry.json` unchanged (checksum before and after).
- [x] 7.7 Move the original shell drivers to `bench/legacy/` with a README naming the results each one produced, and update the paths in `MODELS.md`. Verify: `rg 'bench/(speed|context|gates)/[a-z-]+\.sh' MODELS.md results` finds only `bench/legacy/` paths.
- [ ] 7.8 Re-run one existing Linux result with the new driver on the Linux host (for example a depth-sweep cell) and compare it with the recorded TSV. Verify: the numbers are within the noise band recorded for the old run, and the comparison is noted in `bench/legacy/README.md`.

## 8. Model registration

- [x] 8.1 Add the `mtp` registry key and the engine `mtp_args` setting (validated as a list of single-line strings), and render them for llama.cpp engines and the header-macro path. Treat `mtp` with `cmd` or with a `custom` engine as an error (D7). Verify: tests cover the default args, per-engine args, the error cases, and a legacy `extra_args` entry rendering byte-identically.
- [x] 8.2 Change `llms add` to write `"mtp": true` instead of `--spec-type draft-mtp`. Verify: update `test_local_add_*` so an MTP probe asserts `mtp` is true and `extra_args` has no `--spec-type`.
- [x] 8.3 Read `general.sampling.*` in `probe_gguf`, record it as `embedded_sampling` in the entry, and show it in the `add` output without rendering it. Verify: the synthetic GGUF smoke test includes `general.sampling.temp` and asserts it is recorded and absent from the rendered command.
- [x] 8.4 Add unregistered-model discovery (GGUF groups, MLX snapshots with `config.json` + `.safetensors`) and expose it as `llms ls --unregistered`. Verify: tests on a fake HF cache cover a partial MLX download (hidden), a registered GGUF (hidden), and a sharded GGUF (listed once).

## 9. Serving controls and clients

- [x] 9.1 Require a positive `ctx` for `custom`-engine entries in rendering and `ensure_clients` (D8). Verify: tests for a missing `ctx` (render error naming the entry) and a present one (`contextWindow` equals `ctx`).
- [x] 9.2 Accept a string `reasoning` in the `use()` answer validation. Verify: a test with `content: null` and `reasoning: "ok"` passes, and the malformed-answer tests still fail as before.
- [x] 9.3 Add `llms unload [NAME]` (all models → `POST /api/models/unload`; one model → `/api/models/unload/<id>`, after checking the name is registered). Verify: tests with a mocked opener for both forms and for an unknown name.
- [x] 9.4 Add a port check before `start`/`restart` that refuses when the listen address is bound and the managed service is not active. Verify: a test binds a socket on a temporary port and asserts that `llms start` fails without calling the runner.
- [x] 9.5 Add `use --sync-clients`: run `ensure_clients`, then set pi `settings.json` `defaultProvider`/`defaultModel` with an atomic write, only after a valid warm-up answer. Verify: tests check that other keys are kept and that nothing is written when the warm-up fails.

## 10. Docs, examples and CI

- [x] 10.1 Update `README.md`: per-platform requirements, a macOS install path (brew llama-swap/cmake/ninja, `build-engine metal`), launchd service steps, `llms logs`, a `newsyslog` rotation example, and the stack-benchmark method. Verify: every command in the macOS section has been run on this Mac.
- [x] 10.2 Add to `config/settings.example.json` a Mac Metal engine with `mtp_args`, an MLX `custom` engine, and `service_manager`/`service_path`; add an MLX entry with `ctx` to `registry.example.json`. Verify: a test loads both examples with `_comment` keys removed through `Settings` and `render_config` without errors.
- [ ] 10.3 Add a GitHub Actions workflow with an `ubuntu-latest` and `macos-latest` matrix (D14). Verify: the workflow passes on both runners in a pushed branch.

## 11. End-to-end host verification

- [ ] 11.1 Linux host: upgrade, `llms render` (the diff is only `--chdir=` → `-C`), `llms doctor --live`, `llms restart`, and one served request per engine. Verify: record the outputs in the PR.
- [x] 11.2 Mac host: the full migration in design.md step 2, using the shared registry. `llms add` an MTP GGUF, then `llms use`, then a tool-calling request through llama-swap, then `llms unload`. Verify: every step succeeds and `llms doctor --live` is green.
