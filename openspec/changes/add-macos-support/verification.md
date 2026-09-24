# Manual verification log (for the PR description)

## 4.7 launchd round trip (macOS 27, Apple Silicon)

Isolated instance: `--config-dir $TMP/rt`, `unit=llama-swap-roundtrip`,
listen `127.0.0.1:18099`, llama-swap v243 (the `llms.bak/bin` binary).

| Step | Observed |
|---|---|
| `llms install-service` | plist written to `<config>/launchd/llms.llama-swap-roundtrip.plist`; no `launchctl` calls; printed bootstrap + LaunchAgents copy hint |
| `llms start` | `launchctl bootstrap`; returned after `/running` and `/v1/models` answered |
| `llms restart` | passed live argv/exe verification (`KERN_PROCARGS2` + `proc_pidpath`); PID 69979 -> 70055 |
| `kill -9` (uptime < 10 s) | state `spawn scheduled` (launchd throttle), respawned within ~1 s after throttle, `runs = 3` |
| `kill -9` (uptime > 10 s) | respawned immediately (PID 70143 -> 70256) |
| `llms logs -n 3` | tailed `~/Library/Logs/llms/llama-swap-roundtrip.log` |
| `llms stop` | `launchctl bootout`; job unloaded (print rc 113), still unloaded 3 s later, no stray llama-swap |

Behaviour to note: while a job is `spawn scheduled` (throttled respawn after a
crash), every verified operation including `stop` is refused as an unstable
state -- the same fail-closed rule the systemd backend applies to
`activating`. Recovery from a crash loop is `launchctl bootout gui/<uid>/<label>`.

## 11.2 Mac host migration (this Mac)

- `make install`; llama-swap v243 release binary -> `~/.local/bin`; `brew install ninja`
  (cmake already present); `bin/build-engine metal metal` -> `~/opt/llama.cpp-metal` (6b790a9).
- `llms init --server ~/opt/llama.cpp-metal/bin/llama-server`; engines `metal` and
  `mlx-vlm` (custom) in settings.json; `gpu_memory_mib` 28753 (Metal budget).
- `llms add cyber-tiel <UD-Q4_K_XL.gguf> --no-restart`: `mtp: true` recorded, embedded
  sampling (temp 0.6, top_p 0.95, top_k 20, min_p 0) reported and stored, not rendered.
- `llms doctor`: all green, including `launchd-domain` and `service-path:engine:mlx-vlm`.
- `llms install-service && llms start`; `llms doctor --live` green.
- `llms use cyber-tiel` (33 s cold); tool-calling request through llama-swap ->
  `finish_reason: tool_calls`, `get_weather({"city":"Paris"})`; MTP drafting 30/36 accepted.
- `llms use cyber-tiel-mlx` (mlx_vlm, needs `useModelName`); tool call -> `tool_calls` OK.
- `llms unload`: no llama-server / mlx_vlm.server left running.

## 7.5 / 7.6 stack drivers, functional runs (tiny parameters; not measurements)

All run to completion against `cyber-tiel` through the stack: kv-probe, coding-profile,
depth_sweep, kv_sweep (production vs q8_0-V arm; cache types read back from the render),
backend_sweep (metal), ctx_probe (derive -> verify -> stretch, plus --test), soak_ctx,
gates (G1 PASS 23 s; G5 REASONS; G4 toolcall 20/22; G3 copy 87.9% acc TG 63.9 vs
control 44.3, prose 50.5% acc TG 46.7 vs control 42.6). Production settings/registry/
header/rendered config checksums identical before and after; no stray processes.

Not measurements: the host was swapping throughout (a 16 GB Virtualization VM plus
other apps; ~21.6-26 GB of swap used). See design.md "Resolved During Implementation".

## Follow-up: MLX engine consolidated on oMLX (2026-09-24)

mlx_vlm was retired and uninstalled (`uv tool uninstall mlx-vlm`) after oMLX measured
3-5x faster decode on the same weights (results/speed/mac-engine-compare.md). The host
registry now holds a single `cyber-tiel`: MLX oQ4e (non-MTP build), served by oMLX with
mtp_enabled=false. The GGUF and mlx_vlm entries above were removed with `llms rm`;
their weights are still in the HF cache.
