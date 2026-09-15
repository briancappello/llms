# Muse-Glimmer-30B (meta-models GGUF) -- agentic-coding setup + benchmark

Meta Superintelligence `muse-glimmer` arch (dense 30B, GQA 32Q/2KV + sliding-window 2048 on 3/4 layers => cheap KV). Needs llama.cpp >= b10353; served on our source build 169e4a7 (arch confirmed present). Q4_K_XL Dynamic build (0.2% degradation) + **DFlash drafter** (`-md`, auto-detected as spec type `draft-dflash`, block_size 16).

**Serving config** (registry `muse-glimmer`): Meta sampling `--temp 1.0 --top-p 0.95 --top-k 64 --min-p 0.0`; `-c 262144 -np 2 --no-kv-unified` = full 131072 per request (card warns -c splits across slots); f16 K/V; `--jinja` mandatory; reasoning always-on (default strength high) -> `reasoning_content`. VRAM loaded: **24.4 GiB / 32.6** (~8 GiB free). Vision mmproj not wired (text-only).

## Agentic generation (coding-profile.py, 5 tasks, Meta sampling temp 1.0)

| metric | value |
|---|---:|
| TG t/s (median, with DFlash) | 64.51 |
| DFlash acceptance % (median) | 66.3 |
| reasoning tok (total, 5 tasks) | 3642 |
| reasoning tok (median/task) | 640 |
| answer tok (total) | 1476 |
| think ratio (median) | 0.76 |
| wall s (total) | 84.8 |
| hit 3000-tok cap | 0 |

## Tool-calling (toolcall.py, 22 scenarios, temp 0.6)

- pass **21/22 (95.5%)**, false-positive 0.0%, malformed JSON 0. breakdown: {'pass': 21, 'wrong_tool': 1}

## Deterministic PP/TG at depth (kv-probe.py, temp 0, 2 reps)

| depth | PP t/s | TG t/s | DFlash acc% |
|---:|---:|---:|---:|
| 4096 | 831.5 | 64.22 | 68.5 |
| 32768 | 810.8 | 48.8 | 45.6 |
| 65536 | 749.5 | 50.49 | 50.3 |

## Notes

- **DFlash pays but tapers with depth**: acceptance ~68% @4k (TG 64) -> ~46-50% @32-64k (TG ~49-50). Still net-positive vs no-draft. Base (no drafter, gate test) was ~35 t/s.
- **PP is flat with depth** (831 @4k -> 749 @64k, only -10%) thanks to sliding-window attention -- vs the dense qwen35 cold-fusion's -20% (894 -> 713). Good for agents re-prefilling large context.
- **Reasons at length** (median 640 thinking tok/task at temp 1.0), as the card warns; default strength `high`. Drop to `medium` via `--chat-template-kwargs '{"reasoning_strength":"medium"}'` or cap with `--reasoning-budget N` if turns feel slow.
- Comparability: the deterministic kv-probe (temp 0) is directly comparable to cold-fusion/bigbang; the coding-profile used Meta's temp 1.0 (vs temp 0.6 for the qwen models), so its thinking-token / TG numbers are not apples-to-apples there.
- The old `muse-glimmer-30b` entry (unsloth UD-Q4_K_XL, no DFlash) also loads now on 169e4a7 -- its 'UNSERVABLE' note was stale. `muse-glimmer` (this entry) is the tuned/preferred one.
