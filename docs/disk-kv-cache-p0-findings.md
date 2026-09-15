# Disk KV Cache — P0 findings & threads to pull

Status: **P0 gate ran — result is NO-GO on the stock primitive for our models.**
Date of investigation: 2026-08-18. Companion to `docs/disk-kv-cache-spec.md`
(read that first for the goal/design). This doc records what we learned so we can
resume cold in a month without re-deriving it.

---

## TL;DR

- The disk-KV primitive (`POST /slots/:id?action=save|restore` +
  `llama_state_seq_save_file/_load_file`) **works** on our binary — verified
  end-to-end on a plain-attention model (restore → next request reused the whole
  prefix, `prompt_n=1`).
- It **does not work for any model we actually serve.** Every `cold-fusion`,
  `fable-fusion`, `qwen38-27b`, `bigbang`, `kat`, `ornith`, `qwopus` entry is a
  **Qwen3.5/3.6 "qwen35/qwen35moe" hybrid = Gated-DeltaNet recurrent + attention**
  arch (~75% of layers hold a *recurrent* state, only ~16/64 layers cache KV).
  llama.cpp's slot save/restore **does not restore the recurrent state**, so after
  a restore the next request logs `forcing full prompt re-processing due to lack of
  cache data (…hybrid/recurrent memory)` and re-prefills the entire prompt.
- This is a **known, still-open upstream problem.** The fix (PR **#24785**,
  recurrent shrink/expand) is **OPEN, conflicting, not merged**, and is **core**
  llama.cpp surgery (memory-recurrent + context), not a `server.cpp` tweak.
- Net: to get disk KV cache for our stack we must **carry a core llama.cpp patch**
  (rebase #24785 or port BeeLlama's mechanism), or wait for upstream. Since 100% of
  our llama.cpp-served models are this arch, there are **zero beneficiaries on
  stock** today (ds4 is excluded — it has its own cache).
- My earlier working hypothesis (M-RoPE) was a **partial red herring** — see
  "Ruled out / red herrings". The operative trigger is the **recurrent** state.

---

## Environment under test

- Binary: `~/opt/llama.cpp-vulkan/bin/llama-server`, **version `0.1.2-dev (build 1,
  commit 169e4a7)`**, Vulkan/RADV, AMD Radeon AI PRO R9700 (gfx1201).
- Source tree: `~/src/llama.cpp` at `169e4a7` (== `origin/master` at the time;
  shallow clone, so `git log`/`blame` history is not available locally).
- Serving stack: `/home/brian/dev/llms` (llama-swap → llama-server; see `README.md`,
  `MODELS.md`). P0 target = `cold-fusion` with its real flags: `-fa on -c 196608
  -np 2 --kv-unified --cache-type-k f16 --cache-type-v f16 --spec-type draft-mtp`.

---

## What P0 tested and what happened

Method: prefill a prompt on a slot, `save` it, force the KV to be lost (real model
swap, or `erase`, or process restart), `restore`, then send the same/continued
prompt and read `timings.prompt_n` (tokens actually prefilled) vs `timings.cache_n`
(tokens reused). Prefill is skipped iff `cache_n` ≈ prefix length.

| scenario (cold-fusion unless noted) | result |
|---|---|
| Full swap-survival: 42k prompt, save (2.94 GB, 920 ms), swap→bigbang→back, restore (2.37 s), continue | **NO reuse** — `prompt_n=42439`, `cache_n=0` (full re-prefill). Correctness OK (greedy output matched a cold run). |
| `erase` then `restore`, same prompt, explicit `id_slot=0` | **NO reuse** — `cache_n=0` |
| same, **without** `id_slot` | NO reuse |
| `--cache-ram 0 --no-cache-idle-slots` (disable RAM prompt-cache machinery) | NO reuse |
| **without** `--spec-type draft-mtp` | NO reuse |
| `--no-kv-unified` | NO reuse |
| `-fa off` | NO reuse |
| **Live** reuse (send same 42k prompt twice, NO restore) | **REUSE WORKS** — `prompt_n=4`, `cache_n=42423` |
| **`stories260K`** (plain-attention, non-recurrent, non-mrope) restore→reuse | **REUSE WORKS** — `prompt_n=1`, `cache_n=37` |

Save/restore file round-trips fine (`n_saved`/`n_restored` correct, bytes exact);
the failure is purely that the restored state is **not reusable** for our arch.

---

## Root cause (the real one)

1. `restore` **does** reload the token list *and* the attention KV cells
   (`state_read_meta: cell_count=8037, dest_seq_id=0`) and sets
   `slot.prompt.tokens`. Slot selection even confirms it:
   `get_availabl: checking sim = 1.000 (8036/8036)`.
2. On the next completion, prompt-processing computes the reuse point, then hits
   the **context-checkpoint gate** (`server-context.cpp:3255` `if (pos_min >=
   pos_min_thold)`) and finds **no checkpoint** (restore clears
   `slot.prompt.checkpoints`, and the state file carries none) →
   **`do_reset`** at `server-context.cpp:3285-3290`:
   `forcing full prompt re-processing due to lack of cache data (…hybrid/recurrent
   memory)` → `n_past = 0` → `memory_seq_rm [0, end)` → full re-prefill.
3. Why the gate trips at all for a model with `n_swa=0`: these models keep a
   **recurrent (Gated DeltaNet) state** in ~75% of layers. That recurrent memory
   cannot be partially reused from a restored prefix, and its `pos_min` exceeds
   `pos_min_thold`, which is exactly what the reset is for (per upstream #20225).

### Why LIVE reuse works but RESTORE doesn't (important nuance)
- In the **live** path, the first prefill *creates context checkpoints* (we see
  `created context checkpoint 1/2 of 32 (pos_min=7519 / 8031)` in the logs — and on
  our **Vulkan** build they create fine, no crash). A second identical request
  finds a checkpoint and reuses it. So multi-turn reuse **within one process**
  already works for us.
- In the **restore** path, the on-disk state file contains **no checkpoints**, and
  restore clears the slot's checkpoints, so the gate has nothing to fall back on →
  full reprocess. The recurrent state is the thing that would need to be
  reconstructed (shrink/expand) or checkpointed to disk.
- Consequence: the disk cache's whole reason for existing (**surviving process
  death / model swaps**, where live checkpoints are gone) is exactly the case the
  primitive can't serve for our arch.

---

## Ruled out / red herrings (don't re-test these)

- **M-RoPE `llama_kv_cell_ext` save/load** (`src/llama-kv-cache.cpp:2268-2270`
  TODO, deferred from PR #16825). This is a *real* separate gap, and our models are
  M-RoPE, but it is **not** what fails P0: `apply_ubatch` already reconstructs the
  ext from positions (`llama-kv-cache.cpp:1129-1134`), and the decisive signal is
  the **recurrent** reset, confirmed by the plain-attention model working and by
  upstream #24785/#20225 naming the recurrent state explicitly. Track it, but it's
  secondary.
- **Flash attention** (`-fa on/off`): no effect.
- **MTP / `--spec-type draft-mtp`**: no effect.
- **`--kv-unified` vs `--no-kv-unified`**: no effect.
- **RAM prompt cache** (`--cache-ram`, `--cache-idle-slots`,
  `[TAG_IDLE_SLOT_CLEAR]` at `server-context.cpp:2353-2369`): initially suspected;
  disabling it changed nothing. Not the cause.
- **`id_slot` routing / wrong slot**: ruled out — response echoes `id_slot:0` and
  the slot does the work; slot selection honors explicit id (`server-context.cpp:
  1493`).
- **`--ctx-checkpoints` crashing on AMD (#20176)**: does **not** reproduce on our
  **Vulkan** build (checkpoints create fine; that issue is HIP/ROCm). So checkpoints
  are usable for us — good news for the live path, doesn't fix restore.

---

## Upstream status (checked via `gh`, 2026-08-18)

The fix for *our* problem:
- **PR #24785 — `server: add recurrent state shrink/expand for prompt cache
  (#22746)` — OPEN, `CONFLICTING`/`DIRTY`, `REVIEW_REQUIRED`.** Created 2026-06-18,
  last activity 2026-08-16 (reviewer asked for a rebase). +390/-0 across 7 files
  incl. **core** (`include/llama.h`, `src/llama-context.cpp`,
  `src/llama-memory-recurrent.cpp`) + `tools/server/server-context.cpp`. Backports
  `recurrent_shrink`/`recurrent_expand` from the **BeeLlama fork
  (`Anbeeld/beellama.cpp`)**. *This is the thing to watch / carry.*

Context issues:
- **#22746** — "Qwen 3.6 27B forcing full prompt re-processing due to lack of cache
  data" — our exact symptom + models. Marked CLOSED/COMPLETED, but the real fix
  (#24785) is **not merged**; closure looks like triage, not resolution.
- **#20225** — "Qwen 3.5 27B (hybrid attention + Mamba2/SSM) full prompt
  re-processing every turn … recurrent memory's `pos_min` exceeds
  `pos_min_thold`." Same root cause, reported on an **R9700** like ours.
- **#20176 (OPEN)** — "Qwen 3.5 Loading checkpoints causes a crash" on AMD. Relevant
  if we lean on checkpoints; appears HIP-specific (our Vulkan build is fine).
- **#27068 (OPEN)** — "failed slot restore leaves corrupted K/V" (hybrid + Gemma3).
  Relevant to G4 (silent-fallback) hardening.
- **M-RoPE save/load TODO** — PR **#16825** ("store mrope data in KV cell", merged
  Oct 2025) *deferred* KV save/load of the cell ext to a follow-up; **no follow-up
  PR exists** as of now. Our build still carries the TODO.
- **#26499 (OPEN)** — `LLAMA_N_RS_SEQ` env override for recurrent-state rollback
  count (speculation perf). Peripheral, but names the same recurrent machinery.

Reference implementation that already solves it: **BeeLlama —
`github.com/Anbeeld/beellama.cpp`** (`recurrent_shrink`/`recurrent_expand`).

---

## Side finding (already applied, keep): GPU pin is env-only now

While reproducing standalone I discovered `-dev Vulkan0` is unsafe: RADV
enumeration order is not stable and `Vulkan0` can resolve to the **iGPU**
(49 GB shared RAM, accepts `-ngl 999`, then runs ~11 tok/s). Proven:
```
# no pin:   Vulkan0 = iGPU (RAPHAEL), Vulkan1 = R9700
# with pin: Vulkan0 = R9700 only
```
The real pin is `MESA_VK_DEVICE_SELECT=1002:7551!` (exclusive) in the llama-swap
systemd unit (`~/.config/systemd/user/llama-swap.service`). **Change made:** removed
`-dev Vulkan0` from the `server` macro in `config/config.header.yaml` and documented
why. Verified: model loads on GPU[0]=R9700 (33.6/34.2 GB used), iGPU idle.
**Optional follow-up not yet done:** also export the var in `~/.zshenv` and/or
`~/.config/environment.d/10-gpu.conf` so ad-hoc CLI/bench runs (outside systemd)
can't land on the iGPU either.

---

## How to reproduce (fast, ~2 min once a model is loaded)

Requires `--slots --slot-save-path <dir>` on the server command (add to the
`common` macro, `bin/llm render`, restart). Then drive the native endpoint through
llama-swap's `/upstream/<model>/…`:

```
# 1. cold prefill on slot 0
POST /upstream/cold-fusion/completion {"prompt": <~8k-tok text>, "id_slot":0, "cache_prompt":true, "n_predict":2, "temperature":0}
#    -> timings.prompt_n == full, cache_n == 0   (cold)
# 2. save, erase, restore
POST /upstream/cold-fusion/slots/0?action=save    {"filename":"repro.bin"}
POST /upstream/cold-fusion/slots/0?action=erase   {}
POST /upstream/cold-fusion/slots/0?action=restore {"filename":"repro.bin"}   # n_restored == n_saved
# 3. same prompt again on slot 0
POST /upstream/cold-fusion/completion {"prompt": <same>, "id_slot":0, "cache_prompt":true, "n_predict":2, "temperature":0}
#    BROKEN arch (qwen35): prompt_n == full, cache_n == 0   + server log:
#       "forcing full prompt re-processing due to lack of cache data (…hybrid/recurrent memory)"
#    WORKING arch (stories260K): prompt_n == 1, cache_n == full-1
```
To see the server's reasoning, add `--log-file <path> --verbose` to the model's
`extra_args` and grep the log for: `state_read_meta`, `checking sim`,
`forcing full prompt re-processing`, `cached n_tokens =`, `pos_min`.
Control model for the "it works on plain attention" check:
`llama-server --hf-repo ggml-org/test-model-stories260K -ngl 0 -np 2 --slots
--slot-save-path <dir>` (tiny, runs on CPU, no GPU contention).

The throwaway repro scripts from this session lived in `/tmp/opencode/`
(`p0_kvcache.py`, `p0_probe.py`, `repro_swap.py`, `repro_direct.py`,
`live_reuse.py`) — `/tmp` is ephemeral; the recipe above is the durable version.

---

## Threads to pull on next (prioritized)

1. **Decide go / shelve.** With 100% of llama.cpp-served models on the recurrent
   arch, stock has zero beneficiaries. Either commit to carrying a core patch or
   shelve until #24785 lands. This is the gating decision.
2. **Evaluate carrying the fix.** Fetch full history of `~/src/llama.cpp` (it's a
   shallow clone), then assess rebasing **PR #24785** onto our `169e4a7`, or porting
   `recurrent_shrink`/`recurrent_expand` from **BeeLlama (`Anbeeld/beellama.cpp`)**.
   Files it touches: `src/llama-memory-recurrent.{cpp,h}`, `src/llama-context.{cpp,h}`,
   `include/llama.h`, `tools/server/server-context.cpp`. Our build pipeline
   (`~/dev/my-llama/build-backends.sh`) builds from master, so this becomes a
   maintained patch to rebase each rebuild (the spec's "Option A" cost, but deeper).
3. **Re-run P0 on a patched build** using the recipe above. Gate = restore →
   `cache_n` ≈ prefix, `prompt_n` ≈ suffix, on `cold-fusion`.
4. **Watch upstream** (subscribe): #24785 (the fix), #27068 (restore corruption /
   G4 hardening), and any #16825 follow-up for M-RoPE save/load. If #24785 merges,
   bump the pinned llama.cpp commit and re-test instead of patching.
5. **Separate the M-RoPE variable (optional, for certainty).** To be 100% sure the
   recurrent state is the *only* blocker, test a pure-attention **M-RoPE** model
   (e.g. a Qwen2.5-VL text-only run). If it restores fine, M-RoPE is fully cleared;
   if not, the `kv_cell_ext` TODO is a second patch we'd also need.
6. **If/when the primitive works,** the rest of the spec (Option A vs B, LRU/budget,
   `cfghash`, `llm kvcache` subcommands) is unblocked and unchanged — none of that
   was the problem; the primitive was.

---

## Repo/config state after this session

- **Kept (intended):** removed `-dev Vulkan0` from the `server` macro in
  `config/config.header.yaml`; comment rewritten to point at the
  `MESA_VK_DEVICE_SELECT` pin. Verified working.
- **Reverted to pristine:** `cold-fusion` registry entry (all experiment flags
  removed, `--spec-type draft-mtp` restored, `kv_unified: true`); the temporary
  `--slots --slot-save-path` addition to the `common` macro removed. `bin/llm
  render` + llama-swap restart done; production healthy (fable-fusion ready).
- **Left on disk (harmless):** empty dir `~/.cache/llama-swap/kvcache` and a stale
  `~/.cache/llama-swap/kvcache/p0-cf.bin`/`repro.bin` if present — safe to delete.
- The large pre-existing uncommitted diff in the repo (`MODELS.md`, `bin/llm`,
  `Makefile`, `registry.json`, etc.) predates this session and is unrelated.

---

## Key code references (llama.cpp @ 169e4a7)

- Reset that defeats us: `tools/server/server-context.cpp:3255` (gate),
  `:3285-3290` (`do_reset`, the log line).
- Slot restore worker: `tools/server/server-context.cpp:2501-2564`
  (`slot->prompt.tokens = restored`, KV via `llama_state_seq_load_file`).
- Prompt-reuse / `get_common_prefix`: `server-context.cpp:3123-3320`;
  `tools/server/server-common.cpp:680-728`.
- Slot selection (honors `id_slot`, similarity, LRU): `server-context.cpp:1490-1600`.
- Idle-slot clear (`[TAG_IDLE_SLOT_CLEAR]`, ruled out): `server-context.cpp:2353-2369`.
- KV state read (single-seq path + M-RoPE ext TODO): `src/llama-kv-cache.cpp:2217-2331`
  (TODO at `:2268-2270`); `apply_ubatch` ext handling at `:1129-1134`.
- State-file format constants: `include/llama.h:41-49` (`ggsq`, `LLAMA_STATE_SEQ_VERSION`).
- Slots HTTP routes: `tools/server/server.cpp:271-273`; handlers at
  `server-context.cpp:4534-4565`, `5143-5241`.
