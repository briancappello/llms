# Spec: Disk-backed KV cache for the llama-swap serving stack

Status: **Draft for implementation planning.** This document is a *specification*,
not an implementation plan. A follow-up agent should read it end-to-end and
produce (a) an architecture decision, (b) a phased implementation plan, and
(c) a test plan. Everything marked **[OPEN]** is a decision the planning agent
owns; everything marked **[REQUIRED]** is a hard constraint.

Repo: `/home/brian/dev/llms`. Read `README.md` and `MODELS.md` first for how the
serving stack works; this spec assumes that context.

---

## 1. Problem

Chat/completion APIs are stateless: agent clients (opencode, pi) resend the whole
conversation every turn. Our box serves one model at a time on a single GPU and
**swaps models on demand** (llama-swap kills the old `llama-server` and starts the
new one). Two consequences:

1. **Every model swap throws away the KV cache.** Bouncing cold-fusion → bigbang →
   cold-fusion re-prefills cold-fusion's entire context from token zero on the way
   back.
2. **Every fresh long prompt pays full prefill**, even if a near-identical prefix
   was processed seconds ago in a now-dead process.

Prefill is not free. Measured on this box (build `169e4a7`, Vulkan, R9700):
- dense 27B (`cold-fusion`, f16 KV): ~700–900 tok/s → a 64k prompt ≈ **80–90 s**.
- dense 30B (`muse-glimmer`): ~750–830 tok/s.
- A3B MoE (`bigbang`): ~1800–2200 tok/s → 64k ≈ **~30 s**.

So a swap-back or a long re-prompt stalls the user for tens of seconds to a
minute-plus. `ds4` (the DeepSeek-V4 engine in `~/dev/ds4-hip`) already solves this
with a disk KV cache; we want the same for the llama.cpp-served models.

## 2. Goal

A **transparent, disk-backed KV cache** that lets a `llama-server` restore a
previously-computed KV state for a matching token prefix instead of re-prefilling
it — surviving process death and model swaps — bounded by a **fixed disk budget
with LRU eviction**.

### 2.1 Success criteria (must all hold)
- **G1 — Swap survival:** after cold-fusion → bigbang → cold-fusion, the second
  cold-fusion request with the same (or longer) conversation prefill only the new
  suffix, not the whole prompt.
- **G2 — Bounded disk:** total cache bytes never exceed a configured budget
  (default target **150 GiB**); the oldest unused entries are evicted first; an
  entry currently in use is never evicted.
- **G3 — Transparent:** no client changes. pi/opencode/benches/remote all benefit
  automatically. The `model` request field and responses are unchanged.
- **G4 — Never wrong:** a restore only ever happens against a byte-compatible
  model+config, and a restored prefix is an exact token-prefix of the request.
  Any mismatch, corruption, or error **degrades silently to normal prefill** — the
  cache is an optimization and must never fail a user request.
- **G5 — Net win:** on a cache hit, time-to-first-token is dominated by restore
  (~sub-second for the file sizes here) plus suffix prefill, and is far below cold
  prefill.

### 2.2 Non-goals (v1)
- Cross-machine / shared cache.
- Delta/incremental KV snapshots (v1 stores full-sequence state).
- Multi-user isolation (single human user across all machines).
- Caching for models where prefix semantics don't hold cleanly — see §7 (SWA).
- Replacing `ds4`'s own cache (ds4 is **excluded**, it manages its own).

## 3. What already exists (do not reinvent)

The hard part — serializing/restoring a KV cache — is **already implemented in
llama.cpp** and present in our build. Verify each of these on our binary
(`~/opt/llama.cpp-vulkan/bin/llama-server`) as part of P0:

- **Disk slot state:** `--slot-save-path DIR` + `POST /slots/:id?action=save|restore`
  (with a `filename`) → `llama_state_seq_save_file` / `llama_state_seq_load_file`.
  Serializes a sequence's KV **and tokens** to a file and restores it. This is the
  low-level primitive the whole design sits on. (`tools/server/server.cpp` routes
  `GET /slots`, `POST /slots/:id_slot`.)
- **RAM prompt cache:** `--cache-ram N` (default 8192 MiB) + `--cache-idle-slots` —
  in-memory prompt cache across requests. **Dies on process exit** (so it does not
  survive swaps). Useful as a complementary hot tier, not the persistence layer.
- **Prefix reuse within a live slot:** slot LCP matching, `--cache-reuse N`,
  `--slot-prompt-similarity`. This is what makes a *restored* slot only prefill the
  divergent suffix — we rely on it, we do not build it.

**The gap this project fills:** an automatic, content-addressed, **disk** cache
that (a) survives restarts/swaps and (b) is keyed by the token prefix so a matching
request transparently restores instead of re-prefilling — plus the budget/LRU/GC
machinery around it.

## 4. Reference design: how `ds4` does it (copy the good ideas)

Source of truth: `~/dev/ds4-hip/README.md` § "Disk KV Cache" and `ds4_server.c`.
Salient decisions we should mirror:

- **Key = `SHA1(token_ids)`** (each token id hashed as a little-endian u32), file
  named `<sha1>.kv`. The key is tokens, never text.
- **Self-describing files.** A fixed header records: magic/version, quant bits,
  save reason (cold / continued-interval / evict / shutdown), cached token count,
  the **context size the snapshot was written for**, timestamps, payload size. The
  directory is the source of truth; any index is a rebuildable cache.
- **One live in-RAM checkpoint; disk is the resume mechanism** when a different
  session displaces the live one or after a restart.
- **Written with plain `read`/`write`, not `mmap`** — deliberately, so restoring
  doesn't add VM mappings alongside the already-mmap'd weights.
- **Boundary-aligned saves** (`--kv-cache-boundary-align-tokens`) and
  **interval saves** (`--kv-cache-continued-interval-tokens 10000`) so a *growing*
  conversation resumes from the nearest boundary and prefills only the delta.
- **LRU within a byte budget** (`--kv-disk-space-mb`).

## 5. The economics the implementer must size against

KV bytes/token are set by cached-layers × KV-heads × head_dim and the cache type;
they are independent of weight quant (see `MODELS.md`). Approximate full-sequence
snapshot sizes:

| model | KV/token | 64k snapshot | 150 GiB holds | restore @ ~7 GB/s | cold re-prefill 64k |
|---|---:|---:|---:|---:|---:|
| `cold-fusion` (dense, f16 K/V) | ~68 KiB | **~4.3 GiB** | ~35 | ~0.6 s | ~85 s |
| `qwen38-27b` (f16 K / q8_0 V) | ~52 KiB | ~3.2 GiB | ~45 | ~0.5 s | ~90 s |
| `muse-glimmer` (GQA 2-KV + SWA) | ~13 KiB global | ~0.8 GiB | ~180 | ~0.1 s | ~85 s |
| A3B MoE (`bigbang`, `kat`, …) | ~20–24 KiB | ~1.3 GiB | ~110 | ~0.2 s | ~30 s |

Typical agentic contexts are 8–32k (0.5–2 GiB), so a 150 GiB budget realistically
holds **dozens to a few hundred** session snapshots. Restore beats re-prefill by
~50–150×. **The GBs are fine; the risk is *redundant* GBs — see §6.**

## 6. Key design decisions (converged; the plan may refine)

- **D1 — Snapshot granularity is the whole ballgame.** `llama_state_seq_save_file`
  writes the **full** sequence state, not a delta. Saving every turn of a growing
  conversation produces overlapping near-duplicate snapshots (8k, 16k, 32k, 64k…)
  and blows the budget. **[REQUIRED]** v1 snapshots **sparsely**: on swap-out, and
  optionally at aligned intervals (e.g. every +16k tokens) — never once per turn.
  Supersede shorter prefixes of the same session (a snapshot whose tokens are a
  prefix of a newer one is redundant and should be dropped or aged out first).
- **D2 — No database; the directory is the truth.** Encode identity in the
  filename: `<cfghash>.<ntokens>.<prefixsha>.kv`. LRU by file `atime`; budget =
  Σ file sizes; reconcile = list the dir. (A sqlite index is allowed later if hit
  counts / richer policy are needed, but must be a rebuildable cache, not
  authoritative.)
- **D3 — Cache key = `(cfghash, SHA1(token_prefix))`.** `cfghash` **[REQUIRED]**
  folds in everything that changes the KV bytes or the state-file format:
  model file path + blob sha, quant, `--cache-type-k`, `--cache-type-v`, `-c`
  (ctx), rope/YaRN params, `-fa`, `--kv-unified`/`--no-kv-unified`, and the
  llama.cpp build commit (the state-file format can change across versions). A
  changed flag ⇒ new `cfghash` ⇒ old snapshots are simply never matched and age
  out. Provide a GC to purge stale `cfghash`es.
- **D4 — Restore = longest matching prefix**, then let the server's LCP reuse
  prefill only the suffix. To find the longest match without storing token lists,
  snapshot only at **aligned boundaries** and, on lookup, hash the incoming prompt
  at those same boundaries (prefix[:8k], [:16k], …) and pick the longest that hits.
  Bounded number of hash lookups, DB-free.
- **D5 — Placement is [OPEN] and is the primary architecture decision** — see §8.
- **D6 — Full-state snapshots in v1** (native `/slots` files). Delta snapshots are
  a v2 optimization requiring a llama.cpp patch.

## 7. Model-specific caveats

- **SWA models (`muse-glimmer`, arch has sliding-window attention on 3/4 layers):**
  windowed layers do **not** hold a full prefix, so "restore a prefix" is only
  partially valid. **[REQUIRED]** exclude SWA models from the cache in v1 (or prove
  the state-file round-trips correctly for them in P0 before including).
- **Draft / MTP / spec-decode models:** several of our entries run `--spec-type
  draft-mtp` or a `-md` draft model. The draft's state may or may not be part of
  `llama_state_seq_save_file`. Verify round-trip in P0; if the draft state isn't
  captured, decide whether that's acceptable (it only affects speculative accept
  rate, not correctness).
- **`ds4`:** exclude — it has its own disk KV cache and is a non-llama.cpp engine.
- **`-np 2`:** models run with 2 slots. Snapshots are per-sequence/slot. The design
  must track *which* slot holds a given session and snapshot/restore that slot.
  Consider running the cached path at `-np 1` if slot-selection races (see §8).

## 8. Architecture — the primary [OPEN] decision

The core hazard is **ordering**: to skip prefill, the KV must be *restored into the
slot before the server processes the request*. Two candidate placements:

### Option A — In-server patch (recommended target)
Patch `tools/server/server.cpp` to do it internally, like ds4:
- on request: hash the prompt token prefix, look up the disk dir, restore into the
  slot before prefill, then prefill the suffix;
- after generation and on `SIGTERM` (swap-out): save the slot keyed by the new
  prefix; enforce budget/LRU.
- **Pros:** race-free (the process owns the KV, the request lifecycle, and its own
  shutdown); can save on graceful exit; no external tokenize round-trips; matches
  ds4's proven design.
- **Cons:** a maintained fork. Our build pipeline (`~/dev/my-llama/build-backends.sh`
  builds from upstream `master`) would carry a patch to re-apply/rebase on each
  rebuild. Feasible (we own the pipeline) but ongoing cost.

### Option B — External sidecar (fastest to validate, no fork)
A proxy in front of llama-swap (fold in / chain with the existing
`proxy/` auth proxy; move llama-swap to `:8081`, sidecar owns `:8080` and `:4096`).
Per request it orchestrates via llama-swap's `/upstream/{model}/…`:
1. determine current vs requested model (`/running`); if a swap is imminent, first
   `POST /upstream/{current}/slots/N?action=save` (returns 200 **before** the swap
   kills the backend — race-free save);
2. ensure the requested model is loaded, then `…/slots/N?action=restore` the
   longest-prefix hit **before** forwarding the real request;
3. forward unchanged; on completion, save; evict over budget.
- **Pros:** no llama.cpp fork; iterate fast; survives our from-master rebuilds.
- **Cons / [OPEN] to resolve:** the **restore-before-prefill ordering** — you must
  load the model and restore the slot *without* the real request prefilling first.
  And with `-np 2`, slot selection (LCP) must land the real request on the restored
  slot. Likely needs `-np 1` on the cached path and a careful load→restore→forward
  sequence. This is the main thing the plan must prove out.

**Recommendation for the plan:** validate the primitive and the *value* with a thin
Option-B prototype (P0/P1), and adopt Option A for production if the ordering in B
proves fragile. Do not commit to an architecture before P0 passes.

## 9. Data & formats

- **Snapshot payload:** the native `/slots ... action=save` file (opaque; produced
  by `llama_state_seq_save_file`). Do **not** parse it — treat as a blob; llama.cpp
  validates its own header on load.
- **Filename:** `<cfghash8>.<ntokens>.<prefixsha16>.kv`. All lookup metadata lives
  in the name; `atime` is the LRU clock.
- **Atomic writes [REQUIRED]:** write to `*.tmp`, fsync, rename. A crash leaves a
  `*.tmp` (ignored + GC'd), never a half file under a real name.
- **Prefix hashing:** incremental over token ids at aligned boundaries (§6-D4).

## 10. Eviction, budget, GC

- Budget `B` bytes (config; default 150 GiB), on the NVMe (`/home`, 400+ GiB free).
- On save: write, then while `Σsizes > B` delete the min-`atime` entry that is **not
  currently loaded/in-use** (pin the live session's snapshot).
- `atime` (or a touch/rename) updated on every hit.
- GC command: drop entries whose `cfghash` no longer matches any registered model,
  plus stray `*.tmp`.

## 11. Correctness & failure handling [REQUIRED]

- Restore only when `cfghash` matches the running server exactly (§6-D3).
- After restore, the restored tokens **must** be an exact prefix of the request
  tokens (the server's own reuse enforces this; a mismatch just means a normal
  prefill).
- **Every** failure path (missing model, bad/opaque state file, load error, disk
  full, tokenizer drift) falls back to normal prefill and logs a metric. A cache
  problem must never surface as a user-visible error or a wrong answer.

## 12. Observability [REQUIRED]

Expose: hit/miss counts, prefill-tokens-avoided, bytes on disk vs budget, restore
latency, save latency, evictions, per-model breakdown. Surface via a stats endpoint
and/or a `llm kvcache status` subcommand / `make` target. A correctness self-check
(restore a session, continue, compare first-N greedy tokens vs a cold run) is
valuable as a smoke test.

## 13. Config surface [OPEN placement]

Knobs the plan must site (candidates: `config/config.header.yaml` global block, a
new `config/kvcache.*`, and/or per-model flags in `registry.json`):
`enabled`, `dir`, `budget_mib`, `min_tokens_to_cache` (skip tiny prompts),
`boundary_align_tokens`, `interval_tokens`, `per_model_include/exclude`
(exclude SWA + ds4 in v1). If Option A, this also means adding `--slot-save-path`
(and any new flags) to the `server`/`common` macros the renderer emits
(`bin/llm` `render_config`).

## 14. Interaction with the existing stack

- **llama-swap:** models already run with slots (`-np 2`); Option A/B both need
  `--slot-save-path` on the server command. llama-swap's `--cache-ram` can stay on
  as a hot RAM tier. Note llama-swap sends `SIGTERM` on swap; if Option A saves on
  shutdown, the model's `unloadTimeout` must be long enough to flush ~a few GB
  (~<1 s at NVMe speed, but set generous, e.g. 30 s).
- **`bin/llm`:** likely grows `llm kvcache {status,gc,purge}`; if Option A, the
  renderer adds the slot-save flag; `cfghash` should be derivable from a registry
  entry + macros.
- **`proxy/` (auth):** if Option B, this is where the sidecar logic lands (extend
  it) or chains behind it.
- **Registry/config are the source of truth** — any new serving flags must flow
  through `registry.json` → `llm render`, never hand-edited into `config.yaml`.

## 15. Phased milestones (for the plan to expand into tasks)

- **P0 — Primitive + value gate (GO/NO-GO).** On `cold-fusion` with our exact
  serving flags (`-fa on`, `-c 196608`, f16 K/V, `-np 2`, `--spec-type draft-mtp`):
  manually `save` a slot after a ~16k prompt, kill+restart, `restore`, send the same
  prompt + one more turn, and confirm the server prefills only the suffix
  (inspect `usage`/timings prompt-eval token count). Record file size, save/restore
  latency. **Gate:** restore round-trips correctly and skips prefill. If it doesn't
  work with `-fa`/draft/`-np 2`, find the minimal working flag set before P1.
- **P1 — MVP swap-survival + LRU.** One snapshot per (model, session), saved on
  swap-out, restored on swap-back; 150 GiB LRU. Measure re-prefill time saved on a
  cold-fusion → bigbang → cold-fusion bounce. This alone delivers G1/G2/G5.
- **P2 — General prefix cache.** Boundary-aligned + interval snapshots; longest-
  prefix match for *any* request (not just swaps); supersession of redundant
  shorter prefixes.
- **P3 — (optional) hardening.** Move Option B → Option A if warranted; or delta
  snapshots (patch); richer policy; multi-model tiering.

## 16. Acceptance tests

- **Correctness:** restored+continued session yields identical greedy first-N
  tokens vs a cold run (temp 0). Changing any `cfghash` input (e.g. flip
  `--cache-type-v`) makes prior snapshots un-matched (no restore, no error).
- **Perf:** warm hit prompt-eval tokens ≈ suffix only; TTFT ≪ cold; report the
  ratio per model.
- **Budget:** disk never exceeds `B`; LRU evicts oldest; in-use never evicted;
  crash mid-save leaves only an ignored `*.tmp`.
- **Transparency:** pi/opencode/benches unchanged and correct; ds4 + SWA excluded.

## 17. Open questions for the planning agent

1. **Option A vs B** (§8) — decide after P0; the restore-before-prefill ordering and
   `-np 2` slot selection are the deciding factors.
2. Does `llama_state_seq_save_file` capture draft/MTP state, and does it round-trip
   under `-fa on`? (P0.)
3. Can SWA (`muse-glimmer`) be cached at all usefully, or exclude permanently?
4. Snapshot-on-`SIGTERM` (Option A) vs pre-swap save (Option B) — which is more
   robust given llama-swap's unload timeout?
5. Boundary/interval sizes: what `boundary_align_tokens` / `interval_tokens`
   balance redundancy vs hit rate for real agent traffic?
6. Where does `cfghash` get computed and stored so both the renderer and the
   cache agree?

## 18. References

- Our stack: `README.md`, `MODELS.md`, `config/config.header.yaml` (macros),
  `bin/llm` (`render_config`, registry), `proxy/main.go` (auth proxy),
  `bench/speed/kv-probe.py` + `results/speed/*.md` (measured prefill/decode rates).
- llama.cpp (built at `~/opt/llama.cpp-vulkan`, source `~/src/llama.cpp`,
  commit in `~/opt/llama.cpp-COMMIT`): `tools/server/server.cpp` `/slots` routes;
  `llama_state_seq_save_file` / `llama_state_seq_load_file`; flags
  `--slot-save-path`, `--cache-ram`, `--cache-reuse`, `--slot-prompt-similarity`.
- ds4 reference implementation: `~/dev/ds4-hip/README.md` § "Disk KV Cache";
  `ds4_server.c` (KV cache functions, save-reason enum, boundary/interval knobs).
