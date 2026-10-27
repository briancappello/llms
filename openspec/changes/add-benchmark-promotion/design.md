## Context

See proposal.md for motivation; the requirements are in the two delta specs.

Current state the design builds on:

- `bench/lib/stack.py` builds a variant by applying an RFC 7386 merge patch
  (`merge_patch`) to copies of the production registry entries and engines, then
  renders an isolated config. `Stack.record(model)` returns the provenance every
  driver writes: `variant`, `patch` (the whole variant as JSON), `rendered_cmd`
  and `production` (the service state).
- TSV drivers write those four as `PROVENANCE` columns through `common.TSV`.
  `bench/speed/kv-probe.py` writes one JSON file with a `provenance` object.
- `rendered_cmd` keeps llama-swap macros unexpanded (`${common}`, `${PORT}`), so
  a change to a header macro does not show up in it.
- `ModelManager` already has what promotion needs: `load_registry`, `render`,
  `_verify_service`, `service("restart")`, `wait_ready` and `use` (a warm-up
  that validates the answer). `atomic_write` writes files.
- Production `settings.json` is hand-written JSON. `llms` never writes it today.

## Goals / Non-Goals

**Goals:**

- Production after promotion is the measured variant, proven by recomputing the
  same digest and render the benchmark recorded.
- No partial state: any failure leaves production byte-identical to before.

**Non-Goals:**

- Choosing the winner. The operator names the variant; promotion does not rank
  results.
- Promotion across hosts. A result is promoted on the host that measured it,
  because the base digest covers that host's engines.
- Promoting old results that lack a base digest.
- Adding promotion to the `llms` CLI. It stays a bench tool next to the
  drivers that produce its input.

## Decisions

### D1. Base digest: entry, engines and header macros

`base_digest` is the SHA-256 of a canonical JSON document (`sort_keys=True`,
compact separators) holding:

- `entry`: the model's production registry entry;
- `engines`: a map from engine name to its production definition (or null) for
  the entry's engine and every engine the variant patches;
- `macros`: the `macros` map from the production header.

`Stack.record()` computes it from the production files, not the isolated copy,
and it becomes a fifth `PROVENANCE` column. kv-probe stores it in its
`provenance` object.

Why: `rendered_cmd` alone misses engine `env` (device pins) and header macros.
Alternative considered: storing the whole rendered model block. It would catch
the same drift, but it puts large YAML blobs in every TSV row and still does
not identify which input changed.

### D2. Promotion reads the patch from the results, never from the operator

`promote` takes `RESULTS --variant NAME [--model ID]`. It collects the rows for
that variant (and model, when the file has several), and requires that they all
agree on `patch`, `base_digest` and `rendered_cmd`. Only OK rows count: a
LOAD_FAIL row records the patch too, but it is not evidence that the variant serves.

Why: typing the patch again is the hand copy this change removes.

### D3. Checks before any write

In order, each failure exits non-zero with nothing written:

1. The rows are stack mode and carry a base digest (D2 agreement included).
2. The patch touches no registry entry except the measured model.
3. The current production base digest equals the recorded one.
4. Apply the patch in memory with `stack.merge_patch` (the same function the
   benchmark used) and render it with `render_config`. The model's `cmd` must
   equal `rendered_cmd`.
5. Compare every other chat model's rendered `cmd` before and after the patch.
   If any changed, list them, and require `--accept-shared-engine`.

### D4. Writes, restart, warm-up, rollback

Snapshot the bytes of `settings.json`, `registry.json` and the rendered output.
Then:

- registry: written the way `ModelManager._commit` writes it (`indent=2`,
  `sort_keys=True`);
- settings: raw JSON loaded, `engines.<name>` merge-patched, written with
  `indent=2`, so every other key and engine survives. (`Settings` rejects
  `_comment` and unknown keys in a real `settings.json`, so none can exist.)
- render through `ModelManager.render()`.

If the service was active: `service("restart")`, `wait_ready`, `use(model)`.
On any exception after the first write, restore all three snapshots with
`atomic_write`, restart again if the service was active, and re-raise. If the
service was inactive, it stays stopped and the log row says `warmup=skipped`.

Alternative considered: writing through `llms` CLI subcommands. There is none
for engines, and adding one widens `llms` for a bench-only need.

### D5. Promotion log

`results/promotions.tsv` (in the repo, beside the results it points to) is
appended only after success. `common.TSV` truncates on open, so promote writes
the header only when the file is new and appends rows with `csv`. Columns:
`utc`, `host`, `model`, `variant`, `results`, `patch`, `base_digest`,
`rendered_cmd`, `warmup`.

### D6. kv-probe standalone mode

`--server URL` (a llama-server base URL) replaces the stack: `/tokenize` and
`/v1/chat/completions` are taken from that URL, the production service is not
touched, `--model` becomes only a label, and the JSON gets
`provenance = {"mode": "standalone", "server": URL}`. `--server` with
`--variant` is an error. Stack mode adds `"mode": "stack"`. TSV drivers do not
get a standalone mode; promote treats a TSV without `base_digest` as unprovable.

This is taichi's `--tokenize` idea. A base URL replaces it because standalone
mode needs the chat endpoint as well, and the old flag paired with a `--url`
option that the stack rewrite removed.

## Risks / Trade-offs

- [`settings.json` loses hand formatting on its first promotion] → It is valid
  JSON with the same values. The proposal states it, and `git diff` of a backup
  shows only layout.
- [A results file from another host is promoted by mistake] → The base digest
  covers absolute engine paths and entry paths, so it almost always differs and
  promotion refuses.
- [Rollback restart also fails, for example the GPU is wedged] → Files are still
  restored first; the error reports both failures so the operator can run
  `llms restart`.
- [Two engines changed by one variant, one shared] → D3 step 5 compares every
  model's render, so any shared effect is listed, whatever the patch shape.
- [Benchmarks recorded before this change cannot be promoted] → Intended; a
  re-run takes minutes and gives a fresh, comparable measurement.

## Migration Plan

1. Ship the `base_digest` column and kv-probe modes first. Existing results stay
   as they are; drivers write the new column from then on.
2. Ship `promote`. Test on p1 by promoting a `production` variant (no-op, log
   row only), then a real one-flag variant, then promote the previous value back.
3. Rollback of the change itself: revert the commits. Result files with the
   extra column still read correctly, because `read_tsv` uses header names.
