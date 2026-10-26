## Why

Benchmarks find the best configuration for a model, but nothing carries that
result into production. Someone copies flags into `registry.json` or
`settings.json` by hand, so production can drift from what was measured, or
never pick up the winner at all. The taichi host also needs a quick way to
probe a `llama-server` it started itself, which the stack-only rule forbids today.

## What Changes

- Add a promote step: `bench/promote.py RESULTS --variant NAME` applies the
  variant patch recorded in a results file to the production registry and
  engines, re-renders, and restarts the service.
- Before it writes anything, promote checks that production has not changed
  since the measurement. Every result now records a digest of the production
  entry and engine that the variant was applied to. Promote refuses when the
  digest differs, or when the production render does not equal the measured
  command. A patch to an engine that other models share needs an explicit flag.
- After the restart, promote warms the model. If the warm-up fails, it restores
  the previous files and restarts again.
- Every promotion is appended to `results/promotions.tsv`.
- Add a standalone mode to `bench/speed/kv-probe.py` (`--server URL`) for a
  server started outside the stack. Its results are labelled `standalone`, and
  promote refuses them: a standalone finding must be re-measured as a stack
  variant first.
- **BREAKING** (results format): stack result files gain a `base_digest`
  provenance column (TSV) or key (kv-probe JSON). Promote refuses results
  recorded before this change.

## Capabilities

### New Capabilities

- `benchmark-promotion`: applying a measured variant to production, the checks
  that guard it, rollback, and the promotion log.

### Modified Capabilities

- `stack-benchmarks`: "Benchmarks run the production configuration" allows a
  labelled standalone mode, and results record the production base they were
  measured against.

## Impact

- Code: `bench/lib/stack.py` (base digest in `record()`), `bench/lib/common.py`
  (`PROVENANCE`), a new `bench/lib/promote.py` and `bench/promote.py`,
  `bench/speed/kv-probe.py`.
- Host files: promote rewrites `registry.json` and `settings.json` through
  atomic writes. `settings.json` is re-serialised with 2-space JSON, so
  hand-formatting in it is not kept.
- Results: new `results/promotions.tsv`. Existing result files are unchanged.
- No change to `llms` itself, its CLI, or the rendered config format.
