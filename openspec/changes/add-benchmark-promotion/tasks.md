# Tasks

## 1. Base digest in results

- [x] 1.1 Add `base_digest(production_settings, registry, model, variant)` to `bench/lib/stack.py` (D1): canonical JSON of the entry, the entry's engine plus every patched engine (null when absent), and the header `macros`. Verify: unit tests show the digest changes when the entry's `ctx`, the engine's `env`, or a header macro changes, and stays the same when `settings.json` is re-indented or its keys are reordered.
- [x] 1.2 Return `base_digest` from `Stack.record()` and add it to `PROVENANCE` in `bench/lib/common.py`. Verify: a `test_bench_stack.py` test with the stub llama-swap shows `record()` has the key and equals `base_digest` computed from the production files, not the isolated copy.
- [x] 1.3 Check every driver that writes `PROVENANCE` still runs. Verify: `rg -l PROVENANCE bench` lists only drivers that splat `stack.record()` into rows, and the test suite passes.

## 2. kv-probe modes

- [x] 2.1 Record `"mode": "stack"` and `base_digest` in kv-probe's `provenance`. Verify: a test runs `kv-probe` against the stub stack and reads both keys from the output JSON.
- [ ] 2.2 Add `--server URL` standalone mode (D6): `/tokenize` and `/v1/chat/completions` from the URL, no `Stack`, provenance `{"mode": "standalone", "server": URL}`; `--server` with `--variant` is a parse error. Verify: a test against a local stub HTTP server writes a standalone JSON, and a mocked `Stack` is never constructed.

## 3. Promote

- [ ] 3.1 Implement result loading in `bench/lib/promote.py` (D2): TSV and kv-probe JSON; select OK rows by variant and optional model; require agreement on patch, digest and command; refuse standalone results and results without a digest; name the available variants when the requested one is missing. Verify: unit tests for each refusal and for a TSV holding two models.
- [ ] 3.2 Implement the pre-write checks (D3 steps 2-5) against a temporary production config dir. Verify: tests for a patch touching another entry, an edited production entry (digest mismatch), a render mismatch, and a shared-engine change listed and refused without `--accept-shared-engine`.
- [ ] 3.3 Implement writes, restart, warm-up and rollback (D4) with an injected manager. Verify: tests show (a) success writes the patched registry and settings with `_comment` and unknown keys kept; (b) a failing warm-up restores all three files byte-identical and restarts twice; (c) an inactive service is not restarted and the warm-up is skipped; (d) the `production` variant changes no file.
- [ ] 3.4 Append to `results/promotions.tsv` on success only (D5), with the header written once. Verify: two successful promotions give one header and two rows; a failed one adds none.
- [ ] 3.5 Add the `bench/promote.py` CLI (`RESULTS --variant NAME [--model ID] [--config-dir DIR] [--accept-shared-engine]`). Verify: `bench/promote.py --help` lists the options, and a CLI test refuses a pre-digest results file with exit code non-zero.

## 4. Docs

- [ ] 4.1 Document the measure, conclude and promote loop, and standalone mode, in `README.md`'s benchmark section and in kv-probe's docstring. Verify: every command shown in the new text has been run once on p1.

## 5. Host verification (p1)

- [ ] 5.1 On p1, run `kv_sweep.py` (or `depth_sweep.py`) for one model with `production` and one one-flag variant, then promote `production`. Verify: production files are unchanged and `results/promotions.tsv` has a row with `warmup` OK.
- [ ] 5.2 Promote the one-flag variant. Then run a new sweep with a variant that restores the old value, and promote that. Verify: `llms doctor --live` is green after each step, the rendered command matches the recorded one, and the final production files equal the backup taken before 5.1.
- [ ] 5.3 Edit the model's production entry after a sweep and try to promote from it. Verify: promotion refuses with a base digest mismatch, and the files are unchanged.
