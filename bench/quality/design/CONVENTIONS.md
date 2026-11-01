# Design-task quality suite: conventions

Measures "if I ask it to design a non-trivial piece of code, is the result correct, fast, stable, bug-free and
well-engineered" -- not perplexity. Each task is a self-contained Python 3.12 (stdlib only) design problem with a
precise public interface so hidden tests can grade it.

## Task layout: `tasks/<task_id>/`

| file | purpose |
|---|---|
| `prompt.md` | What the model sees. A realistic engineering request: the problem, the EXACT public interface (module `solution`, class/function names, signatures, exception types, semantics, edge-case behaviour, thread-safety requirements, complexity expectations). It must be complete enough that a correct implementation passes every hidden test, without revealing the tests. Do NOT include the deliverable-format instructions; the generator appends them. |
| `reference.py` | A high-quality reference implementation of the interface (stdlib only). It must pass 100% of `test_hidden.py` and is the performance baseline. |
| `test_hidden.py` | pytest suite. `import solution as S` (the grader puts the candidate as `solution.py` on `sys.path`). Groups, by test-name prefix so the grader can score categories: `test_basic_*` (spec'd behaviour), `test_edge_*` (edge cases + error handling/validation stated in the prompt), `test_prop_*` (seeded randomized property tests against a brute-force model; use `random.Random(seed)`, no hypothesis; >= 200 random cases each, still fast), `test_conc_*` (thread-safety / concurrency, only where the prompt requires it), `test_stress_*` (scale/robustness: large inputs, deep nesting, many ops -- must complete well under the timeout with a correct efficient implementation). Every test must finish in < 5 s with the reference; the grader runs with `--timeout=30` per test. No test may depend on wall-clock speed except via generous bounds; perf is measured separately. Tests must not rely on private names, only on the spec'd interface. |
| `bench.py` | `def bench(mod) -> float`: runs a fixed, deterministic, realistic workload against module `mod` (seeded RNG) and returns elapsed seconds (`time.perf_counter`). Should take 0.2-1.5 s with the reference. Must validate a few results so a wrong-but-fast solution cannot win (raise AssertionError if wrong). |
| `mutants/` | 2-3 plausible buggy variants of the reference (`m1.py`, `m2.py`, ...), each a full module with ONE realistic bug (off-by-one, missing lock, wrong merge boundary, unescaped edge, etc.). Each must FAIL at least one hidden test. This proves the tests have teeth. |

## Rules

- Python 3.12, stdlib only, for both the interface and the references. Candidates are told stdlib only.
- Deterministic tests (seeded). No network, no files outside `tmp_path`.
- Clocks / sleeps must be injectable where timing matters (e.g. `clock=time.monotonic`, `sleep=time.sleep` parameters) so tests use fake time and stay fast and deterministic. Concurrency tests may use real threads.
- Validate: `cd tasks/<id> && cp reference.py /tmp/x/solution.py` style: run `uv run --with pytest --with pytest-timeout pytest -q -p no:cacheprovider --timeout=30 test_hidden.py` with the reference importable as `solution` -> all pass; each mutant -> at least one failure; `bench(reference)` returns 0.2-1.5 s. Use the helper `validate.py` in this directory: `python3 validate.py <task_id>`.
