"""Build-pipeline workload: several thousand small tasks with random dependencies, some failures."""
import random
import time


def _graph(mod, rng, n, fail_rate):
    names = [f"step{i}" for i in range(n)]
    deps = {}
    for i, nm in enumerate(names):
        lo = max(0, i - 200)  # mostly local dependencies, like a real pipeline
        k = rng.randint(0, min(i - lo, 4))
        deps[nm] = rng.sample(names[lo:i], k)
    insertion = names[:]
    rng.shuffle(insertion)
    failing = {nm for nm in names if rng.random() < fail_rate}
    g = mod.TaskGraph()
    for nm in insertion:
        def fn(_nm=nm, _fail=nm in failing, **kw):
            if _fail:
                raise RuntimeError(_nm)
            return (len(_nm) + sum(kw.values())) % 1_000_003
        g.add_task(nm, fn, deps[nm])
    return g, names, deps, failing


def _expected(names, deps, failing):
    results, bad, errors = {}, set(), set()
    for nm in names:  # names are already a valid topological order
        if any(d in bad for d in deps[nm]):
            bad.add(nm)
        elif nm in failing:
            bad.add(nm)
            errors.add(nm)
        else:
            results[nm] = (len(nm) + sum(results[d] for d in deps[nm])) % 1_000_003
    return results, errors


def bench(mod) -> float:
    rng = random.Random(4242)
    specs = [_graph(mod, rng, 12000, 0.0), _graph(mod, rng, 12000, 0.0005), _graph(mod, rng, 50000, 0.0)]
    t0 = time.perf_counter()
    r1 = specs[0][0].run(max_workers=8)
    r2 = specs[1][0].run(max_workers=4)
    r3 = specs[0][0].run(max_workers=1, fail_fast=True)
    topo = specs[2][0].topological_order()
    elapsed = time.perf_counter() - t0

    for (g, names, deps, failing), r in ((specs[0], r1), (specs[1], r2), (specs[0], r3)):
        exp_results, exp_errors = _expected(names, deps, failing)
        if r.results != exp_results:
            raise AssertionError("wrong results")
        if set(r.errors) != exp_errors:
            raise AssertionError("wrong errors")
        if len(r.results) + len(r.errors) + len(r.skipped) != len(names):
            raise AssertionError("tasks unaccounted for")
    if not r2.errors or not r2.skipped:
        raise AssertionError("expected some failures and skips")
    _, names, deps, _ = specs[2]
    pos = {n: i for i, n in enumerate(topo)}
    if len(pos) != len(names) or any(pos[d] >= pos[n] for n in names for d in deps[n]):
        raise AssertionError("invalid topological order")
    return elapsed
