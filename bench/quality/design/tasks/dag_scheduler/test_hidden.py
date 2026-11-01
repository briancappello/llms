import heapq
import random
import threading
import time

import pytest

import solution as S


def const(v):
    return lambda **kw: v


def check_invariants(g_names, r):
    names = set(g_names)
    assert r.results.keys() & r.errors.keys() == set()
    assert r.results.keys() & r.skipped == set()
    assert r.errors.keys() & r.skipped == set()
    assert set(r.results) | set(r.errors) | set(r.skipped) == names
    assert len(r.order) == len(set(r.order))
    assert set(r.order) == set(r.results) | set(r.errors)


def assert_valid_cycle(cycle, deps):
    assert isinstance(cycle, list)
    assert len(cycle) >= 2
    assert cycle[0] == cycle[-1]
    assert len(set(cycle[:-1])) == len(cycle) - 1
    for a, b in zip(cycle, cycle[1:]):
        assert b in deps[a], (a, b)


# --------------------------------------------------------------------------------------------- basic
def test_basic_results_passed_as_kwargs():
    g = S.TaskGraph()
    g.add_task("fetch", lambda: 10)
    g.add_task("scale", lambda fetch: fetch * 3, deps=["fetch"])
    g.add_task("offset", lambda fetch: fetch + 1, deps=["fetch"])
    g.add_task("combine", lambda scale, offset: (scale, offset), deps=["scale", "offset"])
    r = g.run()
    assert r.results == {"fetch": 10, "scale": 30, "offset": 11, "combine": (30, 11)}
    assert r.errors == {} and r.skipped == set()
    check_invariants(["fetch", "scale", "offset", "combine"], r)


def test_basic_exact_kwargs_and_identity():
    g = S.TaskGraph()
    obj = object()
    seen = {}
    g.add_task("a", lambda: obj)
    g.add_task("b", lambda: [1])

    def c(**kw):
        seen.update(kw)
        return "c"

    g.add_task("c", c, deps=["b", "a"])
    g.add_task("d", lambda **kw: sorted(kw))
    r = g.run()
    assert set(seen) == {"a", "b"} and seen["a"] is obj and seen["b"] == [1]
    assert r.results["d"] == []


def test_basic_non_identifier_names():
    g = S.TaskGraph()
    g.add_task("fetch-assets", lambda: 2)
    g.add_task("build step", lambda **kw: kw["fetch-assets"] * 5, deps=["fetch-assets"])
    assert g.run().results["build step"] == 10


def test_basic_dependencies_finish_before_dependents_start():
    g = S.TaskGraph()
    log = []
    lock = threading.Lock()

    def mk(name):
        def fn(**kw):
            with lock:
                log.append(("start", name))
            time.sleep(0.01)
            with lock:
                log.append(("end", name))
            return name
        return fn

    deps = {"a": [], "b": [], "c": ["a"], "d": ["a", "b"], "e": ["c", "d"], "f": []}
    for n, ds in deps.items():
        g.add_task(n, mk(n), ds)
    r = g.run(max_workers=4)
    pos = {e: i for i, e in enumerate(log)}
    for n, ds in deps.items():
        for d in ds:
            assert pos[("end", d)] < pos[("start", n)]
    for n, ds in deps.items():
        for d in ds:
            assert r.order.index(d) < r.order.index(n)


def test_basic_topological_order_insertion_tiebreak():
    g = S.TaskGraph()
    g.add_task("c", const(1))
    g.add_task("a", const(1), deps=["c"])
    g.add_task("b", const(1))
    assert g.topological_order() == ["c", "a", "b"]


def test_basic_topological_order_forward_refs():
    g = S.TaskGraph()
    g.add_task("zeta", const(0), deps=["beta", "alpha"])
    g.add_task("beta", const(0), deps=["alpha"])
    g.add_task("gamma", const(0))
    g.add_task("alpha", const(0))
    g.add_task("delta", const(0), deps=["gamma"])
    assert g.topological_order() == ["gamma", "alpha", "beta", "zeta", "delta"]


def test_basic_topological_order_earliest_ready_first():
    g = S.TaskGraph()
    g.add_task("x", const(0))
    g.add_task("y", const(0), deps=["w"])
    g.add_task("z", const(0))
    g.add_task("w", const(0), deps=["x"])
    # after x: ready = {z(2), w(3)} -> z first; then w; then y
    assert g.topological_order() == ["x", "z", "w", "y"]


def test_basic_single_worker_runs_in_topological_order():
    g = S.TaskGraph()
    executed = []

    def rec(n):
        return lambda **kw: executed.append(n) or n

    g.add_task("e", rec("e"), ["b"])
    g.add_task("d", rec("d"))
    g.add_task("b", rec("b"))
    g.add_task("a", rec("a"), ["d", "e"])
    g.add_task("c", rec("c"), ["d"])
    topo = g.topological_order()
    assert topo == ["d", "b", "e", "a", "c"]
    r = g.run(max_workers=1)
    assert r.order == topo
    assert executed == topo


def test_basic_failure_propagates_to_transitive_dependents():
    g = S.TaskGraph()
    err = RuntimeError("transcode failed")

    def bad(src):
        raise err

    g.add_task("src", const(1))
    g.add_task("bad", bad, deps=["src"])
    g.add_task("b", const(2), deps=["bad"])
    g.add_task("c", const(3), deps=["b"])
    g.add_task("ok", const(4), deps=["src"])
    g.add_task("d", const(5), deps=["c", "ok"])
    g.add_task("indep", const(6))
    r = g.run(max_workers=3)
    assert r.errors == {"bad": err} and r.errors["bad"] is err
    assert r.skipped == {"b", "c", "d"}
    assert r.results == {"src": 1, "ok": 4, "indep": 6}
    check_invariants(["src", "bad", "b", "c", "ok", "d", "indep"], r)


def test_basic_multiple_failures_and_order_contains_failed():
    g = S.TaskGraph()

    def boom(**kw):
        raise ValueError("x")

    g.add_task("f1", boom)
    g.add_task("f2", boom)
    g.add_task("s1", const(1), deps=["f1"])
    g.add_task("s2", const(2), deps=["f1", "f2"])
    g.add_task("ok", const(3))
    r = g.run(max_workers=1)
    assert set(r.errors) == {"f1", "f2"}
    assert all(isinstance(e, ValueError) for e in r.errors.values())
    assert r.skipped == {"s1", "s2"}
    assert r.order == ["f1", "f2", "ok"]


def test_basic_fail_fast_stops_new_tasks_running_ones_finish():
    g = S.TaskGraph()

    def bad():
        raise RuntimeError("fail")

    def slow():
        time.sleep(0.3)
        return "slow-done"

    started = []
    g.add_task("bad", bad)
    g.add_task("slow", slow)
    for n in ("c", "d", "e"):
        g.add_task(n, (lambda n: lambda: started.append(n) or n)(n))
    r = g.run(max_workers=2, fail_fast=True)
    assert set(r.errors) == {"bad"}
    assert r.results == {"slow": "slow-done"}
    assert r.skipped == {"c", "d", "e"}
    assert started == []
    assert r.order == ["bad", "slow"]


def test_basic_without_fail_fast_independent_tasks_run():
    g = S.TaskGraph()

    def bad():
        raise RuntimeError("fail")

    g.add_task("bad", bad)
    g.add_task("slow", lambda: time.sleep(0.05) or "s")
    for n in ("c", "d", "e"):
        g.add_task(n, const(n))
    g.add_task("after", const(0), deps=["bad", "c"])
    r = g.run(max_workers=2)
    assert r.results == {"slow": "s", "c": "c", "d": "d", "e": "e"}
    assert r.skipped == {"after"}


def test_basic_fail_fast_single_worker_sequence():
    g = S.TaskGraph()

    def bad():
        raise KeyError("k")

    g.add_task("a", const("A"))
    g.add_task("b", bad)
    g.add_task("c", const("C"))
    g.add_task("d", const("D"), deps=["a"])
    r = g.run(max_workers=1, fail_fast=True)
    assert r.order == ["a", "b"]
    assert r.results == {"a": "A"}
    assert r.skipped == {"c", "d"}


def test_basic_run_twice_reexecutes_and_graph_extendable():
    g = S.TaskGraph()
    calls = []
    g.add_task("a", lambda: calls.append(1) or len(calls))
    r1 = g.run()
    r2 = g.run()
    assert r1.results == {"a": 1} and r2.results == {"a": 2}
    g.add_task("b", lambda a: a * 100, deps=["a"])
    r3 = g.run()
    assert r3.results == {"a": 3, "b": 300}


def test_basic_runresult_type():
    g = S.TaskGraph()
    g.add_task("a", const(1))
    r = g.run()
    assert isinstance(r, S.RunResult)
    assert isinstance(r.results, dict) and isinstance(r.errors, dict)
    assert isinstance(r.skipped, set) and isinstance(r.order, list)


# --------------------------------------------------------------------------------------------- edge
def test_edge_empty_graph():
    g = S.TaskGraph()
    assert g.topological_order() == []
    r = g.run()
    assert r.results == {} and r.errors == {} and r.skipped == set() and r.order == []


def test_edge_duplicate_name_rejected_graph_unchanged():
    g = S.TaskGraph()
    g.add_task("a", const(1))
    with pytest.raises(ValueError):
        g.add_task("a", const(2))
    assert g.run().results == {"a": 1}


def test_edge_duplicate_dependency_rejected():
    g = S.TaskGraph()
    g.add_task("a", const(1))
    with pytest.raises(ValueError):
        g.add_task("b", const(2), deps=["a", "a"])
    assert g.topological_order() == ["a"]
    g.add_task("b", const(2), deps=["a"])  # name still free


def test_edge_type_errors():
    g = S.TaskGraph()
    with pytest.raises(TypeError):
        g.add_task(1, const(1))
    with pytest.raises(TypeError):
        g.add_task("a", "not callable")
    with pytest.raises(TypeError):
        g.add_task("a", const(1), deps="fetch")
    g.add_task("a", const(1))
    assert g.topological_order() == ["a"]


def test_edge_deps_any_iterable():
    g = S.TaskGraph()
    g.add_task("a", const(1))
    g.add_task("b", const(2))
    g.add_task("c", lambda a, b: a + b, deps=(d for d in ["a", "b"]))
    g.add_task("d", lambda c: c, deps=("c",))
    assert g.run().results["d"] == 3


def test_edge_missing_dependency():
    g = S.TaskGraph()
    ran = []
    g.add_task("a", lambda: ran.append("a"))
    g.add_task("b", lambda **kw: ran.append("b"), deps=["a", "ghost", "phantom"])
    g.add_task("c", lambda **kw: ran.append("c"), deps=["nope"])
    with pytest.raises(S.MissingDependencyError) as ei:
        g.run()
    assert isinstance(ei.value, S.GraphError)
    assert ei.value.task == "b" and ei.value.dependency == "ghost"
    with pytest.raises(S.MissingDependencyError):
        g.topological_order()
    assert ran == []


def test_edge_missing_dependency_reported_before_cycle():
    g = S.TaskGraph()
    g.add_task("a", const(0), deps=["b"])
    g.add_task("b", const(0), deps=["a"])
    g.add_task("c", const(0), deps=["missing"])
    with pytest.raises(S.MissingDependencyError) as ei:
        g.topological_order()
    assert (ei.value.task, ei.value.dependency) == ("c", "missing")


def test_edge_cycle_detected_before_running():
    g = S.TaskGraph()
    ran = []
    deps = {"root": [], "a": ["root", "c"], "b": ["a"], "c": ["b"], "leaf": ["c"]}
    for n, ds in deps.items():
        g.add_task(n, (lambda n: lambda **kw: ran.append(n))(n), ds)
    with pytest.raises(S.CycleError) as ei:
        g.run()
    assert isinstance(ei.value, S.GraphError)
    assert_valid_cycle(ei.value.cycle, deps)
    assert set(ei.value.cycle) == {"a", "b", "c"}
    assert ran == []
    with pytest.raises(S.CycleError) as ei2:
        g.topological_order()
    assert_valid_cycle(ei2.value.cycle, deps)


def test_edge_self_dependency_cycle():
    g = S.TaskGraph()
    g.add_task("x", const(0))
    g.add_task("a", const(0), deps=["x", "a"])
    with pytest.raises(S.CycleError) as ei:
        g.topological_order()
    assert ei.value.cycle == ["a", "a"]


def test_edge_two_node_cycle():
    g = S.TaskGraph()
    g.add_task("a", const(0), deps=["b"])
    g.add_task("b", const(0), deps=["a"])
    with pytest.raises(S.CycleError) as ei:
        g.run()
    assert ei.value.cycle in (["a", "b", "a"], ["b", "a", "b"])


def test_edge_invalid_max_workers():
    g = S.TaskGraph()
    ran = []
    g.add_task("a", lambda: ran.append(1))
    for mw in (0, -1):
        with pytest.raises(ValueError):
            g.run(max_workers=mw)
    assert ran == []


def test_edge_exception_types_and_none_results():
    g = S.TaskGraph()

    class Custom(Exception):
        pass

    g.add_task("none", lambda: None)
    g.add_task("uses_none", lambda none: none is None, deps=["none"])
    g.add_task("custom", lambda: (_ for _ in ()).throw(Custom("c")))
    r = g.run()
    assert r.results == {"none": None, "uses_none": True}
    assert isinstance(r.errors["custom"], Custom)


def test_edge_failure_in_diamond_skips_join_only():
    g = S.TaskGraph()

    def bad(top):
        raise RuntimeError

    g.add_task("top", const(1))
    g.add_task("left", bad, deps=["top"])
    g.add_task("right", lambda top: top + 1, deps=["top"])
    g.add_task("join", lambda left, right: 0, deps=["left", "right"])
    g.add_task("right2", lambda right: right * 2, deps=["right"])
    r = g.run(max_workers=2)
    assert r.results == {"top": 1, "right": 2, "right2": 4}
    assert r.skipped == {"join"}


# --------------------------------------------------------------------------------------------- conc
def test_conc_independent_tasks_run_in_parallel():
    g = S.TaskGraph()
    barrier = threading.Barrier(8, timeout=3)
    for i in range(8):
        g.add_task(f"t{i}", (lambda i: lambda: barrier.wait() * 0 + i)(i))
    r = g.run(max_workers=8)
    assert r.errors == {}
    assert r.results == {f"t{i}": i for i in range(8)}


def test_conc_parallel_speedup():
    g = S.TaskGraph()
    for i in range(8):
        g.add_task(f"s{i}", lambda: time.sleep(0.2))
    t0 = time.perf_counter()
    r = g.run(max_workers=8)
    assert time.perf_counter() - t0 < 1.0
    assert len(r.results) == 8


def test_conc_max_workers_never_exceeded():
    for mw in (1, 3):
        g = S.TaskGraph()
        lock = threading.Lock()
        state = {"cur": 0, "peak": 0}

        def work(**kw):
            with lock:
                state["cur"] += 1
                state["peak"] = max(state["peak"], state["cur"])
            time.sleep(0.01)
            with lock:
                state["cur"] -= 1
            return 1

        for i in range(24):
            deps = [f"w{i - 7}"] if i >= 7 and i % 3 == 0 else []
            g.add_task(f"w{i}", work, deps)
        r = g.run(max_workers=mw)
        assert len(r.results) == 24
        assert state["peak"] <= mw
        assert state["peak"] >= min(mw, 2)


def test_conc_no_level_barrier():
    # A long task on one branch must not delay an independent chain: "slow" only finishes once the
    # whole chain c1 -> c2 -> c3 -> c4 has completed.
    g = S.TaskGraph()
    chain_done = threading.Event()

    def slow():
        if not chain_done.wait(3):
            raise AssertionError("independent chain was blocked behind the slow task")
        return "slow"

    g.add_task("slow", slow)
    g.add_task("c1", const(1))
    g.add_task("c2", lambda c1: c1 + 1, deps=["c1"])
    g.add_task("c3", lambda c2: c2 + 1, deps=["c2"])
    g.add_task("c4", lambda c3: chain_done.set() or c3 + 1, deps=["c3"])
    g.add_task("join", lambda slow, c4: (slow, c4), deps=["slow", "c4"])
    r = g.run(max_workers=2)
    assert r.errors == {}
    assert r.results["join"] == ("slow", 4)
    assert r.order.index("c4") < r.order.index("slow")


def test_conc_dependent_starts_while_sibling_still_running():
    g = S.TaskGraph()
    gate = threading.Event()

    def blocker():
        if not gate.wait(3):
            raise AssertionError("dependent of a finished task did not start")
        return 0

    g.add_task("quick", const(5))
    g.add_task("blocker", blocker)
    g.add_task("child", lambda quick: gate.set() or quick, deps=["quick"])
    r = g.run(max_workers=3)
    assert r.errors == {} and r.results["child"] == 5


def test_conc_fail_fast_lets_running_task_finish():
    g = S.TaskGraph()
    both = threading.Barrier(2, timeout=3)

    def bad():
        both.wait()
        raise RuntimeError

    def other():
        both.wait()
        time.sleep(0.1)
        return "ok"

    g.add_task("bad", bad)
    g.add_task("other", other)
    g.add_task("later", const(1))
    r = g.run(max_workers=2, fail_fast=True)
    assert r.results == {"other": "ok"}
    assert set(r.errors) == {"bad"}
    assert r.skipped == {"later"}


# --------------------------------------------------------------------------------------------- prop
def random_dag(rng, n, max_deps=3, shuffle=True):
    names = [f"n{i}" for i in range(n)]
    deps = {}
    for i, nm in enumerate(names):
        k = rng.randint(0, min(i, max_deps))
        deps[nm] = rng.sample(names[:i], k)
    insertion = names[:]
    if shuffle:
        rng.shuffle(insertion)
    return insertion, deps


def model_topo(insertion, deps):
    idx = {n: i for i, n in enumerate(insertion)}
    pending = {n: len(deps[n]) for n in insertion}
    dependents = {n: [] for n in insertion}
    for n in insertion:
        for d in deps[n]:
            dependents[d].append(n)
    heap = [(idx[n], n) for n in insertion if not deps[n]]
    heapq.heapify(heap)
    out = []
    while heap:
        _, n = heapq.heappop(heap)
        out.append(n)
        for m in dependents[n]:
            pending[m] -= 1
            if pending[m] == 0:
                heapq.heappush(heap, (idx[m], m))
    return out


class Fail(Exception):
    pass


def make_fn(name, base, fail, expected_deps):
    def fn(**kw):
        assert set(kw) == set(expected_deps), (name, kw)
        if fail:
            raise Fail(name)
        return (base * 31 + sum(kw.values())) % 1_000_003
    return fn


def sequential_model(insertion, deps, bases, failing):
    results, errors, skipped = {}, set(), set()
    for n in model_topo(insertion, deps):
        if any(d in errors or d in skipped for d in deps[n]):
            skipped.add(n)
        elif n in failing:
            errors.add(n)
        else:
            results[n] = (bases[n] * 31 + sum(results[d] for d in deps[n])) % 1_000_003
    return results, errors, skipped


def test_prop_topological_order_matches_model():
    for seed in range(300):
        rng = random.Random(seed)
        insertion, deps = random_dag(rng, rng.randint(0, 25))
        g = S.TaskGraph()
        for n in insertion:
            g.add_task(n, const(0), deps[n])
        got = g.topological_order()
        assert got == model_topo(insertion, deps), seed
        pos = {n: i for i, n in enumerate(got)}
        assert all(pos[d] < pos[n] for n in insertion for d in deps[n])


def test_prop_run_matches_sequential_evaluation():
    for seed in range(250):
        rng = random.Random(10_000 + seed)
        insertion, deps = random_dag(rng, rng.randint(1, 18))
        bases = {n: rng.randrange(1000) for n in insertion}
        failing = {n for n in insertion if rng.random() < 0.12}
        g = S.TaskGraph()
        for n in insertion:
            g.add_task(n, make_fn(n, bases[n], n in failing, deps[n]), deps[n])
        r = g.run(max_workers=rng.randint(1, 5))
        exp_results, exp_errors, exp_skipped = sequential_model(insertion, deps, bases, failing)
        assert r.results == exp_results, seed
        assert set(r.errors) == exp_errors, seed
        assert all(isinstance(e, Fail) and e.args == (n,) for n, e in r.errors.items())
        assert r.skipped == exp_skipped, seed
        check_invariants(insertion, r)
        pos = {n: i for i, n in enumerate(r.order)}
        for n in r.order:
            for d in deps[n]:
                assert d in pos and pos[d] < pos[n], seed


def test_prop_single_worker_order_equals_topological_order():
    for seed in range(200):
        rng = random.Random(50_000 + seed)
        insertion, deps = random_dag(rng, rng.randint(1, 15))
        failing = {n for n in insertion if rng.random() < 0.1}
        g = S.TaskGraph()
        for n in insertion:
            g.add_task(n, make_fn(n, 1, n in failing, deps[n]), deps[n])
        topo = g.topological_order()
        r = g.run(max_workers=1)
        assert r.order == [n for n in topo if n not in r.skipped], seed


def test_prop_fail_fast_single_worker_model():
    for seed in range(200):
        rng = random.Random(70_000 + seed)
        insertion, deps = random_dag(rng, rng.randint(1, 15))
        failing = {n for n in insertion if rng.random() < 0.15}
        g = S.TaskGraph()
        for n in insertion:
            g.add_task(n, make_fn(n, 1, n in failing, deps[n]), deps[n])
        r = g.run(max_workers=1, fail_fast=True)
        expected_order = []
        for n in model_topo(insertion, deps):
            expected_order.append(n)
            if n in failing:
                break
        assert r.order == expected_order, seed
        assert r.skipped == set(insertion) - set(expected_order)


def test_prop_cycles_detected_with_valid_cycle():
    for seed in range(300):
        rng = random.Random(90_000 + seed)
        n = rng.randint(2, 20)
        insertion, deps = random_dag(rng, n)
        # plant a back edge from an ancestor-reachable node: pick a < b with a path b -> ... -> a
        names = [f"n{i}" for i in range(n)]
        b = rng.randrange(1, n)
        if deps[names[b]]:
            a = rng.choice(deps[names[b]])
        else:
            a = names[rng.randrange(b)]
            deps[names[b]].append(a)
        if names[b] not in deps[a]:
            deps[a].append(names[b])
        g = S.TaskGraph()
        for nm in insertion:
            g.add_task(nm, const(0), deps[nm])
        with pytest.raises(S.CycleError) as ei:
            g.topological_order()
        assert_valid_cycle(ei.value.cycle, deps)
        with pytest.raises(S.CycleError) as ei:
            g.run(max_workers=2)
        assert_valid_cycle(ei.value.cycle, deps)


# --------------------------------------------------------------------------------------------- stress
def test_stress_5000_node_random_dag():
    rng = random.Random(2024)
    insertion, deps = random_dag(rng, 5000, max_deps=4)
    bases = {n: rng.randrange(1000) for n in insertion}
    g = S.TaskGraph()
    for n in insertion:
        g.add_task(n, make_fn(n, bases[n], False, deps[n]), deps[n])
    t0 = time.perf_counter()
    r = g.run(max_workers=8)
    elapsed = time.perf_counter() - t0
    exp, _, _ = sequential_model(insertion, deps, bases, set())
    assert r.results == exp
    assert len(r.order) == 5000
    assert elapsed < 20


def test_stress_deep_chain_and_long_cycle():
    n = 30_000
    g = S.TaskGraph()
    g.add_task("c0", lambda: 0)
    for i in range(1, n):
        g.add_task(f"c{i}", (lambda p: lambda **kw: kw[p] + 1)(f"c{i - 1}"), [f"c{i - 1}"])
    order = g.topological_order()
    assert order == [f"c{i}" for i in range(n)]
    r = g.run(max_workers=4)
    assert r.results[f"c{n - 1}"] == n - 1
    # close the loop: c0 depends on the last node
    g2 = S.TaskGraph()
    for i in range(n):
        g2.add_task(f"c{i}", const(0), [f"c{(i - 1) % n}"])
    with pytest.raises(S.CycleError) as ei:
        g2.topological_order()
    assert len(ei.value.cycle) == n + 1
    assert_valid_cycle(ei.value.cycle, {f"c{i}": [f"c{(i - 1) % n}"] for i in range(n)})


def test_stress_large_topological_order():
    rng = random.Random(77)
    insertion, deps = random_dag(rng, 40_000, max_deps=5)
    g = S.TaskGraph()
    for n in insertion:
        g.add_task(n, const(0), deps[n])
    t0 = time.perf_counter()
    got = g.topological_order()
    assert time.perf_counter() - t0 < 10
    assert got == model_topo(insertion, deps)


def test_stress_wide_fan_in_fan_out():
    g = S.TaskGraph()
    g.add_task("root", const(1))
    for i in range(3000):
        g.add_task(f"m{i}", lambda root, i=i: root + i, ["root"])
    g.add_task("sink", lambda **kw: sum(kw.values()), [f"m{i}" for i in range(3000)])
    r = g.run(max_workers=16)
    assert r.results["sink"] == 3000 + sum(range(3000))
    assert r.order[0] == "root" and r.order[-1] == "sink"
