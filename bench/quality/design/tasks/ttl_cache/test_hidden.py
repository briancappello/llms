import random
import threading
import time

import pytest

import solution as S


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make(maxsize=10, default_ttl=None, t=1000.0):
    clock = FakeClock(t)
    log = []
    cache = S.TTLCache(maxsize, default_ttl, clock=clock, on_evict=lambda k, v, r: log.append((k, v, r)))
    return cache, clock, log


def st(cache):
    s = cache.stats()
    return (s.hits, s.misses, s.evictions, s.expirations)


def run_in_thread(fn, timeout=5.0):
    """Run fn in a daemon thread; fail (instead of hanging) if it deadlocks."""
    box = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout)
    assert not th.is_alive(), "operation deadlocked"
    if "error" in box:
        raise box["error"]
    return box.get("result")


# --------------------------------------------------------------------------------------------- basic
def test_basic_set_get():
    c, _, _ = make()
    c.set("a", 1)
    c.set("b", [2])
    assert c.get("a") == 1
    assert c.get("b") == [2]
    assert c.get("zzz") is None
    assert c.get("zzz", 42) == 42
    assert st(c) == (2, 2, 0, 0)


def test_basic_contains_len_items():
    c, _, _ = make()
    for k in "abc":
        c.set(k, k.upper())
    assert "a" in c and "d" not in c
    assert len(c) == 3
    assert c.items() == [("a", "A"), ("b", "B"), ("c", "C")]


def test_basic_lru_capacity_eviction_order():
    c, _, log = make(maxsize=3)
    for k in "abc":
        c.set(k, k)
    c.set("d", "d")
    assert log == [("a", "a", "capacity")]
    assert c.items() == [("b", "b"), ("c", "c"), ("d", "d")]
    assert st(c) == (0, 0, 1, 0)


def test_basic_get_refreshes_recency():
    c, _, log = make(maxsize=3)
    for k in "abc":
        c.set(k, k)
    assert c.get("a") == "a"
    assert c.items() == [("b", "b"), ("c", "c"), ("a", "a")]
    c.set("d", "d")
    assert log == [("b", "b", "capacity")]
    assert "a" in c and "b" not in c


def test_basic_set_refreshes_recency_and_replaces():
    c, _, log = make(maxsize=3)
    for k in "abc":
        c.set(k, k)
    c.set("a", "A2")
    assert log == [("a", "a", "replaced")]
    assert c.items() == [("b", "b"), ("c", "c"), ("a", "A2")]
    c.set("d", "d")
    assert log[-1] == ("b", "b", "capacity")
    assert st(c) == (0, 0, 1, 0)


def test_basic_queries_do_not_touch_recency_or_stats():
    c, _, log = make(maxsize=3)
    for k in "abc":
        c.set(k, k)
    assert "a" in c
    assert len(c) == 3
    c.items()
    c.stats()
    c.set("d", "d")
    assert log == [("a", "a", "capacity")]
    assert st(c) == (0, 0, 1, 0)


def test_basic_ttl_expiry_boundary():
    c, clock, log = make(default_ttl=5, t=100.0)
    c.set("k", "v")
    clock.t = 104.999
    assert "k" in c
    assert c.get("k") == "v"
    clock.t = 105.0
    assert "k" not in c
    assert c.get("k") is None
    assert log == [("k", "v", "expired")]
    assert st(c) == (1, 1, 0, 1)


def test_basic_explicit_ttl_overrides_default_and_none_never_expires():
    c, clock, _ = make(default_ttl=5, t=0.0)
    c.set("default", 1)
    c.set("short", 2, ttl=1)
    c.set("long", 3, ttl=50)
    c.set("forever", 4, ttl=None)
    clock.t = 1.0
    assert "short" not in c and "default" in c
    clock.t = 5.0
    assert "default" not in c and "long" in c
    clock.t = 1e9
    assert [k for k, _ in c.items()] == ["forever"]
    assert c.get("forever") == 4


def test_basic_no_default_ttl_means_never_expires():
    c, clock, _ = make(default_ttl=None)
    c.set("a", 1)
    clock.advance(1e12)
    assert c.get("a") == 1


def test_basic_set_resets_ttl():
    c, clock, _ = make(default_ttl=10, t=0.0)
    c.set("a", 1)
    clock.t = 8.0
    c.set("a", 2)
    clock.t = 17.0
    assert c.get("a") == 2
    clock.t = 18.0
    assert c.get("a") is None


def test_basic_get_does_not_extend_ttl():
    c, clock, _ = make(default_ttl=10, t=0.0)
    c.set("a", 1)
    clock.t = 9.0
    assert c.get("a") == 1
    clock.t = 10.0
    assert c.get("a") is None


def test_basic_delete():
    c, _, log = make()
    c.set("a", 1)
    assert c.delete("a") is True
    assert log == [("a", 1, "deleted")]
    assert c.delete("a") is False
    assert c.delete("never") is False
    assert len(c) == 0
    assert st(c) == (0, 0, 0, 0)


def test_basic_pop():
    c, _, log = make()
    c.set("a", 1)
    c.set("b", 2)
    assert c.pop("a") == 1
    assert log == []
    assert c.pop("a", "dflt") == "dflt"
    assert c.pop("a", None) is None
    with pytest.raises(KeyError):
        c.pop("a")
    assert c.items() == [("b", 2)]
    assert st(c) == (0, 0, 0, 0)


def test_basic_get_or_compute_miss_then_hit():
    c, _, _ = make()
    calls = []

    def factory():
        calls.append(1)
        return object()

    v1 = c.get_or_compute("k", factory)
    v2 = c.get_or_compute("k", factory)
    assert v1 is v2
    assert len(calls) == 1
    assert st(c) == (1, 1, 0, 0)
    assert c.items() == [("k", v1)]


def test_basic_get_or_compute_ttl_and_refresh():
    c, clock, log = make(maxsize=2, default_ttl=100, t=0.0)
    c.get_or_compute("a", lambda: "A", ttl=3)
    c.set("b", "B")
    clock.t = 2.0
    assert c.get_or_compute("a", lambda: "X") == "A"  # hit, becomes MRU
    assert c.items() == [("b", "B"), ("a", "A")]
    clock.t = 3.0
    assert c.get_or_compute("a", lambda: "A2", ttl=None) == "A2"
    assert log == [("a", "A", "expired")]
    clock.t = 1e9
    assert c.get("a") == "A2"


def test_basic_get_or_compute_ttl_measured_after_factory():
    c, clock, _ = make(t=0.0)

    def slow_factory():
        clock.advance(10)
        return "v"

    c.get_or_compute("k", slow_factory, ttl=5)
    clock.t = 14.9
    assert "k" in c
    clock.t = 15.0
    assert "k" not in c


def test_basic_get_or_compute_failure_caches_nothing():
    c, _, _ = make()

    def boom():
        raise RuntimeError("upstream down")

    with pytest.raises(RuntimeError, match="upstream down"):
        c.get_or_compute("k", boom)
    assert "k" not in c
    assert c.get_or_compute("k", lambda: 7) == 7
    assert st(c) == (0, 2, 0, 0)


def test_basic_get_or_compute_inserts_with_capacity_eviction():
    c, _, log = make(maxsize=2)
    c.set("a", 1)
    c.set("b", 2)
    c.get_or_compute("c", lambda: 3)
    assert log == [("a", 1, "capacity")]
    assert c.items() == [("b", 2), ("c", 3)]


def test_basic_stats_type():
    c, _, _ = make()
    s = c.stats()
    assert isinstance(s, S.CacheStats)
    assert (s.hits, s.misses, s.evictions, s.expirations) == (0, 0, 0, 0)
    c.get("x")
    assert s.misses == 0  # snapshot
    assert c.stats() == S.CacheStats(hits=0, misses=1, evictions=0, expirations=0)


def test_basic_purge():
    c, clock, log = make(t=0.0)
    c.set("a", 1, ttl=1)
    c.set("b", 2, ttl=None)
    c.set("c", 3, ttl=2)
    c.set("d", 4, ttl=5)
    clock.t = 2.0
    assert c.purge() == 2
    assert log == [("a", 1, "expired"), ("c", 3, "expired")]
    assert c.items() == [("b", 2), ("d", 4)]
    assert c.purge() == 0
    assert st(c) == (0, 0, 0, 2)


def test_basic_clear():
    c, clock, log = make(t=0.0)
    c.set("a", 1, ttl=1)
    c.set("b", 2)
    c.set("c", 3, ttl=1)
    clock.t = 1.0
    c.clear()
    assert log == [("a", 1, "expired"), ("b", 2, "deleted"), ("c", 3, "expired")]
    assert len(c) == 0 and c.items() == []
    assert st(c) == (0, 0, 0, 2)
    c.set("z", 0)
    assert c.items() == [("z", 0)]


def test_basic_default_clock_works():
    c = S.TTLCache(4, 60.0)
    c.set("a", 1)
    assert c.get("a") == 1


# --------------------------------------------------------------------------------------------- edge
@pytest.mark.parametrize("maxsize", [0, -1])
def test_edge_invalid_maxsize(maxsize):
    with pytest.raises(ValueError):
        S.TTLCache(maxsize, None)


@pytest.mark.parametrize("ttl", [0, -1, -0.5, 0.0])
def test_edge_invalid_default_ttl(ttl):
    with pytest.raises(ValueError):
        S.TTLCache(4, ttl)


@pytest.mark.parametrize("ttl", [0, -3])
def test_edge_invalid_ttl_set_and_compute(ttl):
    c, _, log = make()
    with pytest.raises(ValueError):
        c.set("a", 1, ttl=ttl)
    called = []
    with pytest.raises(ValueError):
        c.get_or_compute("a", lambda: called.append(1), ttl=ttl)
    assert called == []
    assert len(c) == 0 and log == [] and st(c) == (0, 0, 0, 0)


def test_edge_maxsize_one():
    c, _, log = make(maxsize=1)
    c.set("a", 1)
    c.set("b", 2)
    c.set("b", 3)
    assert log == [("a", 1, "capacity"), ("b", 2, "replaced")]
    assert c.items() == [("b", 3)]


def test_edge_expired_lru_removed_as_expiration_on_insert():
    c, clock, log = make(maxsize=2, t=0.0)
    c.set("a", 1, ttl=1)
    c.set("b", 2)
    clock.t = 1.0
    assert len(c) == 1
    c.set("c", 3)
    assert log == [("a", 1, "expired")]
    assert st(c) == (0, 0, 0, 1)


def test_edge_only_lru_examined_on_insert():
    # expired entry in the middle of the order is NOT swept by an insert; live LRU is evicted.
    c, clock, log = make(maxsize=3, t=0.0)
    c.set("a", 1)
    c.set("b", 2, ttl=1)
    c.set("c", 3)
    clock.t = 5.0
    c.set("d", 4)
    assert log == [("a", 1, "capacity")]
    assert len(c) == 2  # b is still stored but expired
    c.set("e", 5)
    assert log[-1] == ("b", 2, "expired")
    assert c.items() == [("c", 3), ("d", 4), ("e", 5)]
    assert st(c) == (0, 0, 1, 1)


def test_edge_set_over_expired_entry_is_expiration_not_replacement():
    c, clock, log = make(t=0.0)
    c.set("a", 1, ttl=1)
    clock.t = 1.0
    c.set("a", 2)
    assert log == [("a", 1, "expired")]
    assert c.get("a") == 2
    assert st(c) == (1, 0, 0, 1)


def test_edge_delete_and_pop_on_expired():
    c, clock, log = make(t=0.0)
    c.set("a", 1, ttl=1)
    c.set("b", 2, ttl=1)
    clock.t = 1.0
    assert c.delete("a") is False
    assert c.pop("b", "d") == "d"
    with pytest.raises(KeyError):
        c.pop("b")
    assert log == [("a", 1, "expired"), ("b", 2, "expired")]
    assert st(c) == (0, 0, 0, 2)


def test_edge_contains_does_not_remove_expired():
    c, clock, log = make(t=0.0)
    c.set("a", 1, ttl=1)
    clock.t = 3.0
    assert "a" not in c
    assert len(c) == 0
    assert c.items() == []
    assert log == [] and st(c) == (0, 0, 0, 0)
    assert c.purge() == 1


def test_edge_falsy_and_none_values():
    c, _, _ = make()
    c.set("n", None)
    c.set("z", 0)
    assert "n" in c
    assert c.get("n", "dflt") is None
    assert c.get("z", "dflt") == 0
    assert c.pop("n", "dflt") is None
    assert st(c) == (2, 0, 0, 0)


def test_edge_varied_hashable_keys():
    c, _, _ = make()
    keys = [1, 1.5, "1", (1, 2), frozenset({3}), None, b"x"]
    for i, k in enumerate(keys):
        c.set(k, i)
    for i, k in enumerate(keys):
        assert c.get(k) == i


def test_edge_on_evict_exception_propagates_state_consistent():
    def cb(k, v, r):
        raise RuntimeError("cb failed")

    c = S.TTLCache(1, None, clock=FakeClock(), on_evict=cb)
    c.set("a", 1)
    with pytest.raises(RuntimeError, match="cb failed"):
        c.set("b", 2)
    assert c.items() == [("b", 2)]
    assert c.stats().evictions == 1


def test_edge_on_evict_called_outside_lock():
    clock = FakeClock(0.0)
    seen = []

    def cb(k, v, r):
        # re-entering the cache from the callback must not deadlock
        seen.append((k, r, c.get("survivor"), len(c), "x" in c))
        c.set("from_cb_" + str(k), 1)

    c = S.TTLCache(2, None, clock=clock, on_evict=cb)
    c.set("survivor", "s")
    c.set("a", 1, ttl=1)
    clock.t = 1.0
    run_in_thread(lambda: c.get("a"))
    assert seen[0] == ("a", "expired", "s", 1, False)
    run_in_thread(lambda: c.delete("survivor"))
    assert seen[-1][1] == "deleted"


def test_edge_factory_may_use_cache():
    c, _, _ = make()
    c.set("dep", 5)

    def factory():
        c.set("side", 1)
        return c.get("dep") * 2

    assert run_in_thread(lambda: c.get_or_compute("k", factory)) == 10
    assert c.get("side") == 1


# --------------------------------------------------------------------------------------------- prop
_OMIT = object()


class Model:
    def __init__(self, maxsize, default_ttl):
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self.entries = []  # [key, value, expires_at] LRU -> MRU
        self.hits = self.misses = self.evictions = self.expirations = 0
        self.log = []

    @staticmethod
    def expired(e, now):
        return e[2] is not None and now >= e[2]

    def idx(self, key):
        for i, e in enumerate(self.entries):
            if e[0] == key:
                return i
        return -1

    def live(self, key, now):
        i = self.idx(key)
        if i < 0:
            return None
        e = self.entries[i]
        if self.expired(e, now):
            del self.entries[i]
            self.expirations += 1
            self.log.append((e[0], e[1], "expired"))
            return None
        return i

    def store(self, key, value, ttl, now):
        ttl = self.default_ttl if ttl is _OMIT else ttl
        i = self.idx(key)
        if i >= 0:
            e = self.entries.pop(i)
            if self.expired(e, now):
                self.expirations += 1
                self.log.append((e[0], e[1], "expired"))
            else:
                self.log.append((e[0], e[1], "replaced"))
        elif len(self.entries) >= self.maxsize:
            e = self.entries.pop(0)
            if self.expired(e, now):
                self.expirations += 1
                self.log.append((e[0], e[1], "expired"))
            else:
                self.evictions += 1
                self.log.append((e[0], e[1], "capacity"))
        self.entries.append([key, value, None if ttl is None else now + ttl])

    def get(self, key, now):
        i = self.live(key, now)
        if i is None:
            self.misses += 1
            return "<default>"
        self.hits += 1
        e = self.entries.pop(i)
        self.entries.append(e)
        return e[1]

    def delete(self, key, now):
        i = self.live(key, now)
        if i is None:
            return False
        e = self.entries.pop(i)
        self.log.append((e[0], e[1], "deleted"))
        return True

    def pop(self, key, now):
        i = self.live(key, now)
        if i is None:
            return "<default>"
        return self.entries.pop(i)[1]

    def get_or_compute(self, key, value, fail, ttl, now):
        i = self.live(key, now)
        if i is not None:
            self.hits += 1
            e = self.entries.pop(i)
            self.entries.append(e)
            return e[1]
        self.misses += 1
        if fail:
            return "<raised>"
        self.store(key, value, ttl, now)
        return value

    def purge(self, now):
        dead = [e for e in self.entries if self.expired(e, now)]
        self.entries = [e for e in self.entries if not self.expired(e, now)]
        for e in dead:
            self.log.append((e[0], e[1], "expired"))
        self.expirations += len(dead)
        return len(dead)

    def clear(self, now):
        for e in self.entries:
            if self.expired(e, now):
                self.expirations += 1
                self.log.append((e[0], e[1], "expired"))
            else:
                self.log.append((e[0], e[1], "deleted"))
        self.entries = []

    def items(self, now):
        return [(e[0], e[1]) for e in self.entries if not self.expired(e, now)]

    def stats(self):
        return (self.hits, self.misses, self.evictions, self.expirations)


class Boom(Exception):
    pass


def _run_model_case(seed, nops=80):
    rng = random.Random(seed)
    maxsize = rng.randint(1, 6)
    default_ttl = rng.choice([None, 2, 3.5, 6])
    clock = FakeClock(rng.choice([0.0, 1000.0]))
    log = []
    c = S.TTLCache(maxsize, default_ttl, clock=clock, on_evict=lambda k, v, r: log.append((k, v, r)))
    m = Model(maxsize, default_ttl)
    keys = list(range(rng.randint(2, 10)))
    counter = 0
    for step in range(nops):
        op = rng.random()
        key = rng.choice(keys)
        ttl = rng.choice([_OMIT, _OMIT, None, 0.5, 1, 2, 4, 9])
        now = clock.t
        ctx = (seed, step)
        if op < 0.15:
            clock.advance(rng.choice([0.5, 1, 1, 2, 3, 0.25]))
        elif op < 0.40:
            assert c.get(key, "<default>") == m.get(key, now), ctx
        elif op < 0.62:
            counter += 1
            if ttl is _OMIT:
                c.set(key, counter)
            else:
                c.set(key, counter, ttl=ttl)
            m.store(key, counter, ttl, now)
        elif op < 0.68:
            assert c.delete(key) == m.delete(key, now), ctx
        elif op < 0.73:
            assert c.pop(key, "<default>") == m.pop(key, now), ctx
        elif op < 0.85:
            counter += 1
            fail = rng.random() < 0.2
            val = counter

            def factory(val=val, fail=fail):
                if fail:
                    raise Boom()
                return val

            kwargs = {} if ttl is _OMIT else {"ttl": ttl}
            try:
                got = c.get_or_compute(key, factory, **kwargs)
            except Boom:
                got = "<raised>"
            assert got == m.get_or_compute(key, val, fail, ttl, now), ctx
        elif op < 0.89:
            assert (key in c) == any(k == key for k, _ in m.items(now)), ctx
        elif op < 0.92:
            assert len(c) == len(m.items(now)), ctx
        elif op < 0.97:
            assert c.purge() == m.purge(now), ctx
        elif op < 0.98:
            c.clear()
            m.clear(now)
        assert c.items() == m.items(clock.t), ctx
        assert st(c) == m.stats(), ctx
        assert log == m.log, ctx


def test_prop_random_ops_match_model():
    for seed in range(300):
        _run_model_case(seed)


def test_prop_long_sequences_match_model():
    for seed in range(1000, 1200):
        _run_model_case(seed, nops=250)


def test_prop_lru_only_no_ttl_matches_ordereddict():
    from collections import OrderedDict

    for seed in range(250):
        rng = random.Random(seed)
        maxsize = rng.randint(1, 8)
        c = S.TTLCache(maxsize, None, clock=FakeClock())
        od = OrderedDict()
        for _ in range(120):
            k = rng.randrange(12)
            if rng.random() < 0.5:
                if k in od:
                    od.move_to_end(k)
                elif len(od) >= maxsize:
                    od.popitem(last=False)
                od[k] = rng.random()
                c.set(k, od[k])
            else:
                exp = od.get(k)
                if k in od:
                    od.move_to_end(k)
                assert c.get(k) == exp
        assert c.items() == list(od.items())


# --------------------------------------------------------------------------------------------- conc
def test_conc_single_flight_same_key():
    c = S.TTLCache(100, None)
    calls = []
    lock = threading.Lock()
    n = 8
    start = threading.Barrier(n)
    results = [None] * n

    def factory():
        with lock:
            calls.append(1)
        time.sleep(0.3)
        return object()

    def worker(i):
        start.wait()
        results[i] = c.get_or_compute("hot", factory)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert len(calls) == 1
    assert all(r is results[0] for r in results)
    s = c.stats()
    assert s.hits + s.misses == n
    assert s.misses >= 1


def test_conc_different_keys_load_concurrently():
    c = S.TTLCache(100, None)
    both = threading.Barrier(2, timeout=3)

    def factory_for(k):
        def f():
            both.wait()  # deadlocks/BrokenBarrierError if loads are serialized
            return k * 2
        return f

    out = {}

    def worker(k):
        out[k] = c.get_or_compute(k, factory_for(k))

    ts = [threading.Thread(target=worker, args=(k,)) for k in (1, 2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    assert out == {1: 2, 2: 4}


def test_conc_slow_factory_does_not_block_other_ops():
    c = S.TTLCache(100, None)
    c.set("other", 1)
    started = threading.Event()
    release = threading.Event()

    def slow():
        started.set()
        release.wait(5)
        return "slow"

    th = threading.Thread(target=lambda: c.get_or_compute("slow", slow))
    th.start()
    assert started.wait(5)
    try:
        assert run_in_thread(lambda: (c.get("other"), c.set("x", 2), len(c), "slow" in c), 3) == (1, None, 2, False)
    finally:
        release.set()
        th.join(5)
    assert c.get("slow") == "slow"


def test_conc_failed_load_waiters_retry_once():
    c = S.TTLCache(100, None)
    started = threading.Event()
    ok_calls = []
    lock = threading.Lock()

    def failing():
        started.set()
        time.sleep(0.3)
        raise Boom("leader failed")

    def ok():
        with lock:
            ok_calls.append(1)
        time.sleep(0.1)
        return "good"

    leader_err = []
    waiter_res = []

    def leader():
        try:
            c.get_or_compute("k", failing)
        except Boom as e:
            leader_err.append(e)

    def waiter():
        r = c.get_or_compute("k", ok)
        with lock:
            waiter_res.append(r)

    lt = threading.Thread(target=leader)
    lt.start()
    assert started.wait(5)
    ws = [threading.Thread(target=waiter) for _ in range(4)]
    for w in ws:
        w.start()
    for t in [lt, *ws]:
        t.join(5)
    assert len(leader_err) == 1
    assert waiter_res == ["good"] * 4
    assert len(ok_calls) == 1
    assert c.get("k") == "good"
    s = c.stats()
    assert s.hits + s.misses == 6  # 5 get_or_compute calls (one each) + the final get (hit)


def test_conc_waiters_get_same_object_even_if_evicted():
    # maxsize 1: the loaded value may be evicted immediately, waiters still receive the loader's object
    c = S.TTLCache(1, None)
    started = threading.Event()
    marker = object()

    def factory():
        started.set()
        time.sleep(0.2)
        return marker

    res = []
    lt = threading.Thread(target=lambda: res.append(c.get_or_compute("k", factory)))
    lt.start()
    assert started.wait(5)
    ws = [threading.Thread(target=lambda: res.append(c.get_or_compute("k", lambda: "dup"))) for _ in range(3)]
    for w in ws:
        w.start()
    for t in [lt, *ws]:
        t.join(5)
    assert len(res) == 4 and all(r is marker for r in res)


def test_conc_hammer_invariants():
    n_threads, n_ops, maxsize = 8, 4000, 50
    evicted = []
    ev_lock = threading.Lock()

    def cb(k, v, r):
        with ev_lock:
            evicted.append(r)

    c = S.TTLCache(maxsize, None, on_evict=cb)
    gets = [0] * n_threads
    sets = [0] * n_threads
    deletes = [0] * n_threads

    def worker(i):
        rng = random.Random(i)
        for _ in range(n_ops):
            k = rng.randrange(120)
            r = rng.random()
            if r < 0.5:
                c.get(k)
                gets[i] += 1
            elif r < 0.95:
                c.set(k, i)
                sets[i] += 1
            else:
                if c.delete(k):
                    deletes[i] += 1

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(20)
    s = c.stats()
    assert s.hits + s.misses == sum(gets)
    assert len(c) <= maxsize
    assert evicted.count("capacity") == s.evictions
    assert evicted.count("deleted") == sum(deletes)
    # every stored value is still stored, or was removed exactly once
    assert sum(sets) == len(c) + s.evictions + evicted.count("replaced") + sum(deletes)


def test_conc_hammer_get_or_compute_counts():
    c = S.TTLCache(1000, None)
    calls = {}
    lock = threading.Lock()

    def factory_for(k):
        def f():
            with lock:
                calls[k] = calls.get(k, 0) + 1
            time.sleep(0.001)
            return k * 10
        return f

    def worker(i):
        rng = random.Random(i)
        for _ in range(500):
            k = rng.randrange(40)
            assert c.get_or_compute(k, factory_for(k)) == k * 10

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(20)
    assert all(v == 1 for v in calls.values()), calls
    assert len(c) == len(calls) == 40
    s = c.stats()
    assert s.hits + s.misses == 8 * 500


# --------------------------------------------------------------------------------------------- stress
def test_stress_200k_ops_large_cache():
    clock = FakeClock(0.0)
    maxsize = 50_000
    c = S.TTLCache(maxsize, 500.0, clock=clock)
    rng = random.Random(7)
    keyspace = 150_000
    shadow_hits = 0
    t0 = time.perf_counter()
    for i in range(200_000):
        k = rng.randrange(keyspace)
        if i & 1:
            if c.get(k) is not None:
                shadow_hits += 1
        else:
            c.set(k, i)
        if i % 1000 == 0:
            clock.advance(1.0)
    elapsed = time.perf_counter() - t0
    s = c.stats()
    assert s.hits == shadow_hits
    assert s.hits + s.misses == 100_000
    assert s.evictions > 10_000
    assert len(c) <= maxsize
    assert elapsed < 20


def test_stress_capacity_churn():
    c = S.TTLCache(100_000, None, clock=FakeClock())
    for i in range(200_000):
        c.set(i, i)
    assert c.stats().evictions == 100_000
    items = c.items()
    assert len(items) == 100_000
    assert items[0] == (100_000, 100_000) and items[-1] == (199_999, 199_999)
    for i in range(199_999, 99_999, -2):
        assert c.get(i) == i


def test_stress_many_expirations():
    clock = FakeClock(0.0)
    c = S.TTLCache(200_000, 1.0, clock=clock)
    for i in range(100_000):
        c.set(i, i)
    clock.advance(1.0)
    for i in range(0, 100_000, 2):
        assert c.get(i) is None
    assert c.stats().expirations == 50_000
    assert c.purge() == 50_000
    assert len(c) == 0
