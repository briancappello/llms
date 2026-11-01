"""Hot-path workload: skewed key popularity, read-through loads, writes, deletes, TTL expiry on a fake clock."""
import random
import time


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def bench(mod) -> float:
    rng = random.Random(12345)
    n_ops = 600_000
    keyspace = 100_000
    # skewed popularity: a hot set plus a long tail
    keys = [int(rng.paretovariate(1.1)) % keyspace if rng.random() < 0.8 else rng.randrange(keyspace)
            for _ in range(n_ops)]
    kinds = [rng.random() for _ in range(n_ops)]

    clock = _Clock()
    evicted = [0]

    def on_evict(k, v, reason):
        evicted[0] += 1

    cache = mod.TTLCache(20_000, 30.0, clock=clock, on_evict=on_evict)
    loads = 0

    t0 = time.perf_counter()
    for i in range(n_ops):
        k = keys[i]
        r = kinds[i]
        if r < 0.55:
            v = cache.get_or_compute(k, lambda k=k: k * 3)
            if v != k * 3:
                raise AssertionError(f"get_or_compute({k}) returned {v!r}")
        elif r < 0.80:
            v = cache.get(k)
            if v is not None and v != k * 3:
                raise AssertionError(f"get({k}) returned {v!r}")
        elif r < 0.95:
            cache.set(k, k * 3, ttl=None if r < 0.82 else 10.0)
        elif r < 0.98:
            cache.delete(k)
        else:
            cache.pop(k, None)
        if i % 500 == 0:
            clock.t += 1.0
        if i % 200_000 == 199_999:
            loads += len(cache.items())
    elapsed = time.perf_counter() - t0

    s = cache.stats()
    if not (0 < s.hits and 0 < s.misses and s.evictions > 0 and s.expirations > 0):
        raise AssertionError(f"implausible stats {s}")
    if len(cache) > 20_000 or loads <= 0:
        raise AssertionError("cache exceeded maxsize or empty")
    if evicted[0] < s.evictions + s.expirations:
        raise AssertionError("on_evict not called for every eviction/expiration")
    # deterministic spot check of exact LRU/TTL semantics
    c2 = mod.TTLCache(3, 5.0, clock=clock)
    base = clock.t
    c2.set("a", 1)
    c2.set("b", 2)
    c2.set("c", 3, ttl=None)
    c2.get("a")
    c2.set("d", 4)
    if c2.items() != [("c", 3), ("a", 1), ("d", 4)]:
        raise AssertionError(f"LRU order wrong: {c2.items()}")
    clock.t = base + 5.0
    if c2.items() != [("c", 3)]:
        raise AssertionError(f"TTL wrong: {c2.items()}")
    return elapsed
