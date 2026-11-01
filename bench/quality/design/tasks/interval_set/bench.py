"""Benchmark: a download/booking-tracker style workload on IntervalSet."""
import random
import time


def bench(mod) -> float:
    rng = random.Random(20261004)
    IS = mod.IntervalSet
    space = 50_000_000
    t0 = time.perf_counter()

    # Phase 1: fragmented byte-range tracking -- many small adds and some removes.
    s = IS()
    for _ in range(150_000):
        a = rng.randrange(space)
        if rng.random() < 0.75:
            s.add(a, a + rng.randint(1, 2_000))
        else:
            s.remove(a, a + rng.randint(1, 500))
    n_after = len(s)

    # Phase 2: point and range queries.
    hits = 0
    covered = 0
    overl = 0
    gap_total = 0
    for _ in range(150_000):
        x = rng.randrange(space)
        hits += s.contains(x)
        b = x + rng.randint(0, 3_000)
        covered += s.covers(x, b)
        overl += s.overlaps(x, b)
    for _ in range(10_000):
        x = rng.randrange(space)
        gap_total += sum(e - b for b, e in s.gaps(x, x + 20_000))

    # Phase 3: set algebra between two large sets.
    other = IS((i * 997, i * 997 + 400) for i in range(space // 997 // 4))
    u = s | other
    n = s & other
    d = s - other
    elapsed = time.perf_counter() - t0

    # Validation (cheap, done outside the timed region).
    items = list(s)
    assert len(items) == n_after == len(s)
    for (s1, e1), (s2, e2) in zip(items, items[1:]):
        assert s1 < e1 < s2, "not normalized"
    total = sum(e - b for b, e in items)
    assert s.total_length() == total
    assert 0 < hits < 150_000 and 0 < overl and covered <= overl
    assert u.total_length() == total + other.total_length() - n.total_length()
    assert d.total_length() == total - n.total_length()
    assert (d | n) == s
    rng2 = random.Random(1)
    for _ in range(300):
        x = rng2.randrange(space)
        b = x + rng2.randint(0, 3_000)
        g = s.gaps(x, b)
        assert sum(e - c for c, e in g) == (b - x) - (s & IS([(x, b)])).total_length()
        assert s.covers(x, b) == (g == [] )
        assert s.overlaps(x, b) == (g != [(x, b)] and b > x)
    assert gap_total > 0
    return elapsed
