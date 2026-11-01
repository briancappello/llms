"""Gateway-like workload: per-client keyed limiter plus a global bucket, driven by a fake clock."""
import random
import time
from collections import OrderedDict


class _FakeTime:
    def __init__(self):
        self.t = 1000.0
        self.slept = 0.0

    def clock(self):
        return self.t

    def sleep(self, dt):
        assert dt > 0
        self.slept += dt
        self.t += dt


def _model(ops, rate, cap, max_keys, grate, gcap):
    """Straightforward model of the same workload; returns the admitted counts."""
    buckets = OrderedDict()
    g_level, g_last = gcap, 1000.0
    per_ok = global_ok = 0
    for now, key, n in ops:
        b = buckets.get(key)
        if b is None:
            b = buckets[key] = [cap, now]
            if len(buckets) > max_keys:
                buckets.popitem(last=False)
        else:
            buckets.move_to_end(key)
        if now > b[1]:
            b[0] = min(cap, b[0] + (now - b[1]) * rate)
            b[1] = now
        if b[0] >= n - 1e-9:
            b[0] = max(0.0, b[0] - n)
            per_ok += 1
            if now > g_last:
                g_level = min(gcap, g_level + (now - g_last) * grate)
                g_last = now
            if g_level >= n - 1e-9:
                g_level = max(0.0, g_level - n)
                global_ok += 1
    return per_ok, global_ok


def bench(mod) -> float:
    rng = random.Random(2026)
    rate, cap, max_keys = 2.0, 10.0, 3000
    grate, gcap = 600.0, 200.0
    ops = []
    t = 1000.0
    for _ in range(400_000):
        t += rng.choice((0.0, 0.0, 0.0005, 0.001, 0.002))
        # a few heavy hitters plus a long tail of occasional clients
        key = rng.randrange(40) if rng.random() < 0.5 else rng.randrange(40, 12_000)
        ops.append((t, key, rng.choice((1, 1, 1, 2, 0.5))))
    exp_per, exp_global = _model(ops, rate, cap, max_keys, grate, gcap)

    ft = _FakeTime()
    start = time.perf_counter()
    limiter = mod.KeyedRateLimiter(rate, cap, clock=ft.clock, sleep=ft.sleep, max_keys=max_keys)
    gbucket = mod.TokenBucket(grate, gcap, clock=ft.clock, sleep=ft.sleep)
    per_ok = global_ok = 0
    for now, key, n in ops:
        ft.t = now
        if limiter.try_acquire(key, n):
            per_ok += 1
            if gbucket.try_acquire(n):
                global_ok += 1
    # blocking path: a background job draining a slow bucket with acquire()
    slow = mod.TokenBucket(5.0, 5.0, clock=ft.clock, sleep=ft.sleep)
    t0 = ft.t
    for _ in range(80_000):
        assert slow.acquire(1)
    blocked_for = ft.t - t0
    elapsed = time.perf_counter() - start

    assert per_ok == exp_per, (per_ok, exp_per)
    assert global_ok == exp_global, (global_ok, exp_global)
    assert len(limiter) == max_keys
    assert abs(blocked_for - (80_000 - 5) / 5.0) < 1e-3, blocked_for
    return elapsed
