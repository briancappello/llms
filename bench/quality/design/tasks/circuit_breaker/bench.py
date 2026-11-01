"""Order-service-like workload: several breakers around flaky dependencies, wrapped in retry, fake time."""
import random
import time


class _FakeTime:
    def __init__(self):
        self.t = 0.0

    def clock(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class _Down(Exception):
    pass


def bench(mod) -> float:
    rng = random.Random(31337)
    ft = _FakeTime()
    n_deps = 8
    # outage windows per dependency: (start, end) in fake seconds
    outages = [sorted((s, s + rng.uniform(5, 40)) for s in (rng.uniform(0, 9000) for _ in range(15)))
               for _ in range(n_deps)]

    def is_down(dep, t):
        return any(a <= t < b for a, b in outages[dep])

    plan = [(rng.randrange(n_deps), rng.random()) for _ in range(180_000)]

    start = time.perf_counter()
    breakers = [mod.CircuitBreaker(5, 10.0, half_open_max_calls=1, expected_exceptions=(_Down,), clock=ft.clock)
                for _ in range(n_deps)]
    ok = failed = rejected = 0
    for dep, noise in plan:
        ft.t += 0.05
        cb = breakers[dep]

        def downstream(dep=dep, noise=noise):
            if is_down(dep, ft.t) or noise < 0.02:
                raise _Down(dep)
            return dep

        try:
            got = mod.retry(lambda: cb.call(downstream), attempts=3, base_delay=0.01, max_delay=0.1,
                            jitter=lambda d: d, retry_on=(_Down,), sleep=ft.sleep)
            assert got == dep
            ok += 1
        except mod.CircuitOpenError as e:
            assert e.retry_after >= 0
            rejected += 1
        except _Down:
            failed += 1
    # pure-retry hot path with a capped backoff
    sleeps = []
    for _ in range(3000):
        try:
            mod.retry(lambda: (_ for _ in ()).throw(_Down(0)), attempts=20, base_delay=0.001, max_delay=0.5,
                      jitter=lambda d: d, retry_on=(_Down,), sleep=sleeps.append)
        except _Down:
            pass
    elapsed = time.perf_counter() - start

    assert ok + failed + rejected == len(plan)
    assert len(sleeps) == 3000 * 19
    assert abs(sum(sleeps[:19]) - sum(min(0.5, 0.001 * 2 ** i) for i in range(19))) < 1e-9
    # fully determined by the spec (single-threaded, fake time)
    assert (ok, failed, rejected) == (166_607, 3_408, 9_985), (ok, failed, rejected)
    assert sum(cb.opened_count for cb in breakers) == 386
    return elapsed
