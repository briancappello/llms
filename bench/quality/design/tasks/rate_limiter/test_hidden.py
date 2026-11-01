import math
import random
import sys
import threading
from collections import OrderedDict

import pytest

import solution as S

EPS = 1e-9


class FakeTime:
    """Fake monotonic clock; sleep(dt) records dt and advances the clock by dt (thread-safe)."""

    def __init__(self, t: float = 100.0):
        self.t = t
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def clock(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        assert dt > 0, f"sleep called with non-positive value {dt!r}"
        with self._lock:
            self.sleeps.append(dt)
            self.t += dt

    def advance(self, dt: float) -> None:
        with self._lock:
            self.t += dt


def bucket(rate, capacity, ft=None):
    ft = ft or FakeTime()
    return S.TokenBucket(rate, capacity, clock=ft.clock, sleep=ft.sleep), ft


def keyed(rate, capacity, max_keys=10_000, ft=None):
    ft = ft or FakeTime()
    return S.KeyedRateLimiter(rate, capacity, clock=ft.clock, sleep=ft.sleep, max_keys=max_keys), ft


@pytest.fixture
def fast_switching():
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(old)


# ----------------------------------------------------------------------------------------------- basic


def test_basic_starts_full_and_properties():
    b, _ = bucket(2.5, 10)
    assert b.available() == pytest.approx(10)
    assert b.rate == pytest.approx(2.5)
    assert b.capacity == pytest.approx(10)


def test_basic_try_acquire_drains_then_refuses():
    b, ft = bucket(1, 5)
    assert [b.try_acquire() for _ in range(6)] == [True] * 5 + [False]
    assert b.available() == pytest.approx(0, abs=EPS)
    assert ft.sleeps == []


def test_basic_continuous_refill_math():
    b, ft = bucket(2, 4)
    assert b.try_acquire(4)
    ft.advance(0.75)
    assert b.available() == pytest.approx(1.5)
    ft.advance(0.125)
    assert b.available() == pytest.approx(1.75)
    assert b.try_acquire(1.75)
    assert b.available() == pytest.approx(0, abs=EPS)


def test_basic_refill_capped_at_capacity():
    b, ft = bucket(3, 6)
    assert b.try_acquire(1)
    ft.advance(10_000)
    assert b.available() == pytest.approx(6)
    assert b.try_acquire(6)
    assert not b.try_acquire(0.5)


def test_basic_fractional_tokens():
    b, ft = bucket(0.5, 1.5)
    assert b.try_acquire(0.5)
    assert b.try_acquire(0.75)
    assert b.available() == pytest.approx(0.25)
    assert not b.try_acquire(0.5)
    ft.advance(0.5)  # +0.25
    assert b.try_acquire(0.5)


def test_basic_failed_try_acquire_consumes_nothing():
    b, ft = bucket(1, 3)
    assert b.try_acquire(2.5)
    assert not b.try_acquire(1)
    assert not b.try_acquire(1)
    assert b.available() == pytest.approx(0.5)
    ft.advance(0.5)
    assert b.try_acquire(1)


def test_basic_acquire_immediate_does_not_sleep():
    b, ft = bucket(1, 3)
    assert b.acquire(2)
    assert b.acquire(1, timeout=0)
    assert ft.sleeps == []
    assert b.available() == pytest.approx(0, abs=EPS)


def test_basic_acquire_sleeps_exact_computed_wait_once():
    b, ft = bucket(4, 2)
    assert b.try_acquire(2)
    t0 = ft.t
    assert b.acquire(1)
    assert len(ft.sleeps) == 1
    assert ft.sleeps[0] == pytest.approx(0.25)
    assert ft.t == pytest.approx(t0 + 0.25)
    assert b.available() == pytest.approx(0, abs=EPS)


def test_basic_acquire_partial_deficit_wait():
    b, ft = bucket(2, 4)
    assert b.try_acquire(3.5)  # 0.5 left
    assert b.acquire(2, timeout=5)  # deficit 1.5 -> 0.75 s
    assert ft.sleeps == [pytest.approx(0.75)]
    assert b.available() == pytest.approx(0, abs=EPS)


def test_basic_acquire_timeout_fails_fast_without_sleep():
    b, ft = bucket(4, 2)
    assert b.try_acquire(2)
    t0 = ft.t
    assert b.acquire(1, timeout=0.2) is False
    assert ft.sleeps == []
    assert ft.t == t0
    assert b.available() == pytest.approx(0, abs=EPS)
    assert b.acquire(2, timeout=0.49) is False
    assert ft.sleeps == []


def test_basic_acquire_within_timeout_succeeds():
    b, ft = bucket(4, 2)
    assert b.try_acquire(2)
    assert b.acquire(2, timeout=1.0) is True
    assert ft.sleeps == [pytest.approx(0.5)]


def test_basic_acquire_none_timeout_waits_as_long_as_needed():
    b, ft = bucket(0.001, 1)
    assert b.try_acquire(1)
    assert b.acquire(1) is True
    assert len(ft.sleeps) == 1
    assert ft.sleeps[0] == pytest.approx(1000.0)


def test_basic_keyed_isolation():
    k, ft = keyed(1, 2)
    assert k.try_acquire("a", 2)
    assert not k.try_acquire("a")
    assert k.try_acquire("b")
    assert k.try_acquire("b")
    assert not k.try_acquire("b")
    assert k.available("a") == pytest.approx(0, abs=EPS)
    ft.advance(0.5)
    assert k.available("a") == pytest.approx(0.5)
    assert k.available("b") == pytest.approx(0.5)
    assert k.available("c") == pytest.approx(2)
    assert k.rate == pytest.approx(1) and k.capacity == pytest.approx(2) and k.max_keys == 10_000


def test_basic_keyed_lru_eviction():
    k, _ = keyed(1, 3, max_keys=2)
    assert k.try_acquire("a")
    assert k.try_acquire("b")
    assert k.try_acquire("a")  # a is now most recent
    assert k.try_acquire("c")  # evicts b
    assert "b" not in k
    assert "a" in k and "c" in k
    assert len(k) == 2
    assert k.available("b") == pytest.approx(3)
    assert k.available("a") == pytest.approx(1)


def test_basic_keyed_evicted_key_starts_full():
    k, _ = keyed(0.001, 3, max_keys=1)
    assert k.try_acquire("a", 3)
    assert not k.try_acquire("a")
    assert k.try_acquire("b")  # evicts a
    assert "a" not in k
    assert [k.try_acquire("a") for _ in range(4)] == [True, True, True, False]


def test_basic_keyed_acquire_blocks_with_injected_sleep():
    k, ft = keyed(2, 1)
    assert k.try_acquire("x")
    assert k.acquire("x", 1, timeout=1)
    assert ft.sleeps == [pytest.approx(0.5)]
    assert not k.acquire("x", 1, timeout=0.25)
    assert ft.sleeps == [pytest.approx(0.5)]
    assert k.acquire("y", 1, timeout=0)


def test_basic_keyed_len_and_contains():
    k, _ = keyed(1, 1, max_keys=3)
    assert len(k) == 0 and "a" not in k
    for key in ["a", "b", ("tuple", 1), 42]:
        k.try_acquire(key)
    assert len(k) == 3
    assert "a" not in k
    assert ("tuple", 1) in k and 42 in k and "b" in k


# ----------------------------------------------------------------------------------------------- edge


@pytest.mark.parametrize("rate,capacity", [(0, 1), (-1, 1), (1, 0), (1, -2), (math.nan, 1), (1, math.nan),
                                           (math.inf, 1), (1, math.inf)])
def test_edge_invalid_rate_or_capacity(rate, capacity):
    ft = FakeTime()
    with pytest.raises(ValueError):
        S.TokenBucket(rate, capacity, clock=ft.clock, sleep=ft.sleep)
    with pytest.raises(ValueError):
        S.KeyedRateLimiter(rate, capacity, clock=ft.clock, sleep=ft.sleep)


@pytest.mark.parametrize("tokens", [0, -1, -0.5, 5.0001, 6, math.nan])
def test_edge_invalid_tokens(tokens):
    b, ft = bucket(1, 5)
    with pytest.raises(ValueError):
        b.try_acquire(tokens)
    with pytest.raises(ValueError):
        b.acquire(tokens)
    with pytest.raises(ValueError):
        b.acquire(tokens, timeout=1)
    assert b.available() == pytest.approx(5)
    assert ft.sleeps == []


def test_edge_tokens_equal_capacity_allowed():
    b, ft = bucket(1, 2.5)
    assert b.try_acquire(2.5)
    assert b.acquire(2.5)
    assert ft.sleeps == [pytest.approx(2.5)]


@pytest.mark.parametrize("timeout", [-0.001, -5, math.nan])
def test_edge_invalid_timeout(timeout):
    b, ft = bucket(1, 1)
    with pytest.raises(ValueError):
        b.acquire(1, timeout=timeout)
    assert b.available() == pytest.approx(1)
    k, _ = keyed(1, 1)
    with pytest.raises(ValueError):
        k.acquire("a", 1, timeout=timeout)
    assert "a" not in k


def test_edge_timeout_zero_behaves_like_try_acquire():
    b, ft = bucket(1, 1)
    assert b.acquire(1, timeout=0) is True
    assert b.acquire(1, timeout=0) is False
    ft.advance(0.5)
    assert b.acquire(0.5, timeout=0) is True
    assert ft.sleeps == []


def test_edge_wait_exactly_equal_to_timeout_succeeds():
    b, ft = bucket(4, 1)
    assert b.try_acquire(1)
    assert b.acquire(1, timeout=0.25) is True
    assert ft.sleeps == [0.25]


def test_edge_clock_backwards_no_refill_no_loss():
    b, ft = bucket(1, 4)
    assert b.try_acquire(4)
    ft.advance(2)
    assert b.available() == pytest.approx(2)
    ft.advance(-1.5)  # clock jumps back
    assert b.available() == pytest.approx(2)
    assert b.try_acquire(1)
    ft.advance(1.0)  # still before the latest time seen: no new tokens
    assert b.available() == pytest.approx(1)
    ft.advance(1.0)  # 0.5 past the latest time seen
    assert b.available() == pytest.approx(1.5)


def test_edge_float_tolerance_single_sleep():
    b, ft = bucket(3, 1)
    for _ in range(50):
        assert b.acquire(1, timeout=1) is True
    # first acquire is immediate, the 49 others need exactly one sleep of 1/3 s each
    assert len(ft.sleeps) == 49
    assert all(s == pytest.approx(1 / 3) for s in ft.sleeps)
    b2, ft2 = bucket(10, 1)
    assert b2.try_acquire(1)
    ft2.advance(0.1)
    assert b2.try_acquire(1)


def test_edge_early_wakeup_rechecks_and_sleeps_remainder():
    ft = FakeTime()
    calls = []

    def early_sleep(dt):
        assert dt > 0
        calls.append(dt)
        ft.t += dt - 0.25 if len(calls) == 1 else dt  # first sleep wakes 0.25 s early

    b = S.TokenBucket(1, 2, clock=ft.clock, sleep=early_sleep)
    assert b.try_acquire(2)
    assert b.acquire(2, timeout=10) is True
    assert len(calls) == 2
    assert calls[0] == pytest.approx(2.0)
    assert calls[1] == pytest.approx(0.25)
    assert b.available() == pytest.approx(0, abs=EPS)


def test_edge_early_wakeup_respects_original_deadline():
    ft = FakeTime()
    calls = []

    def early_sleep(dt):
        calls.append(dt)
        ft.t += dt - 0.25 if len(calls) == 1 else dt

    b = S.TokenBucket(1, 2, clock=ft.clock, sleep=early_sleep)
    assert b.try_acquire(2)
    # deadline is start+2.0; after waking at +1.75 the remaining 0.25 s ends exactly at the deadline
    assert b.acquire(2, timeout=2.0) is True
    assert calls == [pytest.approx(2.0), pytest.approx(0.25)]


def test_edge_retakes_after_someone_else_took_tokens_while_sleeping():
    ft = FakeTime()
    holder = {}
    calls = []

    def stealing_sleep(dt):
        calls.append(dt)
        ft.t += dt
        if len(calls) == 1:
            assert holder["b"].try_acquire(1)  # another client grabs the refilled token

    b = S.TokenBucket(1, 1, clock=ft.clock, sleep=stealing_sleep)
    holder["b"] = b
    assert b.try_acquire(1)
    assert b.acquire(1, timeout=1.5) is False  # after the steal, the next token would come at +2 > +1.5
    assert calls == [pytest.approx(1.0)]
    assert b.acquire(1, timeout=1.0) is True
    assert calls == [pytest.approx(1.0), pytest.approx(1.0)]


@pytest.mark.parametrize("max_keys", [0, -1])
def test_edge_keyed_invalid_max_keys(max_keys):
    ft = FakeTime()
    with pytest.raises(ValueError):
        S.KeyedRateLimiter(1, 1, clock=ft.clock, sleep=ft.sleep, max_keys=max_keys)


def test_edge_keyed_invalid_tokens_do_not_create_or_touch_key():
    k, _ = keyed(1, 2, max_keys=2)
    with pytest.raises(ValueError):
        k.try_acquire("x", 0)
    with pytest.raises(ValueError):
        k.acquire("x", 3)
    assert "x" not in k and len(k) == 0
    k.try_acquire("a")
    k.try_acquire("b")
    with pytest.raises(ValueError):
        k.try_acquire("a", -1)  # must not refresh a
    k.try_acquire("c")
    assert "a" not in k and "b" in k and "c" in k


def test_edge_keyed_available_does_not_touch_recency_or_create():
    k, _ = keyed(1, 2, max_keys=2)
    k.try_acquire("a")
    k.try_acquire("b")
    assert k.available("a") == pytest.approx(1)  # read-only, a stays LRU
    assert k.available("zzz") == pytest.approx(2)
    assert "zzz" not in k and len(k) == 2
    k.try_acquire("c")
    assert "a" not in k and "b" in k


def test_edge_keyed_failed_attempt_counts_as_use():
    k, _ = keyed(0.001, 1, max_keys=2)
    assert k.try_acquire("a")
    assert k.try_acquire("b")
    assert not k.try_acquire("a")  # refused, but a is now most recently used
    assert not k.acquire("a", 1, timeout=0)  # refused too
    assert k.try_acquire("c")
    assert "a" in k and "b" not in k


def test_edge_keyed_max_keys_one():
    k, _ = keyed(1, 1, max_keys=1)
    for i in range(20):
        assert k.try_acquire(i)
        assert len(k) == 1 and i in k
    assert 0 not in k


# ----------------------------------------------------------------------------------------------- property


class ModelBucket:
    def __init__(self, rate, cap, now):
        self.rate, self.cap, self.level, self.last = rate, cap, cap, now

    def refill(self, now):
        if now > self.last:
            self.level = min(self.cap, self.level + (now - self.last) * self.rate)
            self.last = now

    def take(self, n, now):
        self.refill(now)
        if self.level >= n - 1e-9:
            self.level = max(0.0, self.level - n)
            return 0.0
        return (n - self.level) / self.rate


def test_prop_bucket_vs_model():
    rng = random.Random(1234)
    for case in range(300):
        rate = rng.choice([0.5, 1, 2, 4, 8, 0.25])
        cap = rng.choice([1, 2, 3.5, 5, 8, 0.75])
        ft = FakeTime(rng.choice([0.0, 100.0, 4096.0]))
        b = S.TokenBucket(rate, cap, clock=ft.clock, sleep=ft.sleep)
        m = ModelBucket(rate, cap, ft.t)
        for step in range(40):
            op = rng.random()
            ctx = f"case={case} step={step}"
            if op < 0.25:
                ft.advance(rng.choice([0.0, 0.125, 0.25, 0.5, 1.0, 3.0, 0.0625]))
            elif op < 0.55:
                n = rng.choice([x for x in (0.25, 0.5, 1, 2, cap) if x <= cap])
                expect = m.take(n, ft.t) == 0.0
                assert b.try_acquire(n) is expect, ctx
            elif op < 0.85:
                n = rng.choice([x for x in (0.25, 0.5, 1, 2, cap) if x <= cap])
                timeout = rng.choice([None, 0, 0.25, 1.0, 4.0, 0.5])
                before = len(ft.sleeps)
                start = ft.t
                wait = m.take(n, start)
                if wait == 0.0:
                    exp_ok, exp_sleeps = True, []
                elif timeout is not None and wait > timeout:
                    exp_ok, exp_sleeps = False, []
                else:
                    exp_ok, exp_sleeps = True, [wait]
                    assert m.take(n, start + wait) == 0.0
                got = b.acquire(n, timeout=timeout)
                assert got is exp_ok, ctx
                assert ft.sleeps[before:] == [pytest.approx(w) for w in exp_sleeps], ctx
            else:
                m.refill(ft.t)
                assert b.available() == pytest.approx(m.level, abs=1e-9), ctx


def test_prop_keyed_vs_model():
    rng = random.Random(98765)
    for case in range(300):
        rate = rng.choice([0.5, 1, 2, 4])
        cap = rng.choice([1, 2, 3])
        max_keys = rng.choice([1, 2, 3, 5])
        ft = FakeTime()
        k = S.KeyedRateLimiter(rate, cap, clock=ft.clock, sleep=ft.sleep, max_keys=max_keys)
        model: OrderedDict = OrderedDict()
        for step in range(50):
            ctx = f"case={case} step={step}"
            op = rng.random()
            key = rng.randrange(7)
            if op < 0.2:
                ft.advance(rng.choice([0.0, 0.125, 0.5, 1.0, 2.0]))
            elif op < 0.7:
                n = rng.choice([x for x in (0.5, 1, 2, 3) if x <= cap])
                if key not in model:
                    model[key] = ModelBucket(rate, cap, ft.t)
                    if len(model) > max_keys:
                        model.popitem(last=False)
                else:
                    model.move_to_end(key)
                expect = model[key].take(n, ft.t) == 0.0
                assert k.try_acquire(key, n) is expect, ctx
            elif op < 0.85:
                if key in model:
                    model[key].refill(ft.t)
                    expect = model[key].level
                else:
                    expect = cap
                assert k.available(key) == pytest.approx(expect, abs=1e-9), ctx
            else:
                assert len(k) == len(model), ctx
                assert all((x in k) == (x in model) for x in range(7)), ctx


def test_prop_refill_formula_random_floats():
    rng = random.Random(7)
    for case in range(400):
        rate = rng.uniform(0.01, 50)
        cap = rng.uniform(0.5, 100)
        ft = FakeTime(rng.uniform(0, 1e5))
        b = S.TokenBucket(rate, cap, clock=ft.clock, sleep=ft.sleep)
        take = rng.uniform(0.01, cap)
        assert b.try_acquire(take)
        dt = rng.uniform(0, 2 * cap / rate)
        ft.advance(dt)
        expect = min(cap, cap - take + dt * rate)
        assert b.available() == pytest.approx(expect, rel=1e-7, abs=1e-6), case


# ----------------------------------------------------------------------------------------------- concurrency


def _run_threads(n, target):
    errors = []
    start = threading.Barrier(n)

    def wrap(i):
        try:
            start.wait()
            target(i)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=wrap, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not any(t.is_alive() for t in threads), "threads hung"
    assert not errors, errors[:3]


def test_conc_no_over_admission_frozen_clock(fast_switching):
    for rep in range(3):
        ft = FakeTime()
        b = S.TokenBucket(1e-12, 3000, clock=ft.clock, sleep=ft.sleep)
        counts = [0] * 16

        def worker(i):
            c = 0
            for _ in range(400):
                if b.try_acquire(1):
                    c += 1
            counts[i] = c

        _run_threads(16, worker)
        assert sum(counts) == 3000, (rep, sum(counts))
        assert not b.try_acquire(1)


def test_conc_no_over_admission_fractional(fast_switching):
    ft = FakeTime()
    b = S.TokenBucket(1e-12, 100, clock=ft.clock, sleep=ft.sleep)
    got = [0.0] * 12

    def worker(i):
        amt = 0.25 * (1 + i % 4)
        s = 0.0
        for _ in range(300):
            if b.try_acquire(amt):
                s += amt
        got[i] = s

    _run_threads(12, worker)
    assert sum(got) <= 100 + 1e-6
    assert sum(got) >= 100 - 1.0 - 1e-6  # leftover is smaller than the largest request
    assert b.available() == pytest.approx(100 - sum(got), abs=1e-6)


def test_conc_blocking_acquire_threads_with_shared_fake_clock(fast_switching):
    ft = FakeTime()
    rate, cap = 10.0, 5.0
    b = S.TokenBucket(rate, cap, clock=ft.clock, sleep=ft.sleep)
    t0 = ft.t
    granted = [0] * 8

    def worker(i):
        for _ in range(25):
            assert b.acquire(1) is True
            granted[i] += 1

    _run_threads(8, worker)
    total = sum(granted)
    assert total == 200
    elapsed = ft.t - t0
    assert total <= cap + rate * elapsed + 1e-6
    assert b.available() <= cap


def test_conc_sleep_does_not_hold_lock():
    ft = FakeTime()
    entered = threading.Event()
    release = threading.Event()

    def blocking_sleep(dt):
        entered.set()
        release.wait(10)
        ft.advance(dt)

    b = S.TokenBucket(1, 2, clock=ft.clock, sleep=blocking_sleep)
    assert b.try_acquire(2)
    result = {}
    waiter = threading.Thread(target=lambda: result.setdefault("ok", b.acquire(2)))
    waiter.start()
    assert entered.wait(5)
    probe_done = {}

    def probe():
        probe_done["avail"] = b.available()
        probe_done["try"] = b.try_acquire(2)

    p = threading.Thread(target=probe)
    p.start()
    p.join(3)
    blocked = p.is_alive()
    release.set()
    p.join(5)
    waiter.join(5)
    assert not blocked, "bucket lock held while sleeping"
    assert probe_done["avail"] == pytest.approx(0, abs=EPS)
    assert probe_done["try"] is False
    assert result["ok"] is True


def test_conc_keyed_sleep_does_not_block_other_keys():
    ft = FakeTime()
    entered = threading.Event()
    release = threading.Event()

    def blocking_sleep(dt):
        entered.set()
        release.wait(10)
        ft.advance(dt)

    k = S.KeyedRateLimiter(1, 1, clock=ft.clock, sleep=blocking_sleep)
    assert k.try_acquire("slow")
    result = {}
    waiter = threading.Thread(target=lambda: result.setdefault("ok", k.acquire("slow")))
    waiter.start()
    assert entered.wait(5)
    probe_done = {}

    def probe():
        probe_done["other"] = k.try_acquire("other")
        probe_done["len"] = len(k)

    p = threading.Thread(target=probe)
    p.start()
    p.join(3)
    blocked = p.is_alive()
    release.set()
    p.join(5)
    waiter.join(5)
    assert not blocked, "limiter lock held while sleeping"
    assert probe_done == {"other": True, "len": 2}
    assert result["ok"] is True


def test_conc_keyed_per_key_exact_admission(fast_switching):
    ft = FakeTime()
    k = S.KeyedRateLimiter(1e-12, 500, clock=ft.clock, sleep=ft.sleep, max_keys=100)
    counts = [[0] * 4 for _ in range(16)]

    def worker(i):
        for j in range(600):
            key = (i + j) % 4
            if k.try_acquire(key):
                counts[i][key] += 1

    _run_threads(16, worker)
    per_key = [sum(counts[i][key] for i in range(16)) for key in range(4)]
    assert per_key == [500] * 4


def test_conc_keyed_eviction_under_contention(fast_switching):
    ft = FakeTime()
    k = S.KeyedRateLimiter(1, 2, clock=ft.clock, sleep=ft.sleep, max_keys=8)

    def worker(i):
        rng = random.Random(i)
        for _ in range(2000):
            k.try_acquire(rng.randrange(40))
            assert len(k) <= 8

    _run_threads(8, worker)
    assert len(k) == 8


# ----------------------------------------------------------------------------------------------- stress


def test_stress_keyed_many_keys_is_constant_time():
    ft = FakeTime()
    k = S.KeyedRateLimiter(1, 1, clock=ft.clock, sleep=ft.sleep, max_keys=20_000)
    for i in range(150_000):
        assert k.try_acquire(i)
    assert len(k) == 20_000
    assert 129_999 not in k and 130_000 in k
    for i in range(130_000, 150_000):
        assert not k.try_acquire(i)


def test_stress_bucket_many_ops_exact_count():
    ft = FakeTime(0.0)
    b = S.TokenBucket(100, 10, clock=ft.clock, sleep=ft.sleep)
    admitted = 0
    for _ in range(200_000):
        ft.t += 0.001  # 0.1 tokens per step
        if b.try_acquire(1):
            admitted += 1
    # 200 s at 100 tokens/s plus the initial 10, give or take float rounding at the boundary
    assert abs(admitted - (10 + 20_000)) <= 2


def test_stress_long_waits_never_busy_spin():
    b, ft = bucket(1e-3, 50)
    assert b.try_acquire(50)
    for _ in range(200):
        assert b.acquire(50, timeout=None)
    assert len(ft.sleeps) == 200
    assert all(s == pytest.approx(50_000) for s in ft.sleeps)
