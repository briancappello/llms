import math
import random
import sys
import threading

import pytest

import solution as S


class FakeTime:
    def __init__(self, t: float = 1000.0):
        self.t = t
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def clock(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        with self._lock:
            self.sleeps.append(dt)
            self.t += dt

    def advance(self, dt: float) -> None:
        with self._lock:
            self.t += dt


class Boom(Exception):
    pass


class Other(Exception):
    pass


def ok(value="ok"):
    return value


def fail(exc=None):
    raise exc or Boom("boom")


def breaker(threshold=3, recovery=10.0, ft=None, **kw):
    ft = ft or FakeTime()
    return S.CircuitBreaker(threshold, recovery, clock=ft.clock, **kw), ft


def fail_n(cb, n, exc_type=Boom):
    for _ in range(n):
        with pytest.raises(exc_type):
            cb.call(fail, exc_type())


def identity(d):
    return d


@pytest.fixture
def fast_switching():
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(old)


# ======================================================================================= breaker: basic


def test_basic_initial_state_and_passthrough():
    cb, _ = breaker()
    assert cb.state is S.State.CLOSED
    assert cb.failure_count == 0 and cb.opened_count == 0
    assert cb.call(ok, 42) == 42
    assert cb.call(lambda a, b=0: a + b, 1, b=2) == 3
    assert {s.value for s in S.State} == {"closed", "open", "half_open"}


def test_basic_exceptions_reraised_unchanged():
    cb, _ = breaker(threshold=10)
    err = Boom("x")
    with pytest.raises(Boom) as ei:
        cb.call(fail, err)
    assert ei.value is err
    assert cb.failure_count == 1


def test_basic_opens_after_threshold_consecutive_failures():
    cb, _ = breaker(threshold=3)
    fail_n(cb, 2)
    assert cb.state is S.State.CLOSED and cb.failure_count == 2
    fail_n(cb, 1)
    assert cb.state is S.State.OPEN
    assert cb.opened_count == 1
    assert cb.failure_count == 0


def test_basic_success_resets_consecutive_count():
    cb, _ = breaker(threshold=3)
    fail_n(cb, 2)
    cb.call(ok)
    assert cb.failure_count == 0
    fail_n(cb, 2)
    assert cb.state is S.State.CLOSED
    fail_n(cb, 1)
    assert cb.state is S.State.OPEN


def test_basic_open_rejects_without_calling_fn():
    cb, ft = breaker(threshold=1, recovery=10)
    fail_n(cb, 1)
    calls = []
    ft.advance(4)
    with pytest.raises(S.CircuitOpenError) as ei:
        cb.call(lambda: calls.append(1))
    assert calls == []
    assert ei.value.retry_after == pytest.approx(6)
    assert isinstance(ei.value, Exception)


def test_basic_half_open_after_timeout_then_success_closes():
    cb, ft = breaker(threshold=2, recovery=5)
    fail_n(cb, 2)
    ft.advance(4.5)
    assert cb.state is S.State.OPEN
    ft.advance(0.5)
    assert cb.state is S.State.HALF_OPEN
    assert cb.call(ok, "probe") == "probe"
    assert cb.state is S.State.CLOSED
    assert cb.failure_count == 0
    fail_n(cb, 1)
    assert cb.state is S.State.CLOSED  # needs 2 fresh failures again


def test_basic_half_open_failure_reopens_and_restarts_timer():
    cb, ft = breaker(threshold=1, recovery=10)
    fail_n(cb, 1)
    ft.advance(12)
    fail_n(cb, 1)  # trial fails
    assert cb.state is S.State.OPEN
    assert cb.opened_count == 2
    ft.advance(9)
    with pytest.raises(S.CircuitOpenError) as ei:
        cb.call(ok)
    assert ei.value.retry_after == pytest.approx(1)
    ft.advance(1)
    assert cb.state is S.State.HALF_OPEN


def test_basic_timer_starts_when_failure_recorded():
    cb, ft = breaker(threshold=1, recovery=10)

    def slow_fail():
        ft.advance(3)  # the call itself takes 3 s
        raise Boom()

    with pytest.raises(Boom):
        cb.call(slow_fail)
    ft.advance(9.5)  # 12.5 s after admission, 9.5 s after the failure
    assert cb.state is S.State.OPEN
    with pytest.raises(S.CircuitOpenError) as ei:
        cb.call(ok)
    assert ei.value.retry_after == pytest.approx(0.5)


def test_basic_unexpected_exceptions_do_not_affect_state():
    cb, _ = breaker(threshold=3, expected_exceptions=(Boom,))
    fail_n(cb, 2)
    with pytest.raises(Other):
        cb.call(fail, Other())
    assert cb.failure_count == 2  # neither incremented nor reset
    fail_n(cb, 1)
    assert cb.state is S.State.OPEN


def test_basic_expected_exception_subclasses_count():
    cb, _ = breaker(threshold=2, expected_exceptions=(OSError,))
    with pytest.raises(ConnectionError):
        cb.call(fail, ConnectionError())
    with pytest.raises(TimeoutError):
        cb.call(fail, TimeoutError())
    assert cb.state is S.State.OPEN


def test_basic_reset_closes():
    cb, _ = breaker(threshold=1)
    fail_n(cb, 1)
    assert cb.state is S.State.OPEN
    cb.reset()
    assert cb.state is S.State.CLOSED
    assert cb.failure_count == 0
    assert cb.opened_count == 1
    assert cb.call(ok) == "ok"


# ======================================================================================= breaker: edge


@pytest.mark.parametrize("kwargs", [
    dict(failure_threshold=0), dict(failure_threshold=-1), dict(recovery_timeout=-0.1),
    dict(recovery_timeout=math.nan), dict(recovery_timeout=math.inf), dict(half_open_max_calls=0),
    dict(expected_exceptions=()), dict(expected_exceptions=(Boom, 3)),
])
def test_edge_breaker_validation(kwargs):
    with pytest.raises(ValueError):
        S.CircuitBreaker(clock=FakeTime().clock, **kwargs)


def test_edge_zero_recovery_timeout_goes_half_open_immediately():
    cb, ft = breaker(threshold=1, recovery=0)
    fail_n(cb, 1)
    assert cb.opened_count == 1
    assert cb.state is S.State.HALF_OPEN
    assert cb.call(ok) == "ok"
    assert cb.state is S.State.CLOSED


def test_edge_open_to_half_open_via_call_without_state_read():
    cb, ft = breaker(threshold=1, recovery=2)
    fail_n(cb, 1)
    ft.advance(2)
    assert cb.call(ok, 7) == 7
    assert cb.state is S.State.CLOSED


def test_edge_half_open_trial_limit_single_thread_reentrant():
    cb, ft = breaker(threshold=1, recovery=1, half_open_max_calls=2)
    fail_n(cb, 1)
    ft.advance(1)
    seen = {}

    def outer():
        def inner():
            with pytest.raises(S.CircuitOpenError) as ei:
                cb.call(ok)  # third concurrent trial -> rejected
            seen["retry_after"] = ei.value.retry_after
            return "inner"
        return cb.call(inner)  # second trial

    assert cb.call(outer) == "inner"
    assert seen["retry_after"] == 0.0
    assert cb.state is S.State.CLOSED


def test_edge_ignored_exception_frees_half_open_slot():
    cb, ft = breaker(threshold=1, recovery=1, expected_exceptions=(Boom,))
    fail_n(cb, 1)
    ft.advance(1)
    with pytest.raises(Other):
        cb.call(fail, Other())
    assert cb.state is S.State.HALF_OPEN
    with pytest.raises(KeyboardInterrupt):
        cb.call(fail, KeyboardInterrupt())
    assert cb.state is S.State.HALF_OPEN
    assert cb.call(ok) == "ok"  # slot was released both times
    assert cb.state is S.State.CLOSED


def test_edge_stale_failure_after_open_is_ignored_reentrant():
    cb, ft = breaker(threshold=2, recovery=10)

    def outer():
        fail_n(cb, 2)  # nested calls open the breaker while this call is in flight
        assert cb.state is S.State.OPEN
        ft.advance(4)
        raise Boom("late")

    with pytest.raises(Boom):
        cb.call(outer)
    assert cb.opened_count == 1
    assert cb.state is S.State.OPEN
    with pytest.raises(S.CircuitOpenError) as ei:
        cb.call(ok)
    assert ei.value.retry_after == pytest.approx(6)  # timer not restarted by the stale failure


def test_edge_stale_success_does_not_close():
    cb, ft = breaker(threshold=1, recovery=10)

    def outer():
        fail_n(cb, 1)
        return "late success"

    assert cb.call(outer) == "late success"
    assert cb.state is S.State.OPEN


def test_edge_reset_makes_in_flight_outcomes_stale():
    cb, _ = breaker(threshold=2)
    fail_n(cb, 1)

    def outer():
        cb.reset()
        raise Boom()

    with pytest.raises(Boom):
        cb.call(outer)
    assert cb.failure_count == 0
    fail_n(cb, 1)
    assert cb.state is S.State.CLOSED


def test_edge_half_open_first_outcome_wins_reentrant():
    cb, ft = breaker(threshold=1, recovery=5, half_open_max_calls=2)
    fail_n(cb, 1)
    ft.advance(5)

    def outer_fails_after_inner_succeeds():
        assert cb.call(ok) == "ok"  # second trial succeeds -> CLOSED
        raise Boom()  # first trial's failure is now stale

    with pytest.raises(Boom):
        cb.call(outer_fails_after_inner_succeeds)
    assert cb.state is S.State.CLOSED
    assert cb.failure_count == 0
    assert cb.opened_count == 1


def test_edge_failure_count_zero_while_open_and_half_open():
    cb, ft = breaker(threshold=2, recovery=1)
    fail_n(cb, 2)
    assert cb.failure_count == 0
    ft.advance(1)
    assert cb.state is S.State.HALF_OPEN
    assert cb.failure_count == 0


# ======================================================================================= retry: basic


def test_basic_retry_returns_first_success():
    ft = FakeTime()
    seq = iter([Boom(), Boom(), "done"])

    def fn():
        v = next(seq)
        if isinstance(v, Exception):
            raise v
        return v

    assert S.retry(fn, attempts=5, base_delay=0.1, max_delay=10, jitter=identity, sleep=ft.sleep) == "done"
    assert ft.sleeps == [pytest.approx(0.1), pytest.approx(0.2)]


def test_basic_retry_exact_capped_sleep_sequence_and_reraise_last():
    ft = FakeTime()
    errs = []

    def fn():
        errs.append(Boom(len(errs)))
        raise errs[-1]

    with pytest.raises(Boom) as ei:
        S.retry(fn, attempts=7, base_delay=0.5, max_delay=5, multiplier=3, jitter=identity, sleep=ft.sleep)
    assert ei.value is errs[-1]
    assert len(errs) == 7
    assert ft.sleeps == [pytest.approx(x) for x in [0.5, 1.5, 4.5, 5, 5, 5]]  # no sleep after the last attempt


def test_basic_retry_on_retry_callback():
    ft = FakeTime()
    log = []
    errs = [Boom("a"), Boom("b")]
    it = iter(errs + ["fine"])

    def fn():
        v = next(it)
        if isinstance(v, Exception):
            raise v
        return v

    jit = lambda d: d / 2  # noqa: E731
    out = S.retry(fn, attempts=3, base_delay=1, max_delay=100, jitter=jit, sleep=ft.sleep,
                  on_retry=lambda n, e, d: log.append((n, e, d, len(ft.sleeps))))
    assert out == "fine"
    assert log == [(1, errs[0], pytest.approx(0.5), 0), (2, errs[1], pytest.approx(1.0), 1)]
    assert ft.sleeps == [pytest.approx(0.5), pytest.approx(1.0)]


def test_basic_retry_non_matching_exception_propagates_immediately():
    ft = FakeTime()
    calls = []

    def fn():
        calls.append(1)
        if len(calls) == 1:
            raise Boom()
        raise Other()

    with pytest.raises(Other):
        S.retry(fn, attempts=5, base_delay=1, max_delay=1, jitter=identity, retry_on=(Boom,), sleep=ft.sleep,
                on_retry=lambda *a: None)
    assert len(calls) == 2
    assert ft.sleeps == [1]


def test_basic_retry_default_full_jitter_bounds():
    ft = FakeTime()
    with pytest.raises(Boom):
        S.retry(lambda: fail(), attempts=12, base_delay=0.01, max_delay=1.0, sleep=ft.sleep)
    assert len(ft.sleeps) == 11
    for i, s in enumerate(ft.sleeps):
        assert 0 <= s <= min(1.0, 0.01 * 2 ** i) + 1e-12
    vals = [S.full_jitter(2.0) for _ in range(2000)]
    assert all(0 <= v <= 2.0 for v in vals)
    assert min(vals) < 0.2 and max(vals) > 1.8  # actually spreads over the range
    assert S.full_jitter(0) == 0


def test_basic_retry_with_breaker_fails_fast_by_default():
    cb, ft = breaker(threshold=2, recovery=10)
    calls = []

    def flaky():
        calls.append(1)
        raise Boom()

    with pytest.raises(S.CircuitOpenError):
        S.retry(lambda: cb.call(flaky), attempts=10, base_delay=1, max_delay=60, jitter=identity,
                sleep=ft.sleep)
    assert len(calls) == 2  # two real calls opened the breaker, third attempt rejected -> not retried
    assert ft.sleeps == [1, 2]


def test_basic_retry_waits_out_breaker_when_listed():
    cb, ft = breaker(threshold=2, recovery=10)
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] <= 2:
            raise Boom()
        return "recovered"

    out = S.retry(lambda: cb.call(flaky), attempts=10, base_delay=1, max_delay=60, jitter=identity,
                  retry_on=(Boom, S.CircuitOpenError), sleep=ft.sleep)
    assert out == "recovered"
    # fail(1s) fail->open(2s) reject(4s) reject(8s: total 15 >= 10) half-open trial succeeds
    assert ft.sleeps == [1, 2, 4, 8]
    assert cb.state is S.State.CLOSED


# ======================================================================================= retry: edge


@pytest.mark.parametrize("kwargs", [
    dict(attempts=0), dict(attempts=-3), dict(base_delay=-1), dict(max_delay=-0.5), dict(base_delay=math.nan),
    dict(max_delay=math.inf), dict(multiplier=0.5), dict(multiplier=math.nan), dict(retry_on=()),
    dict(retry_on=(Boom, "x")),
])
def test_edge_retry_validation_before_calling(kwargs):
    calls = []
    args = dict(attempts=3, base_delay=0.1, max_delay=1.0, sleep=lambda d: None)
    args.update(kwargs)
    with pytest.raises(ValueError):
        S.retry(lambda: calls.append(1), **args)
    assert calls == []


def test_edge_retry_single_attempt_never_sleeps():
    ft = FakeTime()
    with pytest.raises(Boom):
        S.retry(fail, attempts=1, base_delay=1, max_delay=1, sleep=ft.sleep)
    assert ft.sleeps == []
    assert S.retry(lambda: 5, attempts=1, base_delay=0, max_delay=0, sleep=ft.sleep) == 5


def test_edge_retry_zero_delays_still_sleep_between_attempts():
    ft = FakeTime()
    with pytest.raises(Boom):
        S.retry(fail, attempts=4, base_delay=0, max_delay=5, jitter=identity, sleep=ft.sleep)
    assert ft.sleeps == [0, 0, 0]


def test_edge_retry_max_below_base_caps_everything():
    ft = FakeTime()
    with pytest.raises(Boom):
        S.retry(fail, attempts=4, base_delay=3, max_delay=1, jitter=identity, sleep=ft.sleep)
    assert ft.sleeps == [1, 1, 1]


def test_edge_retry_multiplier_one_constant():
    ft = FakeTime()
    with pytest.raises(Boom):
        S.retry(fail, attempts=5, base_delay=0.75, max_delay=10, multiplier=1, jitter=identity, sleep=ft.sleep)
    assert ft.sleeps == [0.75] * 4


def test_edge_retry_callback_or_sleep_exception_aborts():
    calls = []

    def fn():
        calls.append(1)
        raise Boom()

    def bad_cb(n, e, d):
        raise RuntimeError("cb")

    with pytest.raises(RuntimeError, match="cb"):
        S.retry(fn, attempts=5, base_delay=1, max_delay=1, sleep=lambda d: None, on_retry=bad_cb)
    assert len(calls) == 1

    def bad_sleep(d):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        S.retry(fn, attempts=5, base_delay=1, max_delay=1, sleep=bad_sleep)
    assert len(calls) == 2


def test_edge_retry_circuit_open_subclass_rules():
    class MyOpen(S.CircuitOpenError):
        pass

    def raiser(exc):
        def fn():
            calls.append(1)
            raise exc
        return fn

    calls = []
    with pytest.raises(S.CircuitOpenError):
        S.retry(raiser(S.CircuitOpenError(1.0)), attempts=3, base_delay=0, max_delay=0,
                retry_on=(Exception, BaseException), sleep=lambda d: None)
    assert len(calls) == 1
    calls.clear()
    with pytest.raises(MyOpen):
        S.retry(raiser(MyOpen(1.0)), attempts=3, base_delay=0, max_delay=0, retry_on=(S.CircuitOpenError,),
                sleep=lambda d: None)
    assert len(calls) == 3
    calls.clear()
    with pytest.raises(S.CircuitOpenError):  # only the subclass is listed; the base class is not retried
        S.retry(raiser(S.CircuitOpenError(1.0)), attempts=3, base_delay=0, max_delay=0, retry_on=(MyOpen,),
                sleep=lambda d: None)
    assert len(calls) == 1


def test_edge_retry_retries_base_exception_only_if_listed():
    calls = []

    def fn():
        calls.append(1)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        S.retry(fn, attempts=3, base_delay=0, max_delay=0, sleep=lambda d: None)
    assert len(calls) == 1
    with pytest.raises(KeyboardInterrupt):
        S.retry(fn, attempts=3, base_delay=0, max_delay=0, retry_on=(KeyboardInterrupt,), sleep=lambda d: None)
    assert len(calls) == 4


# ======================================================================================= property


class ModelBreaker:
    def __init__(self, threshold, recovery):
        self.th, self.rec = threshold, recovery
        self.state, self.failures, self.opened_at, self.opened = "closed", 0, 0.0, 0

    def refresh(self, now):
        if self.state == "open" and now - self.opened_at >= self.rec:
            self.state, self.failures = "half_open", 0

    def to(self, state, now):
        self.state, self.failures = state, 0
        if state == "open":
            self.opened_at, self.opened = now, self.opened + 1

    def call(self, outcome, t_admit, t_done):
        self.refresh(t_admit)
        if self.state == "open":
            return ("rejected", self.opened_at + self.rec - t_admit)
        if outcome == "ignored":
            return ("ignored", None)
        if self.state == "closed":
            if outcome == "success":
                self.failures = 0
            else:
                self.failures += 1
                if self.failures >= self.th:
                    self.to("open", t_done)
        else:
            self.to("closed" if outcome == "success" else "open", t_done)
        return (outcome, None)


def test_prop_breaker_vs_state_machine_model():
    rng = random.Random(4242)
    for case in range(300):
        th = rng.randint(1, 4)
        rec = rng.choice([0.0, 1.0, 2.5, 5.0, 10.0])
        ft = FakeTime(rng.choice([0.0, 1000.0]))
        cb = S.CircuitBreaker(th, rec, expected_exceptions=(OSError,), clock=ft.clock)
        m = ModelBreaker(th, rec)
        for step in range(60):
            ctx = f"case={case} step={step}"
            r = rng.random()
            if r < 0.2:
                ft.advance(rng.choice([0.25, 0.5, 1.0, 2.5, 5.0, 10.0]))
                continue
            if r < 0.25:
                cb.reset()
                m.to("closed", ft.t)
                continue
            if r < 0.32:
                m.refresh(ft.t)
                assert cb.state.value == m.state, ctx
                assert cb.failure_count == m.failures, ctx
                assert cb.opened_count == m.opened, ctx
                continue
            outcome = rng.choices(["success", "failure", "ignored"], [4, 5, 1])[0]
            duration = rng.choice([0.0, 0.0, 0.5, 1.5])
            t_admit = ft.t
            exp, exp_after = m.call(outcome, t_admit, t_admit + duration)
            invoked = []

            def fn():
                invoked.append(1)
                ft.advance(duration)
                if outcome == "success":
                    return "v"
                raise ConnectionError() if outcome == "failure" else KeyError()

            try:
                got, after = ("success", None) if cb.call(fn) == "v" else ("?", None)
            except S.CircuitOpenError as e:
                got, after = "rejected", e.retry_after
            except ConnectionError:
                got, after = "failure", None
            except KeyError:
                got, after = "ignored", None
            assert got == exp, ctx
            if exp == "rejected":
                assert invoked == [], ctx
                assert after == pytest.approx(exp_after), ctx
            else:
                assert invoked == [1], ctx
            assert cb.failure_count == m.failures, ctx
            assert cb.opened_count == m.opened, ctx


def test_prop_retry_vs_model():
    rng = random.Random(777)
    for case in range(400):
        attempts = rng.randint(1, 8)
        base = rng.choice([0.0, 0.01, 0.1, 0.5, 1.0, 3.0])
        mx = rng.choice([0.0, 0.05, 1.0, 2.0, 30.0])
        mult = rng.choice([1.0, 1.5, 2.0, 3.0, 10.0])
        retry_on = rng.choice([(Boom,), (Exception,), (Boom, Other), (OSError,)])
        script = [rng.choice(["boom", "boom", "other", "ok"]) for _ in range(attempts)]
        ctx = f"case={case}"

        # model
        exp_sleeps, exp_cb = [], []
        exp_result, exp_exc_index = None, None
        for i, s in enumerate(script):
            if s == "ok":
                exp_result = i
                break
            etype = Boom if s == "boom" else Other
            if i == attempts - 1 or not issubclass(etype, retry_on):
                exp_exc_index = i
                break
            d = min(mx, base * mult ** i)
            exp_cb.append((i + 1, i, d))
            exp_sleeps.append(d)

        errs = []
        calls = []

        def fn():
            i = len(calls)
            calls.append(i)
            s = script[i]
            if s == "ok":
                return i
            errs.append((Boom if s == "boom" else Other)(i))
            raise errs[-1]

        sleeps, cbs = [], []
        try:
            got = S.retry(fn, attempts=attempts, base_delay=base, max_delay=mx, multiplier=mult, jitter=identity,
                          retry_on=retry_on, sleep=sleeps.append,
                          on_retry=lambda n, e, d: cbs.append((n, e.args[0], d)))
            got_exc = None
        except (Boom, Other) as e:
            got, got_exc = None, e
        if exp_exc_index is None:
            assert got == exp_result and got_exc is None, ctx
        else:
            assert got_exc is errs[-1] and got_exc.args[0] == exp_exc_index, ctx
        assert sleeps == [pytest.approx(x, rel=1e-9, abs=1e-12) for x in exp_sleeps], ctx
        assert cbs == [(n, i, pytest.approx(d, rel=1e-9, abs=1e-12)) for n, i, d in exp_cb], ctx
        assert len(calls) == (exp_result if exp_result is not None else exp_exc_index) + 1, ctx


# ======================================================================================= concurrency


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


def test_conc_concurrent_failures_open_exactly_once():
    for rep in range(5):
        cb, ft = breaker(threshold=5, recovery=30)
        n = 24
        inside = threading.Barrier(n)

        def failing():
            inside.wait(10)  # all n calls are in flight at the same time
            ft.advance(0.001)
            raise Boom()

        results = []

        def worker(i):
            try:
                cb.call(failing)
            except Boom:
                results.append("boom")

        _run_threads(n, worker)
        assert results == ["boom"] * n
        assert cb.opened_count == 1, rep
        assert cb.state is S.State.OPEN
        assert cb.failure_count == 0
        with pytest.raises(S.CircuitOpenError) as ei:
            cb.call(ok)
        assert 29.9 < ei.value.retry_after <= 30


def test_conc_open_once_with_slow_clock():
    # A clock that yields to other threads makes check-then-transition races visible.
    import time as _time

    for rep in range(3):
        ft = FakeTime()

        def slow_clock():
            _time.sleep(0.0005)
            return ft.t

        cb = S.CircuitBreaker(3, 30.0, clock=slow_clock)
        n = 16
        inside = threading.Barrier(n)

        def failing():
            inside.wait(10)
            raise Boom()

        def worker(i):
            with pytest.raises(Boom):
                cb.call(failing)

        _run_threads(n, worker)
        assert cb.opened_count == 1, rep
        assert cb.state is S.State.OPEN
        assert cb.failure_count == 0


def test_conc_failure_counter_has_no_lost_updates(fast_switching):
    cb, _ = breaker(threshold=10**9)

    def worker(i):
        for _ in range(500):
            with pytest.raises(Boom):
                cb.call(fail)

    _run_threads(16, worker)
    assert cb.failure_count == 16 * 500
    assert cb.state is S.State.CLOSED


def test_conc_half_open_admits_at_most_max_trials():
    cb, ft = breaker(threshold=1, recovery=1, half_open_max_calls=3)
    fail_n(cb, 1)
    ft.advance(1)
    release = threading.Event()
    admitted = []
    rejected = []
    lock = threading.Lock()

    def trial():
        with lock:
            admitted.append(1)
        release.wait(10)
        return "ok"

    def worker(i):
        try:
            cb.call(trial)
        except S.CircuitOpenError as e:
            with lock:
                rejected.append(e.retry_after)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for _ in range(500):
        with lock:
            if len(admitted) + len(rejected) == 12 or (len(admitted) == 3 and len(rejected) == 9):
                break
        threading.Event().wait(0.01)
    with lock:
        snapshot = (len(admitted), len(rejected))
    release.set()
    for t in threads:
        t.join(10)
    assert snapshot == (3, 9)
    assert rejected == [0.0] * 9
    assert cb.state is S.State.CLOSED
    assert cb.opened_count == 1


def test_conc_slow_call_does_not_block_other_callers():
    cb, _ = breaker(threshold=3)
    entered = threading.Event()
    release = threading.Event()

    def slow():
        entered.set()
        release.wait(10)
        return "slow"

    out = {}
    t = threading.Thread(target=lambda: out.setdefault("slow", cb.call(slow)))
    t.start()
    assert entered.wait(5)
    p = threading.Thread(target=lambda: out.setdefault("fast", (cb.call(ok, 1), cb.state, cb.failure_count)))
    p.start()
    p.join(3)
    blocked = p.is_alive()
    release.set()
    t.join(5)
    p.join(5)
    assert not blocked, "breaker lock held while fn runs"
    assert out == {"slow": "slow", "fast": (1, S.State.CLOSED, 0)}


def test_conc_mixed_load_invariants(fast_switching):
    cb, ft = breaker(threshold=3, recovery=0.5, half_open_max_calls=2)
    counts = {"rejected": 0, "ran": 0}
    lock = threading.Lock()

    def worker(i):
        rng = random.Random(i)
        for _ in range(400):
            ran = []

            def fn():
                ran.append(1)
                if rng.random() < 0.5:
                    raise Boom()
                return 1

            try:
                cb.call(fn)
            except (Boom, S.CircuitOpenError) as e:
                if isinstance(e, S.CircuitOpenError):
                    assert e.retry_after >= 0
                    assert not ran
            with lock:
                counts["ran"] += len(ran)
                counts["rejected"] += 1 - len(ran)
            if rng.random() < 0.1:
                ft.advance(0.25)

    _run_threads(8, worker)
    assert counts["ran"] + counts["rejected"] == 8 * 400
    assert cb.opened_count >= 1
    s = cb.state
    assert s in (S.State.CLOSED, S.State.OPEN, S.State.HALF_OPEN)
    if s is not S.State.CLOSED:
        assert cb.failure_count == 0
    # after the dust settles, the breaker still works: wait out the timer and probe successfully
    ft.advance(1)
    assert cb.call(ok) == "ok"
    assert cb.state is S.State.CLOSED


# ======================================================================================= stress


def test_stress_huge_attempt_count_no_overflow():
    sleeps = []
    with pytest.raises(Boom):
        S.retry(fail, attempts=100_000, base_delay=0.001, max_delay=60, jitter=identity, sleep=sleeps.append)
    assert len(sleeps) == 99_999
    assert sleeps[0] == pytest.approx(0.001)
    assert sleeps[-1] == 60 and sleeps[5000] == 60


def test_stress_many_breaker_cycles():
    cb, ft = breaker(threshold=3, recovery=1.0)
    cycles = 0
    for i in range(60_000):
        try:
            if i % 7 < 4:
                cb.call(fail)
            else:
                cb.call(ok)
        except S.CircuitOpenError:
            ft.advance(0.5)
        except Boom:
            pass
    cycles = cb.opened_count
    assert cycles > 1000
    assert cb.state in (S.State.CLOSED, S.State.OPEN, S.State.HALF_OPEN)


def test_stress_many_breakers_and_calls():
    ft = FakeTime()
    cbs = [S.CircuitBreaker(2, 5.0, clock=ft.clock) for _ in range(2000)]
    total = 0
    for rnd in range(50):
        for j, cb in enumerate(cbs):
            if j % 10 == 0:
                try:
                    cb.call(fail)
                except (Boom, S.CircuitOpenError):
                    pass
            else:
                total += cb.call(ok, 1)
        ft.advance(1.0)
    assert total == 50 * 1800
    assert all(cb.state is S.State.CLOSED for j, cb in enumerate(cbs) if j % 10)
    assert all(cb.opened_count >= 1 for j, cb in enumerate(cbs) if j % 10 == 0)
