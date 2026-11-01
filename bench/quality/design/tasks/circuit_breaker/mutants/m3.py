# MUTANT: exceptions outside expected_exceptions skip bookkeeping, leaking the half-open trial slot
"""Resilience utilities: a thread-safe circuit breaker and a retry helper with capped exponential backoff."""
from __future__ import annotations

import enum
import math
import random
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = ["State", "CircuitOpenError", "CircuitBreaker", "full_jitter", "retry"]

T = TypeVar("T")


class State(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised by ``CircuitBreaker.call`` when the call is rejected without invoking the function."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"circuit open; retry after {retry_after:.3f}s")
        self.retry_after = retry_after


# ------------------------------------------------------------------------------------------- validation helpers


def _int_at_least(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


def _finite_at_least(name: str, value: Any, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number >= {minimum}, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or v < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}, got {value!r}")
    return v


def _exception_tuple(name: str, value: Any) -> tuple[type[BaseException], ...]:
    if (not isinstance(value, tuple) or not value
            or not all(isinstance(c, type) and issubclass(c, BaseException) for c in value)):
        raise ValueError(f"{name} must be a non-empty tuple of exception classes, got {value!r}")
    return value


# ------------------------------------------------------------------------------------------- circuit breaker


class CircuitBreaker:
    """CLOSED -> (failure_threshold consecutive failures) -> OPEN -> (recovery_timeout) -> HALF_OPEN -> ...

    Every state transition (and ``reset``) starts a new *epoch*; a call remembers the epoch it was admitted in and
    its outcome is ignored if the epoch has moved on, which makes concurrent/late outcomes harmless.
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0, *,
                 half_open_max_calls: int = 1,
                 expected_exceptions: tuple[type[BaseException], ...] = (Exception,),
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._threshold = _int_at_least("failure_threshold", failure_threshold, 1)
        self._recovery = _finite_at_least("recovery_timeout", recovery_timeout, 0.0)
        self._half_open_max = _int_at_least("half_open_max_calls", half_open_max_calls, 1)
        self._expected = _exception_tuple("expected_exceptions", expected_exceptions)
        self._clock = clock
        self._lock = threading.Lock()
        self._state = State.CLOSED
        self._epoch = 0
        self._failures = 0
        self._opened_at = 0.0
        self._opened_count = 0
        self._trials_in_flight = 0

    # -- internals (self._lock held) ----------------------------------------------------------------------

    def _transition_locked(self, new: State, now: float) -> None:
        self._state = new
        self._epoch += 1
        self._failures = 0
        self._trials_in_flight = 0
        if new is State.OPEN:
            self._opened_at = now
            self._opened_count += 1

    def _refresh_locked(self, now: float) -> None:
        if self._state is State.OPEN and now - self._opened_at >= self._recovery:
            self._transition_locked(State.HALF_OPEN, now)

    def _admit(self) -> tuple[int, bool]:
        """Admit a call or raise CircuitOpenError. Returns (epoch, is_trial)."""
        with self._lock:
            now = self._clock()
            self._refresh_locked(now)
            if self._state is State.CLOSED:
                return self._epoch, False
            if self._state is State.OPEN:
                raise CircuitOpenError(self._opened_at + self._recovery - now)
            if self._trials_in_flight >= self._half_open_max:
                raise CircuitOpenError(0.0)
            self._trials_in_flight += 1
            return self._epoch, True

    def _record(self, epoch: int, is_trial: bool, outcome: str) -> None:
        with self._lock:
            if epoch != self._epoch:
                return  # stale: the breaker moved on since this call was admitted
            if is_trial:
                self._trials_in_flight -= 1
            if outcome == "ignored":
                return
            if self._state is State.CLOSED:
                if outcome == "success":
                    self._failures = 0
                else:
                    self._failures += 1
                    if self._failures >= self._threshold:
                        self._transition_locked(State.OPEN, self._clock())
            elif self._state is State.HALF_OPEN:
                if outcome == "success":
                    self._transition_locked(State.CLOSED, self._clock())
                else:
                    self._transition_locked(State.OPEN, self._clock())

    # -- public API ----------------------------------------------------------------------------------------

    @property
    def state(self) -> State:
        with self._lock:
            self._refresh_locked(self._clock())
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failures

    @property
    def opened_count(self) -> int:
        with self._lock:
            return self._opened_count

    def call(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        epoch, is_trial = self._admit()
        try:
            result = fn(*args, **kwargs)  # never under the lock
        except self._expected:
            self._record(epoch, is_trial, "failure")
            raise
        self._record(epoch, is_trial, "success")
        return result

    def reset(self) -> None:
        with self._lock:
            self._transition_locked(State.CLOSED, self._clock())

    def __repr__(self) -> str:
        return (f"CircuitBreaker(state={self.state.value}, failures={self.failure_count}/{self._threshold}, "
                f"recovery_timeout={self._recovery})")


# ------------------------------------------------------------------------------------------- retry


def full_jitter(delay: float) -> float:
    """'Full jitter' (AWS architecture blog): a uniformly random delay in [0, delay]."""
    return random.uniform(0.0, delay)


def _backoff(i: int, base_delay: float, max_delay: float, multiplier: float) -> float:
    if base_delay == 0.0:
        return 0.0
    try:
        d = base_delay * multiplier ** i
    except OverflowError:
        return max_delay
    return min(max_delay, d)


def _is_retryable(exc: BaseException, retry_on: tuple[type[BaseException], ...]) -> bool:
    if isinstance(exc, CircuitOpenError):
        # Only an explicit CircuitOpenError (sub)class makes an open circuit retryable; `Exception` does not.
        return any(issubclass(c, CircuitOpenError) and isinstance(exc, c) for c in retry_on)
    return isinstance(exc, retry_on)


def retry(fn: Callable[[], T], *,
          attempts: int,
          base_delay: float,
          max_delay: float,
          multiplier: float = 2.0,
          jitter: Callable[[float], float] = full_jitter,
          retry_on: tuple[type[BaseException], ...] = (Exception,),
          sleep: Callable[[float], None] = time.sleep,
          on_retry: Callable[[int, BaseException, float], None] | None = None) -> T:
    """Call ``fn`` up to ``attempts`` times with capped exponential backoff between attempts."""
    attempts = _int_at_least("attempts", attempts, 1)
    base_delay = _finite_at_least("base_delay", base_delay, 0.0)
    max_delay = _finite_at_least("max_delay", max_delay, 0.0)
    multiplier = _finite_at_least("multiplier", multiplier, 1.0)
    retry_on = _exception_tuple("retry_on", retry_on)

    for i in range(attempts):
        try:
            return fn()
        except BaseException as exc:
            if i == attempts - 1 or not _is_retryable(exc, retry_on):
                raise
            delay = jitter(_backoff(i, base_delay, max_delay, multiplier))
            if on_retry is not None:
                on_retry(i + 1, exc, delay)
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
