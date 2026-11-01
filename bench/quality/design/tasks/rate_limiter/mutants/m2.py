# MUTANT: eviction is FIFO by creation, uses do not refresh recency
"""Thread-safe token-bucket rate limiting with injectable time.

``TokenBucket`` is a single continuously refilling bucket; ``KeyedRateLimiter`` keeps one bucket per key with
LRU eviction so memory stays bounded.
"""
from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable

__all__ = ["TokenBucket", "KeyedRateLimiter"]

#: Absolute tolerance (in tokens) used when deciding whether a request is satisfiable.
EPSILON = 1e-9


def _positive_finite(name: str, value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}") from None
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    return v


def _check_tokens(tokens: float, capacity: float) -> float:
    try:
        t = float(tokens)
    except (TypeError, ValueError):
        raise ValueError(f"tokens must be a number, got {tokens!r}") from None
    if not (0 < t <= capacity):  # also rejects nan
        raise ValueError(f"tokens must satisfy 0 < tokens <= capacity ({capacity}), got {tokens!r}")
    return t


def _check_timeout(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    try:
        t = float(timeout)
    except (TypeError, ValueError):
        raise ValueError(f"timeout must be None or a number >= 0, got {timeout!r}") from None
    if not t >= 0:  # also rejects nan
        raise ValueError(f"timeout must be None or a number >= 0, got {timeout!r}")
    return t


class TokenBucket:
    """A token bucket refilling continuously at ``rate`` tokens/s up to ``capacity``. Starts full."""

    __slots__ = ("_rate", "_capacity", "_clock", "_sleep", "_lock", "_level", "_last")

    def __init__(self, rate: float, capacity: float, *,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._rate = _positive_finite("rate", rate)
        self._capacity = _positive_finite("capacity", capacity)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._level = self._capacity
        self._last = clock()

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def capacity(self) -> float:
        return self._capacity

    # -- internals (call with self._lock held) -------------------------------------------------------------

    def _refill_locked(self, now: float) -> None:
        if now > self._last:
            self._level = min(self._capacity, self._level + (now - self._last) * self._rate)
            self._last = now
        # A clock that went backwards earns nothing and does not move _last back.

    def _take_locked(self, tokens: float, now: float) -> float:
        """Refill to ``now`` and take ``tokens`` if possible. Returns 0.0 on success, else the wait in seconds."""
        self._refill_locked(now)
        if self._level >= tokens - EPSILON:
            self._level = max(0.0, self._level - tokens)
            return 0.0
        return (tokens - self._level) / self._rate

    # -- public API --------------------------------------------------------------------------------------------

    def try_acquire(self, tokens: float = 1) -> bool:
        t = _check_tokens(tokens, self._capacity)
        with self._lock:
            return self._take_locked(t, self._clock()) == 0.0

    def acquire(self, tokens: float = 1, timeout: float | None = None) -> bool:
        t = _check_tokens(tokens, self._capacity)
        to = _check_timeout(timeout)
        return self._acquire_validated(t, to)

    def _acquire_validated(self, tokens: float, timeout: float | None) -> bool:
        deadline: float | None = None
        while True:
            with self._lock:
                now = self._clock()
                if deadline is None and timeout is not None:
                    deadline = now + timeout
                wait = self._take_locked(tokens, now)
            if wait == 0.0:
                return True
            if deadline is not None and now + wait > deadline:
                return False
            self._sleep(wait)  # never under the lock

    def available(self) -> float:
        with self._lock:
            self._refill_locked(self._clock())
            return self._level

    def __repr__(self) -> str:
        return f"TokenBucket(rate={self._rate!r}, capacity={self._capacity!r})"


class KeyedRateLimiter:
    """One ``TokenBucket`` per key, at most ``max_keys`` tracked keys, least-recently-used key evicted first."""

    def __init__(self, rate: float, capacity: float, *,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 max_keys: int = 10_000) -> None:
        self._rate = _positive_finite("rate", rate)
        self._capacity = _positive_finite("capacity", capacity)
        if isinstance(max_keys, bool) or not isinstance(max_keys, int) or max_keys < 1:
            raise ValueError(f"max_keys must be an int >= 1, got {max_keys!r}")
        self._max_keys = max_keys
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._buckets: OrderedDict[Hashable, TokenBucket] = OrderedDict()

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def capacity(self) -> float:
        return self._capacity

    @property
    def max_keys(self) -> int:
        return self._max_keys

    def _use(self, key: Hashable) -> TokenBucket:
        """Return the bucket for ``key``, creating it and evicting the LRU key if needed; marks the key as used."""
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(self._rate, self._capacity, clock=self._clock, sleep=self._sleep)
                self._buckets[key] = bucket
                if len(self._buckets) > self._max_keys:
                    self._buckets.popitem(last=False)
            return bucket

    def try_acquire(self, key: Hashable, tokens: float = 1) -> bool:
        t = _check_tokens(tokens, self._capacity)
        return self._use(key).try_acquire(t)

    def acquire(self, key: Hashable, tokens: float = 1, timeout: float | None = None) -> bool:
        t = _check_tokens(tokens, self._capacity)
        to = _check_timeout(timeout)
        return self._use(key)._acquire_validated(t, to)  # bucket lock only; limiter lock is not held

    def available(self, key: Hashable) -> float:
        with self._lock:
            bucket = self._buckets.get(key)
        return self._capacity if bucket is None else bucket.available()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._buckets

    def __repr__(self) -> str:
        return (f"KeyedRateLimiter(rate={self._rate!r}, capacity={self._capacity!r}, "
                f"max_keys={self._max_keys!r}, tracked={len(self)})")
