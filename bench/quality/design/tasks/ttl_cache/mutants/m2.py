"""Thread-safe LRU cache with per-entry TTL, lazy expiry and single-flight loading."""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

_DEFAULT: Any = object()
_MISSING: Any = object()


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int
    expirations: int


class _Entry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, expires_at: float | None) -> None:
        self.value = value
        self.expires_at = expires_at


class _Flight:
    """One in-progress load of a key; waiters block on `done`."""
    __slots__ = ("done", "ok", "value")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.ok = False
        self.value: Any = None


def _check_ttl(ttl: float | None, what: str = "ttl") -> None:
    if ttl is not None and not ttl > 0:
        raise ValueError(f"{what} must be > 0 or None, got {ttl!r}")


class TTLCache(Generic[K, V]):
    def __init__(self, maxsize: int, default_ttl: float | None = None, *,
                 clock: Callable[[], float] = time.monotonic,
                 on_evict: Callable[[K, V, str], None] | None = None) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize!r}")
        _check_ttl(default_ttl, "default_ttl")
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._clock = clock
        self._on_evict = on_evict
        self._lock = threading.Lock()
        self._data: OrderedDict[K, _Entry] = OrderedDict()  # LRU first, MRU last
        self._inflight: dict[K, _Flight] = {}
        self._hits = self._misses = self._evictions = self._expirations = 0

    # ---- internals (call with the lock held) ---------------------------------------------------
    @staticmethod
    def _expired(entry: _Entry, now: float) -> bool:
        return entry.expires_at is not None and now >= entry.expires_at

    def _resolve_ttl(self, ttl: Any) -> float | None:
        if ttl is _DEFAULT:
            return self._default_ttl
        _check_ttl(ttl)
        return ttl

    def _live(self, key: K, now: float, events: list) -> _Entry | None:
        """Return the live entry for key, removing it as an expiration if it is expired."""
        entry = self._data.get(key)
        if entry is None:
            return None
        if self._expired(entry, now):
            del self._data[key]
            self._expirations += 1
            events.append((key, entry.value, "expired"))
            return None
        return entry

    def _store(self, key: K, value: V, ttl: float | None, now: float, events: list) -> None:
        old = self._data.pop(key, None)
        if old is not None:
            if self._expired(old, now):
                self._expirations += 1
                events.append((key, old.value, "expired"))
            else:
                events.append((key, old.value, "replaced"))
        elif len(self._data) >= self._maxsize:
            lru_key, lru = self._data.popitem(last=False)
            if self._expired(lru, now):
                self._expirations += 1
                events.append((lru_key, lru.value, "expired"))
            else:
                self._evictions += 1
                events.append((lru_key, lru.value, "capacity"))
        self._data[key] = _Entry(value, None if ttl is None else now + ttl)

    def _fire(self, events: list) -> None:
        cb = self._on_evict
        if cb is not None:
            for key, value, reason in events:
                cb(key, value, reason)

    # ---- public API ----------------------------------------------------------------------------
    def get(self, key: K, default: Any = None) -> Any:
        events: list = []
        with self._lock:
            entry = self._live(key, self._clock(), events)
            if entry is not None:
                self._hits += 1
                result = entry.value
            else:
                self._misses += 1
                result = default
        self._fire(events)
        return result

    def set(self, key: K, value: V, ttl: Any = _DEFAULT) -> None:
        ttl = self._resolve_ttl(ttl)
        events: list = []
        with self._lock:
            self._store(key, value, ttl, self._clock(), events)
        self._fire(events)

    def delete(self, key: K) -> bool:
        events: list = []
        with self._lock:
            entry = self._live(key, self._clock(), events)
            if entry is not None:
                del self._data[key]
                events.append((key, entry.value, "deleted"))
        self._fire(events)
        return entry is not None

    def pop(self, key: K, default: Any = _MISSING) -> Any:
        events: list = []
        with self._lock:
            entry = self._live(key, self._clock(), events)
            if entry is not None:
                del self._data[key]
        self._fire(events)
        if entry is not None:
            return entry.value
        if default is _MISSING:
            raise KeyError(key)
        return default

    def get_or_compute(self, key: K, factory: Callable[[], V], ttl: Any = _DEFAULT) -> V:
        ttl = self._resolve_ttl(ttl)
        counted = False
        while True:
            events: list = []
            with self._lock:
                entry = self._live(key, self._clock(), events)
                if entry is not None:
                    if not counted:
                        self._hits += 1
                    self._data.move_to_end(key)
                    return entry.value  # no events possible on this path
                if not counted:
                    self._misses += 1
                    counted = True
                flight = self._inflight.get(key)
                leader = flight is None
                if leader:
                    flight = self._inflight[key] = _Flight()
            self._fire(events)

            if not leader:
                flight.done.wait()
                if flight.ok:
                    return flight.value
                continue  # loader failed: retry from the top

            try:
                value = factory()
            except BaseException:
                with self._lock:
                    del self._inflight[key]
                flight.done.set()
                raise
            events = []
            with self._lock:
                self._store(key, value, ttl, self._clock(), events)
                del self._inflight[key]
                flight.value = value
                flight.ok = True
            flight.done.set()
            self._fire(events)
            return value

    def purge(self) -> int:
        events: list = []
        with self._lock:
            now = self._clock()
            dead = [k for k, e in self._data.items() if self._expired(e, now)]
            for k in dead:
                events.append((k, self._data.pop(k).value, "expired"))
            self._expirations += len(dead)
        self._fire(events)
        return len(dead)

    def clear(self) -> None:
        events: list = []
        with self._lock:
            now = self._clock()
            for k, e in self._data.items():
                if self._expired(e, now):
                    self._expirations += 1
                    events.append((k, e.value, "expired"))
                else:
                    events.append((k, e.value, "deleted"))
            self._data.clear()
        self._fire(events)

    def items(self) -> list[tuple[K, V]]:
        with self._lock:
            now = self._clock()
            return [(k, e.value) for k, e in self._data.items() if not self._expired(e, now)]

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(self._hits, self._misses, self._evictions, self._expirations)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            entry = self._data.get(key)  # type: ignore[arg-type]
            return entry is not None and not self._expired(entry, self._clock())

    def __len__(self) -> int:
        with self._lock:
            now = self._clock()
            return sum(1 for e in self._data.values() if not self._expired(e, now))
