# Ticket: in-process LRU cache with per-entry TTL for the pricing service hot path

## Context

The pricing service resolves product/price lookups against a slow upstream on every request. We want an
in-process cache in front of it: bounded in size (LRU eviction), with a time-to-live per entry, safe to share
between the request-handler threads, and with "single-flight" loading so that a burst of requests for the same
cold key triggers exactly one upstream call instead of one per request. Some cached values hold resources
(pooled connections, file handles), so the owner needs an eviction callback to release them.

Please implement it as a standalone module `solution` (Python 3.12, standard library only).

## Public interface

```python
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterator, TypeVar
import time

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int     # removals because the cache was full ("capacity" only)
    expirations: int   # removals of expired entries ("expired" only)

class TTLCache(Generic[K, V]):
    def __init__(self, maxsize: int, default_ttl: float | None = None, *,
                 clock: Callable[[], float] = time.monotonic,
                 on_evict: Callable[[K, V, str], None] | None = None) -> None: ...

    def get(self, key: K, default=None): ...
    def set(self, key: K, value: V, ttl: float | None = <DEFAULT>) -> None: ...
    def delete(self, key: K) -> bool: ...
    def pop(self, key: K, default=<MISSING>): ...
    def get_or_compute(self, key: K, factory: Callable[[], V], ttl: float | None = <DEFAULT>) -> V: ...
    def purge(self) -> int: ...
    def clear(self) -> None: ...
    def items(self) -> list[tuple[K, V]]: ...
    def stats(self) -> CacheStats: ...
    def __contains__(self, key: object) -> bool: ...
    def __len__(self) -> int: ...
```

`<DEFAULT>` and `<MISSING>` are private sentinels of your choosing. For `ttl`, *omitting* the argument means
"use `default_ttl`", while passing `ttl=None` explicitly means "this entry never expires".

## Terminology

- The cache **stores** entries. Each stored entry has a key, a value and an expiry time `expires_at`
  (or no expiry).
- When an entry is stored at clock time `now` with an effective ttl `t`, `expires_at = now + t`. An entry
  with a ttl of `None` never expires.
- An entry is **expired** at time `now` iff `now >= expires_at` (so an entry stored at 100.0 with ttl 5 is
  still live at 104.999 and is expired at exactly 105.0). An entry that is not expired is **live**.
- Stored entries are kept in a **recency order** from least recently used (LRU) to most recently used (MRU).
  Expired entries that have not been removed yet are still stored and still occupy their place in that
  order (and still count towards `maxsize`).
- The clock is `clock()` (seconds, float, monotonic). Every operation that needs "now" reads it under the
  cache's lock at the moment it does its work.

## Validation

- `maxsize < 1` -> `ValueError`.
- `default_ttl` that is not `None` and `<= 0` -> `ValueError`.
- Any explicitly passed `ttl` (in `set` / `get_or_compute`) that is not `None` and `<= 0` -> `ValueError`,
  raised before the cache is touched (no stats change, no callbacks, factory not called).

## Lazy expiry (exact rules)

Expired entries are removed **only lazily**, exactly in the situations listed below, and never at any other
time (no background thread, no opportunistic sweeping). Every such removal increments `expirations` by one
and calls `on_evict(key, value, "expired")`.

1. An operation that looks up a specific key (`get`, `set`, `delete`, `pop`, `get_or_compute`) and finds that
   key stored but expired removes it first, then proceeds exactly as if the key were absent.
2. When a *new* key is inserted (by `set` or by `get_or_compute` storing a computed value) and the number of
   stored entries already equals `maxsize`, exactly one entry is removed first: the LRU stored entry. If that
   entry is expired, the removal is an expiration (reason `"expired"`); otherwise it is a capacity eviction:
   `evictions += 1` and `on_evict(key, value, "capacity")`. (Only that single LRU entry is examined; other
   expired entries elsewhere in the order stay where they are.)
3. `purge()` removes every expired stored entry (in LRU -> MRU order) and returns how many it removed.
4. `clear()` removes every stored entry in LRU -> MRU order; expired ones count as expirations with reason
   `"expired"`, live ones are reported with reason `"deleted"`.

## Operation semantics

- `get(key, default=None)`: if a live entry exists: `hits += 1`, the entry becomes MRU, return its value.
  Otherwise (absent, or expired and therefore removed per rule 1): `misses += 1`, return `default`.
- `set(key, value, ttl=<DEFAULT>)`: store `value` under `key` with the effective ttl, as the MRU entry, with a
  fresh `expires_at`. If a live entry for `key` already exists it is replaced and `on_evict(key, old_value,
  "replaced")` is called (no stat changes). If the key is new, rule 2 applies before inserting. `set` never
  changes `hits`/`misses`.
- `delete(key) -> bool`: if a live entry exists, remove it, call `on_evict(key, value, "deleted")` and return
  `True`; otherwise return `False` (an expired entry is removed per rule 1). No hit/miss accounting.
- `pop(key, default=<MISSING>)`: if a live entry exists, remove it and return its value. **No** `on_evict`
  call for this removal (ownership of the value passes to the caller). Otherwise return `default`, or raise
  `KeyError(key)` if no default was given (an expired entry is removed per rule 1). No hit/miss accounting.
- `key in cache`, `len(cache)`, `cache.items()` and `cache.stats()` are **pure queries**: they never remove
  entries, never change recency, never change stats and never call `on_evict`.
  - `key in cache` is `True` iff a live entry for `key` exists.
  - `len(cache)` is the number of live entries.
  - `items()` returns a new list of `(key, value)` pairs for the live entries, LRU first, MRU last.
  - `stats()` returns a `CacheStats` snapshot.
- `get_or_compute(key, factory, ttl=<DEFAULT>)`:
  - If a live entry exists: `hits += 1`, it becomes MRU, return its value (factory not called).
  - Otherwise `misses += 1` (an expired entry is removed per rule 1) and the value is loaded with
    **single-flight** semantics: if no other thread is currently loading this key, this call becomes the
    *loader*: it calls `factory()` **without holding the cache lock**, then stores the result exactly like
    `set(key, result, ttl)` would (including rule 2 / `"replaced"`), with `expires_at` computed from the clock
    reading taken *after* `factory()` returned, and returns the result.
  - If another thread is already loading the same key, this call waits for that load to finish. If the load
    succeeded, it returns the loader's result (the very same object) without calling its own factory and
    without consulting the cache again. If the loader's factory raised, the exception is propagated **only**
    to the loader; nothing is stored; each waiting call then retries from the beginning (re-checks the cache;
    one of them becomes the new loader and calls *its own* factory; the rest wait for it, and so on).
  - Each `get_or_compute` call records exactly one hit or one miss in total, decided by its first check;
    waiting and retrying never add further hits or misses.
  - Loads of different keys must proceed concurrently (a slow factory for key A must not block operations
    on other keys, nor a concurrent load of key B). The factory may itself use the cache (e.g. call `get` or
    `set` on other keys) without deadlocking.
- `purge() -> int` and `clear()`: see rules 3 and 4. `clear()` does not affect in-flight loads.

## Eviction callback

`on_evict(key, value, reason)` with `reason` in `{"expired", "capacity", "replaced", "deleted"}` is called
exactly for the removals described above, once per removed entry, in the order the removals happened, in the
thread that performed the operation, **after** the operation's state change is complete and **without
holding the cache lock** (the callback may call back into the cache). If a callback raises, the exception
propagates to the caller of that operation (the cache state is already consistent); any remaining callbacks
for that operation are skipped.

## Thread-safety and complexity

- All public methods may be called concurrently from many threads; every operation is atomic with respect to
  the others (except that `factory()` and `on_evict` run outside the lock as described).
- `get`, `set`, `delete`, `pop`, `__contains__` and the cache part of `get_or_compute` must be O(1) average
  time, independent of `maxsize` and the number of stored entries (this sits on a hot path with caches of
  100k+ entries). `purge`, `clear`, `items` and `__len__` may be O(n).
