# Token-bucket rate limiter for the API gateway

## Context

Our API gateway currently throttles clients with an ad-hoc "requests in the last second" counter that lets bursts of
2x the limit through at window boundaries and is not thread-safe (we have seen over-admission under load). We want to
replace it with a small, well-tested token-bucket library that the gateway's worker threads can share. It must be
usable for a single global limit (one bucket) and for per-client limits (one bucket per client id, bounded memory).

Time must be injectable: production passes nothing (real monotonic clock and real sleep), our tests pass a fake clock
and a fake sleep.

## Deliverable

A single module `solution` (Python 3.12, standard library only) exporting `TokenBucket` and `KeyedRateLimiter`.

## `TokenBucket`

```python
class TokenBucket:
    def __init__(self, rate: float, capacity: float, *,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None: ...

    @property
    def rate(self) -> float: ...        # tokens per second, as given
    @property
    def capacity(self) -> float: ...    # maximum number of stored tokens, as given

    def try_acquire(self, tokens: float = 1) -> bool: ...
    def acquire(self, tokens: float = 1, timeout: float | None = None) -> bool: ...
    def available(self) -> float: ...
```

Semantics:

- **Construction.** `rate` and `capacity` must be finite numbers `> 0`; otherwise raise `ValueError` (this includes
  `0`, negatives, `nan` and `inf`). The bucket starts **full** (`capacity` tokens).
- **Refill.** Tokens refill continuously at `rate` tokens per second of `clock()` time, capped at `capacity`. That is,
  the fill level at time `t` is `min(capacity, level_at_last_update + (t - t_last_update) * rate)`. Fractional
  tokens are allowed everywhere (the level is a float, `tokens` arguments may be floats).
- **Clock going backwards.** If `clock()` returns a value smaller than the latest time the bucket has already seen,
  treat the elapsed time as zero: no refill, no removal of tokens, and the bucket keeps its latest time (it must not
  "re-earn" the same interval later).
- **`tokens` argument.** For `try_acquire` and `acquire`, `tokens` must satisfy `0 < tokens <= capacity`; otherwise
  raise `ValueError` (also for `nan`). Validation happens before any state change.
- **`try_acquire(tokens)`** — non-blocking. If at least `tokens` tokens are available now, remove them and return
  `True`; otherwise return `False` and change nothing (a failed attempt consumes nothing). Never sleeps.
- **`acquire(tokens, timeout=None)`** — blocking. `timeout` is `None` (wait as long as needed) or a number `>= 0`
  seconds (negative or `nan` -> `ValueError`). The deadline is `clock()` read at the start of the call plus
  `timeout`.
  - If the tokens are available now, take them and return `True` without sleeping.
  - Otherwise compute the wait until enough tokens will have refilled: `(tokens - available) / rate` seconds.
    If that point in time is **after** the deadline, return `False` immediately without sleeping and without
    changing anything (fail fast: there is no point waiting for something that cannot happen in time). If it is at
    or before the deadline, call the injected `sleep(wait)` **once with that computed wait** — no polling loops, no
    small fixed sleep steps — and then try again.
  - `sleep` may return early (real sleeps can wake slightly early) and other threads may take the tokens in the
    meantime, so after waking re-check and, if still short, repeat the same procedure (recompute the wait, check it
    against the original deadline, sleep again or give up). A request that is granted always actually removes the
    tokens.
  - `sleep` must never be called with a value `<= 0`, and must never be called while holding an internal lock
    (other threads must be able to use the bucket while one thread waits).
  - `timeout=0` behaves like `try_acquire`.
- **`available()`** — returns the current fill level (after refilling up to `clock()`), without consuming anything.
- **Floating-point tolerance.** Use an absolute tolerance of `1e-9` tokens when deciding whether enough tokens are
  available (a request for `t` tokens is satisfiable if `level >= t - 1e-9`; on success the level becomes
  `max(0.0, level - t)`). This guarantees that sleeping exactly the computed wait is always enough, e.g. with
  `rate=3` a wait of `1/3` s must satisfy a 1-token deficit after a single sleep.
- **Thread safety.** All methods may be called concurrently from many threads. Each `try_acquire`/`acquire` grant
  must be atomic with respect to the refill/check/consume sequence: the total number of tokens handed out over any
  period can never exceed `capacity + rate * elapsed`. No fairness/FIFO ordering between waiting threads is required.

## `KeyedRateLimiter`

Per-key limits (for example one bucket per client id) with bounded memory.

```python
class KeyedRateLimiter:
    def __init__(self, rate: float, capacity: float, *,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 max_keys: int = 10_000) -> None: ...

    @property
    def rate(self) -> float: ...
    @property
    def capacity(self) -> float: ...
    @property
    def max_keys(self) -> int: ...

    def try_acquire(self, key: Hashable, tokens: float = 1) -> bool: ...
    def acquire(self, key: Hashable, tokens: float = 1, timeout: float | None = None) -> bool: ...
    def available(self, key: Hashable) -> float: ...
    def __len__(self) -> int: ...                 # number of keys currently tracked
    def __contains__(self, key: object) -> bool: ...  # is this key currently tracked
```

Semantics:

- `rate`/`capacity` are validated exactly like `TokenBucket`; `max_keys` must be an `int >= 1`, otherwise
  `ValueError`.
- Every key has its own independent bucket with the same `rate`/`capacity`, behaving exactly like a `TokenBucket`
  (same refill, tolerance, blocking and timeout semantics). Keys never affect each other's tokens.
- A key's bucket is created on its first `try_acquire`/`acquire` and starts full.
- **Recency / eviction.** Each `try_acquire` or `acquire` call for a key counts as a *use* of that key, whether or
  not it is granted. When a use of a new key would make the number of tracked keys exceed `max_keys`, the
  least-recently-used key is evicted (forgotten). An evicted key that is used again gets a fresh, full bucket.
- `available(key)` is a read-only query: for a tracked key it returns that bucket's current level; for an untracked
  key it returns `capacity`. It never creates a key and does **not** count as a use (does not change recency).
- Argument validation (`tokens`, `timeout`) happens before anything else: an invalid call raises `ValueError`
  without creating the key or changing its recency.
- A blocking `acquire` must not hold the limiter-wide lock while sleeping: other keys (and the same key) must stay
  usable. If the key is evicted while a thread is blocked on it, that thread may finish against the bucket it
  started with.
- Thread-safe, like `TokenBucket`.
- **Complexity.** `try_acquire`, `acquire` (excluding sleeping), `available`, `len` and `in` must be O(1)
  amortised, independent of the number of tracked keys (the gateway tracks tens of thousands of clients).
