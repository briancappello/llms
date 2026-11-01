# Resilience utilities: circuit breaker + retry with backoff

## Context

Our order service calls several flaky downstream services (payments, inventory, a fraud scorer). Today every call
site has its own hand-rolled `for i in range(3): try/except: time.sleep(1)` loop. When payments has an outage we
hammer it with retries from every worker thread, which slows its recovery and ties up our threads. We want one
small, shared, well-tested module with:

1. a **circuit breaker** that stops calling a dependency after repeated failures, then probes it carefully, and
2. a **retry helper** with capped exponential backoff and jitter,

that compose cleanly. Time must be injectable: production passes nothing (real monotonic clock / real sleep); our
tests pass fakes.

## Deliverable

A single module `solution` (Python 3.12, standard library only) exporting `State`, `CircuitOpenError`,
`CircuitBreaker`, `full_jitter` and `retry`.

## `State` and `CircuitOpenError`

```python
class State(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitOpenError(Exception):
    def __init__(self, retry_after: float) -> None: ...
    retry_after: float   # attribute, see below
```

## `CircuitBreaker`

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0, *,
                 half_open_max_calls: int = 1,
                 expected_exceptions: tuple[type[BaseException], ...] = (Exception,),
                 clock: Callable[[], float] = time.monotonic) -> None: ...

    @property
    def state(self) -> State: ...
    @property
    def failure_count(self) -> int: ...
    @property
    def opened_count(self) -> int: ...

    def call(self, fn: Callable[..., T], /, *args, **kwargs) -> T: ...
    def reset(self) -> None: ...
```

Validation (raise `ValueError`): `failure_threshold` must be an `int >= 1`; `recovery_timeout` a finite number
`>= 0`; `half_open_max_calls` an `int >= 1`; `expected_exceptions` a non-empty tuple whose items are all exception
classes (subclasses of `BaseException`).

Terminology: a call's **outcome** is *success* (fn returned), *failure* (fn raised an exception that is an instance
of one of `expected_exceptions`, subclasses included), or *ignored* (fn raised anything else).

State machine:

- **CLOSED** (initial). `call` invokes `fn(*args, **kwargs)` and returns its result / re-raises its exception
  unchanged.
  - success: the consecutive-failure counter resets to 0.
  - failure: the counter increases by 1; when it reaches `failure_threshold` the breaker transitions to **OPEN**.
    The recovery timer starts at `clock()` read when that failure is recorded (i.e. after `fn` raised).
  - ignored: the exception propagates, nothing changes (the counter is neither incremented nor reset).
- **OPEN**. `call` does **not** invoke `fn`; it raises `CircuitOpenError(retry_after)` where `retry_after` is the
  number of seconds left until `recovery_timeout` has elapsed since the breaker opened (`> 0`). Once
  `clock() - opened_at >= recovery_timeout` the breaker is **HALF_OPEN** (the transition happens lazily, the next
  time `call` or the `state` property is used; `state` must never report OPEN when the timeout has elapsed).
- **HALF_OPEN**. Up to `half_open_max_calls` trial calls may be *in flight* at the same time; a `call` beyond that
  limit raises `CircuitOpenError(0.0)` without invoking `fn`. When a trial call completes:
  - success: transition to **CLOSED** (counter 0).
  - failure: transition back to **OPEN**; the recovery timer restarts at `clock()` read when this failure is
    recorded.
  - ignored: the exception propagates, the state stays HALF_OPEN and the trial slot is freed (also for
    `BaseException`s such as `KeyboardInterrupt`; slots must never leak).
- **Stale outcomes.** The outcome of a call only affects the breaker if the breaker has not changed state (and has
  not been `reset()`) since that call was admitted. Example: with `failure_threshold=3`, ten concurrent calls that
  all fail open the breaker exactly once — the three failures that cross the threshold open it, the seven failures
  that complete afterwards are ignored by the breaker (they still raise to their callers) and do not restart the
  recovery timer. Likewise, once one trial call has closed or re-opened the breaker, the outcomes of the other
  in-flight trial calls are ignored.

Observability:

- `state` — current `State` (applying the lazy OPEN -> HALF_OPEN transition).
- `failure_count` — current consecutive-failure counter. It is reset to 0 on every state transition and by
  `reset()` (so it is 0 while OPEN or HALF_OPEN).
- `opened_count` — total number of transitions into OPEN since construction (CLOSED -> OPEN and
  HALF_OPEN -> OPEN). Not affected by `reset()`.

`reset()` forces the breaker to CLOSED with counter 0, forgets any in-flight trial calls, and makes outcomes of
calls admitted before the reset stale (ignored).

Thread safety: all methods may be called concurrently from many threads; state and counters must stay consistent
(no lost updates, no double transitions). `fn` must be invoked **without holding any internal lock**: a slow call
must not block other callers, and `fn` itself may use the same breaker (e.g. call `breaker.call(...)`,
`breaker.state`, `breaker.reset()`) without deadlocking.

## `retry`

```python
def full_jitter(delay: float) -> float: ...   # uniformly random value in [0, delay]

def retry(fn: Callable[[], T], *,
          attempts: int,
          base_delay: float,
          max_delay: float,
          multiplier: float = 2.0,
          jitter: Callable[[float], float] = full_jitter,
          retry_on: tuple[type[BaseException], ...] = (Exception,),
          sleep: Callable[[float], None] = time.sleep,
          on_retry: Callable[[int, BaseException, float], None] | None = None) -> T: ...
```

- Calls `fn()` (no arguments) up to `attempts` times in total and returns the first successful result.
- Backoff: after attempt number `i` fails (0-based: `i = 0` is the first attempt) and another attempt will follow,
  compute `d_i = min(max_delay, base_delay * multiplier ** i)`, then `s = jitter(d_i)`, then call
  `on_retry(i + 1, exc, s)` if `on_retry` is given, then call `sleep(s)` exactly once (even when `s == 0`), then make
  the next attempt. `d_i` must be computed without `OverflowError` for very large `i` (e.g. `attempts=100_000`):
  once `base_delay * multiplier ** i` exceeds `max_delay` (or would overflow) the delay is simply `max_delay`.
  `base_delay == 0` gives `d_i == 0` for all `i`.
- No sleep after the final attempt: if the last attempt fails, re-raise its exception object as-is (not wrapped,
  original traceback) immediately.
- Only exceptions that are instances of a class in `retry_on` are retried. Any other exception propagates
  immediately from the attempt that raised it, without sleeping and without calling `on_retry`.
- Composition with the breaker: `CircuitOpenError` is retried **only if `retry_on` explicitly lists
  `CircuitOpenError`** (or a subclass of it that the raised error is an instance of). Broad classes such as
  `Exception` in `retry_on` do **not** make it retryable — retrying a call to an open circuit just burns the retry
  budget. So `retry(lambda: breaker.call(f), ...)` with the default `retry_on` fails fast on an open breaker,
  while `retry_on=(IOError, CircuitOpenError)` waits out the breaker via the backoff.
- If `on_retry`, `jitter` or `sleep` raise, that exception propagates and no further attempts are made.
- Validation, before `fn` is called (raise `ValueError`): `attempts` an `int >= 1`; `base_delay` and `max_delay`
  finite and `>= 0` (`max_delay < base_delay` is allowed and simply caps every delay); `multiplier` finite and
  `>= 1`; `retry_on` a non-empty tuple of exception classes.
- `retry` holds no shared state and is safe to call concurrently.
