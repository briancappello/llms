# Ticket: dependency-aware parallel task runner for the build/asset pipeline

## Context

Our asset pipeline is a few thousand small steps (fetch, transcode, hash, bundle, upload) with dependencies
between them. Today a shell script runs them one by one in a hand-maintained order, which is slow and breaks
whenever someone adds a step. We want a tiny in-process "build system": register tasks with their
dependencies, then run them in parallel on a thread pool, each task starting as soon as everything it needs
is done, with clear reporting of what succeeded, what failed and what was skipped because something upstream
failed.

Please implement it as a standalone module `solution` (Python 3.12, standard library only).

## Public interface

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

class GraphError(Exception): ...

class MissingDependencyError(GraphError):
    task: str         # the task that declared the dependency
    dependency: str   # the dependency name that is not a registered task

class CycleError(GraphError):
    cycle: list[str]  # see "Validation" below

@dataclass
class RunResult:
    results: dict[str, Any]              # task name -> return value, for tasks that succeeded
    errors: dict[str, BaseException]     # task name -> the exception it raised, for tasks that failed
    skipped: set[str]                    # tasks that were never started
    order: list[str]                     # tasks that ran (succeeded or failed), in completion order

class TaskGraph:
    def __init__(self) -> None: ...
    def add_task(self, name: str, fn: Callable[..., Any], deps: Iterable[str] = ()) -> None: ...
    def topological_order(self) -> list[str]: ...
    def run(self, max_workers: int = 4, *, fail_fast: bool = False) -> RunResult: ...
```

`MissingDependencyError(task, dependency)` and `CycleError(cycle)` take those values as constructor
arguments and expose them as attributes of the same names; the message text is up to you.

## Building the graph: `add_task`

- `name` must be a `str` and `fn` must be callable, otherwise `TypeError`.
- `deps` is an iterable of task names (`str`). Passing a single `str` as `deps` is a `TypeError` (it is
  almost always a bug: `deps="fetch"` instead of `deps=["fetch"]`).
- Registering a name that is already registered -> `ValueError`. Listing the same dependency twice in one
  `deps` -> `ValueError`. When `add_task` raises, the graph is left unchanged.
- Dependencies may refer to tasks that are registered later (forward references are fine); they are only
  checked when the graph is validated. A task may even list itself (that is a cycle, reported at validation).
- Tasks are remembered in **insertion order** (the order of successful `add_task` calls); this order is used
  for all tie-breaking below.

## Validation (done by `topological_order()` and at the start of `run()`)

1. **Missing dependencies first**: scanning tasks in insertion order and each task's deps in the order they
   were given, the first dependency that is not a registered task raises
   `MissingDependencyError(task, dependency)`.
2. **Then cycles**: if the graph has a cycle, raise `CycleError(cycle)` where `cycle` is a list of task names
   `[n0, n1, ..., nk]` with `k >= 1`, `n0 == nk`, `n0 ... n(k-1)` pairwise distinct, and for every `i`,
   `n(i+1)` is one of the declared deps of `n(i)` (i.e. the list follows dependency edges and closes the
   loop). A self-dependency is reported as `[a, a]`. If there are several cycles, any one may be reported.

Validation must not run any task. It must work on graphs of tens of thousands of tasks and on dependency
chains tens of thousands long (no recursion-depth problems) in roughly linear time.

## `topological_order()`

Validates, then returns all task names in an order where every task appears after all its dependencies.
The order is deterministic: at every step, among the tasks whose dependencies have all been emitted already,
emit the one that was **added earliest**. (E.g. tasks added as `c`, `a(deps=[c])`, `b` produce
`["c", "a", "b"]`.)

## `run(max_workers=4, *, fail_fast=False)`

- `max_workers < 1` -> `ValueError`. Then the graph is validated (errors above propagate, nothing runs).
- Each task's `fn` is called on a worker thread as `fn(**{dep: result_of_dep for dep in deps})`: one keyword
  argument per declared dependency, the keyword being the dependency's name (names need not be valid Python
  identifiers; they are passed through `**`), the value being exactly the object that dependency returned.
  A task without dependencies is called as `fn()`.
- A task becomes **ready** when all of its dependencies have *succeeded*. A ready task is started as soon as
  fewer than `max_workers` tasks are running; there are no level-by-level barriers (a slow task only delays
  its own dependents). At most `max_workers` task functions ever execute at the same time.
- When several tasks are ready and slots are free, they are started in insertion order (earliest-added
  first). Consequently, with `max_workers=1` and no failures, tasks run, and `order` lists them, exactly in
  `topological_order()`.
- A task **fails** if `fn` raises an `Exception`; the exception object is stored in `errors[name]`. Every task
  that depends, directly or transitively, on a failed task is never started and is reported in `skipped`.
  With `fail_fast=False`, all tasks not depending on a failure still run.
- With `fail_fast=True`: once the first failure has been recorded, no further task is started (in
  particular, the worker slot freed by the failing task is not reused). Tasks already running are allowed to
  finish and their outcomes are recorded normally in `results` / `errors` / `order`. Every task that was not
  started is in `skipped`.
- `order` lists every task that ran (success or failure) in the order their completions were recorded by the
  scheduler. Skipped tasks are not in `order`.
- Invariant: `results`, `errors` and `skipped` are disjoint and together contain every task exactly once;
  `set(order) == results.keys() | errors.keys()`.
- `run()` blocks until no task is running and returns the `RunResult`. It does not raise because a task
  failed. Exceptions not derived from `Exception` (e.g. `KeyboardInterrupt`) are out of scope.
- `run()` may be called again (it re-runs every task from scratch), and tasks may be added between runs. The
  graph must not be modified while `run()` is executing; a single `TaskGraph` is driven from one thread.

## Performance

Scheduling overhead must be roughly linear in the number of tasks plus dependency edges: a 5,000-task graph
of trivial functions should run in well under a few seconds, and `topological_order()` on 20,000+ tasks
should be fast. Do not rescan all tasks after every completion.
