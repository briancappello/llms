"""Dependency-aware parallel task runner (a tiny in-process build system)."""
import heapq
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


class GraphError(Exception):
    """Base class for graph validation errors."""


class MissingDependencyError(GraphError):
    def __init__(self, task: str, dependency: str) -> None:
        super().__init__(f"task {task!r} depends on unknown task {dependency!r}")
        self.task = task
        self.dependency = dependency


class CycleError(GraphError):
    def __init__(self, cycle: list[str]) -> None:
        super().__init__("dependency cycle: " + " -> ".join(map(repr, cycle)))
        self.cycle = cycle


@dataclass
class RunResult:
    results: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, BaseException] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)
    order: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Task:
    index: int
    fn: Callable[..., Any]
    deps: tuple[str, ...]


class TaskGraph:
    def __init__(self) -> None:
        self._tasks: dict[str, _Task] = {}  # insertion ordered

    def add_task(self, name: str, fn: Callable[..., Any], deps: Iterable[str] = ()) -> None:
        if not isinstance(name, str):
            raise TypeError(f"task name must be str, got {type(name).__name__}")
        if not callable(fn):
            raise TypeError(f"fn for task {name!r} is not callable")
        if isinstance(deps, str):
            raise TypeError(f"deps of task {name!r} must be an iterable of names, not a single str")
        dep_list = tuple(deps)
        for d in dep_list:
            if not isinstance(d, str):
                raise TypeError(f"dependency names must be str, got {type(d).__name__}")
        if len(set(dep_list)) != len(dep_list):
            raise ValueError(f"task {name!r} lists a dependency more than once")
        if name in self._tasks:
            raise ValueError(f"duplicate task name {name!r}")
        self._tasks[name] = _Task(len(self._tasks), fn, dep_list)

    # ---- validation / ordering -----------------------------------------------------------------
    def _dependents(self) -> dict[str, list[str]]:
        dependents: dict[str, list[str]] = {n: [] for n in self._tasks}
        for name, task in self._tasks.items():
            for d in task.deps:
                if d not in dependents:
                    raise MissingDependencyError(name, d)
                dependents[d].append(name)
        return dependents

    def topological_order(self) -> list[str]:
        tasks = self._tasks
        dependents = self._dependents()
        pending = {n: len(t.deps) for n, t in tasks.items()}
        heap = [(t.index, n) for n, t in tasks.items() if not t.deps]
        heapq.heapify(heap)
        out: list[str] = []
        while heap:
            _, n = heapq.heappop(heap)
            out.append(n)
            for m in dependents[n]:
                pending[m] -= 1
                if pending[m] == 0:
                    heapq.heappush(heap, (tasks[m].index, m))
        if len(out) != len(tasks):
            raise CycleError(self._find_cycle(pending))
        return out

    def _find_cycle(self, pending: dict[str, int]) -> list[str]:
        # Every task left with pending > 0 has at least one dependency that is also left over, so
        # walking "some leftover dependency" from a leftover task must eventually revisit a task.
        left = {n for n, c in pending.items() if c > 0}
        node = next(n for n in self._tasks if n in left)
        pos: dict[str, int] = {}
        path: list[str] = []
        while node not in pos:
            pos[node] = len(path)
            path.append(node)
            node = next(d for d in self._tasks[node].deps if d in left)
        return path[pos[node]:] + [node]

    # ---- execution -----------------------------------------------------------------------------
    @staticmethod
    def _invoke(name: str, fn: Callable[..., Any], kwargs: dict[str, Any], done: queue.SimpleQueue) -> None:
        try:
            value = fn(**kwargs)
        except BaseException as exc:  # noqa: BLE001 - reported to the scheduler thread
            done.put((name, False, exc))
        else:
            done.put((name, True, value))

    def run(self, max_workers: int = 4, *, fail_fast: bool = False) -> RunResult:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers!r}")
        self.topological_order()  # validation only
        tasks = self._tasks
        dependents = self._dependents()
        pending = {n: len(t.deps) for n, t in tasks.items()}
        ready = [(t.index, n) for n, t in tasks.items() if not t.deps]
        heapq.heapify(ready)

        res = RunResult()
        done: queue.SimpleQueue = queue.SimpleQueue()
        running = 0
        stop = False
        fatal: BaseException | None = None
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="taskgraph") as pool:
            while True:
                while ready and running < max_workers:
                    _, name = heapq.heappop(ready)
                    task = tasks[name]
                    kwargs = {d: res.results[d] for d in task.deps}
                    pool.submit(self._invoke, name, task.fn, kwargs, done)
                    running += 1
                if running == 0:
                    break
                name, ok, payload = done.get()
                running -= 1
                res.order.append(name)
                if ok:
                    res.results[name] = payload
                    for m in dependents[name]:
                        pending[m] -= 1
                        if pending[m] == 0 and not stop:
                            heapq.heappush(ready, (tasks[m].index, m))
                else:
                    res.errors[name] = payload
                    if not isinstance(payload, Exception):
                        fatal = fatal or payload
                        stop = True
                    elif fail_fast:
                        stop = True
        if fatal is not None:
            raise fatal
        res.skipped = set(tasks) - res.results.keys() - res.errors.keys()
        return res
