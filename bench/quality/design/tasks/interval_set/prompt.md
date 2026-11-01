# Interval set for booking / byte-range tracking

We keep re-implementing "which ranges are covered" logic in several services: the room-booking service tracks
booked minutes, the download manager tracks which byte ranges of a file have arrived, and the replication worker
tracks which log offsets have been applied. Each copy has had its own boundary bugs (adjacent ranges not
merged, off-by-one at range ends, quadratic behaviour once a file has tens of thousands of fragments). We want a
single, well-tested, fast building block.

Please write a module `solution` (one file, `solution.py`, Python 3.12, standard library only) that provides a
mutable set of **half-open integer intervals** `[start, end)`.

## Model

An `IntervalSet` represents a set of integers: integer `x` is *covered* iff some stored interval has
`start <= x < end`. Internally it is always kept **normalized**: a sorted list of disjoint, non-empty intervals
where no two intervals overlap **or touch**. Adding `[1, 3)` and `[3, 5)` must therefore result in the single
interval `[1, 5)`. Two sets that cover the same integers are equal and iterate identically, regardless of the
order of operations that built them.

Coordinates are arbitrary Python `int`s (negative values and very large values are allowed).

## Interface

```python
class IntervalSet:
    def __init__(self, intervals=None) -> None: ...
    def add(self, start: int, end: int) -> None: ...
    def remove(self, start: int, end: int) -> None: ...
    def contains(self, x: int) -> bool: ...
    def overlaps(self, start: int, end: int) -> bool: ...
    def covers(self, start: int, end: int) -> bool: ...
    def gaps(self, start: int, end: int) -> list[tuple[int, int]]: ...
    def total_length(self) -> int: ...
    def copy(self) -> "IntervalSet": ...
    def union(self, other: "IntervalSet") -> "IntervalSet": ...
    def intersection(self, other: "IntervalSet") -> "IntervalSet": ...
    def difference(self, other: "IntervalSet") -> "IntervalSet": ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __eq__(self, other) -> bool: ...
    def __or__(self, other): ...    # same as union
    def __and__(self, other): ...   # same as intersection
    def __sub__(self, other): ...   # same as difference
```

### Semantics

- `IntervalSet(intervals=None)`: `intervals` is an optional iterable of `(start, end)` pairs; the result is the
  same as creating an empty set and calling `add(start, end)` for each pair in order (same validation, same
  exceptions).
- `add(start, end)`: cover every integer in `[start, end)`. Merges with every stored interval it overlaps **or
  touches** (shares an endpoint with). May merge many stored intervals into one.
- `remove(start, end)`: uncover every integer in `[start, end)`. May shrink intervals, delete them, or split one
  interval into two (e.g. removing `[3, 5)` from `[0, 10)` leaves `[0, 3)` and `[5, 10)`). Removing a range that is
  not covered is not an error.
- `contains(x)`: `True` iff `x` is covered. `contains(end)` of a stored interval is `False` (half-open).
- `overlaps(start, end)`: `True` iff at least one integer in `[start, end)` is covered. Touching is not
  overlapping: with `[0, 5)` stored, `overlaps(5, 8)` is `False`.
- `covers(start, end)`: `True` iff every integer in `[start, end)` is covered.
- `gaps(start, end)`: the maximal uncovered sub-ranges of the window `[start, end)`, clipped to the window, sorted,
  as a list of `(s, e)` tuples with `s < e`. For example, with `[2, 4)` and `[6, 8)` stored, `gaps(0, 10)` is
  `[(0, 2), (4, 6), (8, 10)]` and `gaps(3, 7)` is `[(4, 6)]`.
- `total_length()`: the number of covered integers (sum of `end - start` over the stored intervals).
- `__iter__`: yields the normalized intervals as `(start, end)` tuples of ints, in increasing order.
- `__len__`: the number of disjoint stored intervals (not the number of covered integers). An empty set is falsy.
- `__eq__`: two `IntervalSet`s are equal iff they cover exactly the same integers. Comparing with any other type
  returns `NotImplemented` (so `IntervalSet() == []` is `False`). The class is mutable and therefore must not be
  hashable.
- `copy()`: an independent copy; mutating either afterwards does not affect the other.
- `union` / `intersection` / `difference` (and `|`, `&`, `-`): return a **new** `IntervalSet`; neither operand is
  modified. `difference` is `self` minus `other`. The named methods raise `TypeError` if `other` is not an
  `IntervalSet`; the operators return `NotImplemented` for other types (so Python raises `TypeError`).
- `__repr__` is up to you.

### Empty ranges

A range with `start == end` is empty and valid:

- `add(s, s)` and `remove(s, s)` do nothing.
- `overlaps(s, s)` is `False`, `covers(s, s)` is `True` (vacuously), `gaps(s, s)` is `[]`.

### Validation

Every method that takes `start, end` (including the constructor) and `contains(x)` validates its arguments
before doing anything else, and leaves the set unchanged when it raises:

- Any coordinate that is not an `int` raises `TypeError`. `bool` is **not** accepted as an int here
  (`add(True, 5)` raises `TypeError`); floats, strings, `None` etc. are rejected too, even if integral.
- If both are ints and `start > end`, raise `ValueError`.

## Performance

These sets get big (hundreds of thousands of operations, tens of thousands of disjoint intervals), so:

- `contains`, `overlaps` and `covers` must be `O(log n)`, where `n` is the number of stored intervals.
- `add` and `remove` must be `O(log n + k)` where `k` is the number of stored intervals they touch, and
  `gaps(start, end)` `O(log n + k)` where `k` is the number of stored intervals inside the window. Keeping sorted
  Python lists and using `bisect` is fine, and the `O(n)` element shifting of `list.insert` / slice assignment is
  acceptable (it is a fast memmove); what is not acceptable is a Python-level linear scan over all intervals per
  operation.
- `len()` is `O(1)`; `union` / `intersection` / `difference` are linear in the total number of intervals
  (`O((n + m) log(n + m))` is also fine).
