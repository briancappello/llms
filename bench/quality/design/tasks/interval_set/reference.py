"""A mutable set of half-open integer intervals [start, end), kept normalized.

Representation: two parallel sorted lists ``_starts`` and ``_ends`` of disjoint, non-empty, non-touching
intervals, so ``_starts[i] < _ends[i] < _starts[i + 1]``. Both lists are therefore strictly increasing, which
lets every query use ``bisect``.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator

__all__ = ["IntervalSet"]


def _check_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")


def _check_range(start: object, end: object) -> None:
    _check_int(start, "start")
    _check_int(end, "end")
    if start > end:  # type: ignore[operator]
        raise ValueError(f"start ({start}) must be <= end ({end})")


class IntervalSet:
    __slots__ = ("_starts", "_ends", "_total")
    __hash__ = None  # type: ignore[assignment]  # mutable

    def __init__(self, intervals: Iterable[tuple[int, int]] | None = None) -> None:
        self._starts: list[int] = []
        self._ends: list[int] = []
        self._total = 0
        if intervals is not None:
            for start, end in intervals:
                self.add(start, end)

    @classmethod
    def _from_normalized(cls, starts: list[int], ends: list[int]) -> IntervalSet:
        new = cls.__new__(cls)
        new._starts = starts
        new._ends = ends
        new._total = sum(e - s for s, e in zip(starts, ends))
        return new

    # ------------------------------------------------------------------ mutation
    def add(self, start: int, end: int) -> None:
        _check_range(start, end)
        if start == end:
            return
        starts, ends = self._starts, self._ends
        # Intervals i..j-1 overlap or touch [start, end): end_i >= start and start_j <= end.
        i = bisect_left(ends, start)
        j = bisect_right(starts, end, lo=i)
        if i < j:
            start = min(start, starts[i])
            end = max(end, ends[j - 1])
            self._total -= sum(ends[t] - starts[t] for t in range(i, j))
        starts[i:j] = [start]
        ends[i:j] = [end]
        self._total += end - start

    def remove(self, start: int, end: int) -> None:
        _check_range(start, end)
        if start == end:
            return
        starts, ends = self._starts, self._ends
        # Intervals i..j-1 share at least one integer with [start, end).
        i = bisect_right(ends, start)
        j = bisect_left(starts, end, lo=i)
        if i >= j:
            return
        new_starts: list[int] = []
        new_ends: list[int] = []
        if starts[i] < start:
            new_starts.append(starts[i])
            new_ends.append(start)
        if ends[j - 1] > end:
            new_starts.append(end)
            new_ends.append(ends[j - 1])
        removed = sum(ends[t] - starts[t] for t in range(i, j))
        kept = sum(e - s for s, e in zip(new_starts, new_ends))
        starts[i:j] = new_starts
        ends[i:j] = new_ends
        self._total += kept - removed

    # ------------------------------------------------------------------ queries
    def contains(self, x: int) -> bool:
        _check_int(x, "x")
        i = bisect_right(self._starts, x) - 1
        return i >= 0 and x < self._ends[i]

    def overlaps(self, start: int, end: int) -> bool:
        _check_range(start, end)
        if start == end:
            return False
        i = bisect_right(self._ends, start)  # first interval ending after start
        return i < len(self._starts) and self._starts[i] < end

    def covers(self, start: int, end: int) -> bool:
        _check_range(start, end)
        if start == end:
            return True
        i = bisect_right(self._starts, start) - 1
        return i >= 0 and self._ends[i] >= end

    def gaps(self, start: int, end: int) -> list[tuple[int, int]]:
        _check_range(start, end)
        result: list[tuple[int, int]] = []
        if start == end:
            return result
        starts, ends = self._starts, self._ends
        cursor = start
        i = bisect_right(ends, start)
        n = len(starts)
        while i < n and starts[i] < end:
            if starts[i] > cursor:
                result.append((cursor, starts[i]))
            cursor = ends[i]
            i += 1
        if cursor < end:
            result.append((cursor, end))
        return result

    def total_length(self) -> int:
        return self._total

    # ------------------------------------------------------------------ protocol
    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(list(zip(self._starts, self._ends)))

    def __len__(self) -> int:
        return len(self._starts)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntervalSet):
            return NotImplemented
        return self._starts == other._starts and self._ends == other._ends

    def __repr__(self) -> str:
        return f"IntervalSet({list(zip(self._starts, self._ends))!r})"

    def copy(self) -> IntervalSet:
        return self._from_normalized(self._starts.copy(), self._ends.copy())

    # ------------------------------------------------------------------ algebra
    @staticmethod
    def _require(other: object) -> IntervalSet:
        if not isinstance(other, IntervalSet):
            raise TypeError(f"expected IntervalSet, got {type(other).__name__}")
        return other

    def union(self, other: IntervalSet) -> IntervalSet:
        other = self._require(other)
        a_s, a_e, b_s, b_e = self._starts, self._ends, other._starts, other._ends
        starts: list[int] = []
        ends: list[int] = []
        i = j = 0
        while i < len(a_s) or j < len(b_s):
            if j >= len(b_s) or (i < len(a_s) and a_s[i] <= b_s[j]):
                s, e = a_s[i], a_e[i]
                i += 1
            else:
                s, e = b_s[j], b_e[j]
                j += 1
            if ends and s <= ends[-1]:
                if e > ends[-1]:
                    ends[-1] = e
            else:
                starts.append(s)
                ends.append(e)
        return self._from_normalized(starts, ends)

    def intersection(self, other: IntervalSet) -> IntervalSet:
        other = self._require(other)
        a_s, a_e, b_s, b_e = self._starts, self._ends, other._starts, other._ends
        starts: list[int] = []
        ends: list[int] = []
        i = j = 0
        while i < len(a_s) and j < len(b_s):
            s = max(a_s[i], b_s[j])
            e = min(a_e[i], b_e[j])
            if s < e:
                starts.append(s)
                ends.append(e)
            if a_e[i] < b_e[j]:
                i += 1
            else:
                j += 1
        return self._from_normalized(starts, ends)

    def difference(self, other: IntervalSet) -> IntervalSet:
        other = self._require(other)
        a_s, a_e, b_s, b_e = self._starts, self._ends, other._starts, other._ends
        starts: list[int] = []
        ends: list[int] = []
        j = 0
        for s, e in zip(a_s, a_e):
            while j < len(b_s) and b_e[j] <= s:
                j += 1
            k = j
            cur = s
            while k < len(b_s) and b_s[k] < e:
                if b_s[k] > cur:
                    starts.append(cur)
                    ends.append(b_s[k])
                cur = max(cur, b_e[k])
                k += 1
            if cur < e:
                starts.append(cur)
                ends.append(e)
        return self._from_normalized(starts, ends)

    def __or__(self, other: object) -> IntervalSet:
        if not isinstance(other, IntervalSet):
            return NotImplemented
        return self.union(other)

    def __and__(self, other: object) -> IntervalSet:
        if not isinstance(other, IntervalSet):
            return NotImplemented
        return self.intersection(other)

    def __sub__(self, other: object) -> IntervalSet:
        if not isinstance(other, IntervalSet):
            return NotImplemented
        return self.difference(other)
