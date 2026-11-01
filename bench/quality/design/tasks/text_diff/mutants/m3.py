"""Minimal line-based diff, patch application and unified-diff rendering.

The diff is Myers' O((N+M)·D) algorithm in its linear-space "middle snake" form, run iteratively (explicit work
stack, no recursion). Before running it we

* map every distinct line to a small int so comparisons are cheap,
* strip the common prefix and suffix, and
* discard lines that occur in only one of the two inputs: they can never be part of a common subsequence, so
  removing them keeps the result minimal while making unrelated inputs trivial.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = ["Op", "PatchError", "apply", "diff", "unified"]

EQUAL, DELETE, INSERT = "equal", "delete", "insert"
_TAGS = (EQUAL, DELETE, INSERT)


class PatchError(ValueError):
    """Raised by apply() when an edit script does not match its input."""


@dataclass(frozen=True)
class Op:
    tag: str
    a_start: int
    a_end: int
    b_start: int
    b_end: int
    lines: tuple[str, ...]


# --------------------------------------------------------------------------- core: matched index pairs
def _middle_split(a: list[int], b: list[int], a0: int, a1: int, b0: int, b1: int) -> tuple[int, int]:
    """Return a point (x, y) (absolute indices) on a minimal edit path from (a0, b0) to (a1, b1).

    Precondition: both ranges are non-empty and their first and last elements differ, which guarantees the
    point is strictly between the corners so callers always make progress.
    """
    n = a1 - a0
    m = b1 - b0
    max_d = (n + m + 1) // 2
    offset = max_d + 2
    size = 2 * max_d + 5
    # vf[offset + k]: furthest x of a forward d-path on diagonal k = x - y (-1: no in-grid path).
    # vb[offset + k]: same for the backward search, in coordinates measured from the end (x' = n - x).
    vf = [-1] * size
    vb = [-1] * size
    vf[offset + 1] = 0  # virtual start: a "down" move from diagonal 1 lands on (0, 0)
    vb[offset + 1] = 0
    delta = n - m
    odd = delta & 1
    for d in range(max_d + 1):
        # Forward pass.
        for k in range(-d, d + 1, 2):
            i = offset + k
            x = vf[i + 1]  # down move from diagonal k + 1
            if x >= 0 and x - k > m:
                x = -1
            right = vf[i - 1]
            if right >= 0 and right + 1 <= n and right + 1 > x:
                x = right + 1
            if x < 0:
                vf[i] = -1
                continue
            y = x - k
            while x < n and y < m and a[a0 + x] == b[b0 + y]:
                x += 1
                y += 1
            vf[i] = x
            if odd:
                kb = delta - k
                if -(d - 1) <= kb <= d - 1:
                    xb = vb[offset + kb]
                    if xb >= 0 and x + xb >= n:
                        return a0 + x, b0 + y
        # Backward pass (on the reversed sequences).
        for kb in range(-d, d + 1, 2):
            i = offset + kb
            x = vb[i + 1]
            if x >= 0 and x - kb > m:
                x = -1
            right = vb[i - 1]
            if right >= 0 and right + 1 <= n and right + 1 > x:
                x = right + 1
            if x < 0:
                vb[i] = -1
                continue
            y = x - kb
            while x < n and y < m and a[a1 - 1 - x] == b[b1 - 1 - y]:
                x += 1
                y += 1
            vb[i] = x
            if not odd:
                k = delta - kb
                if -d <= k <= d:
                    xf = vf[offset + k]
                    if xf >= 0 and xf + x >= n:
                        return a0 + n - x, b0 + m - y
    raise AssertionError("unreachable: Myers search must meet")  # pragma: no cover


def _matches(a: list[int], b: list[int]) -> list[tuple[int, int]]:
    """Index pairs (i, j) with a[i] == b[j] forming a longest common subsequence, in increasing order."""
    pairs: list[tuple[int, int]] = []
    stack = [(0, len(a), 0, len(b))]
    while stack:
        a0, a1, b0, b1 = stack.pop()
        while a0 < a1 and b0 < b1 and a[a0] == b[b0]:
            pairs.append((a0, b0))
            a0 += 1
            b0 += 1
        while a0 < a1 and b0 < b1 and a[a1 - 1] == b[b1 - 1]:
            a1 -= 1
            b1 -= 1
            pairs.append((a1, b1))
        if a0 == a1 or b0 == b1:
            continue
        x, y = _middle_split(a, b, a0, a1, b0, b1)
        stack.append((a0, x, b0, y))
        stack.append((x, a1, y, b1))
    pairs.sort()
    return pairs


def _lcs_pairs(a: Sequence[str], b: Sequence[str]) -> list[tuple[int, int]]:
    ids: dict[str, int] = {}
    ia = [ids.setdefault(line, len(ids)) for line in a]
    ib = [ids.setdefault(line, len(ids)) for line in b]
    n, m = len(ia), len(ib)

    pre = 0
    while pre < n and pre < m and ia[pre] == ib[pre]:
        pre += 1
    suf = 0
    while suf < n - pre and suf < m - pre and ia[n - 1 - suf] == ib[m - 1 - suf]:
        suf += 1

    # Keep only middle lines that also occur in the other side's middle.
    in_a = set(ia[pre:n - suf])
    in_b = set(ib[pre:m - suf])
    map_a = [i for i in range(pre, n - suf) if ia[i] in in_b]
    map_b = [j for j in range(pre, m - suf) if ib[j] in in_a]
    sub = _matches([ia[i] for i in map_a], [ib[j] for j in map_b])

    pairs = [(i, i) for i in range(pre)]
    pairs.extend((map_a[i], map_b[j]) for i, j in sub)
    pairs.extend((n - suf + t, m - suf + t) for t in range(suf))
    return pairs


# --------------------------------------------------------------------------- public API
def diff(a: Sequence[str], b: Sequence[str]) -> list[Op]:
    pairs = _lcs_pairs(a, b)
    ops: list[Op] = []
    i = j = 0
    p = 0
    n, m = len(a), len(b)
    while i < n or j < m:
        if p < len(pairs) and pairs[p] == (i, j):
            start_i, start_j = i, j
            while p < len(pairs) and pairs[p] == (i, j):
                i += 1
                j += 1
                p += 1
            ops.append(Op(EQUAL, start_i, i, start_j, j, tuple(a[start_i:i])))
            continue
        ni, nj = pairs[p] if p < len(pairs) else (n, m)
        if ni > i:
            ops.append(Op(DELETE, i, ni, j, j, tuple(a[i:ni])))
        if nj > j:
            ops.append(Op(INSERT, ni, ni, j, nj, tuple(b[j:nj])))
        i, j = ni, nj
    return ops


def apply(a: Sequence[str], ops: Iterable[Op]) -> list[str]:
    out: list[str] = []
    ai = bi = 0
    n = len(a)
    for op in ops:
        tag = op.tag
        if tag not in _TAGS:
            raise PatchError(f"unknown op tag {tag!r}")
        if op.a_start != ai or op.b_start != bi:
            raise PatchError(f"op {op!r} does not start at a={ai}, b={bi}")
        la = op.a_end - op.a_start
        lb = op.b_end - op.b_start
        lines = tuple(op.lines)
        if la < 0 or lb < 0:
            raise PatchError(f"op {op!r} has a negative range")
        if tag == EQUAL and not (la == lb == len(lines)):
            raise PatchError(f"equal op {op!r} has inconsistent lengths")
        if tag == DELETE and not (lb == 0 and la == len(lines)):
            raise PatchError(f"delete op {op!r} has inconsistent lengths")
        if tag == INSERT and not (la == 0 and lb == len(lines)):
            raise PatchError(f"insert op {op!r} has inconsistent lengths")
        if op.a_end > n:
            raise PatchError(f"op {op!r} extends beyond the end of the input ({n} lines)")
        if tag != INSERT and tuple(a[op.a_start:op.a_end]) != lines:
            raise PatchError(f"op {op!r} does not match the input at lines {op.a_start}..{op.a_end}")
        if tag != DELETE:
            out.extend(lines)
        ai, bi = op.a_end, op.b_end
    return out


def _format_range(start: int, length: int) -> str:
    if length == 1:
        return str(start + 1)
    if length == 0:
        return f"{start},0"
    return f"{start + 1},{length}"


def unified(a: Sequence[str], b: Sequence[str], *, fromfile: str = "a", tofile: str = "b",
            context: int = 3) -> str:
    if not isinstance(context, int) or isinstance(context, bool):
        raise TypeError("context must be an int")
    if context < 0:
        raise ValueError("context must be >= 0")
    ops = diff(a, b)
    change_idx = [k for k, op in enumerate(ops) if op.tag != EQUAL]
    if not change_idx:
        return ""

    # Group op indices into hunks: [first_change, last_change] ranges.
    hunks: list[tuple[int, int]] = []
    first = last = change_idx[0]
    for k in change_idx[1:]:
        gap = ops[k - 1] if k - 1 > last else None  # the equal op between last and k, if any
        if gap is not None and gap.a_end - gap.a_start > 2 * context:
            hunks.append((first, last))
            first = k
        last = k
    hunks.append((first, last))

    out = [f"--- {fromfile}", f"+++ {tofile}"]
    for first, last in hunks:
        body: list[str] = []
        a_lo, b_lo = ops[first].a_start, ops[first].b_start
        a_hi, b_hi = ops[last].a_end, ops[last].b_end
        if first > 0:
            lead = min(context, ops[first - 1].a_end - ops[first - 1].a_start)
            a_lo -= lead
            b_lo -= lead
            body.extend(" " + line for line in a[a_lo:ops[first].a_start])
        for op in ops[first:last + 1]:
            prefix = " " if op.tag == EQUAL else "-" if op.tag == DELETE else "+"
            body.extend(prefix + line for line in op.lines)
        if last + 1 < len(ops):
            trail = min(context, ops[last + 1].a_end - ops[last + 1].a_start)
            body.extend(" " + line for line in a[a_hi:a_hi + trail])
            a_hi += trail
            b_hi += trail
        out.append(f"@@ -{_format_range(a_lo, a_hi - a_lo)} +{_format_range(b_lo, b_hi - b_lo)} @@")
        out.extend(body)
    return "\n".join(out) + "\n"
