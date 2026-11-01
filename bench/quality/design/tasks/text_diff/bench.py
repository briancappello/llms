"""Benchmark: diffing source-file revisions (many repeated lines, scattered edits), apply + unified."""
import random
import time

_COMMON = ["}", "", "    }", "        return None;", "{", "    break;", "    else {", "// ----", "#endif",
           "    i++;", "        continue;", "  default:"]


def _file(rng, n):
    return [rng.choice(_COMMON) if rng.random() < 0.55 else f"stmt_{rng.randrange(3000)};" for _ in range(n)]


def _edit(rng, a, n_edits):
    b = list(a)
    for _ in range(n_edits):
        pos = rng.randrange(len(b) + 1)
        r = rng.random()
        if r < 0.3 and pos < len(b):
            del b[pos:pos + rng.randint(1, 4)]
        elif r < 0.65:
            b[pos:pos] = [rng.choice(_COMMON) if rng.random() < 0.5 else f"new_{rng.randrange(10 ** 6)};"
                          for _ in range(rng.randint(1, 3))]
        elif pos < len(b):
            b[pos] = rng.choice(_COMMON)
    return b


def bench(mod) -> float:
    rng = random.Random(1234567)
    pairs = []
    for size, edits in ((20_000, 60), (8_000, 150), (5_000, 300), (3_000, 30), (12_000, 40), (2_000, 400)):
        a = _file(rng, size)
        pairs.append((a, _edit(rng, a, edits), edits))

    t0 = time.perf_counter()
    results = []
    for _ in range(2):
        for a, b, _ in pairs:
            ops = mod.diff(a, b)
            out = mod.apply(a, ops)
            text = mod.unified(a, b, fromfile="old.c", tofile="new.c")
            results.append((ops, out, text))
    elapsed = time.perf_counter() - t0

    for (ops, out, text), (a, b, edits) in zip(results, pairs + pairs):
        assert out == b, "apply(diff) did not reproduce b"
        n_edit = sum(len(o.lines) for o in ops if o.tag != "equal")
        # Each random edit changes at most 4 + 3 lines; a minimal script can never be longer than that.
        assert n_edit <= 8 * edits, "diff is far from minimal"
        ai = bi = 0
        for o in ops:
            assert (o.a_start, o.b_start) == (ai, bi)
            ai, bi = o.a_end, o.b_end
        assert (ai, bi) == (len(a), len(b))
        assert text.startswith("--- old.c\n+++ new.c\n@@ -")
        minus = sum(1 for line in text.splitlines() if line.startswith("-") and not line.startswith("--- "))
        plus = sum(1 for line in text.splitlines() if line.startswith("+") and not line.startswith("+++ "))
        assert minus + plus == n_edit
    # Exact minimality on a small fixed case.
    a = list("ABCABBA")
    b = list("CBABAC")
    assert sum(len(o.lines) for o in mod.diff(a, b) if o.tag != "equal") == 5
    return elapsed
