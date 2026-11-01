import dataclasses
import difflib
import random

import pytest

import solution as S

Op = S.Op


# ---------------------------------------------------------------- helpers
def lcs_len(a, b):
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    for i in range(n - 1, -1, -1):
        cur = [0] * (m + 1)
        ai = a[i]
        for j in range(m - 1, -1, -1):
            cur[j] = prev[j + 1] + 1 if ai == b[j] else max(prev[j], cur[j + 1])
        prev = cur
    return prev[0]


def edit_count(ops):
    return sum(len(o.lines) for o in ops if o.tag != "equal")


def check_ops(a, b, ops):
    """Every structural guarantee of diff() from the spec."""
    a, b = list(a), list(b)
    assert isinstance(ops, list)
    if not a and not b:
        assert ops == []
        return
    ai = bi = 0
    prev = None
    for op in ops:
        assert isinstance(op, Op)
        assert op.tag in ("equal", "delete", "insert")
        assert (op.a_start, op.b_start) == (ai, bi), (ops, a, b)
        assert isinstance(op.lines, tuple)
        assert len(op.lines) > 0
        if op.tag == "equal":
            assert op.a_end - op.a_start == op.b_end - op.b_start == len(op.lines)
            assert tuple(a[op.a_start:op.a_end]) == op.lines == tuple(b[op.b_start:op.b_end])
        elif op.tag == "delete":
            assert op.b_start == op.b_end
            assert tuple(a[op.a_start:op.a_end]) == op.lines
        else:
            assert op.a_start == op.a_end
            assert tuple(b[op.b_start:op.b_end]) == op.lines
        if prev is not None:
            assert prev.tag != op.tag, ops
            assert not (prev.tag == "insert" and op.tag == "delete"), ops
        prev = op
        ai, bi = op.a_end, op.b_end
    assert (ai, bi) == (len(a), len(b))


def to_opcodes(ops):
    codes = []
    for op in ops:
        if op.tag == "insert" and codes and codes[-1][0] == "delete":
            _, i1, i2, j1, _ = codes[-1]
            codes[-1] = ("replace", i1, i2, j1, op.b_end)
        else:
            codes.append((op.tag, op.a_start, op.a_end, op.b_start, op.b_end))
    return codes


class _FixedMatcher(difflib.SequenceMatcher):
    def __init__(self, codes):
        super().__init__(None, [], [])
        self._codes = codes

    def get_opcodes(self):
        return list(self._codes)


def _rng_fmt(start, stop):
    beginning = start + 1
    length = stop - start
    if length == 1:
        return f"{beginning}"
    if not length:
        beginning -= 1
    return f"{beginning},{length}"


def render_expected(a, b, ops, fromfile="a", tofile="b", context=3):
    """difflib's own hunk grouping/formatting applied to the candidate's ops."""
    if not any(o.tag != "equal" for o in ops):
        return ""
    out = [f"--- {fromfile}", f"+++ {tofile}"]
    for group in _FixedMatcher(to_opcodes(ops)).get_grouped_opcodes(context):
        first, last = group[0], group[-1]
        out.append(f"@@ -{_rng_fmt(first[1], last[2])} +{_rng_fmt(first[3], last[4])} @@")
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                out.extend(" " + line for line in a[i1:i2])
                continue
            if tag in ("replace", "delete"):
                out.extend("-" + line for line in a[i1:i2])
            if tag in ("replace", "insert"):
                out.extend("+" + line for line in b[j1:j2])
    return "".join(line + "\n" for line in out)


def difflib_unified(a, b, fromfile="a", tofile="b", context=3):
    return "".join(line + "\n" for line in difflib.unified_diff(list(a), list(b), fromfile, tofile,
                                                               n=context, lineterm=""))


def unique_lcs_pair(rng, n_max=40):
    """a has unique lines; b keeps a subset of them in order and inserts fresh lines -> minimal diff is unique."""
    n = rng.randint(0, n_max)
    a = [f"line {i}" for i in range(n)]
    b = []
    fresh = 0
    for line in a:
        while rng.random() < 0.15:
            b.append(f"new {fresh}")
            fresh += 1
        if rng.random() < 0.75:
            b.append(line)
    while rng.random() < 0.2:
        b.append(f"new {fresh}")
        fresh += 1
    return a, b


def small_random_pair(rng):
    alpha = "abcdef"[:rng.randint(1, 5)]
    a = [rng.choice(alpha) for _ in range(rng.randint(0, 18))]
    if rng.random() < 0.5:
        b = [rng.choice(alpha) for _ in range(rng.randint(0, 18))]
    else:
        b = a[:]
        for _ in range(rng.randint(0, 5)):
            r = rng.random()
            if r < 0.4 and b:
                del b[rng.randrange(len(b))]
            elif r < 0.8:
                b.insert(rng.randint(0, len(b)), rng.choice(alpha + "z"))
            elif b:
                b[rng.randrange(len(b))] = rng.choice(alpha + "z")
    return a, b


# ---------------------------------------------------------------- basic
def test_basic_identical():
    a = ["x", "y", "z"]
    assert S.diff(a, list(a)) == [Op("equal", 0, 3, 0, 3, ("x", "y", "z"))]


def test_basic_both_empty():
    assert S.diff([], []) == []
    assert S.apply([], []) == []
    assert S.unified([], []) == ""


def test_basic_insert_into_empty_and_delete_all():
    assert S.diff([], ["a", "b"]) == [Op("insert", 0, 0, 0, 2, ("a", "b"))]
    assert S.diff(["a", "b"], []) == [Op("delete", 0, 2, 0, 0, ("a", "b"))]


def test_basic_middle_insert():
    a = ["1", "2", "3", "4"]
    b = ["1", "2", "X", "Y", "3", "4"]
    assert S.diff(a, b) == [
        Op("equal", 0, 2, 0, 2, ("1", "2")),
        Op("insert", 2, 2, 2, 4, ("X", "Y")),
        Op("equal", 2, 4, 4, 6, ("3", "4")),
    ]


def test_basic_middle_delete():
    a = ["1", "2", "3", "4", "5"]
    b = ["1", "5"]
    assert S.diff(a, b) == [
        Op("equal", 0, 1, 0, 1, ("1",)),
        Op("delete", 1, 4, 1, 1, ("2", "3", "4")),
        Op("equal", 4, 5, 1, 2, ("5",)),
    ]


def test_basic_replace_is_delete_then_insert():
    a = ["keep", "old1", "old2", "tail"]
    b = ["keep", "new", "tail"]
    assert S.diff(a, b) == [
        Op("equal", 0, 1, 0, 1, ("keep",)),
        Op("delete", 1, 3, 1, 1, ("old1", "old2")),
        Op("insert", 3, 3, 1, 2, ("new",)),
        Op("equal", 3, 4, 2, 3, ("tail",)),
    ]


def test_basic_myers_paper_example_minimal():
    a = list("ABCABBA")
    b = list("CBABAC")
    ops = S.diff(a, b)
    check_ops(a, b, ops)
    assert edit_count(ops) == 5
    assert S.apply(a, ops) == b


def test_basic_apply_roundtrip():
    a = "the quick brown fox jumps over the lazy dog".split()
    b = "a quick brown cat jumps over the dog today".split()
    ops = S.diff(a, b)
    check_ops(a, b, ops)
    assert S.apply(a, ops) == b
    assert edit_count(ops) == len(a) + len(b) - 2 * lcs_len(a, b)


def test_basic_unified_simple_matches_difflib():
    a = [f"line {i}" for i in range(1, 11)]
    b = a[:]
    b[4] = "changed"
    out = S.unified(a, b, fromfile="old.txt", tofile="new.txt")
    assert out == (
        "--- old.txt\n+++ new.txt\n@@ -2,7 +2,7 @@\n line 2\n line 3\n line 4\n-line 5\n+changed\n"
        " line 6\n line 7\n line 8\n"
    )
    assert out == difflib_unified(a, b, "old.txt", "new.txt")


def test_basic_unified_two_hunks():
    a = [f"l{i}" for i in range(30)]
    b = a[:]
    b[2] = "X"
    del b[20]
    out = S.unified(a, b)
    assert out.count("@@ -") == 2
    assert out == difflib_unified(a, b)


def test_basic_unified_default_names_and_context0():
    a = ["a", "b", "c", "d", "e"]
    b = ["a", "B", "c", "d", "e", "f"]
    out = S.unified(a, b, context=0)
    assert out == "--- a\n+++ b\n@@ -2 +2 @@\n-b\n+B\n@@ -5,0 +6 @@\n+f\n"
    assert out == difflib_unified(a, b, context=0)


def test_basic_unified_empty_sides():
    assert S.unified([], ["x", "y", "z"]) == "--- a\n+++ b\n@@ -0,0 +1,3 @@\n+x\n+y\n+z\n"
    assert S.unified(["x"], []) == "--- a\n+++ b\n@@ -1 +0,0 @@\n-x\n"


def test_basic_unified_identical_is_empty():
    a = ["same"] * 5
    assert S.unified(a, tuple(a)) == ""
    assert S.unified(a, a, context=0) == ""


# ---------------------------------------------------------------- edge
def test_edge_op_is_frozen_dataclass():
    op = Op("equal", 0, 1, 0, 1, ("x",))
    assert dataclasses.is_dataclass(op)
    assert [f.name for f in dataclasses.fields(op)] == ["tag", "a_start", "a_end", "b_start", "b_end", "lines"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        op.tag = "delete"
    assert op == Op(tag="equal", a_start=0, a_end=1, b_start=0, b_end=1, lines=("x",))


def test_edge_all_different():
    a = ["a", "b", "c"]
    b = ["x", "y"]
    assert S.diff(a, b) == [Op("delete", 0, 3, 0, 0, ("a", "b", "c")), Op("insert", 3, 3, 0, 2, ("x", "y"))]


def test_edge_tuple_inputs_and_not_mutated():
    a = ("a", "b", "c")
    b = ["a", "c", "d"]
    b_copy = b[:]
    ops = S.diff(a, b)
    check_ops(a, b, ops)
    assert b == b_copy
    a_list = list(a)
    res = S.apply(a_list, ops)
    assert res == b and a_list == list(a) and res is not a_list
    assert isinstance(S.apply(a, ops), list)


def test_edge_patch_error_is_value_error():
    assert issubclass(S.PatchError, ValueError)


def test_edge_apply_content_mismatch():
    a = ["a", "b", "c"]
    ops = S.diff(a, ["a", "c"])
    with pytest.raises(S.PatchError):
        S.apply(["a", "B", "c"], ops)
    with pytest.raises(S.PatchError):
        S.apply(["x", "b", "c"], ops)


def test_edge_apply_wrong_length_input():
    a = ["a", "b", "c"]
    ops = S.diff(a, ["a", "x", "c"])
    with pytest.raises(S.PatchError):
        S.apply(a + ["extra"], ops)
    with pytest.raises(S.PatchError):
        S.apply(a[:2], ops)
    with pytest.raises(S.PatchError):
        S.apply(["a"], [])


def test_edge_apply_noncontiguous():
    a = ["a", "b", "c", "d"]
    with pytest.raises(S.PatchError):  # gap in a
        S.apply(a, [Op("equal", 0, 1, 0, 1, ("a",)), Op("equal", 2, 4, 1, 3, ("c", "d"))])
    with pytest.raises(S.PatchError):  # gap in b
        S.apply(a, [Op("equal", 0, 1, 0, 1, ("a",)), Op("equal", 1, 4, 2, 5, ("b", "c", "d"))])
    with pytest.raises(S.PatchError):  # does not start at 0
        S.apply(a, [Op("equal", 1, 4, 0, 3, ("b", "c", "d"))])


def test_edge_apply_inconsistent_ops():
    a = ["a", "b"]
    bad = [
        [Op("equal", 0, 2, 0, 1, ("a", "b"))],
        [Op("delete", 0, 2, 0, 1, ("a", "b"))],
        [Op("insert", 0, 1, 0, 1, ("z",)), Op("equal", 1, 2, 1, 2, ("b",))],
        [Op("delete", 0, 1, 0, 0, ("a", "b")), Op("equal", 1, 2, 0, 1, ("b",))],
        [Op("equal", 0, 2, 0, 2, ("a",))],
        [Op("replace", 0, 2, 0, 2, ("a", "b"))],
        [Op("equal", 0, 3, 0, 3, ("a", "b", "c"))],
        [Op("equal", 0, 2, 0, 2, ("a", "b")), Op("delete", 2, 1, 2, 2, ())],
    ]
    for ops in bad:
        with pytest.raises(S.PatchError):
            S.apply(a, ops)


def test_edge_apply_accepts_unmerged_and_empty_ops():
    a = ["a", "b", "c"]
    ops = [
        Op("equal", 0, 1, 0, 1, ("a",)),
        Op("equal", 1, 2, 1, 2, ("b",)),
        Op("insert", 2, 2, 2, 3, ("x",)),
        Op("delete", 2, 3, 3, 3, ("c",)),
        Op("insert", 3, 3, 3, 3, ()),
        Op("insert", 3, 3, 3, 4, ("y",)),
    ]
    assert S.apply(a, ops) == ["a", "b", "x", "y"]
    assert S.apply(a, iter(ops)) == ["a", "b", "x", "y"]


def test_edge_apply_empty_input_with_insert():
    assert S.apply([], [Op("insert", 0, 0, 0, 2, ("p", "q"))]) == ["p", "q"]


def test_edge_unified_negative_context():
    with pytest.raises(ValueError):
        S.unified(["a"], ["b"], context=-1)


def test_edge_unified_hunk_merge_boundary():
    # Equal run of exactly 2*context lines between changes -> one hunk; one more -> two hunks.
    for ctx in (0, 1, 2, 3):
        for gap, hunks in ((2 * ctx, 1), (2 * ctx + 1, 2)):
            a = ["A"] + [f"m{i}" for i in range(gap)] + ["B"]
            b = ["a"] + [f"m{i}" for i in range(gap)] + ["b"]
            out = S.unified(a, b, context=ctx)
            assert out.count("\n@@ ") == hunks, (ctx, gap, out)
            assert out == difflib_unified(a, b, context=ctx)


def test_edge_unified_context_larger_than_file():
    a = ["a", "b", "c"]
    b = ["a", "x", "c"]
    out = S.unified(a, b, context=100)
    assert out == "--- a\n+++ b\n@@ -1,3 +1,3 @@\n a\n-b\n+x\n c\n"


def test_edge_unified_change_at_start_and_end():
    a = [f"{i}" for i in range(12)]
    b = ["first"] + a[1:-1] + ["last"]
    out = S.unified(a, b, context=2)
    assert out == difflib_unified(a, b, context=2)
    assert out.startswith("--- a\n+++ b\n@@ -1,3 +1,3 @@\n-0\n+first\n 1\n 2\n@@ -10,3 +10,3 @@\n")


def test_edge_repeated_lines():
    a = ["}", "", "}", "", "x", "}", ""]
    b = ["}", "", "y", "}", "", "}", ""]
    ops = S.diff(a, b)
    check_ops(a, b, ops)
    assert edit_count(ops) == len(a) + len(b) - 2 * lcs_len(a, b)
    assert S.apply(a, ops) == b


# ---------------------------------------------------------------- property
def test_prop_minimal_and_valid_small_alphabet():
    rng = random.Random(12345)
    for _ in range(600):
        a, b = small_random_pair(rng)
        ops = S.diff(a, b)
        check_ops(a, b, ops)
        assert edit_count(ops) == len(a) + len(b) - 2 * lcs_len(a, b), (a, b, ops)
        assert S.apply(a, ops) == b


def test_prop_minimal_medium_inputs():
    rng = random.Random(777)
    for _ in range(250):
        alpha = [f"w{i}" for i in range(rng.randint(2, 12))]
        a = [rng.choice(alpha) for _ in range(rng.randint(0, 70))]
        b = a[:]
        for _ in range(rng.randint(0, 25)):
            r = rng.random()
            if r < 0.35 and b:
                del b[rng.randrange(len(b))]
            elif r < 0.7:
                b.insert(rng.randint(0, len(b)), rng.choice(alpha))
            elif b:
                i = rng.randrange(len(b))
                j = min(len(b), i + rng.randint(1, 6))
                b[i:j] = b[i:j][::-1]
        ops = S.diff(a, b)
        check_ops(a, b, ops)
        assert edit_count(ops) == len(a) + len(b) - 2 * lcs_len(a, b)
        assert S.apply(a, ops) == b


def test_prop_unique_lcs_ops_and_unified_match_difflib():
    rng = random.Random(4242)
    for _ in range(400):
        a, b = unique_lcs_pair(rng)
        ctx = rng.choice([0, 1, 2, 3, 5])
        ops = S.diff(a, b)
        check_ops(a, b, ops)
        assert S.unified(a, b, fromfile="x/old", tofile="y/new", context=ctx) == \
            difflib_unified(a, b, "x/old", "y/new", ctx), (a, b, ctx)


def test_prop_unified_consistent_with_own_diff():
    rng = random.Random(99)
    for _ in range(400):
        a, b = small_random_pair(rng)
        ctx = rng.choice([0, 1, 2, 3])
        ops = S.diff(a, b)
        assert S.unified(a, b, context=ctx) == render_expected(a, b, ops, context=ctx), (a, b, ctx, ops)


def test_prop_apply_rejects_tampered_content():
    rng = random.Random(5)
    done = 0
    for _ in range(400):
        a, b = small_random_pair(rng)
        if not a:
            continue
        ops = S.diff(a, b)
        i = rng.randrange(len(a))
        tampered = a[:]
        tampered[i] = tampered[i] + "!"
        with pytest.raises(S.PatchError):
            S.apply(tampered, ops)
        done += 1
    assert done > 200


# ---------------------------------------------------------------- stress
def _code_like(rng, n):
    common = ["}", "", "    return None", "{", "    pass", "else:", "# ---"]
    return [rng.choice(common) if rng.random() < 0.4 else f"stmt_{rng.randrange(10 ** 6)}" for _ in range(n)]


def test_stress_large_files_few_changes():
    rng = random.Random(1)
    a = _code_like(rng, 20_000)
    b = a[:]
    for _ in range(50):
        r = rng.random()
        pos = rng.randrange(len(b))
        if r < 0.33:
            del b[pos]
        elif r < 0.66:
            b.insert(pos, f"added_{rng.randrange(10 ** 6)}")
        else:
            b[pos] = rng.choice(["}", "", f"changed_{pos}"])
    ops = S.diff(a, b)
    check_ops(a, b, ops)
    assert edit_count(ops) <= 2 * 50
    assert S.apply(a, ops) == b
    text = S.unified(a, b)
    assert text.startswith("--- a\n+++ b\n@@ ")


def test_stress_large_files_repeated_lines_few_changes():
    rng = random.Random(2)
    alpha = [f"t{i}" for i in range(8)]
    a = [rng.choice(alpha) for _ in range(20_000)]
    b = a[:]
    for _ in range(40):
        pos = rng.randrange(len(b))
        if rng.random() < 0.5:
            del b[pos]
        else:
            b.insert(pos, rng.choice(alpha))
    ops = S.diff(a, b)
    check_ops(a, b, ops)
    assert edit_count(ops) <= 40
    assert S.apply(a, ops) == b


def test_stress_totally_different_files():
    a = [f"old line {i}" for i in range(3_000)]
    b = [f"new line {i}" for i in range(3_000)]
    ops = S.diff(a, b)
    assert ops == [Op("delete", 0, 3000, 0, 0, tuple(a)), Op("insert", 3000, 3000, 0, 3000, tuple(b))]
    assert S.apply(a, ops) == b
    out = S.unified(a, b)
    assert out.startswith("--- a\n+++ b\n@@ -1,3000 +1,3000 @@\n-old line 0\n")


def test_stress_mostly_different_with_shared_lines():
    rng = random.Random(3)
    a = [f"a{i}" if rng.random() < 0.9 else "shared" for i in range(3_000)]
    b = [f"b{i}" if rng.random() < 0.9 else "shared" for i in range(3_000)]
    ops = S.diff(a, b)
    check_ops(a, b, ops)
    assert edit_count(ops) == len(a) + len(b) - 2 * min(a.count("shared"), b.count("shared"))
    assert S.apply(a, ops) == b


def test_stress_large_unique_lcs_unified_matches_difflib():
    rng = random.Random(8)
    a = [f"row {i}" for i in range(15_000)]
    b = []
    for i, line in enumerate(a):
        if rng.random() < 0.003:
            b.append(f"ins {i}")
        if rng.random() > 0.003:
            b.append(line)
    out = S.unified(a, b)
    exp = "".join(line + "\n" for line in difflib.unified_diff(a, b, "a", "b", lineterm=""))
    assert out == exp
