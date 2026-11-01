import random

import pytest

import solution as S

IS = S.IntervalSet


def ivs(s):
    return list(s)


# ---------------------------------------------------------------- brute-force model
UNIVERSE = 200


def model_intervals(cov):
    out = []
    start = None
    for x in range(-5, UNIVERSE + 6):
        if x in cov and start is None:
            start = x
        elif x not in cov and start is not None:
            out.append((start, x))
            start = None
    assert start is None
    return out


def model_gaps(cov, s, e):
    out = []
    start = None
    for x in range(s, e):
        if x not in cov and start is None:
            start = x
        elif x in cov and start is not None:
            out.append((start, x))
            start = None
    if start is not None:
        out.append((start, e))
    return out


def rand_range(rng, lo=0, hi=UNIVERSE):
    a = rng.randint(lo, hi)
    if rng.random() < 0.1:
        return a, a
    b = rng.randint(lo, hi)
    return min(a, b), max(a, b)


def check_state(s, cov):
    assert ivs(s) == model_intervals(cov)
    assert len(s) == len(model_intervals(cov))
    assert s.total_length() == len(cov)


# ---------------------------------------------------------------- basic
def test_basic_empty_set():
    s = IS()
    assert ivs(s) == []
    assert len(s) == 0
    assert not s
    assert s.total_length() == 0
    assert not s.contains(0)


def test_basic_add_disjoint_sorted():
    s = IS()
    s.add(10, 20)
    s.add(0, 5)
    s.add(30, 31)
    assert ivs(s) == [(0, 5), (10, 20), (30, 31)]
    assert len(s) == 3
    assert s.total_length() == 5 + 10 + 1
    assert all(type(t) is tuple for t in s)


def test_basic_add_overlapping_merges():
    s = IS()
    s.add(0, 10)
    s.add(5, 15)
    assert ivs(s) == [(0, 15)]
    s.add(-3, 2)
    assert ivs(s) == [(-3, 15)]
    s.add(2, 4)  # fully inside
    assert ivs(s) == [(-3, 15)]


def test_basic_add_adjacent_merges():
    s = IS()
    s.add(1, 3)
    s.add(3, 5)
    assert ivs(s) == [(1, 5)]
    s.add(0, 1)
    assert ivs(s) == [(0, 5)]
    s.add(6, 8)
    assert ivs(s) == [(0, 5), (6, 8)]
    s.add(5, 6)
    assert ivs(s) == [(0, 8)]


def test_basic_add_bridges_many():
    s = IS([(0, 2), (4, 6), (8, 10), (12, 14), (20, 22)])
    s.add(1, 13)
    assert ivs(s) == [(0, 14), (20, 22)]
    s.add(14, 20)
    assert ivs(s) == [(0, 22)]


def test_basic_remove_splits():
    s = IS([(0, 10)])
    s.remove(3, 5)
    assert ivs(s) == [(0, 3), (5, 10)]
    assert s.total_length() == 8
    assert len(s) == 2


def test_basic_remove_trims_and_deletes():
    s = IS([(0, 5), (10, 15), (20, 25), (30, 35)])
    s.remove(3, 22)
    assert ivs(s) == [(0, 3), (22, 25), (30, 35)]
    s.remove(30, 35)
    assert ivs(s) == [(0, 3), (22, 25)]
    s.remove(-100, 1)
    assert ivs(s) == [(1, 3), (22, 25)]
    s.remove(4, 22)  # nothing covered there
    assert ivs(s) == [(1, 3), (22, 25)]
    s.remove(0, 100)
    assert ivs(s) == []


def test_basic_contains_half_open():
    s = IS([(1, 3), (7, 9)])
    assert [x for x in range(-1, 12) if s.contains(x)] == [1, 2, 7, 8]


def test_basic_overlaps():
    s = IS([(0, 5), (10, 15)])
    assert s.overlaps(4, 6)
    assert s.overlaps(-10, 1)
    assert s.overlaps(6, 11)
    assert s.overlaps(-100, 100)
    assert s.overlaps(11, 12)
    assert not s.overlaps(5, 10)
    assert not s.overlaps(15, 20)
    assert not s.overlaps(-5, 0)


def test_basic_covers():
    s = IS([(0, 5), (10, 15)])
    assert s.covers(0, 5)
    assert s.covers(1, 4)
    assert s.covers(10, 15)
    assert not s.covers(0, 6)
    assert not s.covers(-1, 3)
    assert not s.covers(4, 11)
    assert not s.covers(20, 21)
    s.add(5, 10)
    assert s.covers(0, 15)


def test_basic_gaps_example():
    s = IS([(2, 4), (6, 8)])
    assert s.gaps(0, 10) == [(0, 2), (4, 6), (8, 10)]
    assert s.gaps(3, 7) == [(4, 6)]
    assert s.gaps(2, 4) == []
    assert s.gaps(2, 8) == [(4, 6)]
    assert s.gaps(8, 12) == [(8, 12)]
    assert IS().gaps(-3, 3) == [(-3, 3)]


def test_basic_equality():
    a = IS([(0, 3), (3, 6)])
    b = IS([(4, 6), (0, 2), (1, 5)])
    assert a == b
    assert not (a != b)
    b.remove(2, 3)
    assert a != b
    assert IS() == IS()


def test_basic_union_intersection_difference():
    a = IS([(0, 10), (20, 30)])
    b = IS([(5, 25), (40, 50)])
    assert ivs(a.union(b)) == [(0, 30), (40, 50)]
    assert ivs(a.intersection(b)) == [(5, 10), (20, 25)]
    assert ivs(a.difference(b)) == [(0, 5), (25, 30)]
    assert ivs(b.difference(a)) == [(10, 20), (40, 50)]
    assert a | b == a.union(b)
    assert a & b == a.intersection(b)
    assert a - b == a.difference(b)


def test_basic_union_merges_adjacent_across_operands():
    a = IS([(0, 5), (10, 15)])
    b = IS([(5, 10)])
    u = a | b
    assert ivs(u) == [(0, 15)]
    assert len(u) == 1


def test_basic_copy_independent():
    a = IS([(0, 10)])
    b = a.copy()
    assert a == b and a is not b
    b.add(20, 30)
    a.remove(0, 5)
    assert ivs(a) == [(5, 10)]
    assert ivs(b) == [(0, 10), (20, 30)]


def test_basic_constructor_iterable():
    s = IS(iter([(5, 7), (1, 2), (2, 3)]))
    assert ivs(s) == [(1, 3), (5, 7)]
    assert ivs(IS(None)) == []


# ---------------------------------------------------------------- edge
def test_edge_empty_ranges_noop_and_queries():
    s = IS([(0, 10)])
    s.add(20, 20)
    s.remove(5, 5)
    assert ivs(s) == [(0, 10)]
    assert s.overlaps(5, 5) is False
    assert s.overlaps(50, 50) is False
    assert s.covers(50, 50) is True
    assert s.covers(5, 5) is True
    assert s.gaps(50, 50) == []
    assert s.gaps(5, 5) == []
    assert IS().covers(3, 3) is True
    s2 = IS([(1, 1), (2, 2)])
    assert ivs(s2) == [] and len(s2) == 0


def test_edge_value_error_start_gt_end():
    s = IS([(0, 10)])
    for call in (lambda: s.add(5, 4), lambda: s.remove(5, 4), lambda: s.overlaps(5, 4),
                 lambda: s.covers(5, 4), lambda: s.gaps(5, 4)):
        with pytest.raises(ValueError):
            call()
    with pytest.raises(ValueError):
        IS([(0, 1), (3, 2)])
    assert ivs(s) == [(0, 10)]


def test_edge_type_errors():
    s = IS([(0, 10)])
    bad = [1.0, "1", None, True, False, 2.5]
    for b in bad:
        for call in (lambda: s.add(b, 20), lambda: s.add(0, b), lambda: s.remove(b, 20),
                     lambda: s.remove(0, b), lambda: s.overlaps(b, 20), lambda: s.covers(0, b),
                     lambda: s.gaps(b, 20), lambda: s.contains(b)):
            with pytest.raises(TypeError):
                call()
    with pytest.raises(TypeError):
        IS([(0.0, 1)])
    assert ivs(s) == [(0, 10)]


def test_edge_type_error_before_value_error():
    s = IS()
    with pytest.raises(TypeError):
        s.add(5, 1.0)
    with pytest.raises(TypeError):
        s.remove(True, 0)


def test_edge_bool_not_int_for_contains():
    s = IS([(0, 5)])
    with pytest.raises(TypeError):
        s.contains(True)


def test_edge_negative_and_huge_coordinates():
    big = 10 ** 30
    s = IS([(-big, -big + 3), (big, big + 2), (-1, 1)])
    assert ivs(s) == [(-big, -big + 3), (-1, 1), (big, big + 2)]
    assert s.contains(-big + 2) and not s.contains(-big + 3)
    assert s.contains(big + 1)
    assert s.total_length() == 3 + 2 + 2
    assert s.gaps(-1, 3) == [(1, 3)]


def test_edge_touching_not_overlapping_but_covers_merged():
    s = IS([(0, 5)])
    assert not s.overlaps(5, 8)
    assert not s.overlaps(-3, 0)
    s.add(5, 8)
    assert s.covers(0, 8)
    assert len(s) == 1


def test_edge_remove_exact_boundaries():
    s = IS([(0, 10)])
    s.remove(0, 1)
    s.remove(9, 10)
    assert ivs(s) == [(1, 9)]
    s.remove(1, 9)
    assert ivs(s) == [] and s.total_length() == 0
    s.remove(0, 100)
    assert ivs(s) == []


def test_edge_remove_one_from_middle_then_readd():
    s = IS([(0, 10)])
    s.remove(5, 6)
    assert ivs(s) == [(0, 5), (6, 10)]
    assert not s.contains(5) and s.contains(4) and s.contains(6)
    assert s.gaps(0, 10) == [(5, 6)]
    s.add(5, 6)
    assert ivs(s) == [(0, 10)]


def test_edge_set_ops_do_not_modify_operands():
    a = IS([(0, 10)])
    b = IS([(5, 15)])
    for op in (lambda: a | b, lambda: a & b, lambda: a - b, lambda: a.union(b),
               lambda: a.intersection(b), lambda: a.difference(b)):
        r = op()
        assert r is not a and r is not b
    assert ivs(a) == [(0, 10)] and ivs(b) == [(5, 15)]
    r = a | b
    r.add(100, 200)
    assert ivs(a) == [(0, 10)] and ivs(b) == [(5, 15)]
    r = a - IS()
    r.remove(0, 10)
    assert ivs(a) == [(0, 10)]


def test_edge_set_ops_with_empty():
    a = IS([(0, 10)])
    e = IS()
    assert a | e == a and e | a == a
    assert ivs(a & e) == [] and ivs(e & a) == []
    assert a - e == a
    assert ivs(e - a) == []
    assert ivs(a - a) == []


def test_edge_type_errors_set_ops_and_eq():
    a = IS([(0, 10)])
    for other in ([(0, 1)], {1, 2}, None, 3):
        with pytest.raises(TypeError):
            a.union(other)
        with pytest.raises(TypeError):
            a.intersection(other)
        with pytest.raises(TypeError):
            a.difference(other)
        with pytest.raises(TypeError):
            a | other
        with pytest.raises(TypeError):
            a & other
        with pytest.raises(TypeError):
            a - other
    assert (a == [(0, 10)]) is False
    assert (a != [(0, 10)]) is True
    assert a.__eq__([(0, 10)]) is NotImplemented


def test_edge_unhashable():
    with pytest.raises(TypeError):
        hash(IS())


def test_edge_intersection_touching_is_empty():
    a = IS([(0, 5), (10, 15)])
    b = IS([(5, 10), (15, 20)])
    assert ivs(a & b) == []
    assert a - b == a
    assert ivs(a | b) == [(0, 20)]


# ---------------------------------------------------------------- property tests
def test_prop_random_ops_vs_model():
    rng = random.Random(1234)
    for _seq in range(400):
        s = IS()
        cov = set()
        for _ in range(rng.randint(1, 40)):
            a, b = rand_range(rng)
            if rng.random() < 0.6:
                s.add(a, b)
                cov.update(range(a, b))
            else:
                s.remove(a, b)
                cov.difference_update(range(a, b))
        check_state(s, cov)
        for _ in range(15):
            x = rng.randint(-3, UNIVERSE + 3)
            assert s.contains(x) == (x in cov)
            a, b = rand_range(rng, -3, UNIVERSE + 3)
            r = range(a, b)
            assert s.overlaps(a, b) == any(y in cov for y in r), (list(s), a, b)
            assert s.covers(a, b) == all(y in cov for y in r), (list(s), a, b)
            assert s.gaps(a, b) == model_gaps(cov, a, b), (list(s), a, b)


def test_prop_state_after_each_op():
    rng = random.Random(99)
    for _seq in range(300):
        s = IS()
        cov = set()
        for _ in range(rng.randint(1, 12)):
            a, b = rand_range(rng, 0, 60)
            if rng.random() < 0.55:
                s.add(a, b)
                cov.update(range(a, b))
            else:
                s.remove(a, b)
                cov.difference_update(range(a, b))
            check_state(s, cov)


def test_prop_set_algebra_vs_model():
    rng = random.Random(7)
    for _ in range(300):
        sets = []
        covs = []
        for _k in range(2):
            s = IS()
            cov = set()
            for _ in range(rng.randint(0, 12)):
                a, b = rand_range(rng)
                if rng.random() < 0.7:
                    s.add(a, b)
                    cov.update(range(a, b))
                else:
                    s.remove(a, b)
                    cov.difference_update(range(a, b))
            sets.append(s)
            covs.append(cov)
        a, b = sets
        ca, cb = covs
        before_a, before_b = ivs(a), ivs(b)
        check_state(a | b, ca | cb)
        check_state(a & b, ca & cb)
        check_state(a - b, ca - cb)
        check_state(b - a, cb - ca)
        check_state(a.union(b), ca | cb)
        assert (a == b) == (ca == cb)
        assert ivs(a) == before_a and ivs(b) == before_b


def test_prop_order_independence_and_eq():
    rng = random.Random(5)
    for _ in range(300):
        ranges = [rand_range(rng) for _ in range(rng.randint(0, 15))]
        a = IS(ranges)
        shuffled = ranges[:]
        rng.shuffle(shuffled)
        b = IS()
        for r in shuffled:
            b.add(*r)
        assert a == b
        assert ivs(a) == ivs(b)
        c = a.copy()
        assert c == a
        cov = set()
        for x, y in ranges:
            cov.update(range(x, y))
        check_state(c, cov)


# ---------------------------------------------------------------- stress
def test_stress_many_adds_removes():
    rng = random.Random(2024)
    s = IS()
    space = 10 ** 9
    for i in range(200_000):
        a = rng.randrange(space)
        if rng.random() < 0.6:
            s.add(a, a + rng.randint(1, 5_000))
        else:
            s.remove(a, a + rng.randint(1, 5_000))
    items = ivs(s)
    assert len(items) == len(s)
    assert len(s) > 10_000
    for (s1, e1), (s2, e2) in zip(items, items[1:]):
        assert s1 < e1 < s2
    assert s.total_length() == sum(e - b for b, e in items)


def test_stress_contains_queries_many_intervals():
    s = IS()
    for i in range(50_000):
        s.add(i * 10, i * 10 + 5)
    assert len(s) == 50_000
    rng = random.Random(3)
    hits = 0
    for _ in range(200_000):
        x = rng.randrange(-10, 500_010)
        r = s.contains(x)
        assert r == (0 <= x < 500_000 and x % 10 < 5)
        hits += r
    assert hits > 50_000


def test_stress_range_queries_many_intervals():
    s = IS((i * 10, i * 10 + 5) for i in range(50_000))
    rng = random.Random(4)
    for _ in range(100_000):
        a = rng.randrange(0, 500_000)
        b = a + rng.randint(0, 7)
        exp_cov = all(y % 10 < 5 for y in range(a, b))
        exp_ov = any(y % 10 < 5 for y in range(a, b))
        assert s.covers(a, b) == exp_cov
        assert s.overlaps(a, b) == exp_ov
    for _ in range(20_000):
        a = rng.randrange(0, 499_000)
        g = s.gaps(a, a + 40)
        assert sum(e - b for b, e in g) == sum(1 for y in range(a, a + 40) if y % 10 >= 5)


def test_stress_set_algebra_large():
    a = IS((i * 10, i * 10 + 6) for i in range(50_000))
    b = IS((i * 10 + 4, i * 10 + 8) for i in range(50_000))
    u = a | b
    assert len(u) == 50_000 and u.total_length() == 8 * 50_000
    n = a & b
    assert len(n) == 50_000 and n.total_length() == 2 * 50_000
    d = a - b
    assert len(d) == 50_000 and d.total_length() == 4 * 50_000
    assert (d | n) == a


def test_stress_bulk_merge_and_split():
    s = IS((i * 3, i * 3 + 2) for i in range(100_000))
    assert len(s) == 100_000
    s.add(0, 300_000)
    assert ivs(s) == [(0, 300_000)]
    for i in range(50_000):
        s.remove(i * 6 + 1, i * 6 + 2)
    assert len(s) == 50_001
    assert s.total_length() == 300_000 - 50_000
