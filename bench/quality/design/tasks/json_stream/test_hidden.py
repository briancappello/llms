import json
import random

import pytest

import solution as S


# --------------------------------------------------------------------------------------------- helpers
def same(a, b):
    """Structural equality that also distinguishes int/float/bool and -0.0 from 0.0."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return repr(a) == repr(b)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    return a == b


def assert_same(got, want):
    assert same(got, want), f"\n got: {got!r}\nwant: {want!r}"


def decode_all(chunks, **kw):
    d = S.JSONStreamDecoder(**kw)
    out = []
    for c in chunks:
        out.extend(d.feed(c))
    out.extend(d.close())
    return out


def error_of(chunks, **kw):
    d = S.JSONStreamDecoder(**kw)
    with pytest.raises(S.JSONStreamError) as ei:
        for c in chunks:
            d.feed(c)
        d.close()
    return ei.value


# ----------------------------------------------------------------------------------------------- basic
def test_basic_single_object():
    d = S.JSONStreamDecoder()
    assert_same(d.feed('{"a": 1, "b": [true, false, null], "c": "x"}'), [{"a": 1, "b": [True, False, None], "c": "x"}])
    assert d.close() == []


def test_basic_whitespace_separated_values():
    d = S.JSONStreamDecoder()
    got = d.feed('{"a":1}\n[2]\t"s" \r\n {}')
    assert_same(got, [{"a": 1}, [2], "s", {}])
    assert d.close() == []


def test_basic_back_to_back_containers_and_strings():
    d = S.JSONStreamDecoder()
    assert_same(d.feed('{}[]"x"{"a":1}[[]]""'), [{}, [], "x", {"a": 1}, [[]], ""])


def test_basic_value_returned_by_completing_chunk():
    d = S.JSONStreamDecoder()
    assert d.feed('{"a": [1, 2') == []
    assert d.feed(", 3]") == []
    assert_same(d.feed('}{"b"'), [{"a": [1, 2, 3]}])
    assert_same(d.feed(': 2}'), [{"b": 2}])


def test_basic_number_waits_for_terminator():
    d = S.JSONStreamDecoder()
    assert d.feed("12") == []
    assert d.feed("3") == []
    assert_same(d.feed(" "), [123])
    assert_same(d.feed("-4.5e1\n"), [-45.0])


def test_basic_number_at_end_emitted_on_close():
    d = S.JSONStreamDecoder()
    assert_same(d.feed("1 2 3"), [1, 2])
    assert_same(d.close(), [3])


def test_basic_literals_need_terminator():
    d = S.JSONStreamDecoder()
    assert_same(d.feed("true false null"), [True, False])
    assert_same(d.close(), [None])
    d = S.JSONStreamDecoder()
    assert d.feed("true") == []
    assert_same(d.feed("\n"), [True])


def test_basic_scalar_terminated_by_value_start():
    d = S.JSONStreamDecoder()
    got = d.feed('1[2]3{"a":4}5"s"true[]null{}false"t"-0.5')
    assert_same(got, [1, [2], 3, {"a": 4}, 5, "s", True, [], None, {}, False, "t"])
    assert_same(d.close(), [-0.5])


def test_basic_anything_may_follow_self_delimiting_values():
    assert_same(decode_all(['"a"1 []2 {}true "b"null']), ["a", 1, [], 2, {}, True, "b", None])
    assert_same(decode_all(['[]-1 {}0 ""3']), [[], -1, {}, 0, "", 3])


def test_basic_decoding_matches_json_loads():
    text = ('{"int": 7, "neg": -12, "zero": -0, "f": 1.5, "nz": -0.0, "e": 1e-7, "E": 2.5E+3, '
            '"big": 123456789012345678901234567890, "u": "\\u00e9\\ud83d\\ude00\\n\\t\\"\\\\\\/", '
            '"raw": "é€😀", "dup": 1, "dup": 2, "nest": [[{"x": []}], {}]}')
    assert_same(decode_all([text]), [json.loads(text)])


def test_basic_split_inside_escape_sequence():
    assert_same(decode_all(['"caf\\u00', 'e9"']), ["café"])
    assert_same(decode_all(['"a\\', 'nb"']), ["a\nb"])
    assert_same(decode_all(['"a\\', '"', 'b"']), ['a"b'])
    assert_same(decode_all(['"\\', "\\", '"']), ["\\"])


def test_basic_split_between_surrogate_pair():
    assert_same(decode_all(['"\\ud83d', '\\ude00"']), ["😀"])
    assert_same(decode_all(['"\\ud83d\\', 'ude00"']), ["😀"])


def test_basic_split_inside_number_and_literal():
    assert_same(decode_all(["[-", "1", ".", "2", "5e", "-", "3]"]), [[-1.25e-3]])
    assert_same(decode_all(["tr", "u", "e ", "fa", "lse"]), [True, False])


def test_basic_pending():
    d = S.JSONStreamDecoder()
    assert d.pending() == 0
    d.feed("  [1, 2")
    assert d.pending() == 5
    d.feed(", 3")
    assert d.pending() == 8
    d.feed("]   ")
    assert d.pending() == 0
    d.feed('"ab')
    assert d.pending() == 3
    d.feed('c" 42')
    assert d.pending() == 2
    d.feed(" ")
    assert d.pending() == 0


def test_basic_empty_feed_and_whitespace_only_stream():
    d = S.JSONStreamDecoder()
    assert d.feed("") == []
    assert d.feed(" \t\r\n ") == []
    assert d.feed("") == []
    assert d.close() == []


def test_basic_close_on_empty_stream():
    assert S.JSONStreamDecoder().close() == []


# ------------------------------------------------------------------------------------------------ edge
ERROR_CASES = [
    ("[1,]", 3), ('{"a" 1}', 5), ('{"a":1,}', 7), ("[1 2]", 3), ("01", 1), ("[01]", 2), ("-a", 1),
    ("[1.]", 3), ("1.e5", 2), ("1e ", 2), ('"abc\\x"', 5), ('"\\u12G4"', 5), ('"a\nb"', 2), ('"\t"', 1),
    ("tru x", 3), ("trux", 3), ("nulL", 3), ("nul l", 3), ("[true false]", 6), ("}", 0), ("]", 0), (",", 0),
    (":", 0), ("1,2", 1), ('{"a":1}}', 7), ("{1:2}", 1), ('{"a"}', 4), ("[}", 1), ('{"a":1]', 6), ("NaN", 0),
    ("Infinity", 0), ("-Infinity", 1), ("[-]", 2), ("true1", 4), ("1-2", 1), ("'a'", 0), ("[1]x", 3),
    ('"abc" ,', 6), ("/* c */ 1", 0), ("\u00a01", 0), ('{"a":1 "b":2}', 7), ("[1,,2]", 3), ("-01", 2),
    ("[-01]", 3), ("1e5.0", 3), ("2.5.1", 3), ("+1", 0), ("0x1F", 1), ('{"a":tru}', 8), ("[1]]", 3),
    ('{"a"::1}', 5), ("\ufeff[]", 0), ("[.5]", 1), ("1.5e+-3", 5), ('{"a":1,"b"}', 10), ("nullnull", 4),
    ('[1] 2 {"k": [3, x]}', 16), ("[[[]]]]", 6), ("12 34 5a", 7),
]


@pytest.mark.parametrize("text,offset", ERROR_CASES)
def test_edge_error_offsets(text, offset):
    e = error_of([text])
    assert not isinstance(e, (S.IncompleteJSONError, S.BufferLimitError))
    assert e.offset == offset


def test_edge_error_is_value_error_with_offset():
    e = error_of(["[1,]"])
    assert isinstance(e, ValueError)
    assert issubclass(S.IncompleteJSONError, S.JSONStreamError)
    assert issubclass(S.BufferLimitError, S.JSONStreamError)
    assert issubclass(S.JSONStreamError, ValueError)


def test_edge_error_offset_is_absolute_across_chunks():
    chunks = ['[1] {"a": [tr', "ue, fals", "x]}"]
    d = S.JSONStreamDecoder()
    assert_same(d.feed(chunks[0]), [[1]])
    assert d.feed(chunks[1]) == []
    with pytest.raises(S.JSONStreamError) as ei:
        d.feed(chunks[2])
    assert ei.value.offset == len(chunks[0]) + len(chunks[1])


def test_edge_error_raised_by_the_feed_that_delivers_the_bad_char():
    d = S.JSONStreamDecoder()
    d.feed('["a", ')
    with pytest.raises(S.JSONStreamError) as ei:
        d.feed("x")  # the array is still open; the error must not wait for it to close
    assert ei.value.offset == 6
    d = S.JSONStreamDecoder()
    d.feed('{"k": [1, 2, 3')
    with pytest.raises(S.JSONStreamError) as ei:
        d.feed(" 4")
    assert ei.value.offset == 15


def test_edge_unusable_after_error():
    d = S.JSONStreamDecoder()
    with pytest.raises(S.JSONStreamError):
        d.feed("[1,]")
    with pytest.raises(S.JSONStreamError):
        d.feed("[1]")
    with pytest.raises(S.JSONStreamError):
        d.close()


CLOSE_INCOMPLETE = [
    ("[1, 2", 5), ('"abc', 4), ('{"a":', 5), ("tru", 3), ("-", 1), ("1.", 2), ("1e+", 3), ('"\\u12', 5),
    ('{"a"', 4), ("[[[]]", 5), ('"abc\\', 5), ("  [  ", 5), ("1 2 [", 5), ("fals", 4), ("1E", 2),
]


@pytest.mark.parametrize("text,offset", CLOSE_INCOMPLETE)
def test_edge_close_incomplete(text, offset):
    d = S.JSONStreamDecoder()
    for ch in text:  # also proves no error is raised before close()
        d.feed(ch)
    with pytest.raises(S.IncompleteJSONError) as ei:
        d.close()
    assert ei.value.offset == offset


def test_edge_close_complete_scalars():
    for text, want in [("0", 0), ("-0", 0), ("1.0", 1.0), ("-0.0", -0.0), ("1e2", 100.0), ("false", False),
                       ("null", None), ("12 \n", 12), ("[1] 7", 7)]:
        d = S.JSONStreamDecoder()
        got = d.feed(text)
        got += d.close()
        assert_same(got[-1], want)


def test_edge_feed_or_close_after_close_raises():
    d = S.JSONStreamDecoder()
    d.feed("[1]")
    assert d.close() == []
    with pytest.raises(S.JSONStreamError):
        d.feed("[2]")
    d2 = S.JSONStreamDecoder()
    d2.close()
    with pytest.raises(S.JSONStreamError):
        d2.close()


def test_edge_feed_rejects_non_str():
    d = S.JSONStreamDecoder()
    with pytest.raises(TypeError):
        d.feed(b"[1]")


def test_edge_max_buffer_validation():
    with pytest.raises(ValueError):
        S.JSONStreamDecoder(max_buffer=0)
    with pytest.raises(ValueError):
        S.JSONStreamDecoder(max_buffer=-5)


def test_edge_buffer_limit_exact_size_in_one_chunk():
    d = S.JSONStreamDecoder(max_buffer=10)
    assert d.feed('"' + "a" * 8 + '"') == ["a" * 8]  # exactly 10 characters: fine
    d = S.JSONStreamDecoder(max_buffer=10)
    with pytest.raises(S.BufferLimitError) as ei:
        d.feed('  "' + "a" * 9 + '"')  # 11 characters starting at offset 2
    assert ei.value.offset == 12
    with pytest.raises(S.JSONStreamError):
        d.feed("1 ")


def test_edge_buffer_limit_across_chunks():
    d = S.JSONStreamDecoder(max_buffer=10)
    d.feed("  [1,2,")
    assert d.pending() == 5
    d.feed("3,4")
    assert d.pending() == 8
    with pytest.raises(S.BufferLimitError) as ei:
        d.feed(",5,")
    assert ei.value.offset == 12


def test_edge_buffer_limit_whitespace_and_total_not_counted():
    d = S.JSONStreamDecoder(max_buffer=5)
    got = d.feed("[1]" + " " * 100 + "[2]\n" * 50)
    assert len(got) == 51
    assert d.feed('{"a":1}'[:5]) == []
    assert d.pending() == 5


def test_edge_buffer_limit_unterminated_number():
    d = S.JSONStreamDecoder(max_buffer=5)
    assert d.feed("12345") == []
    assert_same(d.feed(" "), [12345])
    d = S.JSONStreamDecoder(max_buffer=5)
    with pytest.raises(S.BufferLimitError) as ei:
        d.feed("1 123456")
    assert ei.value.offset == 7


def test_edge_strict_rfc8259():
    for text in ["NaN", "-Infinity", "[1,]", "{'a':1}", "// x\n1", "[01]", "+1", ".5", "[0x10]", '{"a":1,}',
                 "[True]", "None"]:
        with pytest.raises(S.JSONStreamError):
            decode_all([text])


def test_edge_string_character_rules():
    assert_same(decode_all(['"\x7f\u2028\u00a0/"']), ["\x7f\u2028\u00a0/"])
    for cc in ["\x00", "\x1f", "\n", "\r"]:
        e = error_of(['["ab' + cc + '"]'])
        assert e.offset == 4


def test_edge_lone_surrogate_escape_decodes_like_json_loads():
    assert_same(decode_all(['"\\ud800" "\\udc00x"']), [json.loads('"\\ud800"'), json.loads('"\\udc00x"')])


def test_edge_nesting_depth_1000():
    text = "[" * 1000 + "1" + "]" * 1000
    got = decode_all([text[i:i + 7] for i in range(0, len(text), 7)])
    v = got[0]
    for _ in range(1000):
        assert isinstance(v, list) and len(v) == 1
        v = v[0]
    assert v == 1


def test_edge_values_from_earlier_feeds_survive_later_error():
    d = S.JSONStreamDecoder()
    assert_same(d.feed('[1] {"a": 2} 3 '), [[1], {"a": 2}, 3])
    with pytest.raises(S.JSONStreamError) as ei:
        d.feed("]")
    assert ei.value.offset == 15


# ------------------------------------------------------------------------------------------- property
ALPHABET = list("abcXYZ019 _-") + ['"', "\\", "/", "\n", "\t", "\r", "\b", "\f", "\x00", "\x01", "\x1f", "\x7f",
                                   "é", "ü", "€", "\u2028", "😀", "𝄞", "中"]
RAW_NUMBERS = ["0", "-0", "1E+5", "1e5", "-0.0e-0", "12.50E3", "0.000001", "1e-7", "-2.5E-05", "123456789012345678901",
               "-98765432109876543210.5", "7", "3.14159", "1E400", "-1e-400", "0e0"]


def rand_str(rng):
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 10)))


def rand_number(rng):
    r = rng.random()
    if r < 0.3:
        return rng.randint(-1000, 1000)
    if r < 0.45:
        return rng.choice([-1, 1]) * rng.randint(0, 10 ** rng.randint(1, 40))
    if r < 0.6:
        return rng.choice([0.0, -0.0, 1e-7, -1e-7, 1.5e300, 5e-324, 0.1, 123.456, 1e16, -2.5e-5, 2.0])
    return rng.uniform(-1, 1) * 10 ** rng.randint(-20, 20)


def rand_value(rng, depth=0):
    r = rng.random()
    if depth >= 4 or r < 0.45:
        k = rng.randrange(4)
        if k == 0:
            return rand_str(rng)
        if k == 1:
            return rand_number(rng)
        return rng.choice([True, False, None])
    if r < 0.72:
        return [rand_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {rand_str(rng): rand_value(rng, depth + 1) for _ in range(rng.randint(0, 4))}


def rand_piece(rng):
    """Text of one top-level value."""
    if rng.random() < 0.12:
        return rng.choice(RAW_NUMBERS)
    v = rand_value(rng)
    seps = rng.choice([(",", ":"), (", ", ": "), (" ,\t", " :\n")])
    indent = rng.choice([None, None, 0, 2, "\t"])
    return json.dumps(v, separators=seps, indent=indent, ensure_ascii=rng.random() < 0.5)


def is_scalar(piece):
    return piece[0] not in '{["'


def rand_stream(rng, n_values):
    """Return (text, pieces as (start, end, text))."""
    parts, pieces, pos = [], [], 0
    ws = " \t\n\r"
    lead = "".join(rng.choice(ws) for _ in range(rng.randint(0, 2)))
    parts.append(lead)
    pos += len(lead)
    prev = None
    for _ in range(n_values):
        p = rand_piece(rng)
        if prev is not None:
            need_ws = is_scalar(prev) and is_scalar(p)
            sep = "".join(rng.choice(ws) for _ in range(rng.randint(1 if need_ws else 0, 3)))
            if need_ws and not sep:
                sep = " "
            parts.append(sep)
            pos += len(sep)
        pieces.append((pos, pos + len(p), p))
        parts.append(p)
        pos += len(p)
        prev = p
    tail = "".join(rng.choice(ws) for _ in range(rng.randint(0, 2)))
    parts.append(tail)
    return "".join(parts), pieces


def rand_chunks(rng, text):
    mode = rng.randrange(4)
    out, i = [], 0
    while i < len(text):
        if mode == 0:
            k = 1
        elif mode == 1:
            k = rng.randint(1, 4)
        elif mode == 2:
            k = rng.randint(1, 40)
        else:
            k = rng.choice([0, 1, 2, 3, 50, 200])
        out.append(text[i:i + k])
        i += k
    return out


def model_state(pieces, consumed, total):
    """(number of values emitted, pending chars) after `consumed` characters were fed."""
    emitted, pending = 0, 0
    for start, end, p in pieces:
        done = end < consumed if is_scalar(p) else end <= consumed
        if done:
            emitted += 1
        elif start < consumed:
            pending = consumed - start
    return emitted, pending


def test_prop_random_streams_random_chunks():
    for seed in range(400):
        rng = random.Random(seed)
        text, pieces = rand_stream(rng, rng.randint(1, 8))
        want = [json.loads(p) for _, _, p in pieces]
        d = S.JSONStreamDecoder()
        got, consumed = [], 0
        for c in rand_chunks(rng, text):
            got.extend(d.feed(c))
            consumed += len(c)
            emitted, pending = model_state(pieces, consumed, len(text))
            assert len(got) == emitted, (seed, consumed, text)
            assert d.pending() == pending, (seed, consumed, text)
        got.extend(d.close())
        assert same(got, want), (seed, text, got, want)


CORPUS = ('{"a":[1,-2.5e-3,true,null],"s":"x\\u00e9\\ud83d\\ude00\\"\\\\\\n"} [] "é😀" -0.0 12'
          '{"k":{"n":[[], {}]}}false\t1E+2 "" 0[3]null"\\/\\b\\f\\r\\t"123456789012345678901234567890 7')


def test_prop_every_split_point():
    want = decode_all([CORPUS])
    assert len(want) == 15
    for k in range(len(CORPUS) + 1):
        assert same(decode_all([CORPUS[:k], CORPUS[k:]]), want), k


def test_prop_every_three_way_split():
    text = '[1.5e3,"a\\u00e9\\n"] -12{"k":"\\ud83d\\ude00"}true 0'
    want = decode_all([text])
    n = len(text)
    for a in range(n + 1):
        for b in range(a, n + 1):
            assert same(decode_all([text[:a], text[a:b], text[b:]]), want), (a, b)


def test_prop_error_offsets_independent_of_chunking():
    rng = random.Random(1234)
    cases = 0
    for text, offset in ERROR_CASES * 5:
        for chunks in (rand_chunks(rng, text), rand_chunks(rng, text)):
            d = S.JSONStreamDecoder()
            consumed = 0
            for c in chunks:
                if consumed + len(c) <= offset:
                    d.feed(c)  # the prefix is valid so far: no error yet
                else:
                    with pytest.raises(S.JSONStreamError) as ei:
                        d.feed(c)
                    assert ei.value.offset == offset, (text, chunks)
                    break
                consumed += len(c)
            cases += 1
    assert cases >= 200


def test_prop_streams_with_error_after_valid_prefix():
    for seed in range(200):
        rng = random.Random(10_000 + seed)
        text, pieces = rand_stream(rng, rng.randint(1, 5))
        bad, off = rng.choice(ERROR_CASES)
        sep = " " if (is_scalar(pieces[-1][2]) or bad[0] in "0123456789-") else rng.choice(["", " "])
        if is_scalar(pieces[-1][2]) and text[-1] not in " \t\r\n":
            text += " "
        full = text + sep + bad
        want_off = len(text) + len(sep) + off
        d = S.JSONStreamDecoder()
        got = []
        with pytest.raises(S.JSONStreamError) as ei:
            for c in rand_chunks(rng, full):
                got.extend(d.feed(c))
            d.close()
        assert not isinstance(ei.value, S.IncompleteJSONError)
        assert ei.value.offset == want_off, (seed, full)
        k = min(len(got), len(pieces))
        assert same(got[:k], [json.loads(p) for _, _, p in pieces][:k])


# --------------------------------------------------------------------------------------------- stress
def test_stress_huge_string_in_small_chunks():
    body = "lorem ipsum \\u00e9 \\\" " * 360_000
    text = '"' + body + '"'
    d = S.JSONStreamDecoder()
    out = []
    for i in range(0, len(text), 64):
        out.extend(d.feed(text[i:i + 64]))
        if not out:
            assert d.pending() == min(i + 64, len(text))
    assert len(out) == 1 and out[0] == json.loads(text)
    assert d.close() == []


def test_stress_many_small_values_in_one_chunk():
    vals = []
    for k in range(200_000):
        vals.append('"abcdefghij"' if k % 3 == 0 else (str(k) if k % 3 == 1 else "[%d]" % k))
    text = " ".join(vals) + " "
    d = S.JSONStreamDecoder()
    got = d.feed(text)
    assert len(got) == 200_000
    assert got[1] == 1 and got[2] == [2] and got[-1] == 199_999 and got[199_998] == "abcdefghij"


def test_stress_one_character_feeds():
    rng = random.Random(7)
    docs = [rand_piece(rng) for _ in range(1500)]
    text = "\n".join(docs) + "\n"
    count = len(docs)
    while len(text) < 150_000:
        text += text
        count *= 2
    d = S.JSONStreamDecoder()
    n = 0
    for ch in text:
        n += len(d.feed(ch))
    n += len(d.close())
    assert n == count


def test_stress_large_array_of_numbers_in_chunks():
    rng = random.Random(99)
    nums = [rng.uniform(-1e6, 1e6) if i % 2 else rng.randint(-10**12, 10**12) for i in range(150_000)]
    text = json.dumps(nums)
    d = S.JSONStreamDecoder()
    out = []
    for i in range(0, len(text), 997):
        out.extend(d.feed(text[i:i + 997]))
    assert len(out) == 1 and out[0] == nums
