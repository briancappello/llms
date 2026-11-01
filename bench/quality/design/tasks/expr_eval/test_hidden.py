import io
import math
import random
import sys
import threading
import tokenize
import types

import pytest

import solution as S


def ev(src, variables=None):
    return S.evaluate(src, variables)


def syntax_pos(src):
    with pytest.raises(S.ExprSyntaxError) as ei:
        S.compile_expr(src)
    return ei.value.position


# ----------------------------------------------------------------------------------------------- basic
PRECEDENCE = [
    ("1+2*3", 7), ("(1+2)*3", 9), ("2*3+4", 10), ("10-4-3", 3), ("100/10/5", 2), ("10/2*5", 25),
    ("2^3^2", 512), ("(2^3)^2", 64), ("-2^2", -4), ("(-2)^2", 4), ("2^-2", 0.25), ("-2^-2", -0.25),
    ("2^-1^2", 0.5), ("-3*2", -6), ("2*-3", -6), ("-2*-2", 4), ("1+2<4", 1), ("1<2+3", 1), ("2*3==6", 1),
    ("1-1==0", 1), ("7%4*2", 6), ("7-4%3", 6), ("2*3^2", 18), ("2+3^2*2", 20), ("8/2^2", 2), ("2^3%5", 3),
    ("-5%3", 1), ("--2^2", 4), ("-(2)^2", -4), ("(1<2)==1", 1), ("3-2+1", 2), ("2^2^0", 2), ("-2^0", -1),
    ("4/2/2", 1), ("1+-2^2", -3), ("3>2+2", 0), ("6/3!=2", 0), ("1+2*3^2-4/2", 17), ("-1^0.5", -1),
]


@pytest.mark.parametrize("src,want", PRECEDENCE)
def test_basic_precedence_and_associativity(src, want):
    assert ev(src) == want


def test_basic_unary_chains():
    assert ev("--3") == 3
    assert ev("-+-3") == 3
    assert ev("---3") == -3
    assert ev("+3") == 3
    assert ev("- - + - 3") == -3
    assert ev("2--3") == 5
    assert ev("2-+-3") == 5
    assert ev("-(-(-(1)))") == -1


def test_basic_whitespace():
    assert ev("  1 +\t2 *\n3\r ") == 7
    assert ev("max ( 1 , 2 )") == 2
    assert ev("abs (-4)") == 4
    assert ev("1<=2") == 1 and ev("1 <= 2") == 1


def test_basic_number_formats():
    for src, want in [("0", 0.0), ("42", 42.0), ("3.25", 3.25), ("5.", 5.0), (".5", 0.5), ("1e3", 1000.0),
                      ("1E3", 1000.0), ("1.5e-3", 0.0015), ("2.5E+2", 250.0), (".5e1", 5.0), ("007", 7.0),
                      ("0.1", 0.1), ("123456789012345678901234567890", 1.2345678901234568e+29), ("5.e1", 50.0)]:
        got = ev(src)
        assert got == want, src


def test_basic_result_is_float():
    for src in ["1", "2+3", "1<2", "max(1, 2)", "if(1, 2, 3)", "abs(-3)", "round(2.4)", "x"]:
        assert type(ev(src, {"x": 3})) is float, src


def test_basic_comparisons():
    table = {"<": [1, 0, 0], "<=": [1, 1, 0], ">": [0, 0, 1], ">=": [0, 1, 1], "==": [0, 1, 0], "!=": [1, 0, 1]}
    for op, wants in table.items():
        for (a, b), want in zip([(1, 2), (2, 2), (3, 2)], wants):
            assert ev(f"{a} {op} {b}") == want, (a, op, b)


def test_basic_functions():
    assert ev("abs(-2.5)") == 2.5
    assert ev("sqrt(2)") == math.sqrt(2)
    assert ev("min(3)") == 3 and ev("min(3, -1, 2)") == -1
    assert ev("max(3)") == 3 and ev("max(3, -1, 7, 2)") == 7
    assert ev("max(1+1, 2*2, 3)") == 4
    assert ev("if(1, 10, 20)") == 10 and ev("if(0, 10, 20)") == 20 and ev("if(-0.5, 10, 20)") == 10
    assert ev("if(2 > 1, 1 < 2, 0)") == 1


def test_basic_round_semantics():
    assert ev("round(2.5)") == 2.0
    assert ev("round(3.5)") == 4.0
    assert ev("round(-2.5)") == -2.0
    assert ev("round(2.675, 2)") == round(2.675, 2)
    assert ev("round(1234.5678, 1)") == 1234.6
    assert ev("round(1250, -2)") == 1200.0
    assert ev("round(1351, -2)") == 1400.0
    assert ev("round(7.25, 1+0)") == 7.2


def test_basic_modulo_and_division():
    assert ev("-7 % 3") == 2
    assert ev("7 % -3") == -2
    assert ev("7.5 % 2") == 1.5
    assert ev("7 / 2") == 3.5
    assert ev("1/3") == 1 / 3


def test_basic_if_is_lazy():
    assert ev("if(1, 2, 1/0)") == 2
    assert ev("if(0, 1/0, 3)") == 3
    assert ev("if(1 < 2, 5, sqrt(-1))") == 5
    assert ev("if(0, missing, 4)") == 4
    assert ev("if(1, 4, missing_fn_arg + 1)") == 4
    assert ev("if(1, if(0, 1/0, 7), 1/0)") == 7


def test_basic_variables():
    e = S.compile_expr("rate * hours + bonus_2 - rate")
    assert e.variables == frozenset({"rate", "hours", "bonus_2"})
    assert isinstance(e.variables, frozenset)
    assert e.evaluate({"rate": 10, "hours": 3, "bonus_2": 5.5}) == 25.5
    assert e.evaluate({"rate": 1, "hours": 1, "bonus_2": 0, "unused": 99}) == 0
    assert S.compile_expr("1 + 2").variables == frozenset()
    assert S.compile_expr("if(a, b, c) + max(d)").variables == frozenset("abcd")
    assert S.compile_expr("Rate + rate").variables == frozenset({"Rate", "rate"})


def test_basic_compile_once_evaluate_many():
    e = S.compile_expr("if(x > 0, sqrt(x), -x) * 2")
    assert [e.evaluate({"x": v}) for v in (4, 0, -3, 9)] == [4, 0, 6, 6]
    assert e.source == "if(x > 0, sqrt(x), -x) * 2"


def test_basic_function_names_without_call_are_variables():
    assert ev("min + max", {"min": 1, "max": 2}) == 3
    assert ev("if + 1", {"if": 4}) == 5
    assert S.compile_expr("abs * 2").variables == frozenset({"abs"})


def test_basic_variables_none_and_any_mapping():
    assert S.compile_expr("2*3").evaluate() == 6
    assert S.compile_expr("2*3").evaluate(None) == 6
    assert ev("a+b", types.MappingProxyType({"a": 1, "b": 2})) == 3


# ------------------------------------------------------------------------------------------------ edge
SYNTAX_ERRORS = [
    ("", 0), ("   ", 3), ("1 +", 3), ("(1+2", 4), ("1+2)", 3), ("()", 1), ("* 2", 0), ("1 2", 2), ("2x", 1),
    ("1 < 2 < 3", 6), ("a == b != c", 7), ("1 <= 2 > 0", 7), ("max(1,)", 6), ("max(,1)", 4), ("1 $ 2", 2),
    ("1e", 0), ("3 + 1e+", 4), (".", 0), ("1..2", 2), ("1 = 2", 2), ("!1", 0), ("abs 1", 4), ("(1))", 3),
    ("1 ^", 3), ("-", 1), ("1 2 $", 2), ("(1 < 2", 6), ("1 <> 2", 3), ("1 =< 2", 2), ("<= 1", 0), ("1,2", 1),
    ("abs(1", 5), ("sqrt(1 2)", 7), ("x y", 2), ("2 ^ * 3", 4), ("a <", 3), ("((1)", 4), ("1 + é", 4),
    ("\u0661", 0), ("1 ! 2", 2), ("2 (3)", 2), ("(", 1), (")", 0), ("1 + + ", 6), ("max(1,,2)", 6),
    ("3.4.5", 3), ("1e+x", 0), ("abs(1)(2)", 6), ("if(1, 2, 3", 10), ("1_000", 1), ("2 ** 3", 3),
    ("x[1]", 1), ("1 + 'a'", 4), ("foo(1) +", 8), ("4 % ", 4), ("1e999", 0), ("2 + 1e400 * 0", 4),
]


@pytest.mark.parametrize("src,pos", SYNTAX_ERRORS)
def test_edge_syntax_error_positions(src, pos):
    assert syntax_pos(src) == pos


ARITY_ERRORS = [
    ("abs()", 0), ("abs(1,2)", 0), ("1 + round(1,2,3)", 4), ("if(1,2)", 0), ("min()", 0), ("max()", 0),
    ("sqrt(4, 1)", 0), ("max(1, sqrt())", 7), ("2*if(1,2,3,4)", 2), ("round()", 0),
]


@pytest.mark.parametrize("src,pos", ARITY_ERRORS)
def test_edge_arity_is_compile_time_syntax_error(src, pos):
    assert syntax_pos(src) == pos


def test_edge_error_classes():
    assert issubclass(S.ExprSyntaxError, S.ExprError)
    assert issubclass(S.ExprNameError, S.ExprError)
    assert issubclass(S.ExprEvalError, S.ExprError)
    assert issubclass(S.ExprError, Exception)
    e = S.ExprSyntaxError("boom", 7)
    assert e.message == "boom" and e.position == 7
    with pytest.raises(S.ExprSyntaxError) as ei:
        S.compile_expr("1 +* 2")
    assert ei.value.position == 3 and isinstance(ei.value.message, str) and ei.value.message


def test_edge_unknown_function_at_compile_time():
    for src, name in [("foo(1)", "foo"), ("1 + bar()", "bar"), ("Abs(1)", "Abs"), ("if(1, 2, nope(3))", "nope")]:
        with pytest.raises(S.ExprNameError) as ei:
            S.compile_expr(src)
        assert ei.value.name == name


def test_edge_syntax_errors_take_precedence_over_call_errors():
    assert syntax_pos("foo(1) +") == 8
    assert syntax_pos("abs(1, 2) + foo(3)") == 0
    with pytest.raises(S.ExprNameError):
        S.compile_expr("foo(1) + abs(1, 2)")
    assert syntax_pos("max(1, 2) + abs(1, 2) + sqrt()") == 12


def test_edge_unknown_variable_at_evaluation():
    e = S.compile_expr("a + b")
    with pytest.raises(S.ExprNameError) as ei:
        e.evaluate({"a": 1})
    assert ei.value.name == "b"
    with pytest.raises(S.ExprNameError) as ei:
        e.evaluate()
    assert ei.value.name == "a"


@pytest.mark.parametrize("src", [
    "1/0", "5 % 0", "1 / (2 - 2)", "sqrt(-1)", "0^-1", "(-8)^(1/3)", "10^400", "1e308 * 10", "-1e308 - 1e308",
    "round(1.5, 0.5)", "x / y", "2^1024", "max(1e308 * 2, 1)", "if(1, 1/0, 2)", "(0-0)^-2",
])
def test_edge_evaluation_errors(src):
    e = S.compile_expr(src)  # compiling never evaluates
    with pytest.raises(S.ExprEvalError):
        e.evaluate({"x": 1, "y": 0})


def test_edge_evaluation_valid_corner_cases():
    assert ev("0^0") == 1
    assert ev("(-2)^3") == -8
    assert ev("(-8)^-1") == -0.125
    assert ev("0 % 5") == 0
    assert ev("sqrt(0)") == 0
    assert ev("1e308 + 1") == 1e308
    assert ev("4^0.5") == 2


def test_edge_errors_reported_in_evaluation_order():
    with pytest.raises(S.ExprNameError):
        ev("u + 1/0")
    with pytest.raises(S.ExprEvalError):
        ev("1/0 + u")
    with pytest.raises(S.ExprEvalError):
        ev("max(2, sqrt(-1), u)")
    with pytest.raises(S.ExprNameError):
        ev("min(u, sqrt(-1))")


def test_edge_non_finite_variable_value():
    e = S.compile_expr("x + 1")
    for bad in (math.inf, -math.inf, math.nan):
        with pytest.raises(S.ExprEvalError):
            e.evaluate({"x": bad})
    assert S.compile_expr("if(0, x, 1)").evaluate({"x": math.inf}) == 1


def test_edge_comparison_chain_allowed_with_parentheses():
    assert ev("(1 < 2) < 3") == 1
    assert ev("1 < (2 < 3)") == 0
    assert ev("if(1 < 2, 3, 4) == 3") == 1
    assert ev("max(1 < 2, 0 == 0)") == 1


def test_edge_compile_rejects_non_string():
    with pytest.raises(TypeError):
        S.compile_expr(b"1+2")


def test_edge_no_python_eval_exec_or_ast():
    with open(S.__file__, encoding="utf-8") as f:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(f.read()).readline)
                if t.type in (tokenize.NAME, tokenize.OP)]
    for i, t in enumerate(toks):
        prev = toks[i - 1].string if i else ""
        nxt = toks[i + 1].string if i + 1 < len(toks) else ""
        if t.type == tokenize.NAME and t.string in ("eval", "exec", "compile", "__import__") \
                and nxt == "(" and prev not in (".", "def"):
            pytest.fail(f"uses builtin {t.string}() at line {t.start[0]}")
        if t.type == tokenize.NAME and t.string == "ast" and prev in ("import", "from"):
            pytest.fail("imports ast")


def test_edge_does_not_execute_python():
    for src in ["__import__('os')", "(1).__class__", "x.y", "[1]", "lambda: 1", "1 if 1 else 2", "a = 1"]:
        with pytest.raises(S.ExprError):
            ev(src, {"x": 1, "a": 1})


# ------------------------------------------------------------------------------------------- property
class RefEvalError(Exception):
    pass


class RefNameError(Exception):
    pass


class Fragile(Exception):
    pass


CMP = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
       "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
PREC = {"<": 1, "<=": 1, ">": 1, ">=": 1, "==": 1, "!=": 1, "+": 2, "-": 2, "*": 3, "/": 3, "%": 3, "^": 5}


def near(a, b):
    return a != b and abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def fin(x):
    if x - x != 0:
        raise RefEvalError("non-finite")
    if abs(x) > 1e9:
        raise Fragile()
    return x


def ref_eval(node, env):
    k = node[0]
    if k == "num":
        return node[1]
    if k == "var":
        if node[1] not in env:
            raise RefNameError(node[1])
        return float(env[node[1]])
    if k == "neg":
        return -ref_eval(node[1], env)
    if k == "pos":
        return ref_eval(node[1], env)
    if k == "bin":
        op = node[1]
        a = ref_eval(node[2], env)
        b = ref_eval(node[3], env)
        if op in CMP:
            if near(a, b):
                raise Fragile()
            return 1.0 if CMP[op](a, b) else 0.0
        if op == "+":
            return fin(a + b)
        if op == "-":
            return fin(a - b)
        if op == "*":
            return fin(a * b)
        if op == "/":
            if b == 0:
                raise RefEvalError("div0")
            if abs(b) < 1e-6:
                raise Fragile()
            return fin(a / b)
        if op == "%":
            if b == 0:
                raise RefEvalError("mod0")
            if abs(b) < 1e-6:
                raise Fragile()
            r = a % b
            if min(abs(r), abs(abs(b) - abs(r))) <= 1e-9 * max(1.0, abs(a)) and r != 0:
                raise Fragile()
            return fin(r)
        if op == "^":
            if a == 0 and b < 0:
                raise RefEvalError("zero to negative power")
            if a < 0 and not float(b).is_integer():
                raise RefEvalError("complex")
            try:
                return fin(a ** b)
            except OverflowError:
                raise RefEvalError("overflow")
        raise AssertionError(op)
    if k == "call":
        name, args = node[1], node[2]
        if name == "if":
            c = ref_eval(args[0], env)
            if c != 0 and abs(c) < 1e-9:
                raise Fragile()
            return ref_eval(args[1] if c != 0 else args[2], env)
        vals = [ref_eval(a, env) for a in args]
        if name == "abs":
            return abs(vals[0])
        if name == "sqrt":
            if vals[0] < 0:
                raise RefEvalError("sqrt")
            return math.sqrt(vals[0])
        if name == "min":
            return min(vals)
        if name == "max":
            return max(vals)
        if name == "round":
            nd = 0 if len(vals) == 1 else vals[1]
            if not float(nd).is_integer():
                raise RefEvalError("round digits")
            y = vals[0] * 10.0 ** nd
            if abs(abs(y - math.floor(y)) - 0.5) < 1e-6:
                raise Fragile()
            return float(round(vals[0], int(nd)))
    raise AssertionError(node)


NUM_TEXTS = {
    0.0: ["0", "0.0", "0.", ".0", "0e0", "0E+3"], 1.0: ["1", "1.0", "1.", "1e0", "0.1e1", "10E-1"],
    2.0: ["2", "2.0", "2e0", ".2e1", "20e-1"], 3.0: ["3", "3.", "3.0", "0.3E1"], 4.0: ["4", "4.00", "40e-1"],
    5.0: ["5", "5.", "0.5e1"], 7.0: ["7", "7.0"], 10.0: ["10", "1e1", "1E1", "10.0"], 12.0: ["12", "1.2e1"],
    0.5: ["0.5", ".5", "5e-1", "0.05E+1"], 2.25: ["2.25", "225e-2"], 0.1: ["0.1", ".1", "1e-1"],
    1.75: ["1.75", "17.5e-1"], 3.5: ["3.5", "35E-1"], 100.0: ["100", "1e2", "1.0e+2"], 0.125: ["0.125", ".125"],
}
VARS = {"x": 1.5, "y": -2.0, "rate_1": 0.25, "Z9": 7.0, "_t": 3.0}


def gen(rng, depth, var_names=tuple(VARS), allow_funcs=True):
    if depth <= 0 or rng.random() < 0.12 * (6 - depth):
        if rng.random() < 0.65:
            v = rng.choice(list(NUM_TEXTS))
            return ("num", v, rng.choice(NUM_TEXTS[v]))
        return ("var", rng.choice(var_names))
    r = rng.random()
    if r < 0.12:
        return ("neg", gen(rng, depth - 1, var_names, allow_funcs))
    if r < 0.15:
        return ("pos", gen(rng, depth - 1, var_names, allow_funcs))
    if r < 0.78 or not allow_funcs:
        op = rng.choice("+-*/%^+-*<") if rng.random() < 0.9 else rng.choice(["<=", ">", ">=", "==", "!="])
        if op == "<":
            op = rng.choice(["<", "<=", ">", ">=", "==", "!="])
        left = gen(rng, depth - 1, var_names, allow_funcs)
        if op == "^":
            e = rng.choice([0.0, 1.0, 2.0, 3.0])
            right = ("num", e, rng.choice(NUM_TEXTS[e]))
            if rng.random() < 0.25:
                e2 = rng.choice([0.0, 1.0, 2.0])
                right = ("bin", "^", ("num", e2, rng.choice(NUM_TEXTS[e2])), right)  # exponent chain a^b^c
            if rng.random() < 0.3:
                right = ("neg", right)
        else:
            right = gen(rng, depth - 1, var_names, allow_funcs)
        return ("bin", op, left, right)
    name = rng.choice(["abs", "min", "max", "sqrt", "round", "if", "if"])
    sub = lambda: gen(rng, depth - 1, var_names, allow_funcs)  # noqa: E731
    if name in ("abs", "sqrt"):
        args = [sub()]
        if name == "sqrt":
            args = [("call", "abs", [args[0]])]
    elif name in ("min", "max"):
        args = [sub() for _ in range(rng.randint(1, 3))]
    elif name == "round":
        args = [sub()]
        if rng.random() < 0.5:
            nd = rng.choice([0.0, 1.0, 2.0])
            args.append(("num", nd, rng.choice(NUM_TEXTS[nd])) if rng.random() < 0.7 else ("neg", ("num", 1.0, "1")))
    else:
        args = [sub(), sub(), sub()]
    return ("call", name, args)


def sp(rng):
    return rng.choice(["", "", " ", "  ", "\t", "\n"])


def render_full(node, rng):
    k = node[0]
    if k == "num":
        return node[2]
    if k == "var":
        return node[1]
    if k in ("neg", "pos"):
        return "(" + ("-" if k == "neg" else "+") + sp(rng) + render_full(node[1], rng) + ")"
    if k == "bin":
        return "(" + render_full(node[2], rng) + sp(rng) + node[1] + sp(rng) + render_full(node[3], rng) + ")"
    return node[1] + "(" + ("," + sp(rng)).join(render_full(a, rng) for a in node[2]) + ")"


def render_min(node, rng):
    """Render with the fewest parentheses the precedence table allows; returns (text, level)."""
    k = node[0]
    if k == "num":
        return node[2], 6
    if k == "var":
        return node[1], 6
    if k == "call":
        return node[1] + sp(rng) + "(" + sp(rng) + ("," + sp(rng)).join(render_min(a, rng)[0] for a in node[2]) + ")", 6

    def wrap(child, ok):
        t, lvl = render_min(child, rng)
        return t if ok(lvl) else "(" + sp(rng) + t + sp(rng) + ")"

    if k in ("neg", "pos"):
        return ("-" if k == "neg" else "+") + sp(rng) + wrap(node[1], lambda l: l >= 4), 4
    op = node[1]
    p = PREC[op]
    if op == "^":
        lt = wrap(node[2], lambda l: l == 6)
        rt = wrap(node[3], lambda l: l >= 4)
    elif p == 1:
        lt = wrap(node[2], lambda l: l > 1)
        rt = wrap(node[3], lambda l: l > 1)
    else:
        lt = wrap(node[2], lambda l: l >= p)
        rt = wrap(node[3], lambda l: l > p)
    return lt + sp(rng) + op + sp(rng) + rt, p


def check_case(src, node, env):
    try:
        want = ref_eval(node, env)
    except Fragile:
        return False
    except RefEvalError:
        with pytest.raises(S.ExprEvalError):
            S.evaluate(src, env)
        return True
    except RefNameError:
        with pytest.raises(S.ExprNameError):
            S.evaluate(src, env)
        return True
    got = S.evaluate(src, env)
    assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-6), (src, got, want)
    return True


def test_prop_fully_parenthesized_trees():
    rng = random.Random(1001)
    done = 0
    while done < 600:
        node = gen(rng, rng.randint(1, 6))
        done += check_case(render_full(node, rng), node, VARS)


def test_prop_minimal_parentheses_trees():
    rng = random.Random(2002)
    done = 0
    while done < 800:
        node = gen(rng, rng.randint(2, 7))
        src = render_min(node, rng)[0]
        done += check_case(src, node, VARS)


def test_prop_minimal_rendering_without_whitespace_matches_full():
    rng = random.Random(3003)
    done = 0
    while done < 400:
        node = gen(rng, rng.randint(2, 6))
        try:
            want = ref_eval(node, VARS)
        except (Fragile, RefEvalError, RefNameError):
            continue
        quiet = random.Random(0)
        quiet.choice = lambda seq: ""  # no whitespace at all
        a = S.evaluate(render_min(node, quiet)[0], VARS)
        b = S.evaluate(render_full(node, rng), VARS)
        assert math.isclose(a, want, rel_tol=1e-9, abs_tol=1e-6)
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)
        done += 1


def test_prop_errors_and_evaluation_order():
    rng = random.Random(4004)
    done = 0
    while done < 400:
        node = gen(rng, rng.randint(2, 5), var_names=("x", "y", "missing", "x"), allow_funcs=rng.random() < 0.5)
        env = {"x": 0.0, "y": 2.0}
        src = render_min(node, rng)[0] if rng.random() < 0.5 else render_full(node, rng)
        done += check_case(src, node, env)


def tree_vars(node, acc):
    if node[0] == "var":
        acc.add(node[1])
    elif node[0] in ("neg", "pos"):
        tree_vars(node[1], acc)
    elif node[0] == "bin":
        tree_vars(node[2], acc)
        tree_vars(node[3], acc)
    elif node[0] == "call":
        for a in node[2]:
            tree_vars(a, acc)
    return acc


def test_prop_variables_set():
    rng = random.Random(5005)
    for _ in range(300):
        node = gen(rng, rng.randint(1, 6))
        e = S.compile_expr(render_min(node, rng)[0])
        assert e.variables == frozenset(tree_vars(node, set()))


def test_prop_compiled_expr_reused_with_random_variables():
    rng = random.Random(6006)
    cases = 0
    for _ in range(40):
        node = gen(rng, rng.randint(2, 6))
        e = S.compile_expr(render_min(node, rng)[0])
        for _ in range(10):
            env = {k: rng.choice([-3.0, -0.5, 0.0, 0.25, 1.0, 2.0, 4.5]) for k in VARS}
            try:
                want = ref_eval(node, env)
            except Fragile:
                continue
            except RefEvalError:
                with pytest.raises(S.ExprEvalError):
                    e.evaluate(env)
                cases += 1
                continue
            assert math.isclose(e.evaluate(env), want, rel_tol=1e-9, abs_tol=1e-6)
            cases += 1
    assert cases >= 200


# --------------------------------------------------------------------------------------------- stress
def test_stress_deep_parentheses():
    assert ev("(" * 2000 + "1" + ")" * 2000) == 1
    assert ev("(" * 100_000 + "x+1" + ")" * 100_000, {"x": 1}) == 2
    assert syntax_pos("(" * 2000 + "1" + ")" * 1999) == 4000


def test_stress_deep_unary_and_power_chains():
    assert ev("-" * 100_001 + "2") == -2
    assert ev("-+" * 50_000 + "3") == 3
    assert ev("^".join(["1"] * 20_000)) == 1
    assert ev("2^" + "^".join(["1"] * 20_000)) == 2
    assert ev("(" * 3000 + "1" + ")^2" * 3000 + " - 2") == -1


def test_stress_deep_function_nesting():
    assert ev("abs(" * 2000 + "-1" + ")" * 2000) == 1
    assert ev("max(1, " * 2000 + "2" + ")" * 2000) == 2
    assert ev("if(1, " * 2000 + "5" + ", 1/0)" * 2000) == 5
    assert ev("if(0, 1/0, " * 20_000 + "6" + ")" * 20_000) == 6


def test_stress_deep_nesting_compiles_and_evaluates_with_low_recursion_limit():
    depth, f = 0, sys._getframe()
    while f is not None:
        depth, f = depth + 1, f.f_back
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(depth + 150)
    try:
        e = S.compile_expr("(" * 5000 + "-x" + ")*1" * 5000)
        assert e.evaluate({"x": 0.5}) == -0.5
        e = S.compile_expr("-(" * 5000 + "x" + "+1)" * 5000)
        assert e.evaluate({"x": 0}) == 0
        assert e.evaluate({"x": 3}) == 3
    finally:
        sys.setrecursionlimit(old)


def test_stress_flat_100k_terms():
    src = " + ".join(f"{i % 7}*x" for i in range(100_000))
    e = S.compile_expr(src)
    assert e.evaluate({"x": 2}) == 2 * sum(i % 7 for i in range(100_000))
    src = "-".join(["1"] * 100_000)
    assert ev(src) == 1 - 99_999


def test_stress_evaluate_compiled_expr_100k_times():
    e = S.compile_expr("if(qty > 10, price * qty * (1 - discount), price * qty) + max(0, fee - credit) ^ 2 / 4")
    total = 0.0
    want = 0.0
    for i in range(100_000):
        qty, price, discount, fee, credit = i % 23, (i % 101) / 4, (i % 5) / 10, i % 9, i % 4
        total += e.evaluate({"qty": qty, "price": price, "discount": discount, "fee": fee, "credit": credit})
        want += (price * qty * (1 - discount) if qty > 10 else price * qty) + max(0, fee - credit) ** 2 / 4
    assert math.isclose(total, want, rel_tol=1e-9)


# ------------------------------------------------------------------------------------------ concurrency
def test_conc_shared_compiled_expr_across_threads():
    src = " + ".join(f"(x * {i % 13} - y / {1 + i % 5}) ^ 2" for i in range(400))
    e = S.compile_expr(src)
    envs = [{"x": t * 0.5, "y": 3.0 - t} for t in range(8)]
    expected = [e.evaluate(env) for env in envs]
    errors = []
    barrier = threading.Barrier(8)

    def worker(t):
        barrier.wait()
        for _ in range(60):
            got = e.evaluate(envs[t])
            if got != expected[t]:
                errors.append((t, got, expected[t]))
                return

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors
    assert len(set(expected)) == 8
