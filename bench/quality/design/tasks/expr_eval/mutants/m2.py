"""Safe arithmetic expression engine for spreadsheet formulas.

``compile_expr`` parses the source once, with an iterative operator-precedence (shunting-yard) parser, into a
flat postfix program; ``Expr.evaluate`` runs that program on a private value stack. Neither step recurses per
nesting level, so arbitrarily deep input is handled in constant Python stack depth. ``if`` is compiled to
conditional jumps, which gives it lazy branch evaluation for free.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

__all__ = ["Expr", "ExprError", "ExprSyntaxError", "ExprNameError", "ExprEvalError", "compile_expr", "evaluate"]


class ExprError(Exception):
    """Base class of every error raised by this module."""


class ExprSyntaxError(ExprError):
    def __init__(self, message: str, position: int) -> None:
        super().__init__(f"{message} at position {position}")
        self.message = message
        self.position = position


class ExprNameError(ExprError):
    def __init__(self, name: str, message: str | None = None) -> None:
        super().__init__(message or f"unknown name {name!r}")
        self.name = name


class ExprEvalError(ExprError):
    pass


# --- opcodes --------------------------------------------------------------------------------------------
(CONST, VAR, NEG, ADD, SUB, MUL, DIV, MOD, POW, LT, LE, GT, GE, EQ, NE,
 ABS, SQRT, MIN, MAX, ROUND1, ROUND2, JUMP_IF_FALSE, JUMP) = range(23)

# binary operator token -> (opcode, precedence, right_assoc)
_BINARY = {
    "<": (LT, 1, False), "<=": (LE, 1, False), ">": (GT, 1, False), ">=": (GE, 1, False),
    "==": (EQ, 1, False), "!=": (NE, 1, False),
    "+": (ADD, 2, False), "-": (SUB, 2, False),
    "*": (MUL, 3, False), "/": (DIV, 3, False), "%": (MOD, 3, False),
    "^": (POW, 5, True),
}
_CMP_PREC = 1
_UNARY_PREC = 6

# function name -> (min args, max args)
_FUNCTIONS = {"abs": (1, 1), "sqrt": (1, 1), "min": (1, None), "max": (1, None), "round": (1, 2), "if": (3, 3)}

# --- lexer ------------------------------------------------------------------------------------------------
_NUM, _IDENT, _OP, _LPAREN, _RPAREN, _COMMA, _END = range(7)
_TOKEN_RE = re.compile(
    r"(?P<num>(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?P<exp>[eE][+-]?[0-9]*)?)"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op><=|>=|==|!=|[-+*/%^<>])"
    r"|(?P<punct>[(),])"
)
_WS_RE = re.compile(r"[ \t\n\r]*")
_PUNCT = {"(": _LPAREN, ")": _RPAREN, ",": _COMMA}


class _Lexer:
    __slots__ = ("src", "pos", "n")

    def __init__(self, src: str) -> None:
        self.src = src
        self.pos = 0
        self.n = len(src)

    def next(self):
        """Return (kind, value, start)."""
        src = self.src
        pos = _WS_RE.match(src, self.pos).end()
        if pos >= self.n:
            self.pos = pos
            return _END, None, pos
        m = _TOKEN_RE.match(src, pos)
        if m is None:
            raise ExprSyntaxError(f"invalid character {src[pos]!r}", pos)
        self.pos = m.end()
        kind = m.lastgroup
        if kind == "exp":  # lastgroup reports the innermost group that matched last
            kind = "num"
        text = m.group()
        if kind == "num":
            exp = m.group("exp")
            if exp is not None and not exp[-1].isdigit():
                raise ExprSyntaxError("malformed number", pos)
            value = float(text)
            if value == math.inf:
                raise ExprSyntaxError("number out of range", pos)
            return _NUM, value, pos
        if kind == "ident":
            return _IDENT, text, pos
        if kind == "op":
            return _OP, text, pos
        return _PUNCT[text], text, pos

    def next_is_lparen(self) -> bool:
        pos = _WS_RE.match(self.src, self.pos).end()
        return pos < self.n and self.src[pos] == "("


class _Group:
    """A nesting level: the whole input, a parenthesis or the argument list of a call."""
    __slots__ = ("base", "func", "pos", "argc", "cmp_seen", "jumps")

    def __init__(self, base: int, func: str | None = None, pos: int = 0) -> None:
        self.base = base          # opstack height when the group was opened
        self.func = func          # function name for call groups, "" for a parenthesis, None for the top level
        self.pos = pos            # position of the function name / '('
        self.argc = 0
        self.cmp_seen = False
        self.jumps: list[int] = []  # code indexes of the jumps emitted for if()


def _parse(src: str):
    lex = _Lexer(src)
    code: list[tuple[int, object]] = []
    names: set[str] = set()
    ops: list[tuple[int, int, bool]] = []      # (opcode, precedence, right_assoc); opcode None for unary plus
    groups = [_Group(0)]
    semantic_error: tuple[int, ExprError] | None = None
    expect_operand = True
    just_opened_call = False

    def reduce_to(base: int, prec: int, right_assoc: bool) -> None:
        while len(ops) > base:
            top = ops[-1]
            if top[1] > prec or (top[1] == prec and not right_assoc):
                ops.pop()
                if top[0] is not None:
                    code.append((top[0], None))
            else:
                break

    while True:
        kind, value, pos = lex.next()
        if expect_operand:
            if kind == _NUM:
                code.append((CONST, value))
                expect_operand = False
            elif kind == _IDENT:
                if lex.next_is_lparen():
                    lex.next()  # the '('
                    groups.append(_Group(len(ops), value, pos))
                    just_opened_call = True
                    continue
                names.add(value)
                code.append((VAR, value))
                expect_operand = False
            elif kind == _OP and value in ("+", "-"):
                ops.append((NEG if value == "-" else None, _UNARY_PREC, True))
            elif kind == _LPAREN:
                groups.append(_Group(len(ops), "", pos))
            elif kind == _RPAREN and just_opened_call:
                g = groups[-1]
                semantic_error = _finish_call(g, code, semantic_error)
                groups.pop()
                expect_operand = False
            else:
                raise ExprSyntaxError("expected a number, name, '(' or unary operator", pos)
            just_opened_call = False
            continue

        # expecting an operator
        g = groups[-1]
        if kind == _OP:
            opcode, prec, right = _BINARY[value]
            reduce_to(g.base, prec, right)
            if prec == _CMP_PREC:
                if g.cmp_seen:
                    raise ExprSyntaxError("comparison operators cannot be chained", pos)
                g.cmp_seen = True
            ops.append((opcode, prec, right))
            expect_operand = True
        elif kind == _COMMA:
            if not g.func:
                raise ExprSyntaxError("unexpected ','", pos)
            reduce_to(g.base, 0, False)
            g.argc += 1
            g.cmp_seen = False
            if g.func == "if":
                if g.argc == 1:
                    g.jumps.append(len(code))
                    code.append((JUMP_IF_FALSE, None))
                elif g.argc == 2:
                    g.jumps.append(len(code))
                    code.append((JUMP, None))
                    code[g.jumps[0]] = (JUMP_IF_FALSE, len(code))
            expect_operand = True
        elif kind == _RPAREN:
            if g.func is None:
                raise ExprSyntaxError("unbalanced ')'", pos)
            reduce_to(g.base, 0, False)
            groups.pop()
            if g.func:
                g.argc += 1
                semantic_error = _finish_call(g, code, semantic_error)
        elif kind == _END:
            if len(groups) > 1:
                raise ExprSyntaxError("missing ')'", pos)
            reduce_to(0, 0, False)
            break
        else:
            raise ExprSyntaxError("expected an operator", pos)

    if semantic_error is not None:
        raise semantic_error[1]
    return code, frozenset(names)


def _finish_call(g: _Group, code: list, semantic_error):
    """Emit the call instruction (or patch if()'s jumps); remember the leftmost name/arity problem."""
    err = None
    spec = _FUNCTIONS.get(g.func)
    if spec is None:
        err = ExprNameError(g.func, f"unknown function {g.func!r}")
    else:
        lo, hi = spec
        if g.argc < lo or (hi is not None and g.argc > hi):
            err = ExprSyntaxError(f"wrong number of arguments for {g.func}()", g.pos)
    if err is not None:
        if semantic_error is None or g.pos < semantic_error[0]:
            return (g.pos, err)
        return semantic_error
    f = g.func
    if f == "if":
        code[g.jumps[1]] = (JUMP, len(code))
    elif f == "abs":
        code.append((ABS, None))
    elif f == "sqrt":
        code.append((SQRT, None))
    elif f == "min":
        code.append((MIN, g.argc))
    elif f == "max":
        code.append((MAX, g.argc))
    elif f == "round":
        code.append((ROUND1 if g.argc == 1 else ROUND2, None))
    return semantic_error


class Expr:
    """A compiled expression. Immutable; ``evaluate`` may be called concurrently from several threads."""

    __slots__ = ("_source", "_code", "_variables")

    def __init__(self, source: str, code: list, variables: frozenset[str]) -> None:
        self._source = source
        self._code = tuple(code)
        self._variables = variables

    @property
    def source(self) -> str:
        return self._source

    @property
    def variables(self) -> frozenset[str]:
        return self._variables

    def __repr__(self) -> str:
        return f"Expr({self._source!r})"

    def evaluate(self, variables: Mapping[str, float] | None = None) -> float:
        env = variables if variables is not None else {}
        code = self._code
        n = len(code)
        stack: list[float] = []
        push = stack.append
        pop = stack.pop
        pc = 0
        while pc < n:
            op, arg = code[pc]
            pc += 1
            if op == CONST:
                push(arg)
                continue
            if op == VAR:
                try:
                    raw = env[arg]
                except KeyError:
                    raise ExprNameError(arg, f"unknown variable {arg!r}") from None
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    raise ExprEvalError(f"variable {arg!r} is not a number") from None
                if v - v != 0.0:
                    raise ExprEvalError(f"variable {arg!r} is not finite")
                push(v)
                continue
            if op <= NE:  # NEG and the binary operators
                if op == NEG:
                    stack[-1] = -stack[-1]
                    continue
                b = pop()
                a = stack[-1]
                if op == ADD:
                    r = a + b
                elif op == SUB:
                    r = a - b
                elif op == MUL:
                    r = a * b
                elif op == DIV:
                    if b == 0.0:
                        raise ExprEvalError("division by zero")
                    r = a / b
                elif op == MOD:
                    if b == 0.0:
                        raise ExprEvalError("modulo by zero")
                    r = a % b
                elif op == POW:
                    r = _pow(a, b)
                elif op == LT:
                    r = 1.0 if a < b else 0.0
                elif op == LE:
                    r = 1.0 if a <= b else 0.0
                elif op == GT:
                    r = 1.0 if a > b else 0.0
                elif op == GE:
                    r = 1.0 if a >= b else 0.0
                elif op == EQ:
                    r = 1.0 if a == b else 0.0
                else:
                    r = 1.0 if a != b else 0.0
                if r - r != 0.0:
                    raise ExprEvalError("result is not finite")
                stack[-1] = r
                continue
            if op == JUMP_IF_FALSE:
                if pop() == 0.0:
                    pc = arg
                continue
            if op == JUMP:
                pc = arg
                continue
            if op == ABS:
                stack[-1] = abs(stack[-1])
            elif op == SQRT:
                x = stack[-1]
                if x < 0.0:
                    raise ExprEvalError("sqrt of a negative number")
                stack[-1] = math.sqrt(x)
            elif op == MIN or op == MAX:
                args = stack[-arg:]
                del stack[-arg:]
                push(min(args) if op == MIN else max(args))
            elif op == ROUND1:
                stack[-1] = float(round(stack[-1]))
            elif op == ROUND2:
                digits = pop()
                if not digits.is_integer():
                    raise ExprEvalError("round() digits must be an integer")
                stack[-1] = float(round(stack[-1], int(digits)))
        return stack[0]


def _pow(a: float, b: float) -> float:
    if a == 0.0 and b < 0.0:
        raise ExprEvalError("zero to a negative power")
    if a < 0.0 and not b.is_integer():
        raise ExprEvalError("negative number to a fractional power")
    try:
        return a ** b
    except OverflowError:
        raise ExprEvalError("result is not finite") from None


def compile_expr(src: str) -> Expr:
    """Parse ``src`` once into a reusable :class:`Expr`."""
    if not isinstance(src, str):
        raise TypeError("expression source must be str")
    code, names = _parse(src)
    return Expr(src, code, names)


def evaluate(src: str, variables: Mapping[str, float] | None = None) -> float:
    """Convenience: ``compile_expr(src).evaluate(variables)``."""
    return compile_expr(src).evaluate(variables)
