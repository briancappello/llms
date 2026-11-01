# Safe formula engine for spreadsheet cells

Our spreadsheet feature lets users type formulas such as `if(qty >= 10, price * 0.9, price) * qty`. Formulas come
from untrusted users, so they must never reach Python's `eval` — we need our own small expression engine. A
formula is compiled once when the cell is edited and then evaluated many times (once per row, on every
recalculation), so evaluation must be cheap and must not re-parse the source. Sheets imported from other tools
contain machine-generated formulas that can be enormous (100 000 terms) or nested thousands of levels deep;
those must work too.

Please implement it as module `solution` (Python 3.12, standard library only).

## Interface

```python
class ExprError(Exception): ...                       # base of every error below

class ExprSyntaxError(ExprError):
    def __init__(self, message: str, position: int): ...  # sets .message and .position

class ExprNameError(ExprError): ...                    # has .name: str (the unknown variable/function)
class ExprEvalError(ExprError): ...

class Expr:
    source: str                     # the source text it was compiled from
    variables: frozenset[str]       # every variable name referenced anywhere in the formula
    def evaluate(self, variables: Mapping[str, float] | None = None) -> float: ...

def compile_expr(src: str) -> Expr: ...
def evaluate(src: str, variables: Mapping[str, float] | None = None) -> float: ...  # compile_expr(src).evaluate(variables)
```

`compile_expr` raises `TypeError` if `src` is not a `str`. `source` and `variables` may be attributes or
properties. `variables=None` means no variables; any `Mapping` works and extra keys are ignored.

## Language

### Tokens

- Whitespace (space, `\t`, `\n`, `\r`) may appear between any two tokens and is ignored.
- Number: `DIGITS ['.' [DIGITS]] [EXP]` or `'.' DIGITS [EXP]`, where `DIGITS` is one or more ASCII digits and
  `EXP` is `e`/`E`, an optional `+`/`-`, and one or more digits. So `42`, `007`, `3.25`, `5.`, `.5`, `1e3`,
  `1.5E-3`, `.5e+1` are numbers. Numbers are read greedily: once the longest prefix matching the mantissa has been
  read, an immediately following `e`/`E` belongs to the number, and if it is not followed by a valid exponent
  (`1e`, `2e+`, `3ex`) the number is malformed. The value is what Python's `float()` gives for the token's text;
  a number whose value is infinite (`1e999`) is an error. There are no negative literals — `-` is an operator.
- Identifier: `[A-Za-z_][A-Za-z0-9_]*` (ASCII only, case-sensitive). An identifier immediately followed by `(`
  (whitespace in between allowed) is a function call; otherwise it is a variable — so `min + 1` is legal and
  refers to a variable named `min`.
- Operators and punctuation: `+ - * / % ^ < <= > >= == != ( ) ,`. Anything else (`=`, `!`, `**` as one token,
  `[`, `'`, `$`, non-ASCII letters or digits, ...) is invalid. `**` is two `*` tokens and is a syntax error.

### Grammar and precedence

```
expr    := sum [ cmp_op sum ]                 cmp_op := '<' | '<=' | '>' | '>=' | '==' | '!='
sum     := term { ('+' | '-') term }
term    := unary { ('*' | '/' | '%') unary }
unary   := ('+' | '-') unary | power
power   := primary [ '^' unary ]
primary := NUMBER | IDENT | IDENT '(' [ expr { ',' expr } ] ')' | '(' expr ')'
```

From loosest to tightest binding:

| level | operators | associativity |
|---|---|---|
| 1 | `<` `<=` `>` `>=` `==` `!=` | non-associative: `a < b < c` and `a == b != c` are syntax errors; `(a < b) < c` is fine |
| 2 | binary `+` `-` | left |
| 3 | `*` `/` `%` | left |
| 4 | unary `+` `-` | prefix, may repeat: `--3 == 3`, `-+-3 == 3`, `2*-3 == -6`, `2--3 == 5` |
| 5 | `^` (power) | right: `2^3^2 == 512` |

`^` binds tighter than unary minus on its left but its exponent may itself start with a unary operator:
`-2^2 == -4`, `(-2)^2 == 4`, `2^-2 == 0.25`, `2^-1^2 == 2^(-(1^2)) == 0.5`, `-2^-2 == -0.25`. Comparisons may
appear anywhere an `expr` is allowed, including inside parentheses and function arguments.

### Semantics

All values are Python floats (IEEE doubles) and `evaluate` always returns a `float`.

- `+ - * /` are Python's float operators. `%` is Python's float `%` (the result has the sign of the divisor:
  `-7 % 3 == 2`, `7 % -3 == -2`). `x ^ y` is Python's `x ** y` on floats.
- Comparisons yield `1.0` (true) or `0.0` (false); `==`/`!=` compare exactly.
- Functions (a fixed table; argument counts are checked at compile time):

| function | args | result |
|---|---|---|
| `abs(x)` | 1 | absolute value |
| `sqrt(x)` | 1 | `math.sqrt(x)` |
| `min(x, ...)` / `max(x, ...)` | 1 or more | smallest / largest argument |
| `round(x)` / `round(x, n)` | 1 or 2 | `float(round(x))` / `float(round(x, int(n)))` — Python's rounding (half to even on the binary value), `n` may be negative |
| `if(c, a, b)` | 3 | `a` if `c != 0` else `b`. **Lazy**: only the chosen branch is evaluated, so `if(1, 2, 1/0) == 2` and `if(0, missing, 4) == 4` |

- Evaluation order is strictly left to right, operands before their operator, all arguments of
  `abs`/`sqrt`/`min`/`max`/`round` left to right before the call; for `if`, the condition first, then only the
  chosen branch. The first error in that order is the one raised (e.g. `u + 1/0` with `u` undefined raises
  `ExprNameError`; `1/0 + u` raises `ExprEvalError`).
- A variable is looked up in `variables` when it is evaluated; a missing name raises `ExprNameError` with
  `.name` set to it. Values are `int` or `float` and are converted with `float()`; a value that is not finite
  (inf/nan) raises `ExprEvalError` when that variable is read.
- `ExprEvalError` is raised for: division or modulo by zero; `sqrt` of a negative number; `0 ^ y` with `y < 0`;
  `x ^ y` with `x < 0` and a non-integer `y`; a `round` digit count that is not integer-valued; and any operator
  or function whose result is not finite (`10^400`, `1e308 * 10`, `-1e308 - 1e308`).
- `compile_expr` never evaluates anything: it never raises `ExprEvalError` (`compile_expr("1/0")` succeeds and
  the error comes from `evaluate`), and it must not raise `ExprNameError` for variables, which are only known at
  evaluation time.

### Compile-time errors

`compile_expr` raises:

1. `ExprSyntaxError` for anything not matching the grammar. `.position` is the 0-based index of the first
   character of the offending token: reading tokens left to right, the first token that cannot continue a valid
   expression, or `len(src)` if the input ends where more is required. An invalid character is a token of its
   own; a malformed or infinite number is reported at its first character. Because the input is processed left
   to right, the leftmost problem wins: `1 2 $` reports position 2 (the `2`), not the `$`. Examples:
   `"" → 0`, `"(1+2" → 4`, `"1+2)" → 3`, `"()" → 1`, `"1 < 2 < 3" → 6`, `"max(1,)" → 6`, `"2x" → 1`,
   `"1..2" → 2`, `"3 + 1e+" → 4`, `"1 = 2" → 2`, `"abs 1" → 4`. A call with an empty argument list (`max()`) is
   grammatical and is then subject to the argument-count check below.
2. Only if the whole input is grammatical: unknown function names raise `ExprNameError` (`.name` = the function
   name, e.g. `foo(1)` or `Abs(1)`), and calls with the wrong number of arguments raise `ExprSyntaxError` whose
   `.position` is the index of the function name (`"1 + round(1,2,3)" → 4`). If several calls are wrong, the one
   whose name starts leftmost is reported.

## Engineering requirements

- No Python code execution of any kind: do not use `eval`, `exec`, `compile` (the builtin), `ast` or
  `__import__`. Write your own tokenizer and parser.
- Compile once, evaluate many: `compile_expr` turns the source into an internal representation;
  `Expr.evaluate` must not tokenize or parse again. Evaluating one compiled 15-operator formula 100 000 times with
  different variables should take well under a second or two.
- Nesting depth is unlimited in practice: 100 000 nested parentheses, 100 000 chained unary minuses, thousands of
  nested function calls or `if`s, and a flat 100 000-term sum must all compile and evaluate (in about a second),
  without `RecursionError`, and also when the interpreter's recursion limit is low. Neither parsing nor evaluation
  may recurse once per nesting level; do not change the recursion limit.
- An `Expr` is immutable after compilation and `evaluate` must be safe to call concurrently from several threads
  on the same `Expr`.
