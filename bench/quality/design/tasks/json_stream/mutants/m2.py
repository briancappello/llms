"""Incremental splitter/decoder for a stream of concatenated JSON values.

The decoder runs a small push-down automaton over the characters as they arrive, so every character is
examined exactly once no matter how the stream is chunked. The text of the value in progress is kept as a
list of chunk slices and joined once, when the value is complete; the complete, already validated text is
then decoded with ``json.loads``.
"""
from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["JSONStreamDecoder", "JSONStreamError", "IncompleteJSONError", "BufferLimitError"]


class JSONStreamError(ValueError):
    """Invalid JSON in the stream. ``offset`` is the absolute character offset where it was detected."""

    def __init__(self, message: str, offset: int) -> None:
        super().__init__(f"{message} (at offset {offset})")
        self.msg = message
        self.offset = offset


class IncompleteJSONError(JSONStreamError):
    """close() was called while a value was still incomplete."""


class BufferLimitError(JSONStreamError):
    """A single top-level value grew beyond ``max_buffer`` characters."""


# --- automaton states -------------------------------------------------------------------------------------
_TOP = 0          # between top-level values
_VALUE = 1        # inside a container, a value is required (after ',' in an array or ':' in an object)
_ARR_START = 2    # just after '[': a value or ']'
_OBJ_START = 3    # just after '{': a key or '}'
_OBJ_KEY = 4      # after ',' in an object: a key
_COLON = 5        # after a key: ':'
_AFTER = 6        # after a value inside a container: ',' or the closing bracket
_STR = 7          # inside a string
_ESC = 8          # after a backslash in a string
_UHEX = 9         # inside the 4 hex digits of a \u escape
_LIT = 10         # inside true / false / null
_N_MINUS = 11     # number: after '-'
_N_ZERO = 12      # number: leading '0'             (may end here)
_N_INT = 13       # number: integer digits          (may end here)
_N_DOT = 14       # number: after '.'
_N_FRAC = 15      # number: fraction digits         (may end here)
_N_E = 16         # number: after 'e' / 'E'
_N_ESIGN = 17     # number: after the exponent sign
_N_EXP = 18       # number: exponent digits         (may end here)
_SCALAR_DONE = 19  # a top-level literal is complete and waits for its terminator

_WS = frozenset(" \t\n\r")
_HEX = frozenset("0123456789abcdefABCDEF")
_SIMPLE_ESC = frozenset('"\\/bfnrt')
_DIGIT = frozenset("0123456789")
_TOP_TERMINATORS = frozenset(" \t\n\r{[\"")
_LITERALS = {"t": "true", "f": "false", "n": "null"}

_STR_RUN = re.compile(r'[^"\\\x00-\x1f]*').match
_WS_RUN = re.compile(r"[ \t\n\r]*").match
_DIGIT_RUN = re.compile(r"[0-9]*").match


class JSONStreamDecoder:
    """Split a stream of concatenated JSON values that arrives in arbitrary chunks.

    Not thread-safe: one decoder serves one stream.
    """

    def __init__(self, *, max_buffer: int = 10_000_000) -> None:
        if isinstance(max_buffer, bool) or not isinstance(max_buffer, int):
            raise TypeError("max_buffer must be an int")
        if max_buffer < 1:
            raise ValueError("max_buffer must be >= 1")
        self._max = max_buffer
        self._state = _TOP
        self._stack: list[str] = []      # open containers, '[' or '{'
        self._str_is_key = False
        self._hex_left = 0
        self._lit = ""                   # remaining characters of the literal being read
        self._offset = 0                 # absolute offset of the next character to be fed
        self._vstart = -1                # absolute offset of the value in progress, -1 if none
        self._parts: list[str] = []      # text of the value in progress from earlier chunks
        self._plen = 0
        self._error: JSONStreamError | None = None
        self._closed = False

    # --- public API ---------------------------------------------------------------------------------------
    def pending(self) -> int:
        """Characters buffered for the top-level value in progress (0 if none)."""
        return self._plen if self._vstart >= 0 else 0

    def feed(self, chunk: str) -> list[Any]:
        """Consume ``chunk``; return the top-level values completed by it, in order."""
        if not isinstance(chunk, str):
            raise TypeError("chunk must be str")
        self._check_usable()
        try:
            return self._scan(chunk)
        except JSONStreamError as e:
            self._error = e
            raise

    def close(self) -> list[Any]:
        """End of stream: return the final top-level number/literal if one was waiting for a terminator."""
        self._check_usable()
        self._closed = True
        state = self._state
        if state == _TOP:
            return []
        if not self._stack and state in (_N_ZERO, _N_INT, _N_FRAC, _N_EXP, _SCALAR_DONE):
            value = self._decode("".join(self._parts))
            self._reset_value()
            return [value]
        self._error = IncompleteJSONError("stream ended inside a value", self._offset)
        raise self._error

    # --- internals ----------------------------------------------------------------------------------------
    def _check_usable(self) -> None:
        if self._error is not None:
            raise JSONStreamError("decoder is unusable after an error: " + self._error.msg, self._error.offset)
        if self._closed:
            raise JSONStreamError("decoder is closed", self._offset)

    def _reset_value(self) -> None:
        self._vstart = -1
        self._parts = []
        self._plen = 0

    def _decode(self, text: str) -> Any:
        try:
            return json.loads(text)
        except (ValueError, RecursionError) as e:  # e.g. integers beyond Python's int-string limit
            raise JSONStreamError(f"cannot decode value: {e}", self._vstart) from None

    def _fail(self, offset: int, message: str) -> None:
        # A value that already overran the size limit before the bad character is reported as such.
        if self._vstart >= 0 and offset > self._vstart + self._max:
            raise BufferLimitError("value exceeds max_buffer", self._vstart + self._max)
        raise JSONStreamError(message, offset)

    def _scan(self, s: str) -> list[Any]:
        out: list[Any] = []
        n = len(s)
        base = self._offset
        stack = self._stack
        state = self._state
        i = 0
        vs = 0  # index in s where the in-progress value's text (in this chunk) starts

        def emit(end: int) -> None:
            # the top-level value spanning [vstart, base + end) is complete
            length = self._plen + end - vs
            if length > self._max:
                raise BufferLimitError("value exceeds max_buffer", self._vstart + self._max)
            text = s[vs:end]
            if self._parts:
                self._parts.append(text)
                text = "".join(self._parts)
            out.append(self._decode(text))
            self._reset_value()

        while i < n:
            c = s[i]

            # ---- strings (hot path) ----
            if state == _STR:
                j = _STR_RUN(s, i).end()
                if j == n:
                    i = n
                    break
                c = s[j]
                i = j + 1
                if c == '"':
                    if not stack:
                        emit(i)
                        state = _TOP
                    else:
                        state = _COLON if self._str_is_key else _AFTER
                elif c == "\\":
                    if i < n:
                        state = _ESC
                else:
                    self._fail(base + j, "control character in string")
                continue
            if state == _ESC:
                if c in _SIMPLE_ESC:
                    state = _STR
                elif c == "u":
                    state = _UHEX
                    self._hex_left = 4
                else:
                    self._fail(base + i, "invalid escape")
                i += 1
                continue
            if state == _UHEX:
                if c not in _HEX:
                    self._fail(base + i, "invalid \\u escape")
                i += 1
                self._hex_left -= 1
                if self._hex_left == 0:
                    state = _STR
                continue

            # ---- whitespace between tokens ----
            if c in _WS and state <= _AFTER:
                i = _WS_RUN(s, i).end()
                continue

            # ---- structure ----
            if state <= _VALUE or state == _ARR_START:
                if state == _ARR_START and c == "]":
                    stack.pop()
                    i += 1
                    if not stack:
                        emit(i)
                        state = _TOP
                    else:
                        state = _AFTER
                    continue
                if state == _TOP:
                    vs = i
                    self._vstart = base + i
                    self._plen = 0
                i += 1
                if c == '"':
                    state = _STR
                    self._str_is_key = False
                elif c == "{":
                    stack.append("{")
                    state = _OBJ_START
                elif c == "[":
                    stack.append("[")
                    state = _ARR_START
                elif c == "-":
                    state = _N_MINUS
                elif c == "0":
                    state = _N_ZERO
                elif c in _DIGIT:
                    state = _N_INT
                    i = _DIGIT_RUN(s, i).end()
                elif c in _LITERALS:
                    state = _LIT
                    self._lit = _LITERALS[c][1:]
                else:
                    if not stack:
                        self._reset_value()
                    self._fail(base + i - 1, f"unexpected character {c!r}")
                continue
            if state == _OBJ_START or state == _OBJ_KEY:
                if c == '"':
                    state = _STR
                    self._str_is_key = True
                elif c == "}" and state == _OBJ_START:
                    stack.pop()
                    if not stack:
                        emit(i + 1)
                        state = _TOP
                    else:
                        state = _AFTER
                else:
                    self._fail(base + i, "expected object key")
                i += 1
                continue
            if state == _COLON:
                if c != ":":
                    self._fail(base + i, "expected ':'")
                state = _VALUE
                i += 1
                continue
            if state == _AFTER:
                top = stack[-1]
                if c == ",":
                    state = _VALUE if top == "[" else _OBJ_KEY
                elif (c == "]" and top == "[") or (c == "}" and top == "{"):
                    stack.pop()
                    if not stack:
                        emit(i + 1)
                        state = _TOP
                    else:
                        state = _AFTER
                else:
                    self._fail(base + i, "expected ',' or closing bracket")
                i += 1
                continue

            # ---- literals ----
            if state == _LIT:
                if c != self._lit[0]:
                    self._fail(base + i, "invalid literal")
                self._lit = self._lit[1:]
                i += 1
                if not self._lit:
                    state = _SCALAR_DONE if not stack else _AFTER
                continue

            # ---- numbers ----
            if state == _N_INT or state == _N_FRAC or state == _N_EXP:
                if c in _DIGIT:
                    i = _DIGIT_RUN(s, i).end()
                    continue
                if state != _N_EXP:
                    if c == "." and state == _N_INT:
                        state = _N_DOT
                        i += 1
                        continue
                    if c == "e" or c == "E":
                        state = _N_E
                        i += 1
                        continue
                # fall through: the number ends before c
            elif state == _N_ZERO:
                if c == ".":
                    state = _N_DOT
                    i += 1
                    continue
                if c == "e" or c == "E":
                    state = _N_E
                    i += 1
                    continue
            elif state == _N_MINUS:
                if c == "0":
                    state = _N_ZERO
                elif c in _DIGIT:
                    state = _N_INT
                else:
                    self._fail(base + i, "invalid number")
                i += 1
                continue
            elif state == _N_DOT:
                if c not in _DIGIT:
                    self._fail(base + i, "invalid number")
                state = _N_FRAC
                i += 1
                continue
            elif state == _N_E:
                if c == "+" or c == "-":
                    state = _N_ESIGN
                elif c in _DIGIT:
                    state = _N_EXP
                else:
                    self._fail(base + i, "invalid number")
                i += 1
                continue
            elif state == _N_ESIGN:
                if c not in _DIGIT:
                    self._fail(base + i, "invalid number")
                state = _N_EXP
                i += 1
                continue

            # a number or literal ended right before c (c is not consumed here)
            if stack:
                state = _AFTER
                continue
            if c not in _TOP_TERMINATORS:
                self._fail(base + i, f"unexpected character {c!r} after number or literal")
            emit(i)
            state = _TOP

        # end of chunk: keep the text of an unfinished value
        self._state = state
        self._offset = base + n
        if self._vstart >= 0:
            if vs < n:
                self._parts.append(s[vs:])
                self._plen += n - vs
            if self._plen > self._max:
                raise BufferLimitError("value exceeds max_buffer", self._vstart + self._max)
        return out
