# Incremental JSON stream splitter for log ingestion

Our log-ingestion service receives JSON documents over raw TCP sockets. Producers just write JSON values one
after another — usually one object per line, but some omit the newline, some send bare heartbeat values
(`7`, `true`), and the socket hands us text in arbitrary pieces: a read can end in the middle of a string, in
the middle of an escape sequence such as `\u00e9`, between the two halves of a surrogate pair, or in the middle
of a number. We need a decoder that we feed text chunks as they arrive and that gives back every complete
top-level value as soon as it is known to be complete.

Please implement it as module `solution` (Python 3.12, standard library only).

## Interface

```python
class JSONStreamError(ValueError):
    offset: int          # absolute character offset in the stream where the problem was detected

class IncompleteJSONError(JSONStreamError): ...
class BufferLimitError(JSONStreamError): ...

class JSONStreamDecoder:
    def __init__(self, *, max_buffer: int = 10_000_000) -> None: ...
    def feed(self, chunk: str) -> list[Any]: ...
    def close(self) -> list[Any]: ...
    def pending(self) -> int: ...
```

All four names must be importable from `solution`. Each exception's constructor signature is up to you, but
every instance must have an integer `.offset` attribute. "Offset" always means the 0-based index of a character
in the whole stream (all chunks fed so far, concatenated), not in the current chunk.

### Grammar

The stream is a sequence of JSON values per RFC 8259, strictly: no `NaN`/`Infinity`, no comments, no trailing
commas, no single quotes, no leading `+` or leading zeros, no hex numbers, no byte-order mark. JSON whitespace is
exactly space, `\t`, `\n`, `\r`; any other character (e.g. U+00A0) is not whitespace. Nesting depth up to 1000
must be supported.

Inside strings: any character except `"`, `\` and U+0000–U+001F may appear unescaped (U+007F, U+2028, non-BMP
characters etc. are fine). Escapes are `\" \\ \/ \b \f \n \r \t` and `\uXXXX` with exactly four hex digits
(either case). `\u` escapes are checked only for syntax: a lone surrogate escape such as `"\ud800"` is accepted.

Between top-level values:

- Values may be separated by any amount of JSON whitespace, including none at all.
- Objects, arrays and strings are self-delimiting, so anything may follow them directly: `{}[]"x"1` is four
  values.
- A top-level number or literal (`true`, `false`, `null`) ends at the first character that cannot continue it.
  That character must be JSON whitespace or the start of an object, array or string (`{`, `[`, `"`); anything
  else is an error at that character. So `1[2]` and `true"x"` are two values each, `1 2` is two values, and
  `12` is one value, while `1-2`, `true1`, `nullnull`, `01` and `1,2` are errors at offset 1, 4, 4, 1 and 1
  respectively.
- Characters that cannot start a value (`]`, `}`, `,`, `:`, letters other than the first letters of `true`,
  `false`, `null`, ...) are errors where they appear.

### `feed(chunk)`

Consumes `chunk` (a `str`; anything else raises `TypeError`; `""` is allowed and does nothing) and returns, in
stream order, every top-level value that became complete with this chunk. A value is complete:

- for an object or array, with its closing bracket; for a string, with its closing quote;
- for a top-level number or literal, only when its terminating character (see above) has been fed. Because more
  digits might follow, a number or literal at the very end of what has been fed so far is not returned yet —
  `feed("12")` returns `[]` and a following `feed(" ")` returns `[12]`. It is returned by `close()` if the
  stream ends there. Inside containers this question does not arise (`[12` simply waits for more input).

Each value is decoded exactly as `json.loads` would decode that value's text: objects become `dict` (a
duplicated key keeps its last value, key order preserved), arrays `list`, integers `int` (arbitrarily large),
numbers with a fraction or exponent `float` (`-0.0` stays `-0.0`, `1E400` becomes `inf` just as `json.loads`
does), `\u` escapes decoded with surrogate pairs combined. You may use `json.loads` /
`json.JSONDecoder.raw_decode` to decode the text of a value once you know where it ends, but locating value
boundaries and validating the input must be done by your own incremental scanner.

### Errors

Invalid input raises `JSONStreamError` from the `feed()` call that delivers the offending character — the error
must not be deferred until a container closes or until `close()`. `.offset` is the index of the first character
at which the stream can no longer be the beginning of a valid stream, i.e. the smallest `i` such that no
continuation of `stream[:i + 1]` is valid. Examples: `[1,]` → 3; `[1.]` → 3 (the `]`, since `1.` could still
become `1.5`); `{"a":tru}` → 8; `"abc\x"` → 5; `[01]` → 2; `"a<newline>b"` → 2; `[1]x` → 3; `{"a" 1}` → 5.

Values completed by earlier `feed()` calls have already been returned and are unaffected. If `feed()` raises,
values completed earlier within that same chunk are lost with it. After any error (including the two subclasses
below), the decoder is unusable: every further `feed()` or `close()` raises `JSONStreamError`.

### `close()`

Signals end of stream. Returns the final top-level number or literal if one is still waiting for its terminator
(so the result is `[]` or a one-element list); trailing whitespace is fine. If a value is unfinished (an open
container or string, a partial escape, or a number/literal that is not yet valid such as `-`, `1.`, `1e+`,
`tru`), raises `IncompleteJSONError` with `.offset` equal to the total number of characters fed. After `close()`
(successful or not), further `feed()` or `close()` calls raise `JSONStreamError`.

### Size limit and `pending()`

`max_buffer` (an `int` ≥ 1, otherwise `ValueError`) caps the size of a single top-level value: the number of
characters from its first character to its last, inclusive (whitespace inside the value counts, whitespace
between values does not, and the terminator of a number/literal is not part of it). Any value longer than
`max_buffer` characters raises `BufferLimitError` from the `feed()` call that delivers its
(`max_buffer`+1)-th character — also if the whole value arrives in one chunk — with `.offset` equal to the
offset of that character (value start + `max_buffer`). A value of exactly `max_buffer` characters is fine, and
there is no limit on the total stream length or on how many values one chunk contains.

`pending()` returns the number of characters of the top-level value currently in progress, from its first
character up to the last character fed (0 when no value is in progress). A top-level number or literal that is
waiting for its terminator is in progress: after `feed("  12")`, `pending()` is 2.

### Performance

This runs on every connection, so total work must be linear in the number of characters fed, independent of
how the input is chunked. In particular, do not re-scan or re-copy the buffered text of a value on every
`feed()`: an 8 MB string value delivered in 64-character chunks, or hundreds of thousands of small values
delivered in one chunk, must each be processed in about a second, not minutes. A decoder is used by one
connection at a time; it need not be thread-safe.
