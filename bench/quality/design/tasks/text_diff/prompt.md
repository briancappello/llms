# Line-based diff / patch library

Our config-management service shows users what changed between two revisions of a file and stores the change
as a patch that is re-applied later. Today it uses `difflib`, which has two problems for us: `SequenceMatcher`
does not produce *minimal* diffs (reviewers complain about bloated diffs, and our patch storage grows), and its
behaviour on large files with many repeated lines (blank lines, `}` lines, ...) is hard to predict. We want our
own small, well-specified diff/patch module.

Please write a module `solution` (one file, `solution.py`, Python 3.12, standard library only). **Do not use
`difflib`** (not even as a fallback); implement the diff algorithm yourself.

Lines are plain `str` values compared with `==`. They do not contain their line terminators; you may assume no
line contains `"\n"`.

## Interface

```python
class PatchError(ValueError): ...

@dataclass(frozen=True)
class Op:
    tag: str            # "equal", "delete" or "insert"
    a_start: int
    a_end: int
    b_start: int
    b_end: int
    lines: tuple[str, ...]

def diff(a: Sequence[str], b: Sequence[str]) -> list[Op]: ...
def apply(a: Sequence[str], ops: Iterable[Op]) -> list[str]: ...
def unified(a: Sequence[str], b: Sequence[str], *, fromfile: str = "a", tofile: str = "b",
            context: int = 3) -> str: ...
```

`Op` must be a frozen dataclass with exactly these fields in this order (so `Op("equal", 0, 2, 0, 2, ("x", "y"))`
and keyword construction both work, and instances compare by value).

### `Op` meaning

An op describes what happens to the half-open ranges `a[a_start:a_end]` and `b[b_start:b_end]`:

- `"equal"`: `a[a_start:a_end] == b[b_start:b_end]`, both ranges have the same length, and `lines` is that
  common content.
- `"delete"`: `a[a_start:a_end]` is removed; `b_start == b_end` (the position in `b` where the removal happens);
  `lines == tuple(a[a_start:a_end])`.
- `"insert"`: `b[b_start:b_end]` is inserted; `a_start == a_end` (the position in `a` where the insertion
  happens); `lines == tuple(b[b_start:b_end])`.

### `diff(a, b)`

Returns the edit script turning `a` into `b`, with these guarantees:

1. **Contiguous cover.** The ops cover both sequences in order with no gaps or overlaps: the first op starts at
   `a_start == 0, b_start == 0`, each op starts where the previous one ended (in both `a` and `b`), and the last
   op ends at `len(a)` and `len(b)`. `diff([], [])` is `[]`.
2. **No empty ops** (every op has a non-empty `lines`), and **no two adjacent ops with the same tag** (they are
   merged).
3. **Delete before insert.** When a `"delete"` and an `"insert"` are adjacent, the `"delete"` comes first. (So
   between two `"equal"` ops, or at the start/end, there is at most one `"delete"` followed by at most one
   `"insert"`.)
4. **Minimal.** The total number of deleted plus inserted lines equals `len(a) + len(b) - 2 * L`, where `L` is
   the length of a longest common subsequence of `a` and `b`. When several minimal scripts exist, any one of
   them is acceptable.

The inputs are not modified. Any `Sequence[str]` (list, tuple, ...) must be accepted. `diff` is deterministic:
the same inputs always produce the same ops.

### `apply(a, ops)`

Applies an edit script to `a` and returns the resulting list of lines (a new list; `a` is not modified).
`apply(a, diff(a, b)) == list(b)` must always hold.

`apply` does not require ops to be merged, non-empty or to follow the delete-before-insert rule; it only checks
that they are consistent with `a`. Raise `PatchError` if any of the following holds:

- an op has a tag other than the three above;
- the ops are not contiguous in the sense of rule 1 (the first op must start at `a_start == 0` and
  `b_start == 0`; each op must start at the previous op's `a_end` / `b_end`), or after the last op the
  `a` position is not `len(a)` (for an empty `ops` list, `a` must be empty);
- an op's ranges are inconsistent with its tag and `lines` (for example `a_end < a_start`, an `"equal"` whose
  two ranges differ in length, a `"delete"` with `b_end != b_start`, an `"insert"` with `a_end != a_start`, or
  a range length different from `len(lines)`);
- an op reaches beyond the end of `a`;
- an `"equal"` or `"delete"` op's `lines` differ from the corresponding lines of `a`.

### `unified(a, b, *, fromfile="a", tofile="b", context=3)`

Renders `diff(a, b)` as unified-diff text, using exactly the same layout conventions as GNU `diff -u` /
`difflib.unified_diff` without timestamps:

- If `a` and `b` are equal (same lines), return `""`.
- `context` must be an `int >= 0`; a negative value raises `ValueError`.
- Otherwise the output consists of the lines below, **each terminated by `"\n"`** (so the result ends with
  `"\n"`), and nothing else (no `"\ No newline at end of file"` markers):
  - `--- {fromfile}`
  - `+++ {tofile}`
  - one or more hunks.
- **Hunks.** A *change* is a maximal run of non-`"equal"` ops of `diff(a, b)`. Each change is shown with up to
  `context` lines of the surrounding `"equal"` ops before and after it. Two consecutive changes go into the same
  hunk iff the `"equal"` op between them has at most `2 * context` lines (then that whole equal run is shown);
  otherwise they are in separate hunks. A hunk's leading context is the last `min(context, n)` lines of the
  `"equal"` op before its first change (`n` = that op's length, 0 if there is none), and its trailing context
  the first `min(context, n)` lines of the `"equal"` op after its last change.
- **Hunk header:** `@@ -{ra} +{rb} @@`, where `ra` describes the hunk's range in `a` and `rb` its range in `b`. A
  range with 0-based start `s` and length `n` is written as `"{s+1}"` if `n == 1`, `"{s},0"` if `n == 0` (i.e.
  the 1-based number of the line *after which* the empty range sits, `0` for the start of the file), and
  `"{s+1},{n}"` otherwise. Example: `@@ -0,0 +1,3 @@` for adding three lines to an empty file,
  `@@ -5 +5,2 @@`, `@@ -7,4 +6,3 @@`.
- **Hunk body**, in op order: context (`"equal"`) lines prefixed with a single space `" "`, deleted lines with
  `"-"`, inserted lines with `"+"`. Because of the delete-before-insert rule, within a change all `-` lines
  come before all `+` lines.

## Performance

Real files are large but revisions usually differ in only a few places. The diff must run in `O((N + M) * D)`
time (`N`, `M` the input lengths, `D` the size of the minimal edit script), e.g. Myers' algorithm; an
`O(N * M)` LCS table is far too slow and too big (two 20,000-line files differing in a few dozen lines must diff
in well under a second or two). Memory must not blow up with `D` either. Also make sure the degenerate case of
two completely unrelated files (e.g. 3,000 lines each with no line in common) finishes in a few seconds;
common prefixes/suffixes and lines that occur in only one of the two inputs are cheap to deal with up front.
Recursion, if you use it, must not hit Python's recursion limit on large inputs.
