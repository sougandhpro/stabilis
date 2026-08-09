"""Reaching files, and holding one in memory once it has been read.

Two jobs that both exist for speed or for safety, and nothing else.

**`_safe` is the whole security model.** The page reads files from the
machine it runs on, so every path from the browser goes through it. It
resolves `..` *before* the check, so a path that climbs out of the root is
refused rather than quietly rebased onto it -- the difference between
saying no and serving the wrong file.

**The table cache is what makes the page reactive.** Every interaction --
retyping a label, dragging a range, switching units -- used to re-read the
whole file, which on a 200 MB export is seconds of wall clock for a change
that touched no bytes on disk. Reading once and keying on
`(path, mtime_ns, size)` turns all of those into an array slice. The key
includes mtime and size rather than a hash because rehashing 200 MB to
decide whether to reuse it would cost more than the read it saves, and a
file that changed underneath us changes at least one of the two in every
case that is not a deliberate forgery.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["safe_path", "listing", "load_table", "column_facts", "timebase",
           "Table", "cache_clear", "cache_bytes", "DATA_SUFFIXES"]

# Extensions offered in the file browser. Not a security control -- `safe_path`
# is -- just a way to keep the listing readable in a directory full of
# unrelated files.
DATA_SUFFIXES = (".dat", ".txt", ".csv", ".tsv", ".phase", ".freq", "")

# What stays resident, budgeted in bytes rather than in files.
#
# A count was right when the page looked at one record at a time. A bench is
# the case that broke it: six measurements from six counters is six files,
# all of them wanted on every solve, and a four-slot cache evicts the one it
# is about to need -- so every solve re-read and re-parsed the lot. Counting
# files also mismeasures the thing actually at risk, since four 2 GB records
# is not the same commitment as four 2 MB ones.
#
# So: keep whatever fits under a byte budget, evicting least-recently-used,
# and always keep at least one table however large it is -- a single record
# bigger than the budget must still be usable, and refusing to cache it
# would mean re-reading it on every keystroke.
CACHE_BYTES = 512 << 20
MIN_CACHED = 1

_CACHE: "OrderedDict[tuple, Table]" = OrderedDict()
_LOCK = threading.Lock()


@dataclass(frozen=True)
class Table:
    """One file, read whole, with everything the page asks about it.

    `rows` is 2-D and read-only. It is handed to several requests at once
    -- `ThreadingHTTPServer` serves the page's parallel fetches on separate
    threads -- so making it immutable is what stops one request's editing
    from being another's input.
    """

    path: str
    labels: tuple[str, ...]
    rows: np.ndarray
    defects: str
    has_header: bool
    n: int
    n_columns: int
    # Whether `_read` dropped a phantom first row, below. Carried because a
    # printed recipe that reads the file back has to drop it too, and
    # "there was a header" is not the same condition as "a row was removed".
    header_row_dropped: bool = False


def safe_path(root: Path, candidate: str) -> Path:
    """Resolve `candidate` under `root`, or refuse."""
    target = (root / candidate).resolve() if candidate else root
    if target != root and root not in target.parents:
        raise PermissionError(f"{candidate!r} is outside {root}")
    return target


def listing(root: Path, rel: str) -> dict:
    """Directories and plausible data files at `rel`, for the picker."""
    here = safe_path(root, rel)
    if not here.is_dir():
        raise NotADirectoryError(str(here))
    dirs, files = [], []
    for entry in sorted(here.iterdir(), key=lambda p: p.name.lower()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            dirs.append(entry.name)
        elif entry.suffix.lower() in DATA_SUFFIXES:
            files.append({"name": entry.name, "size": entry.stat().st_size})
    parent = "" if here == root else str(here.parent.relative_to(root))
    return {"here": "" if here == root else str(here.relative_to(root)),
            "parent": parent, "at_root": here == root,
            "root": str(root), "dirs": dirs, "files": files}


def load_table(path: Path, delimiter: str | None = None) -> Table:
    """The file as columns, from cache when it has not changed on disk."""
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, delimiter)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit

    table = _read(path, delimiter)
    with _LOCK:
        _CACHE[key] = table
        _CACHE.move_to_end(key)
        _evict()
    return table


def _evict() -> None:
    """Drop least-recently-used tables until the budget is met.

    Caller holds `_LOCK`. `MIN_CACHED` is what stops a record larger than
    the whole budget from evicting itself the instant it arrives, which
    would turn every request touching it into a re-read.
    """
    total = sum(t.rows.nbytes for t in _CACHE.values())
    while total > CACHE_BYTES and len(_CACHE) > MIN_CACHED:
        _key, dropped = _CACHE.popitem(last=False)
        total -= dropped.rows.nbytes


def cache_bytes() -> int:
    """Bytes currently held. For the tests, and for answering "why is this big"."""
    with _LOCK:
        return sum(t.rows.nbytes for t in _CACHE.values())


def cache_clear() -> None:
    """Drop everything held. For tests, and for `--no-cache` reasoning."""
    with _LOCK:
        _CACHE.clear()


def _read(path: Path, delimiter: str | None) -> Table:
    """`io.text.read_table`, plus the questions the page needs answered.

    The library reader is used as-is rather than reimplemented: it already
    has the fast numpy path, the tolerant fallback for ragged rows, and the
    defect accounting, and a second parser here would be a second thing to
    keep in agreement with it.
    """
    from ..io.text import read_table

    labels, rows, defects = read_table(path, delimiter=delimiter)
    named = _has_header(labels)
    note = "" if defects.clean and not defects.text_columns else defects.describe()

    # An uncommented header line -- `t,NCS,VSL` with no leading `#` -- is
    # data as far as `np.loadtxt` is concerned, so the reader's fast path
    # declines the file, the tolerant path takes it, and the names come back
    # as one row of NaN sitting on top of the record. Left alone that is a
    # phantom first sample and a handful of phantom defects, and on phase
    # data it puts a gap exactly where the pedestal is measured.
    #
    # Dropping it needs both conditions and is safe under both: the reader
    # has said the file names its columns, *and* the first row parsed as
    # nothing at all. A genuine all-gap first sample in a headed file sits
    # at row 1, not row 0, so it survives.
    dropped = bool(named and rows.shape[0] and not np.isfinite(rows[0]).any())
    if dropped:
        rows = rows[1:]
        note = _without_header_row(note, rows.shape[1])

    rows = np.ascontiguousarray(rows, dtype=np.float64)
    rows.flags.writeable = False
    return Table(
        path=str(path),
        labels=tuple(str(name) for name in labels),
        rows=rows,
        defects=note,
        has_header=named,
        n=int(rows.shape[0]),
        n_columns=int(rows.shape[1]),
        header_row_dropped=dropped,
    )


def _without_header_row(note: str, width: int) -> str:
    """Stop reporting the header line as damage, without hiding real damage.

    The tolerated-defect count included the header's own fields. Discounting
    exactly `width` of them keeps the number meaningful: if it reaches zero
    the file was clean and says so, and if anything is left it is a real
    unreadable field somewhere further down and still gets reported.
    """
    import re

    def fix(match):
        left = int(match.group(1)) - width
        return "" if left <= 0 else f"{left} unreadable fields became gaps"

    out = re.sub(r"(\d+) unreadable fields became gaps", fix, note)
    out = out.replace("; (first at line", " (first at line").strip("; ")
    return "" if out.startswith("(first at line") else out


def _has_header(labels) -> bool:
    """Whether the file named its own columns.

    `io.text._header` falls back to exactly `column 0`, `column 1`, ... when
    the first line is data rather than names, so recognising that pattern
    recognises the fallback. Indirect, but it is the reader's own contract
    and keeps this from being a second header sniffer that could disagree
    with the first.
    """
    return list(labels) != [f"column {i}" for i in range(len(labels))]


def column_facts(table: Table) -> list[dict]:
    """Per column: how much of it is real, its span, and whether it ticks.

    The monotonic flag is what lets the page offer a time column instead of
    making the user find it. Computed on the first 4096 rows: a timebase
    that starts by increasing and stops later is a broken timebase, and the
    page says so when the column is actually selected rather than scanning
    two million rows to decorate a dropdown.
    """
    facts = []
    head = table.rows[:4096]
    for i in range(table.n_columns):
        column = table.rows[:, i]
        finite = np.isfinite(column)
        n_finite = int(finite.sum())
        probe = head[:, i]
        probe = probe[np.isfinite(probe)]
        rising = bool(probe.size > 2 and np.all(np.diff(probe) > 0))
        facts.append({
            "index": i,
            "label": table.labels[i],
            "n_finite": n_finite,
            "min": float(np.min(column[finite])) if n_finite else None,
            "max": float(np.max(column[finite])) if n_finite else None,
            "monotonic": rising,
            "text": n_finite == 0,
        })
    return facts


def timebase(table: Table, index: int, fmt: str | None):
    """`(rate_hz, jitter, note)` from a time column, numeric or text.

    The numeric column goes through `parse_times` rather than straight to
    `infer_rate`, even though it is already an array of floats and the
    shortcut is tempting. **A timestamp is not a number of seconds until
    something says which epoch it counts from.** An MJD column ticks by
    1/86400 per sample at 1 Hz, so handing it to `infer_rate` unconverted
    returns 86400 Hz and every tau in the report is wrong by that factor,
    plausibly and silently. `parse_times` classifies by the range the values
    actually occupy, converts, and refuses with `AmbiguousTimebase` when the
    evidence fits more than one format -- at which point the page's "time
    format" control is the answer, and guessing would not have been.

    Non-finite entries are dropped first: a gap in the timestamps is skipped
    rather than interpolated, which is what `infer_rate` documents.

    A column of ISO timestamps is entirely non-numeric, so `read_table`
    holds NaN for it and the text has to be re-read from disk. That costs a
    second pass over the file and happens only when such a column is chosen.
    """
    from ..io.timebase import infer_rate, parse_times

    column = table.rows[:, index]
    finite = np.isfinite(column)
    if finite.sum() > 2:
        seconds, _epoch, detected = parse_times(column[finite], fmt)
        rate, jitter = infer_rate(seconds)
        return float(rate), float(jitter), f"timetags read as {detected}"

    seconds, _epoch, detected = parse_times(_text_column(table, index), fmt)
    rate, jitter = infer_rate(seconds)
    return float(rate), float(jitter), f"timetags read as {detected}"


def _text_column(table: Table, index: int) -> list[str]:
    """One column as strings, for timestamps the numeric reader could not take.

    Splits the way `io.text` does -- comment markers stripped, then the
    first of comma, semicolon or tab that appears, else whitespace -- so a
    field lands in the same position here as it did there. Rows too short to
    reach the column contribute an empty string rather than shifting every
    later stamp up by one.
    """
    lines = []
    with open(table.path, "r") as fh:
        for raw in fh:
            line = raw.split("//", 1)[0]
            for char in "#%":
                line = line.split(char, 1)[0]
            line = line.strip()
            if line:
                lines.append(line)
    delimiter = next((d for d in (",", ";", "\t") if d in lines[0]), None)
    if len(lines) == table.n + 1:
        lines = lines[1:]           # the header the numeric reader skipped
    out = []
    for line in lines:
        fields = line.split(delimiter) if delimiter else line.split()
        out.append(fields[index].strip() if index < len(fields) else "")
    return out
