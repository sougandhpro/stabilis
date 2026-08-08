"""Generic delimited-text reader.

Handles the shape almost every counter and logging script emits: comment
lines, one to eight columns of comma-, semicolon-, tab- or
whitespace-separated numbers, an optional timetag column, and an optional
sentinel value marking gaps.

The sentinel is opt-in on purpose. Stable32 treats a zero as a gap marker in
frequency data, which is convenient right up until a genuine zero-offset
measurement is silently deleted -- so this reader will not assume it.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..log import record, warn
from ..provenance import file_sha256, new_provenance
from ..registry import register
from ..types import FreqSeries, PhaseSeries

__all__ = ["read_text", "load_phase", "load_frequency", "MAX_COLUMNS"]

MAX_COLUMNS = 8
_DELIMITERS = (",", ";", "\t")
_COMMENT_CHARS = "#%"

# Timetags rarely land on an exact grid. Warn when the spread of sample
# intervals exceeds this fraction of the median interval.
_TIMETAG_JITTER_WARN = 1e-3


@register("reader", "text")
def read_text(
    path: str | Path,
    column: int = 0,
    time_column: int | None = None,
    delimiter: str | None = None,
    comment_chars: str = _COMMENT_CHARS,
    gap_value: float | None = None,
    strict: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, "Defects"]:
    """Read one data column, and optionally a timetag column.

    Returns ``(values, times, defects)``, where `times` is None unless
    `time_column` was given. Values equal to `gap_value` become NaN.

    **The data column is the reference and everything aligns to it.** Real
    exports are not rectangular: a logger writes a timestamp every row but
    a reading only sometimes, a capture is truncated mid-line, a field is
    blank between two delimiters. Short rows are padded and unreadable
    fields become NaN -- which is already the gap representation, so the
    existing `gap_policy` machinery handles the consequences and nothing is
    invented.

    Nothing is tolerated silently: `defects` counts what was papered over
    and names the first offending lines. `strict=True` restores the older
    behaviour of refusing the file outright, for a report pipeline where a
    malformed input should stop the run rather than be repaired.
    """
    path = Path(path)
    rows, defects = _parse_rows(path, comment_chars, delimiter, strict)
    if rows.shape[1] > MAX_COLUMNS:
        raise ValueError(
            f"{path.name}: {rows.shape[1]} columns exceeds the {MAX_COLUMNS}-column limit"
        )
    if column >= rows.shape[1]:
        raise ValueError(
            f"{path.name}: requested column {column} but the file has {rows.shape[1]}"
        )

    values = rows[:, column].copy()
    if gap_value is not None:
        values[values == gap_value] = np.nan
    times = rows[:, time_column].copy() if time_column is not None else None
    return values, times, defects


# Fields that mean "no value here". Case-insensitive. A missing reading is
# a gap, and NaN is already this library's gap representation, so these
# convert rather than raise -- the gap machinery then handles the
# consequences honestly instead of the record being lost to one bad line.
_NULL_FIELDS = frozenset({"", "nan", "na", "n/a", "null", "none", "-", "--", "*", "?"})


class Defects(NamedTuple):
    """What had to be tolerated to read a file, so it can be reported.

    Accepting a malformed record silently is the failure this library
    guards against everywhere else, so the reader counts what it papered
    over and hands it back. `lines` holds the first few offending line
    numbers -- enough to go and look, not so many that the message becomes
    the output.
    """

    short_rows: int
    padded_to: int
    bad_fields: int
    lines: tuple[int, ...]

    @property
    def clean(self) -> bool:
        return self.short_rows == 0 and self.bad_fields == 0

    def describe(self) -> str:
        if self.clean:
            return "no defects"
        parts = []
        if self.short_rows:
            parts.append(f"{self.short_rows} short rows padded to "
                         f"{self.padded_to} columns")
        if self.bad_fields:
            parts.append(f"{self.bad_fields} unreadable fields became gaps")
        where = ", ".join(str(n) for n in self.lines)
        return "; ".join(parts) + (f" (first at line {where})" if where else "")


def _parse_rows(path: Path, comment_chars: str, delimiter: str | None,
                strict: bool = False) -> tuple[np.ndarray, Defects]:
    """Strip comments, sniff the delimiter, return a 2-D float array.

    Two paths. A clean file -- rectangular, all numeric -- goes through
    `np.loadtxt`, which is about five times faster than doing it here. A
    file that is not clean falls back to the tolerant parser below, which
    pads short rows and turns unreadable fields into gaps rather than
    losing the record to one bad line.

    The two must agree exactly on any file both can read, and a test
    asserts it: a fast path that disagrees with the careful one is worse
    than no fast path.
    """
    if delimiter is None:
        delimiter = _sniff_file(path, comment_chars)

    if not strict:
        # Fast path: hand the whole file to numpy. Five times quicker than
        # the loop below, and it is what runs on every well-formed export.
        # Anything it cannot do -- ragged rows, a non-numeric field -- it
        # raises on, and the tolerant path takes over.
        # `comments` as a *list* drops numpy off its C path and costs 2.5x
        # (0.104 s against 0.042 s on a 200k-row file), so try the
        # overwhelmingly common single marker. A file using '%' or '//'
        # instead simply fails to parse here and falls through to the
        # tolerant path, which handles every marker correctly -- slower,
        # but only for files that actually use them.
        try:
            rows = np.loadtxt(path, delimiter=delimiter, comments="#", ndmin=2)
            if rows.size:
                return rows, Defects(0, rows.shape[1], 0, ())
        except Exception:
            pass

    lines, numbers = [], []
    with open(path, "r") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.split("//", 1)[0]
            for ch in comment_chars:
                line = line.split(ch, 1)[0]
            line = line.strip()
            if line:
                lines.append(line)
                numbers.append(lineno)
    if not lines:
        raise ValueError(f"{path.name}: no data rows found")
    split = [ln.split(delimiter) if delimiter else ln.split() for ln in lines]
    width = max(len(row) for row in split)
    ragged = [i for i, row in enumerate(split) if len(row) != width]

    if strict and ragged:
        raise ValueError(
            f"{path.name}: ragged rows; column counts differ "
            f"(first at line {numbers[ragged[0]]}). Reading without "
            "strict=True pads them and reports the count."
        )
    if not ragged:
        try:
            rows = np.asarray(split, dtype=np.float64)
            return rows, Defects(0, width, 0, ())
        except ValueError:
            pass  # non-numeric fields somewhere; the tolerant path handles it
    if strict:
        raise ValueError(
            f"{path.name}: contains fields that are not numbers. Reading "
            "without strict=True turns them into gaps and reports the count."
        )
    return _tolerant(split, numbers, width, path)


def _tolerant(split, numbers, width: int, path: Path) -> tuple[np.ndarray, Defects]:
    """Pad short rows, convert unreadable fields to NaN, count both.

    Deliberately not vectorized: this path only runs on files that already
    failed the fast one, and being able to name the offending line matters
    more here than speed.
    """
    rows = np.full((len(split), width), np.nan)
    bad_fields, offenders = 0, []
    for i, row in enumerate(split):
        if len(row) != width:
            offenders.append(numbers[i])
        for j, field in enumerate(row):
            text = field.strip()
            if text.lower() in _NULL_FIELDS:
                bad_fields += 1
                continue
            try:
                rows[i, j] = float(text)
            except ValueError:
                bad_fields += 1
                offenders.append(numbers[i])
    short = sum(1 for row in split if len(row) != width)
    return rows, Defects(short, width, bad_fields,
                         tuple(sorted(set(offenders))[:5]))


def _sniff_file(path: Path, comment_chars: str) -> str | None:
    """Delimiter from the first data line, without reading the whole file."""
    with open(path, "r") as fh:
        for raw in fh:
            line = raw.split("//", 1)[0]
            for ch in comment_chars:
                line = line.split(ch, 1)[0]
            line = line.strip()
            if line:
                return _sniff(line)
    raise ValueError(f"{path.name}: no data rows found")


def _sniff(line: str) -> str | None:
    """Pick a delimiter from the first data row. None means whitespace."""
    for cand in _DELIMITERS:
        if cand in line:
            return cand
    return None


def load_phase(
    path: str | Path,
    rate_hz: float | None = None,
    column: int = 0,
    time_column: int | None = None,
    delimiter: str | None = None,
    gap_value: float | None = None,
    strict: bool = False,
) -> PhaseSeries:
    """Load phase (time error, seconds) as a `PhaseSeries`.

    Give either `rate_hz` or a `time_column` to infer it from.
    """
    values, times, prov = _load(
        path, column, time_column, delimiter, gap_value, "phase", strict
    )
    rate_hz, prov = _resolve_rate(rate_hz, times, prov)
    return PhaseSeries(x=values, rate_hz=rate_hz, provenance=prov)


def load_frequency(
    path: str | Path,
    rate_hz: float | None = None,
    column: int = 0,
    time_column: int | None = None,
    delimiter: str | None = None,
    gap_value: float | None = None,
    strict: bool = False,
) -> FreqSeries:
    """Load fractional frequency (dimensionless) as a `FreqSeries`."""
    values, times, prov = _load(
        path, column, time_column, delimiter, gap_value, "frequency", strict
    )
    rate_hz, prov = _resolve_rate(rate_hz, times, prov)
    return FreqSeries(y=values, rate_hz=rate_hz, provenance=prov)


def _load(path, column, time_column, delimiter, gap_value, kind, strict=False):
    """Shared read plus provenance, including the input file hash."""
    path = Path(path)
    prov = new_provenance(str(path), file_sha256(path))
    values, times, defects = read_text(
        path, column, time_column, delimiter, gap_value=gap_value, strict=strict)
    prov = record(
        prov,
        "io.text",
        "read_text",
        {
            "kind": kind,
            "column": column,
            "time_column": time_column,
            "gap_value": gap_value,
        },
        f"n={values.size}, gaps={int(np.count_nonzero(~np.isfinite(values)))}",
    )
    if not defects.clean:
        # Loud, because accepting a malformed record quietly is the failure
        # this library guards against everywhere else.
        prov = warn(prov, "io.text", "tolerated_defects",
                    {"short_rows": defects.short_rows,
                     "bad_fields": defects.bad_fields,
                     "lines": list(defects.lines)},
                    defects.describe())
    return values, times, prov


def _resolve_rate(rate_hz, times, prov):
    """Use the given rate, or infer one from timetags and say so."""
    if rate_hz is not None:
        return rate_hz, prov
    if times is None:
        raise ValueError("give rate_hz, or a time_column to infer it from")
    steps = np.diff(times)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if steps.size == 0:
        raise ValueError("timetag column has no usable increasing intervals")
    tau0 = float(np.median(steps))
    jitter = float(steps.std() / tau0) if steps.size > 1 else 0.0
    outcome = f"tau0={tau0:.9g}s from timetags, jitter={jitter:.2e}"
    logfn = warn if jitter > _TIMETAG_JITTER_WARN else record
    prov = logfn(prov, "io.text", "infer_rate", {"n_intervals": int(steps.size)}, outcome)
    return 1.0 / tau0, prov
