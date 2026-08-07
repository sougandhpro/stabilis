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
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read one data column, and optionally a timetag column.

    Returns ``(values, times)``, where `times` is None unless `time_column`
    was given. Values equal to `gap_value` become NaN.
    """
    path = Path(path)
    rows = _parse_rows(path, comment_chars, delimiter)
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
    return values, times


def _parse_rows(path: Path, comment_chars: str, delimiter: str | None) -> np.ndarray:
    """Strip comments, sniff the delimiter, return a 2-D float array."""
    lines = []
    with open(path, "r") as fh:
        for raw in fh:
            line = raw.split("//", 1)[0]
            for ch in comment_chars:
                line = line.split(ch, 1)[0]
            line = line.strip()
            if line:
                lines.append(line)
    if not lines:
        raise ValueError(f"{path.name}: no data rows found")

    if delimiter is None:
        delimiter = _sniff(lines[0])
    split = [ln.split(delimiter) if delimiter else ln.split() for ln in lines]
    width = len(split[0])
    if any(len(row) != width for row in split):
        raise ValueError(f"{path.name}: ragged rows; column counts differ")
    return np.asarray(split, dtype=np.float64)


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
) -> PhaseSeries:
    """Load phase (time error, seconds) as a `PhaseSeries`.

    Give either `rate_hz` or a `time_column` to infer it from.
    """
    values, times, prov = _load(
        path, column, time_column, delimiter, gap_value, "phase"
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
) -> FreqSeries:
    """Load fractional frequency (dimensionless) as a `FreqSeries`."""
    values, times, prov = _load(
        path, column, time_column, delimiter, gap_value, "frequency"
    )
    rate_hz, prov = _resolve_rate(rate_hz, times, prov)
    return FreqSeries(y=values, rate_hz=rate_hz, provenance=prov)


def _load(path, column, time_column, delimiter, gap_value, kind):
    """Shared read plus provenance, including the input file hash."""
    path = Path(path)
    prov = new_provenance(str(path), file_sha256(path))
    values, times = read_text(path, column, time_column, delimiter, gap_value=gap_value)
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
