"""Generic delimited-text writer -- the mirror of `text.py`.

Three formats, and the differences between them are all about what a
*reader* on the other side will do with the bytes:

* ``stabilis`` -- whitespace-separated, with a leading comment block naming
  the columns and carrying `rate_hz` and `epoch_unix`. `load_phase` reads
  that block back, so this is the only format that round-trips a series
  whole rather than just its numbers.
* ``csv`` -- the same thing comma-separated, for a spreadsheet.
* ``stable32`` -- bare numeric columns, no comments at all. Stable32 and
  TimeLab are the audience and neither was asked to tolerate our headers.

**Precision is 17 significant digits by default**, which is the shortest
form guaranteed to reproduce a float64 exactly, so a written record reads
back bit-identical. That is worth the extra characters: this library exists
to argue about the last two digits of a deviation, and an export that
quietly rounds is an export that quietly changes the answer.

**Gaps are NaN and stay NaN**, except in `stable32` format which has no way
to write one. There the writer refuses rather than substituting a zero --
the same refusal `text.py` makes on the reading side, for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..log import record, warn
from ..provenance import Provenance, file_sha256
from ..registry import register
from ..types import FreqSeries, PhaseSeries
from .timebase import MJD_EPOCH_UNIX

__all__ = ["WrittenFile", "FORMATS", "TIME_COLUMNS", "write_text",
           "save_phase", "save_frequency"]

FORMATS = ("stabilis", "csv", "stable32")

# "" is no time column at all: the rate lives in the header instead, which
# is both exact and smaller. The other three exist because some consumers
# want a timetag, and they are ranked here by how much of a picosecond they
# can still resolve -- see the warning in `_time_column`.
TIME_COLUMNS = ("", "elapsed", "unix", "mjd")

_DELIMITERS = {"stabilis": " ", "csv": ",", "stable32": " "}
_TIME_LABELS = {"elapsed": "time_s", "unix": "unix_s", "mjd": "mjd"}

# When float64 quantization of the timetag column reaches this fraction of
# one sample interval, say so. Deliberately the same number as `text.py`'s
# `_TIMETAG_JITTER_WARN`: the writer warns exactly when the reader would,
# so a file this library writes never surprises the library reading it.
_TIMETAG_RESOLUTION_WARN = 1e-3


@dataclass(frozen=True)
class WrittenFile:
    """One file this library wrote, and what it hashes to.

    Returned rather than a bare path because everything downstream that
    cares about a written file cares about its hash: the bundle manifest,
    a bench measurement record, an auditor re-checking that the bytes on
    disk are the bytes that were reported. Computing it once here means
    there is one hashing path rather than one per caller.
    """

    path: str
    sha256: str
    n_bytes: int
    format: str
    provenance: Provenance


@register("writer", "text")
def write_text(
    path: str | Path,
    values: np.ndarray,
    rate_hz: float,
    epoch: float = 0.0,
    kind: str = "phase",
    format: str = "stabilis",
    time_column: str = "",
    gap_value: float | None = None,
    digits: int = 17,
) -> int:
    """Write one data column, optionally preceded by a timetag column.

    Returns the number of bytes written. The low-level half of this module,
    exactly as `read_text` is the low-level half of `text.py`; callers with
    a series in hand want `save_phase` or `save_frequency`, which record
    the write in provenance and hand back its hash.
    """
    if format not in FORMATS:
        raise ValueError(f"unknown format {format!r}; expected one of {FORMATS}")
    if time_column not in TIME_COLUMNS:
        raise ValueError(
            f"unknown time_column {time_column!r}; expected one of {TIME_COLUMNS}")

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("refusing to write an empty series")

    out = _apply_gap_value(values, gap_value, format)
    columns = [out] if not time_column else [
        _time_column(time_column, values.size, rate_hz, epoch, digits), out]

    fmt = f"%.{digits}g"
    sep = _DELIMITERS[format]
    header = ("" if format == "stable32" else
              _header(kind, rate_hz, epoch, time_column, format, sep))

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header)
        np.savetxt(fh, np.column_stack(columns), fmt=fmt, delimiter=sep)
    return Path(path).stat().st_size


def save_phase(
    series: PhaseSeries,
    path: str | Path,
    format: str = "stabilis",
    time_column: str = "",
    gap_value: float | None = None,
    digits: int = 17,
) -> WrittenFile:
    """Write a `PhaseSeries` and return the file's hash and provenance."""
    return _save(series.x, series, path, "phase", format, time_column,
                 gap_value, digits)


def save_frequency(
    series: FreqSeries,
    path: str | Path,
    format: str = "stabilis",
    time_column: str = "",
    gap_value: float | None = None,
    digits: int = 17,
) -> WrittenFile:
    """Write a `FreqSeries` and return the file's hash and provenance."""
    return _save(series.y, series, path, "frequency", format, time_column,
                 gap_value, digits)


def _save(values, series, path, kind, format, time_column, gap_value, digits):
    """Shared write plus the provenance step and the output hash."""
    n_bytes = write_text(path, values, series.rate_hz, series.epoch, kind,
                         format, time_column, gap_value, digits)
    prov = record(
        series.provenance,
        "io.write",
        "write_text",
        {
            "kind": kind,
            "format": format,
            "time_column": time_column,
            "gap_value": gap_value,
            "digits": digits,
        },
        f"n={values.size}, {n_bytes} bytes to {Path(path).name}",
    )
    if time_column in ("unix", "mjd"):
        prov = _warn_timetag_resolution(prov, time_column, series, path)
    return WrittenFile(
        path=str(path),
        sha256=file_sha256(path),
        n_bytes=n_bytes,
        format=format,
        provenance=prov,
    )


def _apply_gap_value(values: np.ndarray, gap_value, format: str) -> np.ndarray:
    """Substitute the gap sentinel, or refuse where none can be written.

    `stable32` format is bare numbers with no NaN spelling, so a gapped
    record cannot be written to it honestly. Writing a zero is what the
    format's own convention would suggest and is exactly the trap
    `text.py` refuses to walk into on the reading side: a genuine
    zero-offset measurement and a missing sample become the same bytes.
    """
    gaps = ~np.isfinite(values)
    if gap_value is None:
        if format == "stable32" and gaps.any():
            raise ValueError(
                f"{int(gaps.sum())} gaps cannot be written in stable32 format, "
                "which has no spelling for a missing sample. Pass an explicit "
                "gap_value, or write 'stabilis' format, which writes NaN and "
                "reads it back as a gap."
            )
        return values

    collides = np.isfinite(values) & (values == gap_value)
    if collides.any():
        raise ValueError(
            f"gap_value {gap_value!r} occurs {int(collides.sum())} times as a "
            "real measurement, so reading this file back would delete those "
            "samples. Choose a sentinel the data cannot take."
        )
    return np.where(gaps, gap_value, values)


def _time_column(kind, n: int, rate_hz: float, epoch: float,
                 digits: int) -> np.ndarray:
    """The requested timetag column, refused if it cannot tell samples apart.

    The check is on the *printed* strings, not on a model of their
    precision, because the printed strings are what a reader gets. The last
    pair is the worst case: float64 spacing grows with magnitude and the
    grid is uniform, so if the final two tags survive as distinct text then
    every earlier pair does too.

    This is not a rounding nicety. `text.py` infers a rate from the median
    timetag interval, so a column of repeated timestamps makes our own
    reader compute a rate of infinity from our own file.
    """
    elapsed = np.arange(n, dtype=np.float64) / rate_hz
    if kind == "elapsed":
        column = elapsed
    elif kind == "unix":
        column = epoch + elapsed
    else:
        column = (epoch + elapsed) / 86400.0 + MJD_EPOCH_UNIX

    fmt = f"%.{digits}g"
    if n >= 2 and fmt % column[-1] == fmt % column[-2]:
        raise ValueError(
            f"{kind} timetags cannot resolve a sample interval of "
            f"{1.0 / rate_hz:g} s at this epoch -- consecutive tags print as "
            f"the same value ({fmt % column[-1]}), so the column would say "
            "the samples are simultaneous. Use time_column='elapsed', which "
            "counts from zero and stays exact, or leave it off and let the "
            "header carry the rate."
        )
    return column


def _warn_timetag_resolution(prov, kind, series, path):
    """Say so when the timetags cannot resolve the sample interval.

    float64 spacing at magnitude `t` is `2**-52 * t`. An MJD timetag at a
    present-day date is ~6e4 days, so consecutive tags are quantized at
    ~1.4e-11 days, about 1.2 microseconds -- fine for a one-second grid and
    useless for a millisecond one. The data column is unaffected; what
    degrades is a rate inferred from the timetags, so the warning names
    that consequence rather than the number alone.
    """
    tau0 = series.tau0_s
    largest = abs(series.epoch) + (series.n - 1) * tau0
    if kind == "mjd":
        largest = largest / 86400.0 + MJD_EPOCH_UNIX
        spacing = np.spacing(largest) * 86400.0
    else:
        spacing = np.spacing(largest)
    ratio = float(spacing / tau0) if tau0 > 0 else np.inf
    if ratio <= _TIMETAG_RESOLUTION_WARN:
        return prov
    return warn(
        prov,
        "io.write",
        "timetag_resolution",
        {"time_column": kind, "spacing_s": f"{spacing:.3e}", "tau0_s": tau0},
        f"{kind} timetags resolve the sample interval to only {ratio:.1e}; "
        "a reader inferring the rate from this column will see jitter. The "
        "header carries the exact rate -- prefer time_column='elapsed'.",
    )


def _header(kind, rate_hz, epoch, time_column, format, sep) -> str:
    """The leading comment block, column names first.

    The names go on the *first* line because that is the one `read_table`
    inspects, and a spreadsheet reads it too. The metadata lines follow and
    are what `load_phase` uses to recover the rate and the epoch -- without
    them an export is a bag of numbers whose sample interval the reader has
    to be told again by hand.
    """
    labels = ([_TIME_LABELS[time_column]] if time_column else []) + [
        "phase_s" if kind == "phase" else "frequency"]
    return (
        f"# {sep.join(labels)}\n"
        f"# stabilis {kind} export, format {format}\n"
        f"# rate_hz = {rate_hz!r}\n"
        f"# epoch_unix = {epoch!r}\n"
    )
