"""Reading a time column, whatever the logger decided to write in it.

Counters and logging scripts write timestamps in whatever their author had
to hand: elapsed seconds, Unix epoch seconds, ISO-8601, MJD. Until now the
reader parsed the whole file as float64, so **an ISO date in any column
made the file unreadable**.

Two things this must not do, both of which would be easy and wrong.

**It must not guess between plausible formats.** A bare `60000` is a
credible MJD (year 2023) and a credible count of elapsed seconds. Where the
evidence does not separate them, the candidates are named and the caller
chooses; a wrong timebase rescales every tau silently.

**It must not defeat the float64 phase guard.** `types.py` warns that
absolute MJD-derived seconds quantize to ~1.5e-11 s and silently destroy
the picoseconds this field cares about. So timestamps set the series
`epoch` and the *rate*, and the data stays as offsets from that epoch.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np

__all__ = ["FORMATS", "parse_times", "infer_rate", "MJD_EPOCH_UNIX",
           "AmbiguousTimebase"]

FORMATS = ("elapsed", "unix", "mjd", "iso")

# MJD 0 is 1858-11-17 00:00 UTC; the Unix epoch is MJD 40587.
MJD_EPOCH_UNIX = 40587.0

# Plausible ranges, used to tell formats apart by what the numbers *are*
# rather than by hope. MJD 40000-70000 spans 1968 to 2050; Unix seconds
# 3e8-3e9 spans 1979 to 2065.
_MJD_RANGE = (40_000.0, 70_000.0)
_UNIX_RANGE = (3.0e8, 3.0e9)

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


class AmbiguousTimebase(ValueError):
    """The timestamps fit more than one format and the caller must say which."""


def _classify(values: np.ndarray) -> list[str]:
    """Which numeric formats these values could plausibly be."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    lo, hi = float(finite.min()), float(finite.max())
    out = []
    if _MJD_RANGE[0] <= lo and hi <= _MJD_RANGE[1]:
        out.append("mjd")
    if _UNIX_RANGE[0] <= lo and hi <= _UNIX_RANGE[1]:
        out.append("unix")
    # Elapsed seconds start at or near zero and increase.
    if lo >= -1.0 and lo < 1e6:
        out.append("elapsed")
    return out


def parse_times(raw, fmt: str | None = None) -> tuple[np.ndarray, float, str]:
    """`(seconds_from_epoch, epoch_unix, format)` from a time column.

    `raw` may be an array of numbers or a sequence of strings. `fmt` forces
    one of `FORMATS`; omitted, the format is inferred and
    `AmbiguousTimebase` is raised when the evidence does not single one out.
    """
    if fmt is not None and fmt not in FORMATS:
        raise ValueError(f"unknown time format {fmt!r}; expected {FORMATS}")

    texts = None
    if not isinstance(raw, np.ndarray) or raw.dtype.kind in "US":
        texts = [str(v).strip() for v in np.asarray(raw).ravel()]
        if fmt == "iso" or (fmt is None and texts and _ISO.match(texts[0])):
            return _parse_iso(texts)
        values = np.array([_maybe_float(t) for t in texts], dtype=np.float64)
    else:
        values = np.asarray(raw, dtype=np.float64)

    if fmt is None:
        candidates = _classify(values)
        if not candidates:
            raise ValueError(
                "time column does not look like elapsed seconds, Unix "
                "seconds, MJD or ISO-8601. Pass fmt= to say what it is."
            )
        if len(candidates) > 1:
            raise AmbiguousTimebase(
                f"time column could be any of {candidates} -- "
                f"values span {values.min():.6g} to {values.max():.6g}. "
                "Pass fmt= to choose; guessing would rescale every tau."
            )
        fmt = candidates[0]

    if fmt == "mjd":
        unix = (values - MJD_EPOCH_UNIX) * 86400.0
    elif fmt == "unix":
        unix = values
    else:
        unix = values                     # elapsed: epoch is the first sample
    epoch = float(unix[0])
    return unix - epoch, epoch, fmt


def _maybe_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return np.nan


def _parse_iso(texts) -> tuple[np.ndarray, float, str]:
    """ISO-8601, via `datetime.fromisoformat`. Naive stamps are read as UTC.

    Assuming UTC for a naive timestamp is a choice, not a fact -- but every
    alternative is worse. Local time depends on the machine that reads the
    file rather than the one that wrote it, which would make the same file
    give different answers in different offices.
    """
    seconds = np.empty(len(texts), dtype=np.float64)
    for i, text in enumerate(texts):
        try:
            when = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            seconds[i] = np.nan
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds[i] = when.timestamp()
    finite = seconds[np.isfinite(seconds)]
    if finite.size == 0:
        raise ValueError("no parseable ISO-8601 timestamps in the time column")
    epoch = float(finite[0])
    return seconds - epoch, epoch, "iso"


def infer_rate(seconds: np.ndarray, jitter_warn: float = 1e-3):
    """`(rate_hz, jitter)` from timestamps, using the median interval.

    Median, not mean: one duplicated or missing stamp should not move the
    rate. `jitter` is the interquartile spread of the intervals over the
    median, so a caller can report how far from a uniform grid the record
    really is rather than being told a reassuring single number.

    Gaps in the timestamps are skipped rather than interpolated. A missing
    *timestamp* is not the same as a missing reading: the row's position in
    time is unknown, and inventing it would slide every later sample.
    """
    finite = np.asarray(seconds, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        raise ValueError("need at least two timestamps to infer a rate")
    steps = np.diff(finite)
    steps = steps[steps > 0]
    if steps.size == 0:
        raise ValueError(
            "timestamps do not increase; the time column may be the wrong "
            "column, or the record may be unsorted")
    median = float(np.median(steps))
    spread = float(np.percentile(steps, 75) - np.percentile(steps, 25))
    return 1.0 / median, spread / median
