"""Turning a file's columns into series the library will accept.

Everything between "the user has picked a file" and "an estimator can run"
lives here: which stretches of the record to keep, what the numbers in it
are in, and how to draw two million points into a few hundred pixels
without lying about them.

**Nothing here estimates anything.** The page's whole job is to hand the
library good input; the moment a variance is wanted, `stats/` computes it.
That boundary is why this can be rewritten freely without any risk to an
answer.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PHASE_UNITS", "FREQ_UNITS", "unit_factor", "envelope",
           "clamp_windows", "build_series", "exclude_between", "crop",
           "reject_outliers", "PREVIEW_POINTS"]

# What the numbers in the file are in, and the factor that turns them into
# what the library wants. Phase is seconds; fractional frequency is
# dimensionless. A file in nanoseconds analysed as seconds is wrong by a
# billion and looks entirely plausible on a log axis, so an unrecognised
# unit is an error rather than a silent 1.0.
PHASE_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}
FREQ_UNITS = {"fractional": 1.0, "ppm": 1e-6, "ppb": 1e-9, "ppt": 1e-15}

# A trace is a picture, and a picture has a few hundred pixel columns.
PREVIEW_POINTS = 1400


def unit_factor(unit, is_phase: bool) -> float:
    """The multiplier into the library's units, refusing anything unknown."""
    table = PHASE_UNITS if is_phase else FREQ_UNITS
    if unit in (None, ""):
        return 1.0
    if unit not in table:
        raise ValueError(
            f"unknown unit {unit!r} for {'phase' if is_phase else 'frequency'} "
            f"data; expected one of {sorted(table)}")
    return table[unit]


def envelope(values: np.ndarray, start: int, stop: int,
             target: int = PREVIEW_POINTS) -> dict:
    """The `[start, stop)` slice reduced to one min/max pair per pixel column.

    **Not stride sampling.** Taking every k-th point would hide exactly the
    outliers this view exists to find -- a one-sample glitch has a (k-1)/k
    chance of vanishing, better than 99% at the reduction used on a large
    record. Keeping each bucket's minimum and maximum preserves the visual
    envelope, so a spike is still a spike and a range dragged over it means
    what it looks like it means.

    Vectorized through `reduceat` rather than a Python loop over buckets:
    this runs on every drag, every zoom and every keystroke that changes an
    expression, and a loop over a thousand buckets is the difference between
    a page that feels live and one that stutters. `fmin`/`fmax` skip NaN, so
    a bucket holding one good sample among gaps still reports it, while a
    bucket that is entirely gap correctly comes back NaN and breaks the line.

    Returns bucket start indices with `lo`/`hi`, or plain samples when the
    slice is already short enough to draw honestly point for point.
    """
    start = max(0, int(start))
    stop = min(int(values.size), int(stop))
    if stop <= start:
        return {"kind": "points", "i0": [], "lo": [], "hi": []}

    chunk = values[start:stop]
    n = chunk.size
    if n <= target:
        index = np.arange(start, stop)
        return {"kind": "points", "i0": index, "lo": chunk, "hi": chunk}

    buckets = max(1, target // 2)
    edges = np.linspace(0, n, buckets + 1).astype(np.intp)
    edges = np.unique(edges)[:-1]
    with np.errstate(invalid="ignore"):
        lo = np.fmin.reduceat(chunk, edges)
        hi = np.fmax.reduceat(chunk, edges)
    return {"kind": "envelope", "i0": edges + start, "lo": lo, "hi": hi}


def clamp_windows(windows, size: int) -> list[tuple[int, int]]:
    """The page's `[start, stop]` pairs as real, ordered, merged indices.

    Overlapping windows are merged rather than kept apart, because two
    overlapping selections of the same record are one selection -- leaving
    them separate would analyse the shared stretch twice and quietly
    double-count it in anything that later averaged the results.
    """
    spans = []
    for window in windows or []:
        lo, hi = (list(window) + [None, None])[:2]
        lo = 0 if lo in (None, "") else max(0, int(lo))
        hi = size if hi in (None, "") else min(size, int(hi))
        if hi - lo >= 2:
            spans.append((lo, hi))
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for lo, hi in spans[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def build_series(values: np.ndarray, rate_hz: float, is_phase: bool,
                 factor: float):
    """A `PhaseSeries` or `FreqSeries` from raw column values.

    Built at full length and cropped afterwards through `edit.select_range`,
    rather than sliced here and constructed short. The library call is what
    writes the selection into the step log, and a range that reached the
    answer without appearing in provenance is exactly the undocumented
    exclusion this project exists to make impossible.

    The constant pedestal is removed **before** the unit conversion, not
    after, and the order is load-bearing. These columns routinely carry one
    -- 0.35 s in the record this was built against, where float64 spacing is
    5.5e-17 s and the fluctuations live below it. Every estimator here
    differences the phase, so subtracting a constant changes no answer and
    buys back the resolution `types.precision_ratio` would otherwise be
    warning about.

    Scaling first would spend that resolution before reclaiming it.
    `x * 1e-12` rounds each element independently to within half an ulp,
    which on a 0.35 pedestal is a perturbation of about 4e-17 s scattered
    across a signal of about 1e-12 s -- and it does not cancel in the
    subtraction that follows, because each element rounded its own way. The
    measured cost was a deviation that moved in the sixth figure depending
    on which unit the same file was read in. Subtracting first leaves the
    multiply acting on numbers whose magnitude is already the signal's, and
    the units then scale the answer exactly.
    """
    from ..types import FreqSeries, PhaseSeries

    values, pedestal = to_library_units(values, factor)
    series = (PhaseSeries(values, rate_hz=rate_hz) if is_phase
              else FreqSeries(values, rate_hz=rate_hz))
    return series, pedestal


def to_library_units(values: np.ndarray, factor: float):
    """`(values, pedestal)` in the library's units, pedestal removed first.

    Shared with the stage-2 preview rather than duplicated there, because a
    plot captioned "value handed to the library" that showed the pedestal
    would be a flat line at 0.35 s with the entire signal invisible inside
    the width of it -- and would be lying about what the library gets.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    pedestal = float(finite[0]) if finite.size else 0.0
    if pedestal:
        values = values - pedestal
    if factor != 1.0:
        values = values * factor
    return values, pedestal


def exclude_between(series, spans, reason: str):
    """Mark everything outside `spans` as a gap, and record that it happened.

    This is the `join=True` reading of several selected windows: one record
    from the first window's start to the last one's end, with the stretches
    nobody selected turned into gaps for the estimators' documented gap
    policy to handle.

    **Nothing is filled in.** Interpolating across an excluded stretch would
    invent the data whose exclusion was the entire point, and there is no
    imputer in the library to borrow -- gap filling is a later milestone,
    validated against records where the removed points are known. The
    alternative reading, one independent result per window, needs no
    masking at all and is what the page does by default.

    Written here rather than in `edit.py` because it is a consequence of a
    multi-window gesture, which is a page idea; the step still goes through
    `log.record`, so the provenance is the same shape as any library edit.
    """
    from ..log import record
    from ..types import PhaseSeries

    values = np.array(series.x if isinstance(series, PhaseSeries) else series.y,
                      dtype=np.float64)
    keep = np.zeros(values.size, dtype=bool)
    for lo, hi in spans:
        keep[lo:hi] = True
    dropped = int((~keep).sum())
    values[~keep] = np.nan

    prov = record(series.provenance, "gui", "exclude_between_windows",
                  {"windows": [list(span) for span in spans],
                   "dropped": dropped, "reason": reason},
                  f"{dropped} points outside {len(spans)} selected windows "
                  "marked as gaps; nothing interpolated")
    from dataclasses import replace
    field = "x" if isinstance(series, PhaseSeries) else "y"
    return replace(series, **{field: values}, provenance=prov)


def crop(series, lo: int, hi: int, reason: str, lines: list):
    """`edit.select_range` when it would change anything, echoed either way."""
    from ..edit import select_range

    if lo <= 0 and hi >= series.n:
        return series
    lines.append(f"series = edit.select_range(series, {lo}, {hi}, "
                 f"by='index', reason={reason!r})")
    return select_range(series, lo, hi, by="index", reason=reason)


def reject_outliers(series, threshold, reason: str, lines: list):
    """`edit.drop_outliers` if a threshold was given, echoed if it ran."""
    from ..edit import drop_outliers

    text = str(threshold or "").strip()
    if not text:
        return series
    lines.append(f"series = edit.drop_outliers(series, "
                 f"threshold={float(text)}, reason={reason!r})")
    return drop_outliers(series, threshold=float(text), reason=reason)
