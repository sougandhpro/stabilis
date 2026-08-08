"""Choosing what part of a record to analyse, and what to throw away.

Excluding data changes the answer, so everything here is recorded. An
auditor's first question about an edited record is *which parts were
removed and why*, and a result that cannot answer it is not evidence.

Two rules the whole module follows.

**Nothing is deleted; points become gaps.** Dropping samples would shorten
the record and slide every later one, so a spike removed at index 500 would
silently re-time everything after it. Marking it NaN preserves the timebase
and hands the consequences to the gap machinery, which already handles them
honestly (see `theory/gaps.md`). Deleting is available only as an explicit
range selection, where the intent is unambiguous.

**Every function takes a `reason` and puts it in provenance.** Not
optional, not defaulted to something bland. "warm-up" and "cable touched at
14:05" are what make an exclusion defensible six months later.

The interactive half of this -- dragging on a plot -- lives in the GUI, and
prints the equivalent call from here. An exclusion that exists only as a
mouse gesture is not evidence.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .log import record, warn
from .types import FreqSeries, PhaseSeries

__all__ = ["BY", "select_range", "drop_outliers", "summary", "MJD_EPOCH_UNIX"]

# Ways of naming a point in the record.
BY = ("index", "seconds", "mjd", "unix")

# MJD 0 is 1858-11-17; the Unix epoch is MJD 40587.
MJD_EPOCH_UNIX = 40587.0


def _values(series):
    return series.x if isinstance(series, PhaseSeries) else series.y


def _with_values(series, values, prov):
    field = "x" if isinstance(series, PhaseSeries) else "y"
    return replace(series, **{field: values}, provenance=prov)


def _to_index(series, value, by: str, default: int) -> int:
    """Turn a position expressed any of four ways into a sample index."""
    if value is None:
        return default
    if by == "index":
        return int(value)
    if by == "seconds":
        return int(round(float(value) * series.rate_hz))
    if by in ("mjd", "unix"):
        # Both are absolute, so they only mean anything against the epoch
        # the record carries. Refusing beats guessing that epoch is zero.
        if not series.epoch:
            raise ValueError(
                f"select_range(by={by!r}) needs the series to carry an epoch, "
                "and this one has none. Load it with a time column, or select "
                "by 'index' or 'seconds' instead."
            )
        unix = (float(value) - MJD_EPOCH_UNIX) * 86400.0 if by == "mjd" \
            else float(value)
        return int(round((unix - series.epoch) * series.rate_hz))
    raise ValueError(f"select_range: `by` must be one of {BY}, got {by!r}")


def select_range(series, start=None, stop=None, by: str = "index",
                 reason: str = ""):
    """The part of `series` from `start` to `stop`, and why.

    `by` names how the bounds are expressed: `index` (samples), `seconds`
    (elapsed from the record's own start), `mjd`, or `unix`. `stop` is
    exclusive, as a Python slice.

    The epoch moves with the window, so the returned series still knows
    when it starts in absolute time -- which is what lets a second
    selection by `mjd` behave the same as the first.
    """
    values = _values(series)
    lo = _to_index(series, start, by, 0)
    hi = _to_index(series, stop, by, values.size)
    lo, hi = max(0, lo), min(int(values.size), hi)
    if hi - lo < 2:
        raise ValueError(
            f"select_range: {lo}..{hi} keeps {max(hi - lo, 0)} points; "
            "need at least 2. Check the units of `start`/`stop` against "
            f"by={by!r}."
        )
    if not reason:
        raise ValueError(
            "select_range: `reason` is required. An exclusion without a "
            "stated reason cannot be defended later, and this library's "
            "output is meant to be."
        )

    prov = record(series.provenance, "edit", "select_range",
                  {"start": lo, "stop": hi, "by": by,
                   "kept": hi - lo, "dropped": int(values.size) - (hi - lo),
                   "reason": reason},
                  f"kept {hi - lo} of {values.size} points")
    out = _with_values(series, values[lo:hi].copy(), prov)
    return replace(out, epoch=series.epoch + lo / series.rate_hz)


def drop_outliers(series, threshold: float = 5.0, method: str = "mad",
                  reason: str = ""):
    """Mark points further than `threshold` robust sigmas from the median.

    **Median absolute deviation, not standard deviation.** A single large
    spike inflates the standard deviation enough to hide itself -- the
    classic masking failure -- while the MAD is unmoved by up to half the
    data being wrong. Scaled by 1.4826 so that for Gaussian data it
    estimates the same sigma, which is what makes `threshold` mean
    "sigmas" rather than an arbitrary knob.

    Applied to the *first difference* for phase data, because a phase
    record wanders: an outlier in phase is a point far from its neighbours,
    not far from the record's median, and testing the level would flag the
    whole end of a drifting record.

    Points become **gaps, never deletions** -- see the module docstring.
    """
    if not reason:
        raise ValueError(
            "drop_outliers: `reason` is required; removing data without a "
            "stated reason is not defensible.")
    if method != "mad":
        raise ValueError(f"drop_outliers: unknown method {method!r}; only "
                         "'mad' is implemented")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    values = np.array(_values(series), dtype=np.float64)
    probe = np.diff(values) if isinstance(series, PhaseSeries) else values
    finite = probe[np.isfinite(probe)]
    if finite.size < 3:
        raise ValueError("drop_outliers: too few finite points to judge")

    centre = np.median(finite)
    mad = np.median(np.abs(finite - centre))
    sigma = 1.4826 * mad

    # A floor on sigma, relative to the record's own magnitude.
    #
    # On synthetic or heavily quantized data more than half the differences
    # can be *identical*, making the MAD exactly zero or a few float ulps.
    # Two cases then have to come out differently, and a simple "sigma == 0
    # so decline" gets the second one wrong:
    #
    #   a clean ramp                 -> nothing is an outlier
    #   a clean ramp with one glitch -> the glitch is obviously an outlier
    #
    # Using the floor *as* sigma rather than bailing out handles both: the
    # clean ramp's deviations are dust and stay under the threshold, while
    # a real glitch is many orders above it.
    scale = float(np.max(np.abs(values[np.isfinite(values)])) or 1.0)
    floor = 16.0 * np.spacing(scale)
    at_floor = sigma < floor
    sigma = max(sigma, floor)

    flag = np.abs(probe - centre) > threshold * sigma
    mask = np.zeros(values.size, dtype=bool)
    if isinstance(series, PhaseSeries):
        # A bad difference implicates the later of the two samples: the
        # step *into* it is what was wrong.
        mask[1:] = flag
    else:
        mask = flag
    mask &= np.isfinite(values)

    values[mask] = np.nan
    note = ("" if not at_floor else
            "; the MAD was at the level of floating-point rounding, so the "
            "scale floor was used instead -- expected on synthetic or "
            "heavily quantized data")
    prov = record(series.provenance, "edit", "drop_outliers",
                  {"threshold": threshold, "method": method,
                   "robust_sigma": float(sigma), "at_floor": bool(at_floor),
                   "removed": int(mask.sum()), "reason": reason},
                  f"{int(mask.sum())} of {values.size} points marked as gaps"
                  + note)
    return _with_values(series, values, prov)


def summary(series) -> dict:
    """What a report has to state about an edited record.

    The excluded fraction matters more than the count: ten gaps in a
    thousand points is a different record from ten in twenty.
    """
    values = np.asarray(_values(series))
    finite = np.isfinite(values)
    runs, longest = 0, 0
    current = 0
    for ok in finite:
        if ok:
            current += 1
            longest = max(longest, current)
        else:
            if current:
                runs += 1
            current = 0
    if current:
        runs += 1
    n_gaps = int((~finite).sum())
    return {
        "n": int(values.size),
        "n_gaps": n_gaps,
        "gap_fraction": float(n_gaps / values.size) if values.size else 0.0,
        "runs": runs,
        "longest_run": int(longest),
        "duration_s": float(values.size / series.rate_hz),
    }
