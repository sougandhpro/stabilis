"""Data conditioning: descriptive statistics and gap segmentation.

Grows in M1 with drift removal, outlier handling and decimation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["basic_stats", "segments", "longest_segment"]


def basic_stats(a: np.ndarray) -> dict:
    """Maximum, minimum, mean, median, sample standard deviation, counts.

    Descriptive only. The sample standard deviation does not converge for
    flicker or random-walk noise, which is the entire reason the Allan
    variance exists -- do not report it as an uncertainty on a clock
    measurement. It is here because Stable32 shows it and because it is a
    useful sanity check on a freshly loaded file.

    NaN gaps are excluded from every statistic and counted separately.
    """
    a = np.asarray(a, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        raise ValueError("series contains no finite values")
    return {
        "n": int(a.size),
        "n_finite": int(finite.size),
        "n_gaps": int(a.size - finite.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
    }


def segments(a: np.ndarray, min_length: int = 1) -> list[tuple[int, int]]:
    """Half-open ``[start, stop)`` index ranges of contiguous finite runs.

    NaN marks a gap, but it cannot say *which kind* of gap. A dropped sample
    on a known grid can be stepped over; a counter that stopped and resumed
    with an unknown phase offset cannot, because the resulting phase step
    becomes a large fake frequency excursion. When the second case is
    possible, analyse each segment independently rather than masking terms.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return []
    finite = np.isfinite(a)
    edges = np.diff(finite.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    stops = np.flatnonzero(edges == -1) + 1
    if finite[0]:
        starts = np.concatenate(([0], starts))
    if finite[-1]:
        stops = np.concatenate((stops, [finite.size]))
    return [
        (int(s), int(e)) for s, e in zip(starts, stops) if e - s >= min_length
    ]


def longest_segment(a: np.ndarray) -> tuple[int, int]:
    """The longest contiguous finite run, as ``[start, stop)``."""
    runs = segments(a)
    if not runs:
        raise ValueError("series contains no finite values")
    return max(runs, key=lambda se: se[1] - se[0])
