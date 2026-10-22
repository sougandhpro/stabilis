"""Data conditioning: descriptive statistics, gap segmentation, drift
removal and decimation.

Outlier handling lives in `edit.py`, because removing a point is an
exclusion and carries a stated reason. What is here changes no sample's
status: a drift is subtracted from every point, and decimation keeps the
timebase honest by changing `rate_hz`. Both are recorded in provenance with
the numbers they used. See `theory/drift.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .log import record
from .provenance import Provenance
from .types import FreqSeries, PhaseSeries

__all__ = ["basic_stats", "segments", "longest_segment", "DriftFit",
           "DRIFT_MODELS", "fit_drift", "remove_drift", "decimate"]

# Which drift models apply to which kind of data. Each fits the trend it
# names; `drift_per_s` is always the fractional frequency drift D in 1/s.
DRIFT_MODELS = {
    "phase": ("quadratic", "diff2", "linear"),
    "frequency": ("linear", "bisection", "mean"),
}


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


@dataclass(frozen=True)
class DriftFit:
    """A fitted deterministic trend, and the drift rate it implies.

    `trend` has the length of the series it was fitted to and no gaps: it
    is what `remove_drift` subtracts. `drift_per_s` is D, the linear
    fractional-frequency drift, in 1/s (`drift_per_day` is D * 86400); it is
    0.0 for the two models that fit no drift (`linear` phase, `mean`
    frequency).

    There is deliberately no standard error on D. The least-squares one
    assumes white residuals, and a clock's residuals are not white: under
    random-walk FM it is too small by a factor that grows with the record.
    """

    model: str
    kind: str
    drift_per_s: float
    trend: np.ndarray
    provenance: Provenance

    @property
    def drift_per_day(self) -> float:
        return self.drift_per_s * 86400.0


def _kind(series) -> str:
    if isinstance(series, PhaseSeries):
        return "phase"
    if isinstance(series, FreqSeries):
        return "frequency"
    raise TypeError(f"expected PhaseSeries or FreqSeries, got {type(series)}")


def _values(series) -> np.ndarray:
    return series.x if isinstance(series, PhaseSeries) else series.y


def _lstsq_poly(v: np.ndarray, degree: int):
    """Least-squares polynomial on the finite samples, in scaled time.

        u[i] = (i - c) / h,   c = (n-1)/2,   h = max(c, 1)
        v[i] ~ sum_k p[k] * u[i]**k

    Fitted in `u`, not in seconds or sample index: a Vandermonde matrix in
    raw index is ill-conditioned by n**degree, and in `u` it is O(1).
    Returns `p`, the trend on every index (gaps included), and `h`, which
    the caller needs to turn `p` back into physical units.
    """
    n = v.size
    c = (n - 1) / 2.0
    h = max(c, 1.0)
    u = (np.arange(n) - c) / h
    ok = np.isfinite(v)
    if ok.sum() <= degree:
        raise ValueError(f"need more than {degree} finite points to fit a "
                         f"degree-{degree} trend, have {int(ok.sum())}")
    a = np.vander(u, degree + 1, increasing=True)
    p = np.linalg.lstsq(a[ok], v[ok], rcond=None)[0]
    return p, a @ p, h


def _fit(v: np.ndarray, kind: str, model: str, tau0: float):
    """(D, trend) for one model. See `fit_drift` for the equations."""
    n = v.size
    t = np.arange(n) * tau0
    if model == "mean":
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            raise ValueError("series contains no finite values")
        return 0.0, np.full(n, finite.mean())
    if model in ("quadratic", "linear"):
        degree = 2 if model == "quadratic" else 1
        p, trend, h = _lstsq_poly(v, degree)
        if kind == "phase" and degree == 2:
            return 2.0 * p[2] / (h * tau0) ** 2, trend
        if kind == "frequency":
            return p[1] / (h * tau0), trend
        return 0.0, trend
    if model == "diff2":
        d2 = v[2:] - 2.0 * v[1:-1] + v[:-2]
        d2 = d2[np.isfinite(d2)]
        if d2.size == 0:
            raise ValueError("diff2 needs three consecutive finite points")
        drift = float(d2.mean()) / tau0 ** 2
        return drift, drift * t ** 2 / 2.0
    # bisection
    half = n // 2
    first, second = v[:half], v[half:2 * half]
    k1, k2 = np.isfinite(first), np.isfinite(second)
    if not (k1.any() and k2.any()):
        raise ValueError("bisection needs finite points in both halves")
    t1, t2 = t[:half][k1].mean(), t[half:2 * half][k2].mean()
    y1, y2 = first[k1].mean(), second[k2].mean()
    drift = (y2 - y1) / (t2 - t1)
    return drift, y1 + drift * (t - t1)


def fit_drift(series, model: str | None = None) -> DriftFit:
    """Fit a deterministic trend to a phase or frequency record.

    Phase models (`x` in seconds, `t[i] = i * tau0`):

        quadratic   x ~ a + b*t + (D/2)*t**2       least squares
        diff2       D = mean(x[i+2] - 2*x[i+1] + x[i]) / tau0**2,
                    trend = (D/2)*t**2
        linear      x ~ a + b*t                    (frequency offset; D = 0)

    Frequency models (`y` dimensionless):

        linear      y ~ a + D*t                    least squares
        bisection   D = (ybar2 - ybar1) / (tbar2 - tbar1), the two halves
                    of the record (an odd last sample sits in neither);
                    trend = ybar1 + D*(t - tbar1)
        mean        y ~ a                          (offset only; D = 0)

    Defaults: `quadratic` for phase, `linear` for frequency. Gaps (NaN) are
    left out of the fit and every mean; each finite sample keeps its true
    time, so a gapped record is fitted correctly rather than closed up.
    `diff2` and `bisection` are Stable32's robust alternatives: they cost
    efficiency on white noise and are less pulled by the ends of a
    random-walk record.
    """
    kind = _kind(series)
    model = model or DRIFT_MODELS[kind][0]
    if model not in DRIFT_MODELS[kind]:
        raise ValueError(f"drift model {model!r} does not apply to {kind} "
                         f"data; choose from {DRIFT_MODELS[kind]}")
    drift, trend = _fit(_values(series), kind, model, series.tau0_s)
    trend = np.array(trend, dtype=np.float64)
    trend.flags.writeable = False
    prov = record(series.provenance, "preprocess", "fit_drift",
                  {"model": model, "kind": kind, "n": int(trend.size)},
                  f"D = {float(drift):.6e} /s ({float(drift) * 86400:.6e} /day)")
    return DriftFit(model, kind, float(drift), trend, prov)


def remove_drift(series, model: str | None = None):
    """`series` minus its fitted trend, with the fit in its provenance.

        x_out = x - trend     (phase)        y_out = y - trend   (frequency)

    Gaps stay gaps. A fitted trend absorbs some of the noise as well as the
    drift, so the variance at the longest tau comes out biased low after
    removal -- the cost the Hadamard variance exists to avoid. See
    `theory/drift.md`.
    """
    fit = fit_drift(series, model)
    values = _values(series) - fit.trend
    prov = record(series.provenance, "preprocess", "remove_drift",
                  {"model": fit.model, "drift_per_s": fit.drift_per_s},
                  f"subtracted the {fit.model} trend")
    field = "x" if isinstance(series, PhaseSeries) else "y"
    return replace(series, **{field: values}, provenance=prov)


def decimate(series, m: int):
    """Resample to `m * tau0`, the way each kind of data must be.

        phase:      x_out[k] = x[k*m]
        frequency:  y_out[k] = mean(y[k*m : (k+1)*m])

    These are the same operation: the phase of the averaged frequency is
    the decimated phase. A trailing partial block is dropped, never
    averaged short -- a shorter average is a sample at a different tau. A
    frequency block containing a gap becomes a gap for the same reason. A
    phase gap survives only if it falls on a kept sample, because phase at
    the kept instants is all the decimated record claims to know.

    Epoch is unchanged: the first output sample starts where the first
    input sample did.
    """
    if int(m) != m or m < 1:
        raise ValueError(f"decimation factor must be a positive integer, got {m}")
    m = int(m)
    v = _values(series)
    if isinstance(series, PhaseSeries):
        out = v[::m].copy()
        field = "x"
    else:
        k = v.size // m
        if k == 0:
            raise ValueError(f"cannot average {v.size} samples in blocks of {m}")
        out = v[:k * m].reshape(k, m).mean(axis=1)
        field = "y"
    if out.size < 2:
        raise ValueError(f"decimating {v.size} samples by {m} leaves {out.size}")
    prov = record(series.provenance, "preprocess", "decimate",
                  {"m": m, "n_in": int(v.size), "n_out": int(out.size)},
                  f"tau0 {series.tau0_s:.6g} s -> {series.tau0_s * m:.6g} s")
    return replace(series, **{field: out}, rate_hz=series.rate_hz / m,
                   provenance=prov)
