"""The dynamic Allan variance: σ_y(τ) as a function of when you looked.

Every statistic in `stats/` returns one number per τ for a whole record,
which presumes the record is stationary. A real oscillator is not: it
warms up, it gets bumped, its enclosure's air conditioning cycles, the GPS
antenna sees a different sky. Galleani and Tavella's DAVAR is the obvious
repair -- evaluate the same estimator over a window and slide the window
along the record, giving a surface σ_y(τ, t) whose ridges are the events.

**It is computed by recomputing, not by sliding.** The literature's update
form adds the entering term and subtracts the expiring one, which is O(1)
per step and loses significance without bound: subtractive cancellation in
a running sum of squares has no self-correcting mechanism, and on a long
record the surface drifts away from the truth in a way that looks like a
physical trend. Direct recomputation over the existing estimator costs
`window/step` times one batch call, which for any window worth plotting is
a fraction of a second -- and it buys two things worth more than the
speed:

* **`davar(x, window=len(x))` is bit-identical to the batch call**, not
  equal to 1e-15. Same code path, same data, same rounding. That identity
  is the acceptance test for this module, and it exists only because
  nothing here reimplements the estimator.
* **Every estimator comes free.** The surface is built by name through the
  registry, so `kind="ohdev"` gives a dynamic Hadamard deviation and a
  third-party plugin gets a dynamic form without writing any code.

Two honesty rules the surface follows:

* **Only whole windows.** A truncated window at either end has fewer
  terms, a shorter longest τ and a different EDOF, so plotting it beside
  the others draws an edge artefact that reads as a physical transient.
  The uncovered half-window at each end is reported rather than filled.
* **A τ a window cannot estimate is NaN, not omitted.** Gaps shorten a
  window's longest contiguous run, so the same τ may be available in one
  window and not the next. Dropping it would make the surface ragged;
  filling it would invent a number. NaN is the third option and the only
  correct one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..log import record
from ..provenance import Provenance
from ..registry import get as registry_get
from ..stats.ci import confidence
from ..stats.core import StabilityResult
from ..tau import averaging_factors, taus_from_afs
from ..types import FreqSeries, PhaseSeries

__all__ = ["DavarSurface", "davar", "MAX_WINDOWS", "STEP_DIVISOR"]

# Ceiling on how many windows one call may evaluate. Each is a full batch
# run, so the cost is linear in this and a default `step` on a long record
# can ask for a great many. The message names `step`, which is the knob.
MAX_WINDOWS = 2048

# Default `step`, as a fraction of the window. Ten steps per window
# resolves an event to a tenth of the averaging span, which is as fine as
# the surface means anything: two windows overlapping by 90 % share 90 %
# of their terms and cannot disagree by much.
STEP_DIVISOR = 10


@dataclass(frozen=True)
class DavarSurface:
    """σ²(τ, t) on a rectangular grid of averaging factors and window times.

    `var` is `(n_windows, n_taus)`, NaN where that window could not
    estimate that τ. `t_s` is the *centre* time of each window in the
    original series' timebase, which is where an event the window
    straddles actually happened.
    """

    kind: str
    af: np.ndarray
    tau_s: np.ndarray
    t_s: np.ndarray
    start_index: np.ndarray
    var: np.ndarray
    n_terms: np.ndarray
    window: int
    step: int
    rate_hz: float
    epoch: float
    scale: float
    provenance: Provenance
    var_lo: np.ndarray | None = None
    var_hi: np.ndarray | None = None
    level: float | None = None
    edof: np.ndarray | None = None
    span_factor: int = 2
    is_variance: bool = True

    @property
    def dev(self) -> np.ndarray:
        """The deviation surface, `sqrt(var)`. NaN stays NaN."""
        return np.sqrt(self.var)

    @property
    def n_windows(self) -> int:
        return int(self.t_s.size)

    def window_result(self, k: int) -> StabilityResult:
        """One window as an ordinary `StabilityResult`, for plotting or export.

        Carries no coefficients -- they were released with the window --
        so `confidence()` refuses it. Ask `davar` for a `level` if you want
        intervals; it computes them while the coefficients still exist.
        """
        keep = np.flatnonzero(np.isfinite(self.var[k]))
        return StabilityResult(
            kind=self.kind,
            af=self.af[keep],
            tau_s=self.tau_s[keep],
            var=self.var[k][keep],
            n_terms=self.n_terms[k][keep],
            rate_hz=self.rate_hz,
            provenance=record(
                self.provenance,
                "stream.davar",
                "window_result",
                {"window": k, "t_s": float(self.t_s[k]),
                 "start_index": int(self.start_index[k])},
                f"{keep.size} points",
            ),
            scale=self.scale,
            span_factor=self.span_factor,
            is_variance=self.is_variance,
        )

    def at_af(self, af: int) -> np.ndarray:
        """The time series of `var` at one averaging factor.

        The other cut through the surface, and the one a stability chart
        over time is made of: how τ = 1 s behaved hour by hour.
        """
        idx = np.flatnonzero(self.af == af)
        if idx.size == 0:
            raise KeyError(f"averaging factor {af} not in this surface: {list(self.af)}")
        return self.var[:, idx[0]]


def davar(
    series: PhaseSeries | FreqSeries,
    window: int,
    step: int | None = None,
    taus="octave",
    kind: str = "oadev",
    max_af: int | None = None,
    gap_policy: str = "auto",
    level: float | None = None,
    method: str = "auto",
    max_windows: int = MAX_WINDOWS,
) -> DavarSurface:
    """Evaluate `kind` over sliding windows of `window` samples.

        starts   = 0, step, 2*step, ...  while start + window <= n
        var[k]   = kind(series[start_k : start_k + window])

    `window` is in **samples**, not seconds, because it has to be an
    integer number of slots for the τ grid to be the same in every window,
    and rounding a duration to samples silently is how two runs of the same
    analysis end up on different grids.

    `step` defaults to `window // STEP_DIVISOR` and is recorded. `level`,
    if given, computes a confidence interval per window while its
    coefficients still exist -- there is no way to add one afterwards.

    Passing `window = series.n` gives exactly one window and returns, bit
    for bit, what calling `kind` on the whole record returns. That is
    tested, and it is the reason the surface is built this way.
    """
    n = series.n
    window = int(window)
    if window < 3:
        raise ValueError(f"window must be at least 3 samples, got {window}")
    if window > n:
        raise ValueError(
            f"window of {window} samples exceeds the {n}-sample record. "
            f"window={n} is the whole record and gives one window, which is "
            "the batch answer."
        )
    step = max(1, window // STEP_DIVISOR) if step is None else int(step)
    if step < 1:
        raise ValueError(f"step must be at least 1 sample, got {step}")

    starts = np.arange(0, n - window + 1, step, dtype=np.int64)
    if starts.size > max_windows:
        raise ValueError(
            f"{starts.size} windows of {window} samples at step {step} "
            f"exceeds the {max_windows} limit; each is a full batch "
            f"evaluation. Raise step to about {int(np.ceil((n - window) / max_windows))} "
            "or raise max_windows deliberately."
        )

    estimator = registry_get("estimator", kind)
    # The frequency domain's `n` is one more in phase, which is what the
    # estimators' tau grids are built on. Matching it here is what makes
    # the window-equals-record case land on the same grid.
    n_grid = window + 1 if isinstance(series, FreqSeries) else window
    grid = averaging_factors(n_grid, taus, max_af=max_af)
    if grid.size == 0:
        raise ValueError(f"no averaging factor fits a window of {window} samples")

    var = np.full((starts.size, grid.size), np.nan)
    terms = np.zeros((starts.size, grid.size), dtype=np.int64)
    lo = np.full_like(var, np.nan) if level is not None else None
    hi = np.full_like(var, np.nan) if level is not None else None
    edof = np.full_like(var, np.nan) if level is not None else None

    scale, span, is_var, refused = 1.0, 2, True, ""
    for k, s in enumerate(starts.tolist()):
        try:
            res = estimator(_window(series, s, window), taus=grid,
                            gap_policy=gap_policy)
        except ValueError as exc:
            refused = refused or str(exc)
            continue
        # `searchsorted` returns `grid.size` for anything past the end, so
        # the bound is checked before the array_equal indexes with it.
        cols = np.searchsorted(grid, res.af)
        if cols.max(initial=0) >= grid.size or not np.array_equal(grid[cols], res.af):
            raise ValueError(
                f"estimator {kind!r} returned averaging factors that are not "
                f"a subset of the requested grid, so its columns cannot be "
                f"aligned across windows. It does not honour an explicit tau "
                f"list -- `wdev` is the shipped example, its scales being "
                f"dyadic by construction."
            )
        var[k, cols], terms[k, cols] = res.var, res.n_terms
        scale, span, is_var = res.scale, res.span_factor, res.is_variance
        if level is not None:
            ci = confidence(res, level=level, method=method)
            lo[k, cols], hi[k, cols], edof[k, cols] = ci.var_lo, ci.var_hi, ci.edof

    live = np.flatnonzero(np.isfinite(var).any(axis=0))
    if live.size == 0:
        raise ValueError(
            f"davar: no window of {window} samples produced any point"
            + (f"; {kind} said: {refused}" if refused else "")
        )
    dropped = int(grid.size - live.size)
    grid, var, terms = grid[live], var[:, live], terms[:, live]
    if level is not None:
        lo, hi, edof = lo[:, live], hi[:, live], edof[:, live]

    # A phase sample is an instant; a frequency sample is an interval, so
    # its window's centre sits half a sample later. Half a sample is
    # nothing on a plot and everything when a surface is being lined up
    # against a log of what happened at the bench.
    phase = isinstance(series, PhaseSeries)
    offset = 0.5 * (window - 1) if phase else 0.5 * window
    end = n - 1 if phase else n
    t_s = series.epoch + (starts + offset) / series.rate_hz

    prov = record(
        series.provenance,
        "stream.davar",
        "evaluate",
        {
            "kind": kind,
            "window": window,
            "step": step,
            "step_default": step == max(1, window // STEP_DIVISOR),
            "n_windows": int(starts.size),
            "taus": taus if isinstance(taus, str) else "explicit",
            "gap_policy": gap_policy,
            "level": level,
            # From each end of the record to the nearest window centre.
            # Symmetric by construction when one window covers everything.
            "uncovered_head_s": float(offset / series.rate_hz),
            "uncovered_tail_s": float(
                (end - int(starts[-1]) - offset) / series.rate_hz
            ),
            "taus_dropped_everywhere": dropped,
        },
        f"{starts.size} windows x {grid.size} taus"
        + (f", {dropped} taus no window could estimate" if dropped else ""),
    )
    return DavarSurface(
        kind=kind,
        af=grid,
        tau_s=taus_from_afs(grid, series.rate_hz),
        t_s=t_s,
        start_index=starts,
        var=var,
        n_terms=terms,
        window=window,
        step=step,
        rate_hz=series.rate_hz,
        epoch=series.epoch,
        scale=scale,
        provenance=prov,
        var_lo=lo,
        var_hi=hi,
        level=level,
        edof=edof,
        span_factor=span,
        is_variance=is_var,
    )


def _window(series: PhaseSeries | FreqSeries, start: int, width: int):
    """One window as a series in its own right, on the parent's timebase.

    The values are **not** re-zeroed. Subtracting the window's first sample
    would leave every statistic unchanged in exact arithmetic and change
    the last bits in floating point, which would cost the
    window-equals-record identity for no benefit.
    """
    shift = start / series.rate_hz
    if isinstance(series, PhaseSeries):
        return PhaseSeries(
            x=series.x[start : start + width],
            rate_hz=series.rate_hz,
            epoch=series.epoch + shift,
            provenance=series.provenance,
        )
    return FreqSeries(
        y=series.y[start : start + width],
        rate_hz=series.rate_hz,
        epoch=series.epoch + shift,
        provenance=series.provenance,
    )
