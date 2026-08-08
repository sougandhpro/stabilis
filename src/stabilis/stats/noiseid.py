"""Power-law noise identification, by two independent routes.

Neither the confidence intervals nor the estimators in this library need a
noise type -- that is the whole design. Noise identification is here
because users want to know what they are looking at, because bias
corrections for TOTVAR and Theo1 genuinely require it (M3, M7), and because
Stable32 prints an `Alpha` column that parity demands.

Two methods, deliberately:

* **lag-1 autocorrelation** (Riley & Greenhall) operates on the data.
* **the modified-Allan log-log slope** operates on the computed statistic.

They share no arithmetic, so agreement is evidence and disagreement is
information. A single method that quietly returns "white FM" for data that
is nothing of the sort is worse than useless on a report, so the answer
carries a confidence flag and both estimates. See theory/noiseid.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..registry import register

__all__ = [
    "NoiseID",
    "ALPHA_NAMES",
    "lag1_alpha",
    "mdev_slope_alpha",
    "identify",
    "identify_curve",
]

# Power-law index alpha of S_y(f) ~ f^alpha, and the usual names.
ALPHA_NAMES = {
    2: "white PM",
    1: "flicker PM",
    0: "white FM",
    -1: "flicker FM",
    -2: "random-walk FM",
    -3: "flicker walk FM",
    -4: "random-run FM",
}

# Riley's stopping rule: below this the series is "sufficiently
# uncorrelated" that the lag-1 value maps onto a noise exponent. Above it,
# difference again.
DELTA_THRESHOLD = 0.25

# Disagreement beyond this between the two methods drops the confidence to
# "low". Half a power law: the point at which the two would round to
# different noise types.
AGREEMENT_TOLERANCE = 0.5


@dataclass(frozen=True)
class NoiseID:
    """An identified noise type, with the evidence for it."""

    alpha: int
    alpha_lag1: float
    alpha_slope: float
    confidence: str
    n_differences: int

    @property
    def name(self) -> str:
        return ALPHA_NAMES.get(self.alpha, f"alpha={self.alpha}")


def _lag1_acf(z: np.ndarray) -> float:
    """Lag-1 autocorrelation about the sample mean."""
    z = z[np.isfinite(z)]
    if z.size < 3:
        raise ValueError("lag-1 autocorrelation needs at least 3 finite points")
    d = z - z.mean()
    denom = float(np.dot(d, d))
    if denom == 0.0:
        return 0.0
    return float(np.dot(d[:-1], d[1:]) / denom)


def lag1_alpha(y: np.ndarray, max_differences: int = 3) -> tuple[float, int]:
    """Noise exponent from the lag-1 autocorrelation of *frequency* data.

        r1    = lag-1 autocorrelation
        delta = r1 / (1 + r1)
        if delta < 1/4:  alpha = -2*(delta + d)
        else:            difference, d += 1, repeat

    `d` counts how many times the series had to be differenced to become
    sufficiently uncorrelated, and each difference steepens the underlying
    power law by 2 -- which is why it enters the answer with the same
    weight as `delta`.

    Returns `(alpha, d)`. Phase input should be converted to frequency
    first: this function is defined on the frequency domain, and mixing the
    two is a two-unit error in alpha, which is two whole noise types.
    """
    z = np.asarray(y, dtype=np.float64)
    z = z[np.isfinite(z)]
    for d in range(max_differences + 1):
        r1 = _lag1_acf(z)
        delta = r1 / (1.0 + r1) if r1 != -1.0 else np.inf
        if delta < DELTA_THRESHOLD:
            return -2.0 * (delta + d), d
        z = np.diff(z)
        if z.size < 3:
            break
    # Still correlated after the difference limit: steeper than the model
    # covers. Report the edge rather than a number that looks measured.
    return -2.0 * max_differences, max_differences


def mdev_slope_alpha(tau_s: np.ndarray, mdev: np.ndarray) -> float:
    """Noise exponent from the modified-Allan log-log slope.

        mdev(tau) ~ tau^mu,      alpha = -2*mu - 1

    Checks: alpha=2 gives mu=-3/2, alpha=0 gives mu=-1/2, alpha=-2 gives
    mu=+1/2.

    **This is the method that can separate white PM from flicker PM.** The
    plain Allan deviation cannot: both give mu = -1, because AVAR's
    effective bandwidth is set by the hardware cutoff f_h rather than by
    tau. MDEV averages over m phase positions, so its bandwidth scales as
    1/(m*tau0) and the degeneracy breaks. Any noise identification built on
    ADEV slopes alone is guessing between those two.
    """
    tau_s = np.asarray(tau_s, dtype=np.float64)
    mdev = np.asarray(mdev, dtype=np.float64)
    ok = np.isfinite(mdev) & (mdev > 0.0) & (tau_s > 0.0)
    if int(np.count_nonzero(ok)) < 2:
        raise ValueError("mdev slope needs at least two positive points")
    mu = float(np.polyfit(np.log(tau_s[ok]), np.log(mdev[ok]), 1)[0])
    return -2.0 * mu - 1.0


@register("noiseid", "lag1")
def identify(series, m: int = 1, mdev_result=None) -> NoiseID:
    """Identify the noise at averaging factor `m`, cross-checked if possible.

    The frequency data is averaged to `m*tau0` first, because the noise
    type is a property of the timescale being looked at and a record can
    plainly be white PM at one second and random-walk FM at one day.

    `mdev_result` is an optional `StabilityResult` from `mdev`; when given,
    its log-log slope provides the independent second opinion and the two
    are compared. Without it the answer still comes back, flagged
    `single-method`, because one method is not evidence.
    """
    from ..types import FreqSeries, PhaseSeries, to_frequency

    if isinstance(series, PhaseSeries):
        y = to_frequency(series).y
    elif isinstance(series, FreqSeries):
        y = series.y
    else:
        y = np.asarray(series, dtype=np.float64)

    m = int(m)
    if m > 1:
        usable = (y.size // m) * m
        if usable < 3 * m:
            raise ValueError(f"too few points to average by {m}")
        y = y[:usable].reshape(-1, m).mean(axis=1)

    alpha_lag1, n_diff = lag1_alpha(y)

    if mdev_result is None:
        alpha_slope = float("nan")
        confidence = "single-method"
        best = alpha_lag1
    else:
        alpha_slope = mdev_slope_alpha(mdev_result.tau_s, mdev_result.dev)
        gap = abs(alpha_slope - alpha_lag1)
        confidence = "high" if gap <= AGREEMENT_TOLERANCE else "low"
        best = alpha_lag1 if confidence == "high" else 0.5 * (alpha_lag1 + alpha_slope)

    alpha = int(np.clip(round(best), -4, 2))
    return NoiseID(
        alpha=alpha,
        alpha_lag1=alpha_lag1,
        alpha_slope=alpha_slope,
        confidence=confidence,
        n_differences=n_diff,
    )


def identify_curve(series, afs) -> list[NoiseID]:
    """Identify the noise at every averaging factor on a tau grid.

    Noise type is a function of tau, not of a record, and reporting one
    number for a whole measurement is the commonest way to get a stability
    analysis wrong.
    """
    out = []
    for m in np.asarray(afs).tolist():
        try:
            out.append(identify(series, int(m)))
        except ValueError:
            out.append(None)
    return out
