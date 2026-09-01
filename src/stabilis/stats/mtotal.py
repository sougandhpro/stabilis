"""MTOT, TTOT and HTOT: the total variances built on 3m subsequences.

NIST SP 1065 sections 5.2.12-5.2.14, eq. (27), (28) and (29), read from the
digital original. TOTVAR (`total.py`) extends the *whole record* once by
reflection; this family instead takes every 3m-point subsequence, removes
its own linear trend, extends *that* by reflection to 9m points, and
averages the resulting variance over all N-3m+1 subsequences.

    subsequence   s = a[n : n+3m]                     3m points
    detrend       s0 = s - c*i,  c from the half-means
    extend        s# = [s0 reversed, s0, s0 reversed]  9m points
    average       abar[j] = mean(s#[j : j+m])
    difference    z[i] = abar[i] - 2 abar[i+m] + abar[i+2m],  i = 0 .. 6m-1

with `a` the phase for MTOT and TTOT and the fractional frequency for HTOT.
At the largest averaging factor `m = N/3` the outer sum has one term left
but the inner one still has 6m, which is the whole point: the estimate does
not collapse at long tau the way the Allan family's does.

**The reflection here is uninverted**, unlike TOTVAR's. SP 1065 writes it
`0x#_{n-l} = 0x_{n+l-1}`, a plain half-point mirror that duplicates the
edge sample. TOTVAR's `2x_1 - x` continues the trend across the boundary
instead; it can, because it has not removed one. Here the trend has already
been taken out of the subsequence, so a mirror does not manufacture a
corner and the inversion would be doing the job twice.

**Cross-checked against a second implementation.** `allantools.mtotdev` and
`allantools.htotdev` return these numbers to every printed digit on the
same records. That is two independent readings of the same published
algorithm agreeing, which is the only interop evidence available: no
Stable32 fixture for this family exists in the repository. See
`theory/mtotal.md` for what that does and does not establish, and for the
disagreement with SP 1065's own Table 11 that it exposes.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..registry import register
from ..types import FreqSeries, PhaseSeries
from .classical import SCALE_ALLAN, SCALE_HADAMARD, _coeffs_ohvar
from .core import MAX_COEFF_BYTES, StabilityResult, run_estimator

__all__ = ["mtotdev", "ttotdev", "htotdev", "total_subsequence_terms"]


def total_subsequence_terms(a: np.ndarray, m: int) -> np.ndarray:
    """The `(N-3m+1, 6m)` array of second differences, SP 1065 Fig. 10.

    Public because it is the whole algorithm, and being able to look at one
    subsequence's extension directly is worth more than any description of
    it. Rows are subsequences, columns the 6m fully overlapping second
    differences of m-point averages within one extended subsequence.

    The detrend divides the difference of the half-means by the distance
    between their centroids, which is `3m - floor(3m/2)`. SP 1065 says
    "dividing by half the interval", exact when `3m` is even; when `m` is
    odd the two halves cannot both be `1.5m` long, and the centroid
    distance is what "half the interval" means in the case the text does
    not distinguish.
    """
    a = np.asarray(a, dtype=np.float64)
    width = 3 * int(m)
    if a.size < width:
        return np.empty((0, 6 * int(m)), dtype=np.float64)

    s = sliding_window_view(a, width)
    half = width // 2
    slope = (s[:, width - half:].mean(axis=1) - s[:, :half].mean(axis=1))
    s0 = s - (slope / (width - half))[:, None] * np.arange(width)

    ext = np.concatenate((s0[:, ::-1], s0, s0[:, ::-1]), axis=1)
    running = np.concatenate(
        (np.zeros((ext.shape[0], 1)), np.cumsum(ext, axis=1)), axis=1)
    mean = (running[:, m:] - running[:, :-m]) / m
    return mean[:, :6 * m] - 2.0 * mean[:, m:7 * m] + mean[:, 2 * m:8 * m]


def _guard_size(n: int, m: int, kind: str) -> None:
    """This family's working set is `(N-3m+1) x 9m`, not `N` per tau.

    The driver's own budget is computed from the record length and is a
    large under-estimate here: the coefficient count alone is `6m(N-3m+1)`,
    which peaks near `m = N/6` at half of `N**2`. A 100 000-point record
    would ask for 40 GB there. Caught with the number rather than by the
    allocator.
    """
    rows = n - 3 * m + 1
    if rows < 1:
        return
    want = rows * 9 * m * 8
    if want > MAX_COEFF_BYTES:
        raise MemoryError(
            f"{kind}: at m={m} on {n} points the extended subsequences are "
            f"{rows} x {9 * m}, about {want / 1e9:.1f} GB, past the "
            f"{MAX_COEFF_BYTES / 1e9:.1f} GB limit. This family is "
            "quadratic in the record length near m = N/6, unlike every "
            "other estimator here; use a coarser tau grid, a max_af, or a "
            "shorter record.")


def _coeffs_mtotvar(x: np.ndarray, m: int, tau0: float) -> np.ndarray:
    """MTOT coefficients: `z / (m*tau0)`, with `scale = 1/2`.

        Mod Tot var = sum(z**2) / (2 (m tau0)**2 (N-3m+1) 6m)

    SP 1065 eq. (27). Flattening the `(N-3m+1, 6m)` block is exactly the
    double sum, and the mean of the squares over the flat array is the
    double sum over its own count -- so the estimator's own contract
    reproduces the published prefactor with nothing left over.
    """
    _guard_size(x.size, m, "mtotdev")
    return (total_subsequence_terms(x, m) / (m * tau0)).ravel()


def _coeffs_ttotvar(x: np.ndarray, m: int, tau0: float) -> np.ndarray:
    """TTOT coefficients: `z` itself, with `scale = 1/6`.

        sigma2_x,total(tau) = (tau**2 / 3) * Mod Tot var(tau)

    SP 1065 eq. (28), the same rescaling that turns MVAR into TVAR. Written
    as a coefficient change rather than a multiplication afterwards so that
    the confidence path and the cross-tau covariance see the array the
    statistic actually averages, exactly as `_coeffs_tvar` does.
    """
    _guard_size(x.size, m, "ttotdev")
    return total_subsequence_terms(x, m).ravel()


def _coeffs_htotvar(x: np.ndarray, m: int, tau0: float) -> np.ndarray:
    """HTOT coefficients: `z` on the *frequency* record, `scale = 1/6`.

        sigma2_TotalH = sum(H**2) / (6 (N-3m+1) 6m)

    SP 1065 eq. (29). The subsequences are of fractional frequency, so the
    trend removed from each is a frequency *drift*, and the second
    difference of frequency averages is the Hadamard one.

    **`m = 1` is the overlapping Hadamard variance, by definition rather
    than by fallback.** At `m = 1` the subsequence is three points, the
    detrend takes one degree of freedom out of them, and the six second
    differences of one-point averages are all drawn from what is left; the
    construction says nothing. Stable32 and `allantools` both return OHVAR
    there and so does this. It is an identity, not a substitution: at
    `m = 1` the second difference of frequency *is* the third difference of
    phase, so `_coeffs_ohvar(x, 1, tau0)` is the same array the general
    branch would produce if the machinery did not degenerate, and a test
    asserts the two agree bit for bit against `allantools`.
    """
    if m == 1:
        return _coeffs_ohvar(x, 1, tau0)
    y = np.diff(x) / tau0
    _guard_size(y.size, m, "htotdev")
    return total_subsequence_terms(y, m).ravel()


@register("estimator", "mtotdev")
def mtotdev(
    series: PhaseSeries | FreqSeries,
    taus: str = "octave",
    max_af: int | None = None,
    keep_coeffs: bool = True,
    gap_policy: str = "auto",
) -> StabilityResult:
    """Modified total deviation. See theory/mtotal.md."""
    return _unordered(run_estimator(
        _coeffs_mtotvar, SCALE_ALLAN, "mtotdev", series, taus, max_af, 3,
        keep_coeffs, gap_policy, None,
    ))


@register("estimator", "ttotdev")
def ttotdev(
    series: PhaseSeries | FreqSeries,
    taus: str = "octave",
    max_af: int | None = None,
    keep_coeffs: bool = True,
    gap_policy: str = "auto",
) -> StabilityResult:
    """Time total deviation, in seconds. See theory/mtotal.md."""
    return _unordered(run_estimator(
        _coeffs_ttotvar, SCALE_HADAMARD, "ttotdev", series, taus, max_af, 3,
        keep_coeffs, gap_policy, None,
    ))


@register("estimator", "htotdev")
def htotdev(
    series: PhaseSeries | FreqSeries,
    taus: str = "octave",
    max_af: int | None = None,
    keep_coeffs: bool = True,
    gap_policy: str = "auto",
) -> StabilityResult:
    """Hadamard total deviation. See theory/mtotal.md."""
    return _unordered(run_estimator(
        _coeffs_htotvar, SCALE_HADAMARD, "htotdev", series, taus, max_af, 3,
        keep_coeffs, gap_policy, None,
    ))


def _unordered(result: StabilityResult) -> StabilityResult:
    """Mark the coefficients as not one time-ordered sequence.

    Every other estimator here returns its terms in record order, and three
    things downstream rely on it: `eta1` reads an autocorrelation off them,
    `taucov` reads the cross-tau covariance off them, and the cross-variance
    pairs them index by index. This family's array is `N-3m+1` blocks of 6m
    terms, each block drawn from a *different* subsequence and mostly from
    its reflection -- consecutive entries across a block boundary are not
    consecutive in time and are not even from the same data. An
    autocorrelation over that is a number with no meaning, so the flag makes
    the four consumers refuse instead of computing one.
    """
    from dataclasses import replace
    return replace(result, coeffs_are_ordered=False)
