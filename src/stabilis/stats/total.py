"""Total variance: the Allan variance with the record extended by reflection.

    Tot var(tau) = 1/(2 tau^2 (N-2)) * sum_{i=2}^{N-1}
                       [x*_{i-m} - 2 x*_i + x*_{i+m}]^2

NIST SP 1065 eq. (25), section 5.2.11, read from the digital original. The
`N` phase values are extended by reflection about **both** endpoints into a
virtual sequence `x*` running from `i = 3-N` to `i = 2N-2`, length `3N-4`,
with the original data in the centre:

    x*_{1-j} = 2 x_1 - x_{1+j}          j = 1 .. N-2
    x*_{N+j} = 2 x_N - x_{N-j}          j = 1 .. N-2

**Why bother.** The Allan variance runs out of data at long tau -- at
`tau = T/2` it has one term left, and the estimate collapses. TOTVAR keeps
`N-2` terms at *every* tau, because the sum always runs over the original
interior points however large `m` gets. That buys a large confidence
advantage at long averaging times, which is exactly where a stability
measurement is usually trying to say something.

**What it costs.** The reflected data is not new information, and TOTVAR
is biased *low* for the steeper FM noises. The correction is in `bias.py`
and needs a noise type, which is why this estimator is coupled to noise
identification in a way the classical family is not.

**The reflection is about the endpoint value, not a mirror.** `2 x_1 - x`
continues the trend across the boundary rather than folding it back; a
plain mirror would put a corner at each end and manufacture a large second
difference exactly where the estimator is most sensitive. That is what
makes the time-reversal test worth having.
"""

from __future__ import annotations

import numpy as np

from ..registry import register
from .core import Shape, run_estimator

__all__ = ["totdev", "totvar_extend"]


def totvar_extend(x: np.ndarray) -> np.ndarray:
    """The doubly reflected virtual sequence `x*`, length `3N-4`.

        x*_{1-j} = 2 x_1 - x_{1+j}
        x*_{N+j} = 2 x_N - x_{N-j}          j = 1 .. N-2

    Public because it is the whole idea of the estimator and worth being
    able to look at directly.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 3:
        raise ValueError(f"totvar needs at least 3 phase points, got {n}")
    # j runs 1..N-2; the left piece is written in increasing index order,
    # which is j descending.
    left = (2.0 * x[0] - x[1:n - 1])[::-1]
    right = 2.0 * x[-1] - x[n - 2:0:-1]
    return np.concatenate([left, x, right])


def _coeffs_totvar(x: np.ndarray, m: int, tau0: float) -> np.ndarray:
    """Second differences of the extended sequence, at every interior point.

        c[k] = (x*_{k-m} - 2 x*_k + x*_{k+m}) / (m * tau0),   k = 2 .. N-1

    so that `totvar = mean(c**2) / 2`, the contract every estimator here
    follows. **The count is `N-2` whatever `m` is** -- that is the property
    the whole estimator exists for, and it is why the term count does not
    fall away at long tau the way the Allan family's does.
    """
    n = x.size
    if n < 3:
        return np.empty(0, dtype=np.float64)
    ext = totvar_extend(x)
    base = n - 2                       # where x[0] sits inside the extension
    i = np.arange(1, n - 1) + base     # the interior points, 0-based
    d = ext[i - m] - 2.0 * ext[i] + ext[i + m]
    return d / (m * tau0)


@register("estimator", "totdev")
def totdev(
    series,
    taus: str = "octave",
    max_af: int | None = None,
    keep_coeffs: bool = True,
    gap_policy: str = "auto",
):
    """Total deviation: TOTDEV, the square root of TOTVAR.

    Equal in expectation to the Allan deviation for white and flicker PM
    and for white FM. For flicker FM and random-walk FM it is biased low
    and needs the correction in `stats.bias.totvar_bias`, which is why
    that correction takes a noise type.
    """
    return run_estimator(
        _coeffs_totvar,
        scale=0.5,
        kind="totdev",
        series=series,
        taus=taus,
        max_af=max_af,
        keep_coeffs=keep_coeffs,
        gap_policy=gap_policy,
        # Same algebraic limit as the Allan family: the estimator is
        # defined past tau = T/2 but the reflected data then outnumbers the
        # real data, and Stable32 stops there too.
        hard_limit_divisor=2,
        shape=Shape(2, False, True),
    )
