"""The classical variance family, computed directly from phase.

Deliberately not routed through the MODWT filter bank. The Haar wavelet
variance equals the overlapping Allan variance, and that identity is the
intellectual core of this library -- but making it the *implementation*
would put boundary handling and shift-equivariance on the critical path for
no user-visible gain. Written directly here, the identity becomes an
independent check of two separate code paths instead of a tautology.

Each `_coeffs_*` function returns coefficients `c` with `variance = scale *
mean(c**2)` exactly, and NaN wherever a term touched a gap. See core.py for
why the scale is kept outside the coefficients.
"""

from __future__ import annotations

import numpy as np

from ..registry import register
from ..types import FreqSeries, PhaseSeries
from .core import StabilityResult, run_estimator

__all__ = ["adev", "oadev", "mdev", "tdev", "ohdev"]

# variance = scale * mean(coeffs**2). Kept out of the coefficients so that
# no irrational divisor destroys exactness -- see core.py.
SCALE_ALLAN = 0.5
SCALE_HADAMARD = 1.0 / 6.0


def _coeffs_oavar(x: np.ndarray, m: int, tau0: float) -> np.ndarray:
    """Overlapping Allan variance, from the second difference of phase.

        d[i] = x[i+2m] - 2*x[i+m] + x[i],        i = 0 .. N-2m-1
        c[i] = d[i] / (m * tau0)
        avar = mean(c**2) / 2 = sum(d**2) / (2 * (N-2m) * (m*tau0)**2)
    """
    d = x[2 * m :] - 2.0 * x[m:-m] + x[: -2 * m]
    return d / (m * tau0)


def _coeffs_avar(x: np.ndarray, m: int, tau0: float) -> np.ndarray:
    """Non-overlapping Allan variance: decimate phase, then second difference.

        xd   = x[::m]
        d[k] = xd[k+2] - 2*xd[k+1] + xd[k]
        c[k] = d[k] / (m * tau0)
    """
    xd = x[::m]
    if xd.size < 3:
        return np.empty(0, dtype=np.float64)
    d = xd[2:] - 2.0 * xd[1:-1] + xd[:-2]
    return d / (m * tau0)


def _coeffs_mvar(x: np.ndarray, m: int, tau0: float) -> np.ndarray:
    """Modified Allan variance: the second difference, averaged over m phases.

        z[i] = x[i+2m] - 2*x[i+m] + x[i]
        S[j] = sum_{i=j}^{j+m-1} z[i],           j = 0 .. N-3m
        c[j] = S[j] / (m**2 * tau0)
        mvar = mean(c**2) / 2 = sum(S**2) / (2 * m**4 * tau0**2 * (N-3m+1))

    The inner average over m phase positions is what makes the effective
    high-frequency cutoff scale as 1/(m*tau0), which is why this estimator
    separates white PM from flicker PM where the Allan variance cannot.
    """
    z = x[2 * m :] - 2.0 * x[m:-m] + x[: -2 * m]
    return _running_sum(z, m) / (m * m * tau0)


def _coeffs_tvar(x: np.ndarray, m: int, tau0: float) -> np.ndarray:
    """Time variance, a rescaling of the modified Allan variance.

        tvar(tau) = (tau**2 / 3) * mvar(tau),    tau = m * tau0

    so with S[j] as in _coeffs_mvar, and scale 1/6, the coefficients reduce
    to

        c[j] = S[j] / m
        tvar = mean(c**2) / 6 = sum(S**2) / (6 * m**2 * (N-3m+1))
    """
    z = x[2 * m :] - 2.0 * x[m:-m] + x[: -2 * m]
    return _running_sum(z, m) / m


def _coeffs_ohvar(x: np.ndarray, m: int, tau0: float) -> np.ndarray:
    """Overlapping Hadamard variance, from the third difference of phase.

        d[i] = x[i+3m] - 3*x[i+2m] + 3*x[i+m] - x[i],   i = 0 .. N-3m-1
        c[i] = d[i] / (m * tau0)
        hvar = mean(c**2) / 6 = sum(d**2) / (6 * (N-3m) * (m*tau0)**2)

    The third difference annihilates linear frequency drift exactly, which
    is why this is the right statistic for a drifting oscillator and why it
    needs no detrending beforehand.
    """
    d = (
        x[3 * m :]
        - 3.0 * x[2 * m : -m]
        + 3.0 * x[m : -2 * m]
        - x[: -3 * m]
    )
    return d / (m * tau0)


def _running_sum(z: np.ndarray, m: int) -> np.ndarray:
    """Sliding sum of m consecutive entries; NaN if the window met a gap.

    Uses a cumulative sum over a gap-zeroed copy plus a cumulative count of
    gaps, so a NaN spoils exactly the windows containing it rather than
    poisoning everything downstream of it.
    """
    ok = np.isfinite(z)
    csum = np.concatenate(([0.0], np.cumsum(np.where(ok, z, 0.0))))
    total = csum[m:] - csum[:-m]
    bad = np.concatenate(([0], np.cumsum(~ok, dtype=np.int64)))
    return np.where(bad[m:] - bad[:-m] == 0, total, np.nan)


@register("estimator", "oadev")
def oadev(
    series: PhaseSeries | FreqSeries,
    taus: str = "octave",
    max_af: int | None = None,
    keep_coeffs: bool = True,
    gap_policy: str = "auto",
) -> StabilityResult:
    """Overlapping Allan deviation. See theory/avar.md."""
    return run_estimator(
        _coeffs_oavar, SCALE_ALLAN, "oadev", series, taus, max_af, 2,
        keep_coeffs, gap_policy,
    )


@register("estimator", "adev")
def adev(
    series: PhaseSeries | FreqSeries,
    taus: str = "octave",
    max_af: int | None = None,
    keep_coeffs: bool = True,
    gap_policy: str = "auto",
) -> StabilityResult:
    """Non-overlapping Allan deviation. See theory/avar.md."""
    return run_estimator(
        _coeffs_avar, SCALE_ALLAN, "adev", series, taus, max_af, 2,
        keep_coeffs, gap_policy,
    )


@register("estimator", "mdev")
def mdev(
    series: PhaseSeries | FreqSeries,
    taus: str = "octave",
    max_af: int | None = None,
    keep_coeffs: bool = True,
    gap_policy: str = "auto",
) -> StabilityResult:
    """Modified Allan deviation. See theory/mvar.md."""
    return run_estimator(
        _coeffs_mvar, SCALE_ALLAN, "mdev", series, taus, max_af, 3,
        keep_coeffs, gap_policy,
    )


@register("estimator", "tdev")
def tdev(
    series: PhaseSeries | FreqSeries,
    taus: str = "octave",
    max_af: int | None = None,
    keep_coeffs: bool = True,
    gap_policy: str = "auto",
) -> StabilityResult:
    """Time deviation, in seconds. See theory/tdev.md."""
    return run_estimator(
        _coeffs_tvar, SCALE_HADAMARD, "tdev", series, taus, max_af, 3,
        keep_coeffs, gap_policy,
    )


@register("estimator", "ohdev")
def ohdev(
    series: PhaseSeries | FreqSeries,
    taus: str = "octave",
    max_af: int | None = None,
    keep_coeffs: bool = True,
    gap_policy: str = "auto",
) -> StabilityResult:
    """Overlapping Hadamard deviation. See theory/hvar.md."""
    return run_estimator(
        _coeffs_ohvar, SCALE_HADAMARD, "ohdev", series, taus, max_af, 3,
        keep_coeffs, gap_policy,
    )
