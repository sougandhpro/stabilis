"""Confidence intervals on a stability estimate.

The sampling law is chi-square. If a variance estimate has `eta` equivalent
degrees of freedom then

    eta * sigma_hat^2 / sigma^2  ~  chi^2(eta)

so a two-sided interval at confidence `level` is

    lower = eta * sigma_hat^2 / chi2_ppf(1 - alpha/2, eta)
    upper = eta * sigma_hat^2 / chi2_ppf(alpha/2, eta),   alpha = 1 - level

**The interval is asymmetric and increasingly so as eta falls**, which is
exactly what happens at long tau, which is where the interesting physics
lives. Anyone quoting `sigma +/- delta` at the last few points is quoting a
symmetric approximation to a distribution that is not remotely symmetric --
at eta = 2 the 68.3% interval runs from 0.55*sigma to 2.2*sigma.

Deviation bounds are the square roots of the variance bounds. That is a
monotone transformation, so the coverage is preserved exactly; it is *not*
the same as `sigma_hat +/- something`, and it is not centred on
`sigma_hat`. See theory/ci.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._specfun import chi2_ppf
from ..log import record
from ..provenance import Provenance
from .core import StabilityResult
from .edof import EDOF_METHODS, edof

__all__ = ["Confidence", "confidence", "chi2_interval", "ONE_SIGMA", "TWO_SIGMA"]

# Nominal coverage of +/-1 and +/-2 standard deviations of a normal. Quoted
# this way rather than as 0.68 and 0.95 because that is what "one sigma"
# means, and the difference shows up when coverage is measured.
ONE_SIGMA = 0.682689492137086
TWO_SIGMA = 0.954499736103642


def chi2_interval(var, eta, level: float = ONE_SIGMA):
    """Chi-square bounds on a variance, given its degrees of freedom.

        [ eta*var / chi2_ppf(1 - alpha/2, eta),
          eta*var / chi2_ppf(alpha/2, eta) ],   alpha = 1 - level

    Note the crossed indices: the *upper* bound on the variance comes from
    the *lower* quantile. Getting that backwards produces an interval that
    is inverted but still looks plausible on a log plot, so there is a test
    asserting lower <= var <= upper.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    var = np.asarray(var, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    alpha = 1.0 - level
    lo = eta * var / chi2_ppf(1.0 - 0.5 * alpha, eta)
    hi = eta * var / chi2_ppf(0.5 * alpha, eta)
    return lo, hi


@dataclass(frozen=True)
class Confidence:
    """Interval on each point of a `StabilityResult`, plus how it was got."""

    level: float
    edof: np.ndarray
    var_lo: np.ndarray
    var_hi: np.ndarray
    method: tuple[str, ...]
    provenance: Provenance

    @property
    def dev_lo(self) -> np.ndarray:
        """Lower deviation bound. Monotone, so coverage is preserved."""
        return np.sqrt(self.var_lo)

    @property
    def dev_hi(self) -> np.ndarray:
        return np.sqrt(self.var_hi)


def confidence(
    result: StabilityResult,
    level: float = ONE_SIGMA,
    method: str = "auto",
) -> Confidence:
    """Confidence interval for every tau of a stability result.

    Requires the coefficient arrays, so the estimator must have been called
    with `keep_coeffs=True` (the default). That is the single-CI-path
    contract from core.py doing its job: one implementation here serves
    every estimator, present and future, because they all report the array
    they averaged.
    """
    if method not in EDOF_METHODS:
        raise ValueError(f"method must be one of {EDOF_METHODS}, got {method!r}")
    if len(result.coeffs) != result.af.size:
        raise ValueError(
            f"{result.kind}: confidence needs the coefficient arrays; "
            "the estimator was called with keep_coeffs=False"
        )

    etas, used = [], []
    for m, c in zip(result.af, result.coeffs):
        eta, how = edof(c, int(m), result.span_factor, method)
        etas.append(eta)
        used.append(how)

    eta = np.asarray(etas, dtype=np.float64)
    lo, hi = chi2_interval(result.var, eta, level)
    fell_back = sum(1 for h, m in zip(used, result.af) if h != method and method != "auto")
    prov = record(
        result.provenance,
        "stats.ci",
        "confidence",
        {
            "level": level,
            "method_requested": method,
            "methods_used": sorted(set(used)),
            "edof_min": float(eta.min()),
            "edof_max": float(eta.max()),
        },
        f"{eta.size} intervals" + (f", {fell_back} fell back" if fell_back else ""),
    )
    return Confidence(
        level=level,
        edof=eta,
        var_lo=lo,
        var_hi=hi,
        method=tuple(used),
        provenance=prov,
    )
