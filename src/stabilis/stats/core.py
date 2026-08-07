"""The shared result type and the estimator driver.

Two contracts live here and both are load-bearing.

**Variances are stored, deviations derived.** `StabilityResult.var` is the
computed quantity; `.dev` is a property. Nothing in this library ever takes
a square root that a caller will then square -- that costs precision and
invites factor-of-two bugs.

**Every estimator returns the coefficient array it averaged**, together
with a scale factor, such that

    variance = scale * mean(c**2)

exactly. One confidence-interval implementation then serves every
estimator, and the wavelet EDOF machinery applies unchanged to the
classical ones. A test re-derives each reported variance from its own
returned coefficients, so the contract cannot silently rot.

The scale is kept *outside* the coefficients rather than folded in as a
1/sqrt(2) or 1/sqrt(6) divisor, and that is a numerical decision, not a
cosmetic one. Folding it in makes every coefficient irrational, which
destroys exactness in cases where the mathematics is exact: for a pure
linear frequency drift the second differences are integers, so the variance
is representable to the last bit -- but only if no irrational division
intervenes. Since the EDOF estimators depend on the coefficients'
correlation structure, which is scale-free, keeping the factor out costs
nothing and buys a bit-exact drift identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..log import record
from ..provenance import Provenance
from ..tau import averaging_factors, taus_from_afs
from ..types import FreqSeries, PhaseSeries, to_phase

__all__ = ["StabilityResult", "run_estimator"]


@dataclass(frozen=True)
class StabilityResult:
    """A stability statistic evaluated over a grid of averaging factors."""

    kind: str
    af: np.ndarray
    tau_s: np.ndarray
    var: np.ndarray
    n_terms: np.ndarray
    rate_hz: float
    provenance: Provenance
    scale: float = 1.0
    coeffs: tuple[np.ndarray, ...] = ()

    @property
    def dev(self) -> np.ndarray:
        """The deviation, sqrt of the stored variance."""
        return np.sqrt(self.var)

    def at(self, af: int) -> float:
        """Variance at one averaging factor. Raises if it was not computed."""
        idx = np.flatnonzero(self.af == af)
        if idx.size == 0:
            raise KeyError(f"averaging factor {af} not in this result: {list(self.af)}")
        return float(self.var[idx[0]])

    def dev_at(self, af: int) -> float:
        """Deviation at one averaging factor."""
        return float(np.sqrt(self.at(af)))


def as_phase(series: PhaseSeries | FreqSeries) -> PhaseSeries:
    """Accept either domain; estimators are written against phase."""
    if isinstance(series, PhaseSeries):
        return series
    if isinstance(series, FreqSeries):
        return to_phase(series)
    raise TypeError(f"expected PhaseSeries or FreqSeries, got {type(series).__name__}")


def run_estimator(
    coeff_fn,
    scale: float,
    kind: str,
    series: PhaseSeries | FreqSeries,
    taus: str = "octave",
    max_af: int | None = None,
    hard_limit_divisor: int = 2,
    keep_coeffs: bool = True,
) -> StabilityResult:
    """Evaluate `coeff_fn` over a tau grid and assemble the result.

    `coeff_fn(x, m, tau0) -> ndarray` returns the coefficients for one
    averaging factor, with non-finite entries wherever a term touched a gap.
    Those are dropped here, so gap handling is automatic and the surviving
    term count -- not the nominal one -- is what reaches the result and
    hence the confidence interval.

    `scale` converts the mean square of the coefficients into the variance:
    1/2 for the Allan family, 1/6 for Hadamard.

    `hard_limit_divisor` is the estimator's algebraic limit: overlapping
    Allan needs m <= (N-1)/2, Hadamard m <= (N-1)/3. It is distinct from the
    tau-grid policy limit, which is a judgement rather than a constraint.
    """
    ps = as_phase(series)
    x, tau0, n = ps.x, ps.tau0_s, ps.n

    hard_limit = (n - 1) // hard_limit_divisor
    requested = max_af
    afs = averaging_factors(n, taus, max_af=max_af)
    afs = afs[afs <= hard_limit]
    if afs.size == 0:
        raise ValueError(
            f"{kind}: no valid averaging factor for n={n} "
            f"(algebraic limit m <= {hard_limit})"
        )

    var, counts, coeff_list, kept = [], [], [], []
    for m in afs:
        c = coeff_fn(x, int(m), tau0)
        c = c[np.isfinite(c)]
        if c.size == 0:
            continue
        var.append(scale * float(np.mean(c**2)))
        counts.append(int(c.size))
        kept.append(int(m))
        if keep_coeffs:
            coeff_list.append(c)

    af = np.asarray(kept, dtype=np.int64)
    prov = record(
        ps.provenance,
        f"stats.{kind}",
        "evaluate",
        {
            "taus": taus,
            "max_af_requested": requested,
            "hard_limit": hard_limit,
            "n": n,
            "gaps": ps.n_gaps,
        },
        f"{af.size} points, af {af[0]}..{af[-1]}" if af.size else "no points",
    )
    return StabilityResult(
        kind=kind,
        af=af,
        tau_s=taus_from_afs(af, ps.rate_hz),
        var=np.asarray(var, dtype=np.float64),
        n_terms=np.asarray(counts, dtype=np.int64),
        rate_hz=ps.rate_hz,
        provenance=prov,
        scale=scale,
        coeffs=tuple(coeff_list),
    )
