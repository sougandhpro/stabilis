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
from ..preprocess import segments
from ..provenance import Provenance
from ..tau import averaging_factors, taus_from_afs
from ..types import FreqSeries, PhaseSeries, phase_from_frequency, to_phase

__all__ = ["StabilityResult", "phase_runs", "run_estimator", "GAP_POLICIES"]

# How a gap is allowed to affect the terms around it.
#
# "mask"    -- keep the record whole; terms whose window touches a gap come
#              back non-finite and are dropped. This assumes the timescale
#              ran continuously across the gap, so the phase on either side
#              is on one common origin. True for a dropped sample on a known
#              grid; false after a counter reset.
# "segment" -- analyse each contiguous run on its own origin and pool the
#              coefficients. Costs only the 2m (or 3m) terms spanning the
#              gap. Required for frequency input, and correct for phase
#              whose origin may have moved.
# "auto"    -- "segment" for frequency, "mask" for phase.
GAP_POLICIES = ("auto", "mask", "segment")


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
    # Width of one term in phase samples, as a multiple of the averaging
    # factor: 2m for the Allan family, 3m for Hadamard and modified. It is
    # the estimator's algebraic limit and also what sets the conservative
    # EDOF, so it is carried rather than re-derived from `kind`.
    span_factor: int = 2

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
    """Accept either domain as a single series; estimators want phase.

    Only valid for ungapped frequency input -- see `to_phase`. Estimators
    use `phase_runs` instead.
    """
    if isinstance(series, PhaseSeries):
        return series
    if isinstance(series, FreqSeries):
        return to_phase(series)
    raise TypeError(f"expected PhaseSeries or FreqSeries, got {type(series).__name__}")


def phase_runs(
    series: PhaseSeries | FreqSeries, gap_policy: str = "auto"
) -> tuple[list[np.ndarray], str]:
    """The contiguous phase runs an estimator should be evaluated over.

    Returns `(runs, policy)` where `policy` is `gap_policy` resolved against
    the input domain. Under "mask" there is exactly one run, the whole
    record with its NaNs intact; under "segment" there is one run per
    stretch of live data, each integrated from its own zero.

    Why frequency data cannot use "mask": integrating y through a NaN gives
    NaN for the entire remainder of the record, so a single missing sample
    at the midpoint would discard half the data and shift the answer by
    percent. The Allan variance of frequency data never needs absolute
    phase, so a gap at y[k] should invalidate only the terms that span it.
    Pooling the runs' coefficients is exact rather than approximate: the
    estimator is a mean of squared coefficients and terms from different
    runs are drawn from the same distribution.
    """
    if gap_policy not in GAP_POLICIES:
        raise ValueError(f"gap_policy must be one of {GAP_POLICIES}, got {gap_policy!r}")

    if isinstance(series, FreqSeries):
        if gap_policy == "mask" and series.n_gaps:
            raise ValueError(
                "gap_policy='mask' cannot be used on gapped frequency data: "
                "integrating through a gap propagates NaN to the end of the "
                "record. Use 'segment' (the default)."
            )
        tau0 = series.tau0_s
        runs = [phase_from_frequency(series.y[a:b], tau0) for a, b in segments(series.y)]
        return runs, "segment"

    if not isinstance(series, PhaseSeries):
        raise TypeError(f"expected PhaseSeries or FreqSeries, got {type(series).__name__}")

    policy = "mask" if gap_policy == "auto" else gap_policy
    if policy == "mask":
        return [series.x], policy
    return [series.x[a:b] for a, b in segments(series.x)], policy


def run_estimator(
    coeff_fn,
    scale: float,
    kind: str,
    series: PhaseSeries | FreqSeries,
    taus: str = "octave",
    max_af: int | None = None,
    hard_limit_divisor: int = 2,
    keep_coeffs: bool = True,
    gap_policy: str = "auto",
) -> StabilityResult:
    """Evaluate `coeff_fn` over a tau grid and assemble the result.

    `coeff_fn(x, m, tau0) -> ndarray` returns the coefficients for one
    averaging factor over one contiguous phase run, with non-finite entries
    wherever a term touched a gap. Those are dropped here, so gap handling
    is automatic and the surviving term count -- not the nominal one -- is
    what reaches the result and hence the confidence interval.

    `scale` converts the mean square of the coefficients into the variance:
    1/2 for the Allan family, 1/6 for Hadamard.

    `hard_limit_divisor` is the estimator's algebraic limit: overlapping
    Allan needs m <= (N-1)/2, Hadamard m <= (N-1)/3. It is distinct from the
    tau-grid policy limit, which is a judgement rather than a constraint.
    Under "segment" it binds on the *longest run*, not the record, because
    no term can straddle a gap.

    `gap_policy` is described at `phase_runs` and in theory/gaps.md.
    """
    runs, policy = phase_runs(series, gap_policy)
    tau0, rate_hz = series.tau0_s, series.rate_hz
    n = series.n + 1 if isinstance(series, FreqSeries) else series.n
    longest = max((r.size for r in runs), default=0)

    hard_limit = (longest - 1) // hard_limit_divisor
    requested = max_af
    afs = averaging_factors(n, taus, max_af=max_af)
    afs = afs[afs <= hard_limit]
    if afs.size == 0:
        raise ValueError(
            f"{kind}: no valid averaging factor for n={n} with longest "
            f"gap-free run {longest} (algebraic limit m <= {hard_limit})"
        )

    var, counts, coeff_list, kept = [], [], [], []
    for m in afs:
        parts = [coeff_fn(r, int(m), tau0) for r in runs]
        c = parts[0] if len(parts) == 1 else np.concatenate(parts)
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
        series.provenance,
        f"stats.{kind}",
        "evaluate",
        {
            "taus": taus,
            "max_af_requested": requested,
            "hard_limit": hard_limit,
            "n": n,
            "gaps": series.n_gaps,
            "gap_policy": policy,
            "runs": len(runs),
            "longest_run": longest,
        },
        f"{af.size} points, af {af[0]}..{af[-1]}" if af.size else "no points",
    )
    return StabilityResult(
        kind=kind,
        af=af,
        tau_s=taus_from_afs(af, rate_hz),
        var=np.asarray(var, dtype=np.float64),
        n_terms=np.asarray(counts, dtype=np.int64),
        rate_hz=rate_hz,
        provenance=prov,
        scale=scale,
        coeffs=tuple(coeff_list),
        span_factor=hard_limit_divisor,
    )
