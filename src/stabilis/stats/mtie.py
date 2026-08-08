"""MTIE and TIE rms -- the two telecom statistics that are not variances.

    MTIE(tau)    = max over all windows of length tau of (max x - min x)
    TIE rms(tau) = sqrt( mean over i of (x[i+m] - x[i])**2 )

Both are defined on phase (time error) data and both answer a question the
Allan family cannot: *how far apart can two clocks get*, which is what a
telecom or timing-distribution spec is written against. ITU-T masks are
MTIE masks.

**Neither is a variance, and they fail that in different ways.** The
results carry `is_variance=False`, and it matters:

* **MTIE is an extreme value.** The maximum of a sum is not the sum of
  maxima, so the bench solver's sum rule is not approximate for MTIE, it is
  false. It also has no chi-square sampling distribution, so the usual
  confidence interval is meaningless -- extreme-value statistics converge
  to a Gumbel law, not a chi-square, and their spread grows with record
  length rather than shrinking.
* **TIE rms is a mean square, so it *does* obey the sum rule for the
  stochastic part.** It is refused for a different reason: it retains any
  frequency offset between the two sources, and that offset is
  deterministic. Two clocks each 1e-12 off can differ by 2e-12 or by zero
  depending on sign, so the quantity does not combine in quadrature the way
  the solver assumes. Lumping the two refusals together as "extreme-value
  statistics" would be wrong about this one.

## The algorithm

Naively MTIE is `O(N*m)` per averaging factor and `O(N**2)` overall, which
is unusable at the record lengths this is actually wanted for. Instead the
running extremes come from **binary doubling**: with `T_j[i]` the extreme
over `x[i : i+2**j]`,

    T_0 = x,    T_j[i] = max(T_{j-1}[i], T_{j-1}[i + 2**(j-1)])

so any window of length `L` is covered by two overlapping blocks of length
`2**j`, `j = floor(log2(L))`:

    window_max[k] = max(T_j[k], T_j[k + L - 2**j])

Overlap is harmless because max is idempotent -- `max(a, a) = a` -- which
is exactly why this works for extremes and would not for a sum.

Averaging factors are visited in increasing order and the table is advanced
one level at a time, so only the current level is held: `O(N)` memory and
`O(N log N)` work for the whole octave grid, entirely in numpy.
"""

from __future__ import annotations

import numpy as np

from ..log import record
from ..registry import register
from ..tau import averaging_factors, taus_from_afs
from ..types import FreqSeries, PhaseSeries
from .core import StabilityResult, phase_runs

__all__ = ["mtie", "tierms", "running_extremes"]


def running_extremes(x: np.ndarray, length: int, level: int = 0,
                     table: tuple | None = None):
    """Max and min over every window of `length` consecutive samples.

    Returns `(window_max, window_min, level, table)`; pass the last two
    back in on the next call with a larger `length` to reuse the doubling
    work. `window_max` has `x.size - length + 1` entries.

    Non-finite samples propagate, deliberately: a window touching a gap
    comes back NaN and the caller drops it, which is the same convention
    `run_estimator` uses for coefficients spoiled by a gap.
    """
    if length < 1 or length > x.size:
        raise ValueError(
            f"running_extremes: window {length} outside 1..{x.size}")
    hi, lo = (np.asarray(x, dtype=np.float64),) * 2 if table is None else table
    top = int(np.floor(np.log2(length)))
    while level < top:
        step = 1 << level
        hi = np.maximum(hi[:-step], hi[step:])
        lo = np.minimum(lo[:-step], lo[step:])
        level += 1
    block = 1 << level
    tail = length - block
    n = x.size - length + 1
    return (np.maximum(hi[:n], hi[tail:tail + n]),
            np.minimum(lo[:n], lo[tail:tail + n]),
            level, (hi, lo))


def _series_to_phase(series):
    if isinstance(series, PhaseSeries):
        return series
    if isinstance(series, FreqSeries):
        raise TypeError(
            "mtie: needs phase (time error) data. MTIE is a spread of time "
            "errors in seconds, so integrating frequency to get it would "
            "invent an integration constant the answer depends on. Convert "
            "explicitly with types.to_phase if that is what you want."
        )
    raise TypeError(f"expected PhaseSeries, got {type(series).__name__}")


@register("estimator", "mtie")
def mtie(series, taus: str = "octave", max_af: int | None = None,
         gap_policy: str = "auto") -> StabilityResult:
    """Maximum time interval error: the largest peak-to-peak in any window.

        MTIE(m*tau0) = max_k [ max(x[k..k+m]) - min(x[k..k+m]) ]

    A window of averaging time `tau = m*tau0` spans `m+1` phase samples.
    The result is a time in seconds, not a variance -- see the module
    docstring for what that rules out.
    """
    ps = _series_to_phase(series)
    runs, policy = phase_runs(ps, gap_policy)
    longest = max((r.size for r in runs), default=0)
    afs = averaging_factors(ps.n, taus, max_af=max_af)
    afs = afs[afs <= longest - 1]
    if afs.size == 0:
        raise ValueError(
            f"mtie: no valid averaging factor for n={ps.n} with longest "
            f"gap-free run {longest} (needs m <= {longest - 1})"
        )

    values, counts, kept = [], [], []
    states = [(0, None) for _ in runs]
    for m in afs:
        best, windows = -np.inf, 0
        for i, run in enumerate(runs):
            if run.size < m + 1:
                continue
            level, table = states[i]
            hi, lo, level, table = running_extremes(run, int(m) + 1, level, table)
            states[i] = (level, table)
            spread = hi - lo
            good = spread[np.isfinite(spread)]
            if good.size:
                best = max(best, float(good.max()))
                windows += int(good.size)
        if windows:
            # Stored squared so `.dev` is the MTIE itself -- one container,
            # one plotting path. `is_variance=False` is what keeps anything
            # from treating this number as a variance.
            values.append(best**2)
            counts.append(windows)
            kept.append(int(m))

    af = np.asarray(kept, dtype=np.int64)
    prov = record(ps.provenance, "stats.mtie", "evaluate",
                  {"taus": taus, "max_af_requested": max_af, "n": ps.n,
                   "gap_policy": policy, "runs": len(runs)},
                  f"{af.size} points, af {af[0]}..{af[-1]}" if af.size else "none")
    return StabilityResult(
        kind="mtie", af=af, tau_s=taus_from_afs(af, ps.rate_hz),
        var=np.asarray(values, dtype=np.float64),
        n_terms=np.asarray(counts, dtype=np.int64),
        rate_hz=ps.rate_hz, provenance=prov, span_factor=1,
        run_lengths=tuple(int(r.size) for r in runs),
        is_variance=False,
    )


@register("estimator", "tierms")
def tierms(series, taus: str = "octave", max_af: int | None = None,
           gap_policy: str = "auto") -> StabilityResult:
    """RMS time interval error over intervals of length tau.

        TIE rms(m*tau0) = sqrt( mean_i [ (x[i+m] - x[i])**2 ] )

    The root mean square of the *first* difference of phase, so unlike the
    Allan family it does not remove a constant frequency offset -- a clock
    running fast by a fixed amount contributes `offset * tau` and comes to
    dominate at long tau. That is a property of the definition, not a
    defect, and it is why the number is used for timing distribution
    specifications where the offset is exactly what matters.
    """
    ps = _series_to_phase(series)
    runs, policy = phase_runs(ps, gap_policy)
    longest = max((r.size for r in runs), default=0)
    afs = averaging_factors(ps.n, taus, max_af=max_af)
    afs = afs[afs <= longest - 1]
    if afs.size == 0:
        raise ValueError(
            f"tierms: no valid averaging factor for n={ps.n} with longest "
            f"gap-free run {longest} (needs m <= {longest - 1})"
        )

    values, counts, kept, coeff_list = [], [], [], []
    for m in afs:
        parts = [r[int(m):] - r[:-int(m)] for r in runs if r.size > m]
        parts = [p[np.isfinite(p)] for p in parts]
        pooled = np.concatenate(parts) if parts else np.empty(0)
        if pooled.size == 0:
            continue
        values.append(float(np.mean(pooled**2)))
        counts.append(int(pooled.size))
        kept.append(int(m))
        coeff_list.append(pooled)

    af = np.asarray(kept, dtype=np.int64)
    prov = record(ps.provenance, "stats.tierms", "evaluate",
                  {"taus": taus, "max_af_requested": max_af, "n": ps.n,
                   "gap_policy": policy, "runs": len(runs)},
                  f"{af.size} points, af {af[0]}..{af[-1]}" if af.size else "none")
    return StabilityResult(
        kind="tierms", af=af, tau_s=taus_from_afs(af, ps.rate_hz),
        var=np.asarray(values, dtype=np.float64),
        n_terms=np.asarray(counts, dtype=np.int64),
        rate_hz=ps.rate_hz, provenance=prov, coeffs=tuple(coeff_list),
        span_factor=1,
        run_lengths=tuple(int(r.size) for r in runs),
        is_variance=False,
    )
