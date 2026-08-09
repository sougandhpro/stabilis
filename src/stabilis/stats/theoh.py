"""ThêoBR and ThêoH -- Thêo1 with its bias removed, and spliced onto AVAR.

Thêo1 reaches 75 % of the record but is biased against AVAR for every noise
type except white FM. `stats.bias.theo1_bias` removes that bias **if you
already know the noise type**. ThêoBR removes it without knowing, which is
the point of it, by measuring the ratio off the data itself:

    ThêoBR(m, tau0, N) = [ 1/(n+1) * sum_{i=0}^{n}
                             AVAR(m = 9+3i, tau0, N)
                           / Theo1(m = 12+4i, tau0, N) ] * Theo1(m, tau0, N)

    n = floor(N/6) - 3

NIST SP 1065 eq. (33), §5.2.16, from Howe 2006. The two averaging factors
track each other **at equal tau** -- `0.75(12+4i) = 9+3i` -- which is the
argument that settled Thêo1's tau convention; see `theo1.py`.

ThêoH then puts AVAR and ThêoBR on one axis, eq. (34):

    ThêoH(m, tau0, N) = AVAR(m)   for   m tau0 <= k
                        ThêoBR(m) for   0.75 m tau0 > k,  m even

with `k` the largest available AVAR tau at or below 20 % of the record. The
result is one curve from tau0 to 0.75 T with no noise-type assumption
anywhere in it, which is what SP 1065 means by calling it "the best
statistic available for estimating the stability level and type of noise of
a frequency source".

**The cost is real and it is in the bias ratio, not in the statistic.** The
sum runs over about `N/6` Thêo1 evaluations at averaging factors reaching
`2N/3`, so it grows as roughly `N^3`. Howe's Fast TheoBR (McGee & Howe
2008) reduces it by phase-averaging the record first; that is not
implemented, so `theobr` estimates its own cost and refuses rather than
disappearing for an hour, naming decimation as the way through.

**The ratio is noisy on one record, and that noise is common to every
point.** Measured over 12 realizations of white FM, where the true ratio is
1.00 by definition:

    N = 1024    mean 1.07    single records 0.53 - 1.68
    N = 2048    mean 1.01    single records 0.47 - 1.69
    N = 4096    mean 0.91    single records 0.50 - 1.23

The scatter barely improves with record length, because the average is
dominated by AVAR terms at `m` up to `N/2`, which have a handful of degrees
of freedom however long the record is. ThêoBR multiplies *every* tau by one
draw from that distribution, so the error is a common vertical shift of the
whole curve rather than scatter between points -- and it does not average
away by reading several taus. `bias_ratio` is public so the number can be
reported alongside the curve; it is in provenance for the same reason.

Averaging the ratio term by term is what eq. (33) prints. Pooling instead
-- `sum(AVAR) / sum(Theo1)` over the same terms -- was measured at the same
mean with about half the spread (0.82-1.32 against 0.53-1.68 at N = 1024).
That is recorded rather than substituted: the published definition is what
a parity claim against Stable32 has to mean.
"""

from __future__ import annotations

import numpy as np

from ..log import record
from ..registry import register
from ..types import FreqSeries, PhaseSeries
from ..tau import MAX_TAUS
from .classical import oadev
from .core import StabilityResult
from .theo1 import _theo1_one, theo1

__all__ = ["theobr", "theoh", "bias_ratio", "bias_terms", "MAX_BIAS_OPS"]

# Ceiling on the element-operations the bias ratio may cost, past which
# `theobr` refuses rather than appearing to hang. Measured throughput here
# is about 1.2e8 of these per second (8.7 s at N = 4097, where the sum
# costs 1.06e9), so this is roughly half a minute, reached at around
# N = 6500 phase points. The cost grows as N^3, so the next octave of
# record length is eight times the wait.
MAX_BIAS_OPS = 4e9


def bias_terms(n: int) -> tuple[np.ndarray, np.ndarray]:
    """`(allan_afs, theo1_afs)` for the eq. (33) average: 9+3i and 12+4i.

    `n_terms = floor(N/6) - 3` of each. They are returned together because
    the pairing is the whole content of the equation -- term `i` of one
    against term `i` of the other, at the same tau.
    """
    count = n // 6 - 3
    if count < 0:
        raise ValueError(
            f"theobr: needs at least {6 * 3} phase points for the bias "
            f"average of eq. (33), got {n}")
    i = np.arange(count + 1, dtype=np.int64)
    return 9 + 3 * i, 12 + 4 * i


def bias_ratio(series: PhaseSeries) -> tuple[float, int]:
    """`(ratio, terms)` -- the eq. (33) average of AVAR over Thêo1.

    The number ThêoBR multiplies Thêo1 by. Returned on its own because it
    is interesting in itself: it is a noise-type estimate that needed no
    noise identification, and SP 1065 reports it as "Thêo1 bias" in exactly
    this form (1.00 for white FM, 1.87 for flicker FM, 2.70 for random-walk
    FM -- `stats.bias.THEO1_BIAS_PARAMS` carries the published asymptotes).
    """
    x = np.asarray(series.x, dtype=np.float64)
    allan_afs, theo_afs = bias_terms(x.size)

    ops = float(np.sum((theo_afs / 2.0) * np.maximum(x.size - theo_afs, 0)))
    if ops > MAX_BIAS_OPS:
        raise ValueError(
            f"theobr: the eq. (33) bias average over {theo_afs.size} terms "
            f"reaching m={theo_afs[-1]} would cost about {ops:.1e} element "
            f"operations on {x.size} points, past the {MAX_BIAS_OPS:.0e} "
            "limit. It grows as N^3. Decimate the record first -- SP 1065 "
            "notes phase averaging has negligible effect on the bias "
            "removal -- or use theo1 with stats.bias.theo1_bias if the "
            "noise type is known.")

    # In chunks, because the eq. (33) average wants ~N/6 explicit averaging
    # factors and `tau.averaging_factors` caps one request at MAX_TAUS. The
    # cap is right -- an accidental thousand-tau grid is a hang -- and this
    # is the one caller that legitimately needs more than it.
    got = {}
    for start in range(0, allan_afs.size, MAX_TAUS):
        allan = oadev(series, taus=allan_afs[start:start + MAX_TAUS],
                      keep_coeffs=False)
        got.update({int(m): v for m, v in zip(allan.af, allan.var)})

    ratios = []
    for m_a, m_t in zip(allan_afs, theo_afs):
        if int(m_a) not in got:
            continue
        value, kept = _theo1_one(x, int(m_t), series.tau0_s)
        if kept and np.isfinite(value) and value > 0.0:
            ratios.append(got[int(m_a)] / value)
    if not ratios:
        raise ValueError(
            f"theobr: no term of the eq. (33) average is computable on "
            f"{x.size} points")
    return float(np.mean(ratios)), len(ratios)


@register("estimator", "theobr")
def theobr(series, taus="octave", max_af: int | None = None,
           gap_policy: str = "auto") -> StabilityResult:
    """Bias-removed Thêo1, SP 1065 eq. (33).

    An unbiased estimate of AVAR over the widest range of tau available
    from the record, **without explicit knowledge of the noise type** --
    the bias is measured from the data rather than looked up.
    """
    if not isinstance(series, PhaseSeries):
        if isinstance(series, FreqSeries):
            raise TypeError(
                "theobr: needs phase data. Convert with types.to_phase if "
                "that is what you mean.")
        raise TypeError(f"expected PhaseSeries, got {type(series).__name__}")

    ratio, terms = bias_ratio(series)
    base = theo1(series, taus=taus, max_af=max_af, gap_policy=gap_policy)
    prov = record(base.provenance, "stats.theoh", "theobr",
                  {"bias_ratio": ratio, "bias_terms": terms},
                  f"Theo1 scaled by {ratio:.4f}, measured over {terms} terms")
    from dataclasses import replace
    return replace(base, kind="theobr", var=base.var * ratio, provenance=prov,
                   coeffs=())


@register("estimator", "theoh")
def theoh(series, taus="octave", max_af: int | None = None,
          gap_policy: str = "auto") -> StabilityResult:
    """ThêoH, SP 1065 eq. (34): AVAR at short tau, ThêoBR beyond it.

    One curve from `tau0` to 0.75 of the record, with no noise-type
    assumption in any part of it. The join is at `k`, the largest available
    AVAR tau at or below 20 % of the record length; AVAR supplies every
    point up to it and ThêoBR every point past it.

    The two halves are different estimators, so the crossover is recorded
    in provenance and `n_terms` changes meaning across it -- Allan terms
    below, Thêo1 pair counts above.
    """
    if not isinstance(series, PhaseSeries):
        if isinstance(series, FreqSeries):
            raise TypeError(
                "theoh: needs phase data. Convert with types.to_phase if "
                "that is what you mean.")
        raise TypeError(f"expected PhaseSeries, got {type(series).__name__}")

    record_s = (series.n - 1) * series.tau0_s
    allan = oadev(series, taus=taus, max_af=max_af, gap_policy=gap_policy,
                  keep_coeffs=False)
    below = allan.tau_s <= 0.2 * record_s
    if not below.any():
        raise ValueError(
            f"theoh: no Allan tau falls at or below 20% of the {record_s:g} s "
            "record, so there is no short-tau half to splice. Use theobr.")
    k = float(allan.tau_s[below][-1])

    upper = theobr(series, taus=taus, max_af=max_af, gap_policy=gap_policy)
    above = upper.tau_s > k
    if not above.any():
        raise ValueError(
            f"theoh: no ThêoBR tau falls past the {k:g} s crossover, so "
            "there is no long-tau half to splice. Use oadev.")

    prov = record(upper.provenance, "stats.theoh", "theoh",
                  {"crossover_s": k, "record_s": record_s,
                   "allan_points": int(below.sum()),
                   "theobr_points": int(above.sum())},
                  f"AVAR to {k:g} s, ThêoBR to {upper.tau_s[-1]:g} s")
    return StabilityResult(
        kind="theoh",
        af=np.concatenate([allan.af[below], upper.af[above]]),
        tau_s=np.concatenate([allan.tau_s[below], upper.tau_s[above]]),
        var=np.concatenate([allan.var[below], upper.var[above]]),
        n_terms=np.concatenate([allan.n_terms[below], upper.n_terms[above]]),
        rate_hz=series.rate_hz, provenance=prov, span_factor=1,
    )
