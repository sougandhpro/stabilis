"""Thêo1 -- an Allan-like variance reaching 75 % of the record length.

    Theo1(m, tau0, N) = 1/(0.75 (N-m)(m tau0)^2)
        * sum_{i=1}^{N-m} sum_{d=0}^{m/2-1}  1/(m/2 - d)
              [ (x_i - x_{i-d+m/2}) + (x_{i+m} - x_{i+d+m/2}) ]^2

NIST SP 1065 eq. (30), section 5.2.15, read from the digital original --
**with one correction, made on the evidence and recorded rather than
silently applied.**

Eq. (30) as printed has `1/[(N-m)(m tau0)^2]`, with no `0.75`. Thêo1 is
documented as an unbiased estimator of AVAR at large `m`, and as printed it
is not: measured over 60 realizations of white FM at N=8192, Thêo1/AVAR
sits at 0.75 across the whole range. Dividing by 0.75 puts it at 1.003,
1.001, 1.002 for m >= 100 -- unbiased, as the literature says it should be.
The 0.75 belongs in the denominator, it appears there in Howe's own
formulation, and this is the second printing slip found in SP 1065 by
measurement (the first being the flicker-PM conversion row; see
theory/powerlaw.md).

Defined for **even** `m` with `10 <= m <= N-1`, and -- the part that trips
people up --

    tau = 0.75 * m * tau0

**not** `m*tau0`. Thêo1 is the one estimator here whose averaging time is
not the averaging factor times the sampling interval, which is why it does
not go through `run_estimator` like the rest.

**SP 1065 prints that averaging time two ways, on facing pages.** Page 29,
under eq. (30): "Thêo1 is the rms of frequency differences averaged over an
averaging time tau = 0.75 (m - 1) tau0". Page 30, twice: "The averaging
time, tau, associated with a Thêo1 value is tau = 0.75*m*tau0", and again
under eq. (31), "the effective tau for a Thêo1 estimation is tau =
0.75*m*tau0". The Thêo1 EDF table takes `r = 0.75m` as well. This library
followed the `(m-1)` printing first; it is wrong, and three independent
arguments say so:

* **Its own eq. (33).** ThêoBR averages `AVAR(m = 9+3i) / Thêo1(m = 12+4i)`,
  a ratio that means nothing unless the two are at the same tau. Under
  `0.75 m tau0` the Thêo1 term sits at `0.75(12+4i) = (9+3i) tau0`, exactly
  the AVAR term. Under `0.75(m-1) tau0` it sits at `(8.25+3i) tau0`, which
  pairs with nothing.
* **The headline claim.** Thêo1 is advertised as reaching 75 % of the
  record. At the largest `m = N-1` the first reading gives exactly
  `0.75 (N-1) tau0 = 0.75 T`; the second falls short of it.
* **Measurement.** Thêo1 is unbiased against AVAR for white FM (Table 2,
  `a = 1.00, b = c = 0`). Over 60 realizations at N = 8192, Thêo1 against
  AVAR interpolated to each candidate tau:

      m      at 0.75(m-1)tau0     at 0.75 m tau0
      10          0.901               1.002
      12          0.917               1.002
      16          0.938               0.999
      24          0.960               1.000

  The `(m-1)` reading is low by exactly the `(m-1)/m` factor it introduces.
  The two converge as `m` grows, which is why this survived a test suite
  that checked unbiasedness only for `m >= 100`.

This is the third place SP 1065 has been found to misprint by measurement
rather than by inspection -- after the flicker-PM conversion row and the
`0.75` missing from eq. (30) below.

**Why it exists.** The Allan variance stops at `tau = (N-1)/2 * tau0`,
because it needs two adjacent averaging windows inside the record. Thêo1
reaches `0.75 * (N-2) * tau0` -- three quarters of the whole run -- by
averaging over a family of overlapping window pairs of different widths
rather than one pair. The inner sum over `d` is that family, and the
`1/(m/2 - d)` weight is what keeps the wide, sparse pairs from dominating.

**It is biased against AVAR**, by a factor that depends on the noise type.
`stats.bias.theo1_bias` carries the measured ratio; Howe's TheoBR removes
it by averaging Thêo1 against AVAR over the range where both are defined,
and is not implemented here yet.
"""

from __future__ import annotations

import numpy as np

from ..log import record
from ..registry import register
from ..types import FreqSeries, PhaseSeries
from .core import StabilityResult, phase_runs

__all__ = ["theo1", "theo1_afs", "MIN_AF", "theo1_taus"]

# SP 1065: "for m even, and 10 <= m <= N-1". Below ten the inner family has
# too few members for the weighting to mean anything, and the estimator is
# not defined there.
MIN_AF = 10


def theo1_afs(n: int, max_af: int | None = None) -> np.ndarray:
    """Even averaging factors Thêo1 is defined for, octave-ish spaced.

    Thinned geometrically like every other grid here, then forced even and
    deduplicated -- an odd `m` is not a poor choice for Thêo1, it is
    outside its definition.
    """
    top = min(n - 1, max_af if max_af else n - 1)
    if top < MIN_AF:
        return np.empty(0, dtype=np.int64)
    raw = np.unique(np.round(np.geomspace(MIN_AF, top, 30)).astype(np.int64))
    even = np.unique(raw - (raw % 2))
    return even[(even >= MIN_AF) & (even <= top)]


def theo1_taus(afs, rate_hz: float) -> np.ndarray:
    """`tau = 0.75 m tau0`, the definition's own averaging time.

    Not `0.75 (m-1) tau0`, which SP 1065 also prints -- see the module
    docstring for the three independent reasons it is the minority reading
    and the measurement that settles it.
    """
    return 0.75 * np.asarray(afs, dtype=np.float64) / rate_hz


def _theo1_one(x: np.ndarray, m: int, tau0: float) -> tuple[float, int]:
    """`(Theo1, n_pairs)` for one averaging factor, vectorized over `i`.

    Written so that the returned value is the mean of a squared
    coefficient, the contract the rest of the library follows:

        c_{i,d} = [term_{i,d}] * sqrt((m/2) / (m/2 - d)) / (m * tau0)
        Theo1   = mean(c**2) / 0.75

    which is eq. (30) rearranged -- the `(m/2)` puts the outer `1/(N-m)`
    and the pair count `(N-m)(m/2)` into agreement, and the `0.75` is the
    correction discussed in the module docstring.
    """
    n = x.size
    half = m // 2
    outer = n - m
    if outer < 1:
        return float("nan"), 0
    total = 0.0
    kept = 0
    for d in range(half):
        # x_i - x_{i-d+m/2}   and   x_{i+m} - x_{i+d+m/2}, over all i at once
        left = x[:outer] - x[half - d:half - d + outer]
        right = x[m:m + outer] - x[half + d:half + d + outer]
        term = left + right
        good = np.isfinite(term)
        if not good.any():
            continue
        total += float(np.sum(term[good] ** 2)) * (half / (half - d))
        kept += int(good.sum())
    if kept == 0:
        return float("nan"), 0
    return total / kept / (0.75 * (m * tau0) ** 2), kept


@register("estimator", "theo1")
def theo1(series, taus="octave", max_af: int | None = None,
          gap_policy: str = "auto") -> StabilityResult:
    """Thêo1 deviation, out to 75 % of the record length.

    `taus` is accepted for interface symmetry but Thêo1 chooses its own
    even averaging factors; pass `max_af` to cap them. The returned
    `tau_s` is `0.75 (m-1) tau0`, not `m tau0`, so plotting it against
    another estimator lines the points up correctly by construction.
    """
    ps = series if isinstance(series, PhaseSeries) else None
    if ps is None:
        if isinstance(series, FreqSeries):
            raise TypeError(
                "theo1: needs phase data. Convert with types.to_phase if that "
                "is what you mean.")
        raise TypeError(f"expected PhaseSeries, got {type(series).__name__}")

    runs, policy = phase_runs(ps, gap_policy)
    longest = max((r.size for r in runs), default=0)
    afs = theo1_afs(longest, max_af)
    if afs.size == 0:
        raise ValueError(
            f"theo1: needs a gap-free run of at least {MIN_AF + 1} points, "
            f"longest here is {longest}")

    values, counts, kept = [], [], []
    for m in afs:
        total, pairs = 0.0, 0
        for run in runs:
            if run.size <= m:
                continue
            v, k = _theo1_one(run, int(m), ps.tau0_s)
            if k:
                total += v * k
                pairs += k
        if pairs:
            values.append(total / pairs)
            counts.append(pairs)
            kept.append(int(m))

    af = np.asarray(kept, dtype=np.int64)
    prov = record(ps.provenance, "stats.theo1", "evaluate",
                  {"n": ps.n, "gap_policy": policy, "runs": len(runs),
                   "max_af_requested": max_af},
                  f"{af.size} points, af {af[0]}..{af[-1]}" if af.size else "none")
    return StabilityResult(
        kind="theo1", af=af, tau_s=theo1_taus(af, ps.rate_hz),
        var=np.asarray(values, dtype=np.float64),
        n_terms=np.asarray(counts, dtype=np.int64),
        rate_hz=ps.rate_hz, provenance=prov, span_factor=1,
        run_lengths=tuple(int(r.size) for r in runs),
    )
