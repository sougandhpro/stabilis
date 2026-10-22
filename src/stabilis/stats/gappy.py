"""The wavelet variance of a gappy record, without filling the gaps.

Mondal & Percival, "Wavelet variance analysis for gappy time series",
Ann. Inst. Stat. Math. 62 (2010) 943-966, eqs. 8 and 9. The imputation-free
alternative to `impute.fill_gaps`: nothing is invented, and the estimator
stays unbiased conditional on the gap pattern.

`wavelet_variance` drops every MODWT coefficient whose filter touches a
gap, which is unbiased too but throws away every coefficient within `L_j`
of a gap. At deep levels `L_j` is thousands of samples, so a record with
one gap per thousand loses most of its long-scale coefficients. The
estimators here keep every observed *pair* instead. Expanding `W_t**2`
into its pair products and averaging each pair over the times it is
observed,

    v = -(1/2) sum_{l,l'} h_l h_l' * mean_t[ (y_{t-l} - y_{t-l'})**2 ]   (9)
    u =        sum_{l,l'} h_l h_l' * mean_t[  y_{t-l} * y_{t-l'}      ]   (8)

each mean running over `t = L_j-1 .. N-1` where both samples exist. That
is the paper's `beta_hat_{l,l'}` weighting written as a mean: its
`1/beta_hat` is the observed fraction of that window, `C/M_j`. With no
gaps every mean is over all `M_j` times and both collapse to the MODWT
estimator `mean(W_t**2)` -- a test holds them to it.

`v` is the default. It is the semivariogram form, so it needs only
stationary *increments* and is blind to a constant offset; `u` needs `y`
itself stationary and is centred on its observed mean first, which the
paper's Remark 1 asks for. Both can come out negative on a short or very
gappy record. A negative variance is reported, not clipped: clipping would
bias the estimator the paper proves unbiased.

**The cost is per lag, not per pair.** For a lag `k = l' - l` the pair
products form one sequence `g_k[s]` regardless of `l`; a level's window
over `t` is then a window over `s = t - l`, read off one cumulative sum in
O(1). A level costs O(L_j * N), the same as its convolution, rather than
O(L_j**2 * N).

**No confidence interval yet.** The paper's interval needs a multitaper
estimate of the spectrum of the per-time terms `Z_t` at zero frequency,
and forming `Z_t` costs O(L_j**2) per sample because `beta_hat` depends on
the pair. Declared in `VALIDATION-DEBT.md`; see `theory/gappy.md`.
"""

from __future__ import annotations

import numpy as np

from ..log import record
from ..types import FreqSeries, PhaseSeries, to_frequency
from .core import StabilityResult
from .wavelet import FILTERS, _SPAN, modwt

__all__ = ["level_filter", "gappy_wavelet_variance", "ESTIMATORS",
           "MAX_PAIR_OPS"]

ESTIMATORS = ("v", "u")

# Element operations a call may cost: about L_j * N per level. 4e9 is tens
# of seconds of numpy; beyond it, ask for fewer levels rather than hang.
MAX_PAIR_OPS = 4.0e9


def level_filter(name: str, level: int) -> np.ndarray:
    """The level-`j` MODWT wavelet filter `h_{j,l}`, `l = 0 .. L_j-1`.

        L_j = (2^j - 1)(L - 1) + 1,     sum_l h_{j,l}**2 = 2^-j

    Read off the transform itself: `modwt` applied to a unit impulse
    returns `W_j[t] = h_{j,t}`. Taken this way rather than by a separate
    cascade so that the filter this estimator uses is, by construction, the
    one `wavelet_variance` uses. `validate.expected.wavelet_kernel` builds
    it the other way, and a test holds the two equal.
    """
    taps = ((1 << level) - 1) * (2 * FILTERS[name] - 1) + 1
    impulse = np.zeros(2 * taps)
    impulse[0] = 1.0
    return modwt(impulse, name, level)[level - 1][:taps].copy()


def _pair_means(y: np.ndarray, ok: np.ndarray, k: int, taps: int, form: str):
    """Mean over the observed times of every lag-`k` pair at one level.

        g_k[s] = (y[s] - y[s-k])**2  ('v')   or   y[s] * y[s-k]  ('u')
        pair (l, l+k):  s = t - l,   t = taps-1 .. N-1
        mean_l = sum(g_k[s0..s1]) / count(s0..s1),   s0 = taps-1-l

    Returns `(means, counts)` for `l = 0 .. taps-1-k`. A pair never
    observed together has count 0 and mean NaN; the caller refuses it.
    """
    both = ok[k:] & ok[:y.size - k]
    a, b = y[k:], y[:y.size - k]
    g = np.where(both, (a - b) ** 2 if form == "v" else a * b, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(g)))
    cc = np.concatenate(([0], np.cumsum(both)))
    l = np.arange(taps - k)
    lo = taps - 1 - l - k          # s0 - k, the index into g
    hi = y.size - l - k            # one past s1 - k
    counts = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (cs[hi] - cs[lo]) / counts, counts


def _level(y: np.ndarray, ok: np.ndarray, h: np.ndarray, form: str):
    """One level of eq. 8 or 9. Returns `(estimate, fewest pair count)`.

        v = -sum_{k>=1} sum_l h_l h_{l+k} mean_l(k)
        u =  sum_l h_l**2 mean_l(0) + 2 sum_{k>=1} sum_l h_l h_{l+k} mean_l(k)

    The double sum over `(l, l')` is folded onto `l < l'`: the summand is
    symmetric, which turns the `-1/2` of eq. 9 into `-1`, and the `l = l'`
    terms of eq. 9 are identically zero.
    """
    taps = h.size
    total, fewest = 0.0, y.size
    for k in range(0 if form == "u" else 1, taps):
        means, counts = _pair_means(y, ok, k, taps, form)
        fewest = min(fewest, int(counts.min()))
        weight = h[:taps - k] * h[k:]
        if k == 0:
            total += float(np.sum(weight * means))
        else:
            total += (2.0 if form == "u" else -1.0) * float(np.sum(weight * means))
    return total, fewest


def gappy_wavelet_variance(series, name: str = "haar",
                           levels: int | None = None,
                           estimator: str = "v") -> StabilityResult:
    """Wavelet variance by scale for a record with gaps, eq. 9 (or 8).

        v_j = -(1/2) sum_{l,l'} h_{j,l} h_{j,l'} beta_hat_{l,l'}
                     * (1/M_j) sum_t (y_{t-l} - y_{t-l'})**2 d_{t-l} d_{t-l'}
        beta_hat_{l,l'} = M_j / sum_t d_{t-l} d_{t-l'},   M_j = N - L_j + 1

    `d` is 1 where `y` was observed and 0 at a gap; `t` runs from `L_j - 1`
    to `N - 1`. Same scale and units as `wavelet_variance` -- fractional
    frequency, `tau_j = 2^{j-1} tau0`, Haar giving `avar = 2 * nu^2` -- and
    equal to it to rounding when there are no gaps.

    A level is dropped, and the drop recorded, when some pair of taps is
    never observed together inside the window: `beta_hat` is then
    undefined, which is the paper's own assumption failing rather than a
    numerical accident. Carries no coefficients, so `confidence` refuses
    it; see the module docstring.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    if name not in FILTERS:
        raise ValueError(f"unknown wavelet filter {name!r}; have {sorted(FILTERS)}")
    fs = to_frequency(series) if isinstance(series, PhaseSeries) else series
    if not isinstance(fs, FreqSeries):
        raise TypeError(f"expected a series, got {type(series).__name__}")
    y = np.array(fs.y, dtype=np.float64)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        raise ValueError("gappy_wavelet_variance: fewer than 2 observed samples")
    centre = 0.0
    if estimator == "u":
        centre = float(y[ok].mean())   # Remark 1: u is not offset-invariant
    y = np.where(ok, y - centre, 0.0)

    n, width = y.size, 2 * FILTERS[name]
    top = int(np.floor(np.log2(n)))
    top = top if levels is None else min(levels, top)
    tops = [j for j in range(1, top + 1) if ((1 << j) - 1) * (width - 1) + 1 <= n - 1]
    cost = sum((((1 << j) - 1) * (width - 1) + 1) * n for j in tops)
    if cost > MAX_PAIR_OPS:
        raise MemoryError(
            f"gappy_wavelet_variance: {len(tops)} levels over {n} points is "
            f"about {cost:.1e} operations, past the {MAX_PAIR_OPS:.0e} "
            "budget. Pass `levels=` to stop short of the coarsest scales.")

    variances, counts, kept, dropped = [], [], [], []
    for j in tops:
        h = level_filter(name, j)
        value, fewest = _level(y, ok, h, estimator)
        if fewest == 0:
            dropped.append(j)
            continue
        variances.append(value)
        counts.append(n - h.size + 1)
        kept.append(1 << (j - 1))
    if not kept:
        raise ValueError("gappy_wavelet_variance: no level has every pair of "
                         "taps observed together; the record is too gappy")

    prov = record(fs.provenance, "stats.gappy", "gappy_wavelet_variance",
                  {"filter": name, "estimator": estimator,
                   "n_gaps": int(n - ok.sum()), "centred_on": centre,
                   "levels": len(kept), "levels_dropped": dropped},
                  f"{len(kept)} scales by Mondal-Percival eq. "
                  f"{9 if estimator == 'v' else 8}"
                  + (f"; levels {dropped} dropped, a tap pair never "
                     "observed together" if dropped else ""))
    return StabilityResult(
        kind=f"wvar_gappy[{name}]",
        af=np.asarray(kept, dtype=np.int64),
        tau_s=np.asarray(kept, dtype=np.float64) / fs.rate_hz,
        var=np.asarray(variances, dtype=np.float64),
        n_terms=np.asarray(counts, dtype=np.int64),
        rate_hz=fs.rate_hz, provenance=prov, span_factor=_SPAN[name],
    )
