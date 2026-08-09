"""The MODWT, and the wavelet variance the Allan variance is a case of.

**The Haar wavelet variance *is* the Allan variance.** For a fractional
frequency series,

    avar(tau_j) = 2 * nu^2_Haar(tau_j)

exactly, not approximately. That identity is the intellectual centre of
this library: the classical Allan variance is the `K = 1` member of one
framework, and higher-order Daubechies filters are the rest of it.

What the higher orders buy:

* **vanishing moments.** A filter with `K` vanishing moments annihilates
  any polynomial trend of degree `< K`. Haar (`K=1`) kills a constant
  offset; D4 (`K=2`) kills a *linear frequency drift* with no detrending
  step at all, which is otherwise a separate and contestable operation.
* **convergence for steeper noise.** The wavelet variance converges for
  `alpha > -2K-1` rather than AVAR's `alpha > -3`, so flicker-walk and
  random-run FM stop diverging.

**The filters are computed, not transcribed.** Daubechies coefficients are
the sort of thing that gets copied wrong, and a wrong low-order digit
produces a filter that still looks like a filter. Here they come from a
spectral factorization of the half-band polynomial, and the tests then
*certify* them against the defining algebra -- sum to sqrt(2), unit norm,
orthonormality at even shifts, and `K` vanishing moments. A filter passing
all of those is the right filter however it was obtained.

**MODWT, not DWT.** The maximal-overlap transform does not decimate, so it
is defined at every sample and is **shift-equivariant**: rotating the input
rotates the coefficients and leaves the variance unchanged. The decimating
DWT is not, and a variance that depends on where you started the record is
not a measurement. That is the property most easily got wrong here, so it
has its own test.

Percival & Walden, *Wavelet Methods for Time Series Analysis*, CUP 2000,
for the transform and the unbiased estimator; Percival & Guttorp 1994 for
the Allan equivalence.
"""

from __future__ import annotations

from math import comb, sqrt

import numpy as np

from ..log import record
from ..registry import register
from ..types import FreqSeries, PhaseSeries
from .core import MAX_COEFF_BYTES, StabilityResult

__all__ = ["FILTERS", "daubechies", "wavelet_filters", "modwt",
           "wavelet_variance", "wdev"]

# Filter families by name, keyed to the number of vanishing moments K.
# The length is 2K, so Haar is the two-tap K=1 case.
FILTERS = {"haar": 1, "d4": 2, "d6": 3, "d8": 4}

# Width of one coefficient's support, as a multiple of the averaging factor
# -- the `span_factor` the conservative EDOF counts non-overlapping terms
# with. The equivalent filter at level j is L_j = (2^j - 1)(L - 1) + 1 taps
# long and m = 2^{j-1}, so L_j/m -> 2(L - 1). Haar gives 2, which is the
# Allan family's own value: the same identity as `avar = 2 nu^2`, showing
# up in the error bar.
_SPAN = {name: 2 * (2 * k - 1) for name, k in FILTERS.items()}


def daubechies(k: int) -> np.ndarray:
    """The extremal-phase Daubechies scaling filter `g` of length `2k`.

    By spectral factorization of the half-band polynomial

        P(y) = sum_{j<k} C(k-1+j, j) y^j,    y = (1 - cos w)/2

    keeping the roots inside the unit circle (which is what makes it
    minimum- or "extremal"-phase) and multiplying by `(1 + z^-1)^k`, the
    factor that supplies the `k` vanishing moments.

    Normalized to Percival & Walden's convention: `||g|| = 1` and
    `sum(g) = sqrt(2)`.
    """
    if k < 1:
        raise ValueError(f"daubechies: k must be >= 1, got {k}")
    if k == 1:
        return np.array([1.0, 1.0]) / sqrt(2.0)

    coefficients = np.array([comb(k - 1 + j, j) for j in range(k)][::-1],
                            dtype=np.float64)
    inside = []
    for y in np.roots(coefficients):
        # y = -(z - 2 + 1/z)/4  =>  z^2 - (2 - 4y) z + 1 = 0
        b = 2.0 - 4.0 * y
        disc = np.sqrt(complex(b * b - 4.0))
        first, second = (b + disc) / 2.0, (b - disc) / 2.0
        inside.append(first if abs(first) < 1.0 else second)

    poly = np.array([1.0 + 0.0j])
    for z in inside:
        poly = np.convolve(poly, [1.0, -z])
    for _ in range(k):
        poly = np.convolve(poly, [1.0, 1.0])
    g = np.real(poly)
    return g / np.linalg.norm(g)


def wavelet_filters(name: str) -> tuple[np.ndarray, np.ndarray]:
    """`(g, h)` -- scaling and wavelet filters -- for a named family.

    The wavelet filter is the quadrature mirror of the scaling filter,
    `h_l = (-1)^l g_{L-1-l}`, which is what makes `sum(h) = 0` and gives
    the transform its differencing character.
    """
    if name not in FILTERS:
        raise ValueError(
            f"unknown wavelet filter {name!r}; have {sorted(FILTERS)}")
    g = daubechies(FILTERS[name])
    length = g.size
    h = np.array([(-1) ** l * g[length - 1 - l] for l in range(length)])
    return g, h


def modwt(x: np.ndarray, name: str = "haar", levels: int | None = None):
    """MODWT wavelet coefficients `W_1 .. W_J`, circular, undecimated.

        W_j[t] = sum_l h~_l V_{j-1}[(t - 2^{j-1} l) mod N]
        V_j[t] = sum_l g~_l V_{j-1}[(t - 2^{j-1} l) mod N]

    with `h~ = h/sqrt(2)`, `g~ = g/sqrt(2)`. Every level has `N`
    coefficients -- nothing is decimated, which is the whole point.

    The boundary is handled circularly here; `wavelet_variance` then
    discards the coefficients that wrapped, rather than pretending the
    record joins up.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        raise ValueError(f"modwt: need at least 2 points, got {n}")
    g, h = wavelet_filters(name)
    g, h = g / sqrt(2.0), h / sqrt(2.0)

    max_levels = int(np.floor(np.log2(n)))
    levels = max_levels if levels is None else min(levels, max_levels)

    out, v = [], x
    for j in range(1, levels + 1):
        step = 1 << (j - 1)
        w = np.zeros(n)
        v_next = np.zeros(n)
        for l in range(g.size):
            shifted = np.roll(v, step * l)
            w += h[l] * shifted
            v_next += g[l] * shifted
        out.append(w)
        v = v_next
    return out


def _boundary(n: int, filter_length: int, level: int) -> int:
    """How many leading coefficients at `level` were touched by the wrap.

        L_j = (2^j - 1)(L - 1) + 1

    Percival & Walden. Dropping them is what makes the estimator
    *unbiased*: those coefficients mix the end of the record with the
    beginning, and the record does not actually join up.
    """
    return min(n, (( 1 << level) - 1) * (filter_length - 1) + 1)


def wavelet_variance(series, name: str = "haar", levels: int | None = None,
                     unbiased: bool = True, keep_coeffs: bool = True,
                     gap_policy: str = "auto") -> StabilityResult:
    """Wavelet variance by scale, from a `FreqSeries` or `PhaseSeries`.

    The scale is `tau_j = 2^{j-1} tau0`, and for fractional frequency with
    the Haar filter `avar(tau_j) = 2 * nu^2(tau_j)` exactly. `wdev` returns
    the deviation on the Allan scale so the two are directly comparable on
    one plot; this returns the wavelet variance itself.

    `unbiased=True` drops the coefficients affected by the circular
    boundary. It is the default because the alternative silently treats the
    end of the record as adjacent to its beginning.

    The retained coefficients are carried on the result, so `confidence`
    works here through the same single path it serves every other estimator
    with: `eta1` is an EDOF from the autocorrelation of the terms being
    averaged, and at scale `j` those terms are exactly these coefficients.
    `keep_coeffs=False` drops them, and the interval with them.

    **Gaps are masked, and only masked.** A gap is a non-finite sample; the
    filter spreads it across every coefficient whose support touches it, and
    those coefficients are dropped. That is `gap_policy='mask'` arriving by
    a different route, and it is why `'segment'` is refused rather than
    accepted and ignored: analysing each run separately would give each its
    own dyadic grid, and pooling variances across grids that do not line up
    is a decision this function has no basis to make for the caller.
    """
    if gap_policy not in ("auto", "mask"):
        raise ValueError(
            f"wavelet_variance: gap_policy={gap_policy!r} is not available "
            "here. The MODWT masks gaps -- it drops the coefficients a gap "
            "touched -- and cannot segment, because each run would land on "
            "its own dyadic grid. Use gap_policy='mask', or split the record "
            "yourself and analyse the runs you want.")
    if isinstance(series, PhaseSeries):
        from ..types import to_frequency
        fs = to_frequency(series)
    elif isinstance(series, FreqSeries):
        fs = series
    else:
        raise TypeError(f"expected a series, got {type(series).__name__}")

    y = np.asarray(fs.y, dtype=np.float64)
    coeffs = modwt(y, name, levels)
    filter_length = 2 * FILTERS[name]
    if keep_coeffs:
        want = len(coeffs) * y.size * 8
        if want > MAX_COEFF_BYTES:
            raise MemoryError(
                f"wvar[{name}]: {len(coeffs)} levels over {y.size} points "
                f"would retain about {want / 1e9:.1f} GB of coefficients, "
                f"past the {MAX_COEFF_BYTES / 1e9:.1f} GB limit. They are "
                "kept only for the confidence interval, so pass "
                "keep_coeffs=False if you do not need bounds, or `levels=` "
                "to stop short of the coarsest scales."
            )

    variances, counts, scales, kept, retained = [], [], [], [], []
    masked = 0
    for j, w in enumerate(coeffs, start=1):
        start = _boundary(y.size, filter_length, j) - 1 if unbiased else 0
        usable = w[start:]
        after_boundary = usable.size
        usable = usable[np.isfinite(usable)]
        masked += after_boundary - usable.size
        if usable.size < 2:
            continue
        variances.append(float(np.mean(usable ** 2)))
        counts.append(int(usable.size))
        scales.append((1 << (j - 1)) / fs.rate_hz)
        kept.append(1 << (j - 1))
        if keep_coeffs:
            retained.append(usable)

    if not kept:
        raise ValueError(
            f"wavelet_variance: no scale survives the {name} filter's "
            f"boundary on {y.size} points; use a shorter filter or more data")

    prov = record(fs.provenance, "stats.wavelet", "wavelet_variance",
                  {"filter": name, "vanishing_moments": FILTERS[name],
                   "unbiased": unbiased, "levels": len(kept),
                   "gap_policy": "mask", "n_gaps": int(fs.n_gaps),
                   "coefficients_masked": int(masked)},
                  f"{len(kept)} scales, tau {scales[0]:g}..{scales[-1]:g} s"
                  + (f", {masked} coefficients masked by "
                     f"{fs.n_gaps} gaps" if masked else ""))
    return StabilityResult(
        kind=f"wvar[{name}]",
        af=np.asarray(kept, dtype=np.int64),
        tau_s=np.asarray(scales, dtype=np.float64),
        var=np.asarray(variances, dtype=np.float64),
        n_terms=np.asarray(counts, dtype=np.int64),
        rate_hz=fs.rate_hz, provenance=prov,
        span_factor=_SPAN[name], coeffs=tuple(retained),
    )


@register("estimator", "wdev")
def wdev(series, name: str = "haar", levels: int | None = None,
         unbiased: bool = True, keep_coeffs: bool = True, taus=None,
         max_af=None, gap_policy: str = "auto") -> StabilityResult:
    """Wavelet deviation on the **Allan scale**: `sqrt(2 * nu^2)`.

    Scaled so that `wdev(..., "haar")` and `oadev` are the same number at
    the same tau, which is what makes the Allan variance visibly the `K=1`
    case rather than a separate thing that happens to look similar.

    `taus` and `max_af` are accepted and ignored: the MODWT's scales are
    dyadic by construction, one per level. They are in the signature so that
    the estimator registry can call every member the same way; the CLI does
    not offer the flags, because offering one and ignoring it is worse than
    not having it.
    """
    result = wavelet_variance(series, name=name, levels=levels,
                              unbiased=unbiased, keep_coeffs=keep_coeffs,
                              gap_policy=gap_policy)
    from dataclasses import replace
    # The coefficients are left on their own scale. `eta1` works from their
    # autocorrelation, which the factor of 2 cancels out of, and rescaling
    # arrays the caller may inspect to match a deviation they did not ask
    # for would be the more surprising choice.
    return replace(result, var=result.var * 2.0, kind=f"wdev[{name}]")
