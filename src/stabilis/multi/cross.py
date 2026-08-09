"""The cross-variance: a mean of products where the auto case is a square.

Every estimator in `stats/` returns the coefficient array it averaged,
with the contract `variance = scale * mean(c**2)`. Replace one factor with
the other channel's coefficients and the whole family comes with it:

    cross_var = scale * mean(c_a * c_b)

cross-AVAR, cross-MVAR, cross-HVAR and cross-TVAR are then one function
rather than four, and a plugin estimator gets its cross form for free.
That is the payoff for keeping the coefficients on the result object, and
it is why nothing in this module knows what a second difference is.

**Why the product rejects instrument noise.** With `x_a = x_c + n_a` and
`x_b = x_c + n_b`, the operator is linear so `d_a = d_c + e_a` and
likewise for b, giving

    E[d_a d_b] = E[d_c^2] + E[d_c e_b] + E[e_a d_c] + E[e_a e_b]
               = E[d_c^2]

when the two instrument noises are independent of each other and of the
clock. Each cross term has expectation zero -- but only its *expectation*.
The finite-sample estimate carries their scatter, which is what `sem`
measures and what stops the rejection being unbounded.

**The error bar is not chi-square.** `stats/ci.py` intervals rest on
`sigma_hat^2` being a sum of squares of a Gaussian, which a sum of
products is not: it is unbounded below, and its mean is the quantity of
interest rather than a scale. What replaces it is Isserlis' theorem. For
jointly Gaussian coefficients,

    Cov(c_a[i] c_b[i], c_a[j] c_b[j]) = R_aa(k) R_bb(k) + R_ab(k) R_ba(k)

with `k = i - j`, so summing over the pairs `i, j` of terms,

    Var(cross_var) = (1/M) sum_k (1 - |k|/M) [R_aa(k) R_bb(k)
                                              + R_ab(k) R_ba(k)]

truncated at `4m` lags for the reason `eta1` truncates. All four sequences
come out of the same FFT the EDOF machinery already uses, so this needs no
new numerics -- only the observation that the *cross*-correlation of the
two coefficient arrays is as computable as the auto.

Setting `b = a` collapses it to `2 sigma^4 / eta` with `eta` exactly
`eta1`: the chi-square error bar is the auto case of this one, not a
parallel construction. `cross_variance(a, a)` therefore returns the
auto-variance bit for bit *and* its interval.

**Two estimators were rejected on measurement before this one.**

*A Bartlett-weighted sum over the products' own autocovariance*, which
needs no Gaussian assumption at all. Measured against Monte-Carlo scatter
at 0.89 / 0.73 / 0.56 / 0.37 / 0.30 of truth from `m = N/64` to `N/4`, and
lengthening the lag limit made it worse rather than better, which rules
out truncation bias. The products are heavy-tailed where the coefficients
are Gaussian, so each extra lag of their sample autocovariance adds more
noise than signal and the sum shrinks towards zero.

*The equal-shape closed form* `(auto_a auto_b + cross^2)/eta`, which
follows from the expression above if `R_aa`, `R_bb` and `R_ab` are taken
to have the same shape and differ only in scale. Simpler, and right at
both ends of the tau range -- but measured 0.81 to 0.88 of truth at
intermediate tau across three seeds, error bars 15-20% too narrow. That is
exactly where the clock and the instrument contribute comparably and their
correlation shapes therefore differ most, so the approximation fails in
the middle rather than at an extreme. `cross_sem` keeps it, because it is
still the relation that defines the reported `eta`.

`tests/test_cross.py` keeps both and measures all three on the same
realizations, so none of this can quietly rot.

See theory/cross.md, which also states the assumption that cannot be
checked from the data: independence of the two channels' noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..log import record, warn
from ..provenance import Provenance
from ..registry import get
from ..stats.edof import DEFAULT_LAG_LIMIT, eta1, eta3
from .align import common_support, overlap

__all__ = ["CrossResult", "cross_variance", "cross_sem", "implied_eta",
           "lagged_covariance", "isserlis_var", "pair_sem"]


def lagged_covariance(coeffs, kmax: int) -> np.ndarray:
    """`R[i, j, k]`, the biased sample cross-covariance at lags 0..kmax.

        R_ij(k) = (1/M) sum_t c_i[t + k] c_j[t]

    Zero-padded past `2M` so the circular correlation cannot wrap, and
    divided by `M` rather than `M - k` -- the biased normalization, for the
    reasons `stats.edof._autocorrelation` gives: it shrinks the noisiest
    high-lag estimates towards zero and keeps the sequence positive
    semi-definite. The diagonal `R[i, i]` is exactly that function's input,
    unnormalized.

    Not demeaned, and that is deliberate rather than an oversight: a second
    or third difference of phase has zero mean by construction, so
    subtracting a sample mean would remove signal and add variance.

    Computed one row at a time. The spectra of `M` channels over a long
    record are already large, and forming the full `M x M` outer product in
    the frequency domain at once turns "large" into "several times the
    record". One row is `M` spectra, which is what was already held.
    """
    c = np.atleast_2d(np.asarray(coeffs, dtype=np.float64))
    n_ch, n_terms = c.shape
    nfft = 1 << int(np.ceil(np.log2(2 * n_terms)))
    spec = np.fft.rfft(c, nfft, axis=1)
    out = np.empty((n_ch, n_ch, kmax + 1))
    for i in range(n_ch):
        out[i] = np.fft.irfft(spec[i] * np.conj(spec), nfft,
                              axis=1)[:, : kmax + 1] / n_terms
    return out


def isserlis_var(r: np.ndarray, n_terms: int, pairs) -> float:
    """Variance of the mean of the pair-averaged products, exactly.

    Given `R = lagged_covariance(...)` and a list of channel pairs, returns

        Var(mean_p mean_t c_ip[t] c_jp[t])
            = (1/(P^2 M)) sum_{p,q} sum_k (1 - |k|/M)
              [R_{ip iq}(k) R_{jp jq}(k) + R_{ip jq}(k) R_{jp iq}(k)]

    which is Isserlis' theorem with nothing approximated. Both signs of `k`
    are covered by the factor of two on `k >= 1`: the summand at `-k` for
    the ordered pair `(p, q)` equals the summand at `+k` for `(q, p)`, and
    every ordered pair is visited.

    Multiply by `scale**2` to get the variance of the *variance*; this
    function works in the coefficients' own units so it does not need to
    know the estimator.

    One pair `[(0, 1)]` is the two-channel cross-variance; `[(0, 0)]` is the
    auto case and returns `2 sigma^4 / eta1` exactly. A negative total is
    possible once the lag sum is truncated, and is returned as such -- the
    caller decides what to do with it, because "the covariance estimate
    failed" and "the covariance is zero" are different statements.
    """
    kmax = r.shape[2] - 1
    weights = np.empty(kmax + 1)
    weights[0] = 1.0
    weights[1:] = 2.0 * (1.0 - np.arange(1, kmax + 1) / n_terms)
    i = np.fromiter((p[0] for p in pairs), dtype=int, count=len(pairs))
    j = np.fromiter((p[1] for p in pairs), dtype=int, count=len(pairs))

    total = 0.0
    for q in range(i.size):
        summand = (r[i, i[q]] * r[j, j[q]] + r[i, j[q]] * r[j, i[q]])
        total += float((summand @ weights).sum())
    return total / (n_terms * i.size * i.size)


@dataclass(frozen=True)
class CrossResult:
    """A cross-variance over a tau grid, with both auto-variances beside it.

    The autos are carried rather than left to the caller because every
    question worth asking of a cross-variance is a comparison with them:
    how much of the record is common (`correlation`), and how far below the
    single-channel floor the measurement reached (`rejection_db`).
    """

    kind: str
    af: np.ndarray
    tau_s: np.ndarray
    # Signed. Negative entries are real results, not failures -- see
    # `dev`. Nothing in this module clips them or takes an absolute value.
    var: np.ndarray
    sem: np.ndarray
    auto_a: np.ndarray
    auto_b: np.ndarray
    eta: np.ndarray
    n_terms: np.ndarray
    rate_hz: float
    scale: float
    provenance: Provenance
    # Which EDOF estimator each point actually used, since a fallback at
    # the long-tau end is exactly the sort of decision a report must be
    # able to answer for.
    edof_method: tuple[str, ...] = ()

    @property
    def negative(self) -> np.ndarray:
        """Where the estimate came out below zero."""
        return self.var < 0.0

    @property
    def dev(self) -> np.ndarray:
        """Deviation where one exists, NaN where the variance is negative.

        NaN rather than zero, and rather than the absolute value: a
        negative cross-variance means the common part is smaller than the
        scatter of the estimate, which is a statement about resolution, not
        a deviation of any size. Plotting `sqrt(|var|)` on a log axis would
        draw a noise floor that is not there.
        """
        return np.sqrt(np.where(self.var > 0.0, self.var, np.nan))

    @property
    def correlation(self) -> np.ndarray:
        """`var / sqrt(auto_a * auto_b)`: the fraction the records share.

        Near 1 when both channels see the same dominant clock, near 0 when
        each is dominated by its own instrument. Not bounded to [-1, 1] in
        finite samples, and deliberately not clipped -- an excursion past 1
        says the estimate is noisy or the noises are correlated, which is
        exactly the thing worth seeing.
        """
        return self.var / np.sqrt(self.auto_a * self.auto_b)

    @property
    def rejection_db(self) -> np.ndarray:
        """How far below the single-channel variance this estimate resolves.

            rejection = 10 * log10(sqrt(auto_a * auto_b) / sem)

        A power ratio of the single-channel floor to the cross estimate's
        own standard error, so 10log10 and not 20. When the channels are
        instrument-dominated, `cross_var^2` is negligible beside `auto_a *
        auto_b` and `sem` tends to `sqrt(auto_a * auto_b / eta)`, whence
        `rejection = 5 * log10(eta)` -- the familiar `5 log10 M` law of
        cross-spectrum measurement, here a consequence rather than a
        substitute. It is reported from the measured error bar, so it
        degrades honestly when the terms overlap or the record is short.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return 10.0 * np.log10(np.sqrt(self.auto_a * self.auto_b) / self.sem)


def cross_sem(auto_a: float, auto_b: float, cross: float, eta: float) -> float:
    """Standard error of a cross-variance, from Isserlis.

        sem = sqrt((auto_a * auto_b + cross**2) / eta)

    The equal-shape closed form: what `isserlis_var` reduces to when
    `R_aa`, `R_bb` and `R_ab` differ only in scale. **Not what
    `cross_variance` computes** -- it measured 15-20% narrow at
    intermediate tau, where the clock and the instrument contribute
    comparably and their shapes therefore differ most.

    It is kept because it is the relation that *defines* the reported
    `eta`. Inverted, `eta = (auto_a auto_b + cross^2) / sem^2` turns the
    exact variance back into a degrees-of-freedom number comparable with
    every other one in the library, which is what a reader wants beside a
    chi-square interval. Finite and positive whatever the sign of `cross`,
    and equal to `sigma^2 * sqrt(2 / eta)` when the two channels are one
    channel.
    """
    return float(np.sqrt((auto_a * auto_b + cross * cross) / eta))


def implied_eta(auto_a: float, auto_b: float, cross: float, sem: float) -> float:
    """`cross_sem` inverted: the degrees of freedom this error bar implies."""
    return float("inf") if sem <= 0.0 else float(
        (auto_a * auto_b + cross * cross) / (sem * sem))


def cross_variance(a, b, estimator: str = "oadev", taus="octave",
                   max_af: int | None = None, gap_policy: str = "auto",
                   edof_method: str = "auto",
                   lag_limit: int = DEFAULT_LAG_LIMIT) -> CrossResult:
    """Cross-variance of two records of the same clock, by two channels.

    The two records are cropped to their common window and given a common
    gap pattern before either is estimated, so the coefficient arrays
    correspond term by term; see `align.py` for why both steps are
    necessary and why neither interpolates.

    `estimator` names any registered estimator that keeps coefficients.
    The cross form of a statistic that is not a variance -- MTIE, TIE rms
    -- does not exist, and is refused rather than computed.
    """
    aligned = overlap([a, b])
    sa, sb = common_support(aligned.series)
    fn = get("estimator", estimator)
    # `keep_coeffs` is deliberately not passed. Estimators that have it
    # default it True, and the ones that do not -- MTIE, TIE rms -- have no
    # cross form anyway, so naming it here would turn a clear refusal from
    # `_check_pairable` into a TypeError about a keyword argument.
    ra = fn(sa, taus=taus, max_af=max_af, gap_policy=gap_policy)
    rb = fn(sb, taus=taus, max_af=max_af, gap_policy=gap_policy)
    _check_pairable(ra, rb, estimator)

    n_tau = int(ra.af.size)
    var = np.empty(n_tau)
    sem = np.empty(n_tau)
    eta = np.empty(n_tau)
    methods = []
    for i, m in enumerate(ra.af):
        p = ra.coeffs[i] * rb.coeffs[i]
        var[i] = ra.scale * float(p.mean())
        sem[i], used = pair_sem(ra, rb, i, int(m), var[i], edof_method,
                                lag_limit)
        eta[i] = implied_eta(ra.var[i], rb.var[i], var[i], sem[i])
        methods.append(used)

    prov = record(
        ra.provenance, "multi.cross", "cross_variance",
        {"estimator": estimator, "n": aligned.n, "rate_hz": aligned.rate_hz,
         "points": n_tau, "edof_method": edof_method, "lag_limit": lag_limit,
         "offsets": list(aligned.offsets), "trimmed": list(aligned.trimmed)},
        f"{n_tau} points, af {int(ra.af[0])}..{int(ra.af[-1])}",
    )
    # Not a caveat, an assumption: the rejection is worth exactly what this
    # is worth, and no arrangement of the two records can test it. A shared
    # reference, a shared supply or one thermal environment leaves a term
    # that does not cancel and moves the answer with no symptom.
    prov = warn(
        prov, "multi.cross", "independence_assumed", {"channels": 2},
        "the two channels' noise is assumed independent of each other and "
        "of the clock; a shared reference, supply or thermal environment "
        "breaks this silently")
    n_neg = int((var < 0.0).sum())
    if n_neg:
        prov = record(
            prov, "multi.cross", "negative_variance", {"points": n_neg},
            f"{n_neg} of {n_tau} points came out negative, which is ordinary "
            "at long tau and means the common part is below the scatter of "
            "the estimate")
    return CrossResult(
        kind=f"cross_{estimator}", af=ra.af, tau_s=ra.tau_s, var=var, sem=sem,
        auto_a=ra.var, auto_b=rb.var, eta=eta, n_terms=ra.n_terms,
        rate_hz=aligned.rate_hz, scale=ra.scale, provenance=prov,
        edof_method=tuple(methods),
    )


def pair_sem(ra, rb, i: int, m: int, cross: float, method: str,
             lag_limit: int) -> tuple[float, str]:
    """The exact error bar for one pair at one averaging factor.

    Returns `(sem, method_used)`. The second element exists because
    "silently fell back" is precisely the decision an ISO-17025 report has
    to be able to answer for.

    Two paths, and the choice is the same one `stats.edof.edof` makes. With
    enough terms, `isserlis_var` over the four lagged covariances, which
    approximates nothing. Below `MIN_TERMS_FOR_ETA1` the sample covariances
    are mostly noise and summing them over lags is worse than useless, so
    it falls back to the equal-shape closed form driven by `eta3` -- a
    lower bound on the degrees of freedom and therefore an upper bound on
    the error bar, which is the direction a fallback must err in.

    A truncated lag sum can come out negative, which means the covariance
    *estimate* failed rather than that the covariance is zero. Rather than
    report a confident zero, that case takes the conservative path too and
    says so.
    """
    from ..stats.edof import DATA_DRIVEN_METHODS, MIN_TERMS_FOR_ETA1

    if method not in DATA_DRIVEN_METHODS:
        raise ValueError(
            f"cross_variance: edof_method must be one of "
            f"{DATA_DRIVEN_METHODS}, got {method!r}. The noise-type-based "
            "estimators (greenhall, totvar) are derived for a sum of "
            "squares and do not carry over to a sum of products."
        )
    ca, cb = ra.coeffs[i], rb.coeffs[i]
    n_terms = int(ca.size)
    if method != "eta3" and n_terms >= MIN_TERMS_FOR_ETA1:
        kmax = int(min(lag_limit * m, n_terms - 1))
        r = lagged_covariance(np.stack((ca, cb)), kmax)
        total = isserlis_var(r, n_terms, [(0, 1)])
        if total > 0.0:
            return ra.scale * float(np.sqrt(total)), "isserlis"
    eta = float(np.sqrt(eta3(n_terms, m, ra.span_factor)
                        * eta3(n_terms, m, rb.span_factor)))
    return cross_sem(ra.var[i], rb.var[i], cross, eta), "eta3"


def _check_pairable(ra, rb, estimator: str) -> None:
    """Everything that must hold before two coefficient arrays may be multiplied."""
    if not ra.is_variance:
        raise ValueError(
            f"{estimator!r} does not return a variance, so it has no cross "
            "form. MTIE is an extreme value and TIE rms retains a "
            "deterministic offset; neither is a mean of squares, so "
            "replacing one square with a product means nothing."
        )
    if not ra.coeffs or not rb.coeffs:
        raise ValueError(
            f"{estimator!r} returned no coefficient arrays, so there is "
            "nothing to multiply. Its `keep_coeffs` must default to True."
        )
    if not np.array_equal(ra.af, rb.af):
        raise ValueError(
            f"the two records produced different averaging factors "
            f"({list(ra.af)} against {list(rb.af)}) despite being aligned. "
            "This should be impossible after `common_support`; report it."
        )
    bad = [int(m) for i, m in enumerate(ra.af)
           if ra.coeffs[i].size != rb.coeffs[i].size]
    if bad:
        raise ValueError(
            f"coefficient arrays differ in length at averaging factors {bad}, "
            "so their terms are not the same terms. Alignment should have "
            "prevented this; report it."
        )
