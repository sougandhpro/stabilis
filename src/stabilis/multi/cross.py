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

with `k = i - j`. Taking the three correlation *shapes* to be the same --
they are the same estimator at the same averaging factor, so this is far
weaker than assuming a noise type -- the double sum collapses onto the
same `sum rho^2` that defines the EDOF, and

    Var(cross_var) = (auto_a * auto_b + cross_var^2) / eta

which is the familiar `2 sigma^4 / eta` when `b` is `a`. So the error bar
needs no new machinery: `eta` is `eta1` from the coefficients, the
estimator already validated in `stats/edof.py`, with `eta3` as its usual
fallback.

The obvious alternative -- a Bartlett-weighted sum over the *products'*
own autocovariance, which needs no Gaussian assumption at all -- was
implemented first and measured against Monte-Carlo scatter at
0.89 / 0.73 / 0.56 / 0.37 / 0.30 of truth at `m = N/64` through `N/4`, and
lengthening the lag limit made it worse rather than better. The products
are heavy-tailed where the coefficients are Gaussian, so their sample
autocovariance is too noisy to sum over many lags; each extra lag adds
more noise than signal, and the sum shrinks towards zero. The form above
holds to about 10% over the same range. `tests/test_cross.py` keeps the
rejected estimator and measures both, so the finding cannot quietly rot.

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

__all__ = ["CrossResult", "cross_variance", "cross_sem"]


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
        substitute. It is reported from the measured `eta`, so it degrades
        honestly when the terms overlap or the record is short.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return 10.0 * np.log10(np.sqrt(self.auto_a * self.auto_b) / self.sem)


def cross_sem(auto_a: float, auto_b: float, cross: float, eta: float) -> float:
    """Standard error of a cross-variance, from Isserlis.

        sem = sqrt((auto_a * auto_b + cross**2) / eta)

    Finite and positive whatever the sign of `cross`, and equal to
    `sigma^2 * sqrt(2 / eta)` when the two channels are one channel, which
    is the auto-variance result `stats/ci.py` is built on. See the module
    docstring for the derivation and for the estimator this replaced.
    """
    return float(np.sqrt((auto_a * auto_b + cross * cross) / eta))


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
        eta[i], used = _pair_edof(ra, rb, i, int(m), edof_method, lag_limit)
        sem[i] = cross_sem(ra.var[i], rb.var[i], var[i], eta[i])
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


def _pair_edof(ra, rb, i: int, m: int, method: str,
               lag_limit: int) -> tuple[float, str]:
    """One EDOF for the pair: the geometric mean of the two channels'.

    Geometric rather than arithmetic because `eta` enters as a divisor of
    the *product* `auto_a * auto_b`, so it is the product of the two
    channels' effective counts that the algebra asks for. When the channels
    are alike the choice does not matter; when they are not, the geometric
    mean is the one that keeps `sem` equal to the auto-variance answer in
    the limit `b -> a`.

    Falls back exactly as `stats.edof.edof` does, and reports which.
    """
    from ..stats.edof import DATA_DRIVEN_METHODS, MIN_TERMS_FOR_ETA1

    if method not in DATA_DRIVEN_METHODS:
        raise ValueError(
            f"cross_variance: edof_method must be one of "
            f"{DATA_DRIVEN_METHODS}, got {method!r}. The noise-type-based "
            "estimators (greenhall, totvar) are derived for a sum of "
            "squares and do not carry over to a sum of products."
        )
    n_terms = int(ra.coeffs[i].size)
    if method == "eta3" or (method == "auto" and n_terms < MIN_TERMS_FOR_ETA1):
        return eta3(n_terms, m, ra.span_factor), "eta3"
    return float(np.sqrt(eta1(ra.coeffs[i], m, lag_limit)
                         * eta1(rb.coeffs[i], m, lag_limit))), "eta1"


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
