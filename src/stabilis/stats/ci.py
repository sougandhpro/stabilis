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
from ..log import record, warn
from ..provenance import Provenance
from .core import StabilityResult, require_ordered_coeffs
from .edof import EDOF_METHODS, edof
from .greenhall import GREENHALL_RUN_POLICIES, greenhall_edof_pooled
from .edof import eta3, theo1_edof, totvar_edof

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
    # The power law assumed at each tau, when one was assumed at all. None
    # for the data-driven methods, which is the whole point of them.
    alpha: np.ndarray | None = None
    # Per tau, why the requested method was not the one used -- empty
    # string where it was. A report has to be able to answer "why is this
    # row different", and "it silently fell back" is not an answer.
    declined: tuple[str, ...] = ()

    @property
    def dev_lo(self) -> np.ndarray:
        """Lower deviation bound. Monotone, so coverage is preserved."""
        return np.sqrt(self.var_lo)

    @property
    def dev_hi(self) -> np.ndarray:
        return np.sqrt(self.var_hi)


# Averaged frequency points below which the lag-1 autocorrelation is not
# worth believing. Its standard error is about 1/sqrt(n), so at n = 30 that
# is 0.18 -- already comparable to the 0.25 threshold noiseid uses to
# decide whether to difference again. Below that the identification is a
# coin toss, and a coin toss fed to Greenhall produces a confident error
# bar from nothing. Measured symptom: at N = 1000, m = 128 the identifier
# returned white PM for white FM data and the EDOF jumped from 43 to 420.
MIN_POINTS_FOR_NOISE_ID = 30

# Outside the -4..2 range every noise-ID method works in, so it can never
# collide with a real answer.
_UNIDENTIFIED = 99


def alpha_per_tau(result: StabilityResult, alpha=None, series=None) -> np.ndarray:
    """The power law to assume at each averaging factor of `result`.

    Either given -- as one integer for the whole curve, or one per tau --
    or identified from the data, which is what Stable32's `Alpha` column
    does. Noise type is a function of tau, not of a record, so the
    identified route runs per averaging factor rather than once.

    **To reproduce a Stable32 run exactly, pass its `Alpha` column.** The
    identified route deliberately refuses where Stable32 would answer: past
    `MIN_POINTS_FOR_NOISE_ID` averaged points there is not enough data to
    tell the power laws apart, and this library declines rather than
    guessing. Those points come back with a sentinel and the caller falls
    back to the data-driven EDOF, which needs no noise type at all.
    """
    from .noiseid import identify_curve

    if alpha is not None:
        out = np.broadcast_to(np.asarray(alpha, dtype=np.int64), result.af.shape)
        return np.array(out)
    if series is None:
        raise ValueError(
            "confidence: method='greenhall' assumes a power law, so it needs "
            "one. Pass alpha=<int or one per tau>, or series=<the data> to "
            "identify it per tau"
        )
    ids = identify_curve(series, result.af)
    n_freq = series.n - 1 if hasattr(series, "x") else series.n
    return np.asarray(
        [
            _UNIDENTIFIED
            if n is None or n_freq // int(m) < MIN_POINTS_FOR_NOISE_ID
            else n.alpha
            for m, n in zip(result.af.tolist(), ids)
        ],
        dtype=np.int64,
    )


def _theo1_etas(result, alphas, run_policy):
    """SP 1065 Table 9 per tau, pooled over runs -- and ThêoH's two halves.

    Thêo1 is a sum over a triangular set of lag pairs, not a mean of one
    coefficient per term, so there is no array for `eta1` to read an
    autocorrelation from. Table 9 is the only route, and it needs a noise
    type, which is why `alpha` or `series` is not optional here.

    **ThêoH is two estimators spliced**, and one number from either formula
    would be wrong on half the curve. Below the crossover recorded in its
    provenance the points are overlapping Allan, so the Greenhall model
    supplies those; above it they are ThêoBR and Table 9 supplies them.
    Which was used lands in `Confidence.method` per tau, because a curve
    whose error bars come from two different models must be able to say so.

    Runs are pooled by summing degrees of freedom, the same principle the
    Greenhall path uses: a run shorter than the averaging factor
    contributed no terms and contributes no degrees of freedom.
    """
    runs = result.run_lengths or (int(result.n_terms.max()),)
    crossover = _theoh_crossover(result)
    etas, used, declined = [], [], []
    for i, m in enumerate(result.af.tolist()):
        if alphas[i] == _UNIDENTIFIED:
            etas.append(eta3(int(result.n_terms[i]), m, result.span_factor))
            used.append("eta3")
            declined.append(
                f"too few averaged points to identify the noise at m={m}")
            continue
        if crossover is not None and result.tau_s[i] <= crossover:
            eta, how, why = _theoh_allan_eta(result, i, int(alphas[i]),
                                             m, runs, run_policy)
            etas.append(eta)
            used.append(how)
            declined.append(why)
            continue
        total = sum(theo1_edof(int(alphas[i]), m, length)
                    for length in runs if length > m)
        etas.append(max(total, 1.0))
        used.append("theo1")
        declined.append("")
    return etas, used, declined


def _theoh_crossover(result: StabilityResult):
    """The tau at which ThêoH stops being Allan, or None if not a ThêoH."""
    if result.kind != "theoh":
        return None
    for step in reversed(result.provenance.steps):
        if step.action == "theoh":
            return float(step.params["crossover_s"])
    return None


def _theoh_allan_eta(result, i, alpha, m, runs, run_policy):
    """The Allan half of ThêoH: Greenhall's model for overlapping AVAR.

    Declines to the conservative count rather than guessing when the record
    is segmented and the policy is strict, or when the noise is too steep
    for a second difference to be stationary -- the same two ways the
    ordinary Greenhall path declines.
    """
    terms = result.run_terms[i] if i < len(result.run_terms) else None
    if terms is None or terms.size == 0:
        return (eta3(int(result.n_terms[i]), m, 2), "eta3",
                "ThêoH's Allan half kept no per-run term counts")
    try:
        eta, how = greenhall_edof_pooled(
            alpha, 2, m, runs, terms, False, True, run_policy)
    except ValueError as exc:
        return (eta3(int(result.n_terms[i]), m, 2), "eta3",
                str(exc).removeprefix("greenhall: ").split(";")[0])
    if how != "greenhall":
        return (eta3(int(result.n_terms[i]), m, 2), "eta3",
                f"record is in {len(runs)} runs, policy is 'fallback'")
    return eta, "greenhall", ""


def _greenhall_etas(result, alphas, run_policy):
    """EDOF per tau by the model-based route, with the reason where not.

    Declines rather than guesses in three situations: the noise could not
    be identified, the identified law is too steep for this estimator to
    have a stationary difference of, or the record is segmented and the
    caller asked for the strict reading. Each falls back to the
    data-driven estimator and says so.
    """
    etas, used, declined = [], [], []
    shape = result.shape
    for i, m in enumerate(result.af.tolist()):
        why = ""
        if alphas[i] == _UNIDENTIFIED:
            why = f"too few averaged points to identify the noise at m={m}"
        else:
            try:
                eta, how = greenhall_edof_pooled(
                    int(alphas[i]), shape.d, m, result.run_lengths, result.run_terms[i],
                    shape.modified, shape.overlapped, run_policy,
                )
            except ValueError as exc:
                why = str(exc).removeprefix("greenhall: ").split(";")[0]
            else:
                if how == "greenhall":
                    etas.append(eta)
                    used.append(how)
                    declined.append("")
                    continue
                why = f"record is in {len(result.run_lengths)} runs, policy is 'fallback'"
        eta, how = edof(result.coeffs[i], m, result.span_factor, "auto")
        etas.append(eta)
        used.append(how)
        declined.append(why)
    return etas, used, declined


def confidence(
    result: StabilityResult,
    level: float = ONE_SIGMA,
    method: str = "auto",
    alpha=None,
    series=None,
    run_policy: str = "pool",
    live_terms=None,
) -> Confidence:
    """Confidence interval for every tau of a stability result.

    Requires the coefficient arrays, so the estimator must have been called
    with `keep_coeffs=True` (the default). That is the single-CI-path
    contract from core.py doing its job: one implementation here serves
    every estimator, present and future, because they all report the array
    they averaged.

    `method="greenhall"` is the Stable32 parity path and the only one that
    assumes a power law; `alpha` and `series` are how that assumption gets
    supplied, and both the assumed law and any point where the method was
    declined come back on the result and go into provenance. `auto` never
    selects it -- see edof.py.

    `live_terms` is for a record whose gaps were filled. Imputation
    manufactures terms, and the paper it comes from measures the resulting
    EDOF inflation at up to 6x for white noise -- an interval computed from
    the filled term count is too narrow, and wrongly so in the optimistic
    direction. Pass `impute.live_terms(result, mask)` and every eta is
    scaled by the fraction of its terms that touched no imputed sample.
    Where that fraction is zero the interval is **not** computed: NaN
    bounds and a named warning, because at that tau every term this
    estimate rests on contains invented data and there is no honest error
    bar to print. See theory/impute.md.
    """
    if not result.is_variance:
        raise ValueError(
            f"confidence: {result.kind!r} is not a variance, so it has no "
            "chi-square sampling distribution and this interval would be "
            "meaningless. An extreme value like MTIE converges to a Gumbel "
            "law, and its spread grows with record length rather than "
            "shrinking -- the opposite of what an error bar here implies."
        )
    if method not in EDOF_METHODS:
        raise ValueError(f"method must be one of {EDOF_METHODS}, got {method!r}")
    if run_policy not in GREENHALL_RUN_POLICIES:
        raise ValueError(
            f"run_policy must be one of {GREENHALL_RUN_POLICIES}, got {run_policy!r}"
        )
    # `totvar` and `theo1` are closed-form in the noise type and the record
    # geometry and never look at the terms, so they are the two methods a
    # coefficient-free estimator can use. The Thêo family has no coefficient
    # array at all -- Thêo1 is a sum over a triangular set of lag pairs
    # rather than a mean of one coefficient per term -- which is why it had
    # no interval before Table 9 was implemented.
    if method not in ("totvar", "theo1"):
        if len(result.coeffs) != result.af.size:
            raise ValueError(
                f"{result.kind}: confidence needs the coefficient arrays; "
                "the estimator was called with keep_coeffs=False"
            )
        require_ordered_coeffs(result, "the confidence interval")

    alphas = None
    if method == "totvar":
        alphas = alpha_per_tau(result, alpha, series)
        record_s = float(result.n_terms[0] + 2) / result.rate_hz
        etas, used, declined = [], [], []
        for i, (tau, a) in enumerate(zip(result.tau_s, alphas)):
            # An unidentified point declines to the data-driven estimator
            # and says so, exactly as the Greenhall path does. Before this
            # it raised, so one unidentifiable tau at the long end took the
            # whole curve's error bars with it -- and the long end is where
            # identification fails first, so that was every real record.
            if int(a) == _UNIDENTIFIED:
                eta, how = edof(result.coeffs[i], int(result.af[i]),
                                result.span_factor, "auto")
                etas.append(eta)
                used.append(how)
                declined.append(
                    "too few averaged points to identify the noise at "
                    f"m={int(result.af[i])}")
                continue
            if int(a) in (2, 1):
                # SP 1065 defines these against the overlapping ADEV's edf
                # plus two, so the Allan count is what it is built on.
                allan = eta3(int(result.n_terms[0]), 1, result.span_factor)
                etas.append(totvar_edof(int(a), float(tau), record_s,
                                        overlapping_edof=allan))
            else:
                etas.append(totvar_edof(int(a), float(tau), record_s))
            used.append("totvar")
            declined.append("")
    elif method == "theo1":
        alphas = alpha_per_tau(result, alpha, series)
        etas, used, declined = _theo1_etas(result, alphas, run_policy)
    elif method == "greenhall":
        if result.shape is None:
            raise ValueError(
                f"{result.kind}: this estimator declares no Shape, so the "
                "model-based EDOF cannot know its difference order. Use "
                "method='auto'"
            )
        alphas = alpha_per_tau(result, alpha, series)
        etas, used, declined = _greenhall_etas(result, alphas, run_policy)
    else:
        etas, used = [], []
        for m, c in zip(result.af, result.coeffs):
            eta, how = edof(c, int(m), result.span_factor, method)
            etas.append(eta)
            used.append(how)
        declined = [""] * len(etas)


    eta = np.asarray(etas, dtype=np.float64)
    live, starved = None, ()
    if live_terms is not None:
        live = np.asarray(live_terms, dtype=np.float64)
        if live.shape != eta.shape:
            raise ValueError(
                f"confidence: live_terms has {live.size} entries but the "
                f"result has {eta.size} taus. It must be the per-tau live "
                "count for *this* result -- impute.live_terms(result, mask)")
        if np.any(live > result.n_terms):
            raise ValueError(
                "confidence: live_terms exceeds n_terms at some tau, so it "
                "is not a count of this result's own terms")
        eta = eta * live / result.n_terms
        starved = tuple(int(m) for m, k in zip(result.af.tolist(), live) if k == 0)

    if np.all(eta > 0.0):
        lo, hi = chi2_interval(result.var, eta, level)
    else:
        # chi2_ppf has no answer at eta = 0 and should not be asked for one:
        # the quantile of a chi-square with no degrees of freedom is not a
        # wide interval, it is no interval. Evaluate the taus that have one.
        ok = eta > 0.0
        lo = np.full(eta.shape, np.nan)
        hi = np.full(eta.shape, np.nan)
        lo[ok], hi[ok] = chi2_interval(result.var[ok], eta[ok], level)
    fell_back = sum(1 for h in used if h != method and method != "auto")
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
            "alpha_assumed": None if alphas is None else alphas.tolist(),
            "run_policy": run_policy if method == "greenhall" else None,
            "declined": sorted({d for d in declined if d}),
            "live_term_fraction": None if live is None else
                float(np.sum(live) / np.sum(result.n_terms)),
            "no_live_terms_at_af": list(starved),
        },
        f"{eta.size} intervals" + (f", {fell_back} fell back" if fell_back else "")
        + (f", no interval at af {list(starved)}: every term there rests on "
           "imputed data" if starved else ""),
    )
    if starved:
        prov = warn(
            prov, "stats.ci", "confidence",
            {"no_live_terms_at_af": list(starved)},
            f"at af {list(starved)} every term contains imputed samples, so "
            "the degrees of freedom are zero and no interval is quoted. The "
            "point estimate is still there; it just has no error bar that "
            "is not made of invented data.")
    return Confidence(
        level=level,
        edof=eta,
        var_lo=lo,
        var_hi=hi,
        method=tuple(used),
        provenance=prov,
        alpha=alphas,
        declined=tuple(declined),
    )
