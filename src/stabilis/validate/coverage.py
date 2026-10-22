"""The coverage tier: do the error bars cover the truth as often as claimed.

Monte-Carlo, so every tolerance widens with the measurement's own standard
error and trial counts below `MIN_COVERAGE_TRIALS` are raised rather than
believed. `_coverage` is the tier's entry point and fixes its order.
"""

from __future__ import annotations

import numpy as np

from ..stats import ONE_SIGMA, TWO_SIGMA, confidence, oadev, theo1
from ..stats.edof import eta1
from ..stats.greenhall import greenhall_edof
from ..stats.wavelet import wavelet_variance
from ..synth import power_law_freq, power_law_phase
from .expected import expected_wvar
from .check import MIN_COVERAGE_TRIALS, SEEDS, Check
from .roundtrip import (
    _bench_hat_round_trip,
    _cross_sees_under_the_instrument,
    _decomposition_round_trip,
    _h_alpha_round_trip,
    _imputer_against_the_published_criterion,
    _rejection_gain_from_extra_channels,
)

__all__ = []


def _coverage(trials: int) -> list[Check]:
    """Tier 3. The measurement no competing tool publishes.

    The acceptance band widens with the Monte-Carlo standard error, because
    a measured coverage is itself an estimate. Fixing the band at the
    published spread would make the report fail spuriously whenever it was
    run at a lower trial count -- and a report that cries wolf is not
    evidence any more than one that cannot fail.
    """
    out = []
    trials = max(trials, MIN_COVERAGE_TRIALS)

    # Generate each ensemble once and score both confidence levels against
    # it. Regenerating per level would double the cost for nothing -- the
    # realizations do not depend on the level.
    ensembles = {}
    for alpha in (2, 0, -2):
        rng = np.random.default_rng(SEEDS["coverage"] + alpha)
        runs = [oadev(power_law_freq(512, alpha, rate_hz=1.0, rng=rng),
                      taus="octave") for _ in range(trials)]
        ensembles[alpha] = (runs, np.mean([r.var for r in runs], axis=0))

    for level, label in ((ONE_SIGMA, "68.27%"), (TWO_SIGMA, "95.45%")):
        worst_lo, worst_hi = 1.0, 0.0
        for runs, truth in ensembles.values():
            hits = np.zeros(runs[0].af.size)
            for r in runs:
                c = confidence(r, level=level)
                hits += (c.var_lo <= truth) & (truth <= c.var_hi)
            cov = hits / trials
            worst_lo = min(worst_lo, float(cov.min()))
            worst_hi = max(worst_hi, float(cov.max()))

        # Systematic allowance (how far the estimator itself strays at the
        # last octave) plus four Monte-Carlo standard errors.
        systematic = 0.09 if level == ONE_SIGMA else 0.07
        se = float(np.sqrt(level * (1.0 - level) / trials))
        lo_bound = level - systematic - 4.0 * se
        hi_bound = min(1.0, level + systematic + 4.0 * se)
        ok = lo_bound <= worst_lo and worst_hi <= hi_bound
        vacuous = " The upper bound saturates at 1.0 here, so overcoverage " \
                  "is caught by the 68.27% check rather than this one." \
                  if hi_bound >= 1.0 else ""
        out.append(Check(
            "coverage", f"measured containment at nominal {label}",
            "Fraction of realizations whose interval contains the ensemble "
            "truth, over white PM, white FM and random-walk FM at N=512. "
            "This validates the EDOF estimator and the chi-square inversion "
            "jointly, end to end -- either being wrong shows up here and "
            "nowhere else.",
            f"within [{lo_bound:.3f}, {hi_bound:.3f}] = nominal +/- "
            f"{systematic:.2f} systematic +/- 4 Monte-Carlo standard errors "
            f"({se:.3f} at {trials} trials). The systematic part is how far "
            "the estimator strays at the last octave, where eta is small "
            f"and the autocorrelation is barely resolved.{vacuous}",
            f"{worst_lo:.3f} to {worst_hi:.3f} against nominal {label}",
            ok,
        ))

    runs = ensembles[0][0]  # white FM, already generated above
    v = np.array([r.var for r in runs])
    eta_true = 2.0 * v.mean(axis=0) ** 2 / v.var(axis=0)
    ratios = []
    for i, af in enumerate(runs[0].af):
        if int(af) > 512 // 32:
            continue
        est = np.median([eta1(r.coeffs[i], int(af)) for r in runs])
        ratios.append(est / eta_true[i])
    worst = max(abs(r - 1.0) for r in ratios)
    # eta_true is itself a variance-of-a-variance, whose relative standard
    # error is sqrt(2/trials) for a Gaussian sample. The estimator's own
    # error and the truth's uncertainty add in quadrature.
    truth_se = float(np.sqrt(2.0 / trials))
    bound = float(np.hypot(0.10, 4.0 * truth_se))
    out.append(Check(
        "coverage", "eta1 against Monte-Carlo truth",
        "EDOF is defined by Var(sigma_hat^2) = 2 sigma^4 / eta, and both "
        "sides are measurable. No published table is involved -- every such "
        "table assumes a noise type, which is the assumption this estimator "
        "exists to avoid.",
        f"within {bound:.0%} for m up to N/32, being 10% for the estimator "
        f"combined in quadrature with four standard errors of the "
        f"Monte-Carlo truth ({truth_se:.0%} at {trials} trials). The last "
        "octave is looser and is covered by the coverage checks above.",
        f"worst deviation {worst:.0%} over m = 1..16",
        worst < bound,
    ))
    out += _greenhall_against_truth(ensembles, trials, truth_se)
    out.append(_h_alpha_round_trip(ensembles, trials))
    out.append(_wavelet_level_against_exact_expectation(trials))
    out.append(_imputer_against_the_published_criterion(trials))
    out.append(_theo1_edf_coverage(trials))
    out.append(_decomposition_round_trip(trials))
    out.append(_bench_hat_round_trip())
    out.append(_cross_sees_under_the_instrument(trials))
    out.append(_rejection_gain_from_extra_channels(trials))
    out.extend(_fitted_parameter_coverage(trials))
    return out


def _fitted_parameter_coverage(trials: int) -> list[Check]:
    """Fit a known curve many times; do the stated intervals contain it?

    Two claims scored on one ensemble, because they are the same
    measurement read twice.

    **The M4 pass criterion**: a maximum-likelihood fit recovers a known
    parameter from synthesized noise within its stated uncertainty. White FM
    at a known `h_alpha` has `oadev` variance `h0/(2 tau)` exactly, so both
    fitted parameters have a truth to be scored against and no reference
    table is involved.

    **What the cross-tau correlation is worth**, which is the claim no
    competing tool can make: the same intervals recomputed with the
    correlation deleted under-cover, and by how much. That number is the
    argument for `stats/taucov.py` existing, and stating it as a measured
    coverage rather than as a ratio of error bars is what makes it evidence.

    Scored on the *slope*, not on the amplitude. Both are fitted, but the
    slope is the parameter a customer reads off a specification -- "is this
    oscillator white-FM limited out to a day" -- and it is where the
    correlation bites hardest, since a coherent tilt of the whole curve is
    exactly what correlated points produce.
    """
    from ..fitting import fit as run_fit
    from .._specfun import norm_ppf

    reps = max(trials, MIN_COVERAGE_TRIALS)
    truth = {"A": 1.0e-22, "b": -1.0}
    h_alpha = 2.0 * truth["A"]
    level = ONE_SIGMA
    z = float(norm_ppf(0.5 * (1.0 + level)))

    rng = np.random.default_rng(SEEDS["coverage"] + 900)
    inside = {"A": 0, "b": 0}
    inside_naive = {"A": 0, "b": 0}
    ratios, failures = [], 0
    for _ in range(reps):
        fs = power_law_freq(4096, 0, rate_hz=1.0, rng=rng, h_alpha=h_alpha)
        got = run_fit(oadev(fs, taus="octave", max_af=256), "powerlaw")
        if not got.converged:
            failures += 1
            continue
        for name in ("A", "b"):
            miss = abs(got.params[name] - truth[name])
            inside[name] += miss <= z * got.stderr[name]
            inside_naive[name] += miss <= z * got.stderr_naive[name]
        ratios.append(got.understatement["b"])

    scored = reps - failures
    se = float(np.sqrt(level * (1.0 - level) / scored))
    # Four Monte-Carlo standard errors and no systematic allowance, unlike
    # the per-point coverage row above. That row needs one because the EDOF
    # estimator itself strays at the last octave; this one was measured at
    # 1500 trials and came back at +0.50 and -1.00 standard errors, so
    # there is nothing to allow for -- and adding an allowance anyway would
    # have made the row unable to fail. Deleting the correlation moves the
    # slope's coverage by 5.8 standard errors, which this band catches and
    # a wider one did not.
    band = 4.0 * se
    measured = {k: inside[k] / scored for k in inside}
    naive_b = inside_naive["b"] / scored

    covered = Check(
        "coverage", "a maximum-likelihood fit recovers a known parameter",
        "White FM synthesized at a known IEEE 1139 coefficient has "
        "`oadev` variance h0/(2 tau) exactly, so a two-parameter power-law "
        "fit has a truth for both its amplitude and its slope. Scored as "
        f"containment of that truth by the stated interval over {scored} "
        "realizations at N=4096 -- the whole chain, from the calibrated "
        "synthesizer through the estimator, the degrees of freedom, the "
        "cross-tau covariance and the chi-square likelihood, back to the "
        "number that went in. Every converged fit is scored, including the "
        "ones whose goodness of fit came out poor: dropping those would be "
        "selecting on the outcome being measured.",
        f"both parameters contained {level:.4f} of the time, within "
        f"{band:.3f} (four Monte-Carlo standard errors)",
        "; ".join(f"{k} {v:.3f}" for k, v in measured.items())
        + (f"; {failures} fits did not converge" if failures else ""),
        all(abs(v - level) <= band for v in measured.values()),
    )

    # Two conditions, not one. "The naive interval under-covers" is only
    # evidence for this machinery if the corrected interval does *not* --
    # otherwise a run in which both were broken in the same way would pass
    # while proving nothing.
    margin = 3.0 * se
    deficit = Check(
        "coverage", "ignoring the cross-tau correlation under-covers",
        "The same fits, with the same fitted values, scored against the "
        "interval that would be quoted if the taus were treated as "
        "independent -- which is what every other tool reports. The points "
        "of an overlapping estimator come from the same phase samples, so "
        "that interval is too narrow, and this measures by how much rather "
        "than asserting it.",
        f"the uncorrelated interval contains the true slope less than "
        f"{level - margin:.3f} of the time, and the correlation-aware one "
        f"is within {band:.3f} of {level:.4f}",
        f"slope containment {naive_b:.3f} uncorrelated against "
        f"{measured['b']:.3f} corrected; mean error-bar ratio "
        f"{float(np.mean(ratios)):.2f}",
        naive_b < level - margin and abs(measured["b"] - level) <= band,
    )
    return [covered, deficit]


def _theo1_edf_coverage(trials: int) -> Check:
    """SP 1065 Table 9, measured. A published table is worth its coverage.

    The Thêo family had no confidence interval at all until this table was
    implemented -- Thêo1 is a sum over a triangular set of lag pairs rather
    than a mean of one coefficient per term, so there is no array for the
    data-driven estimators to read an autocorrelation from, and Table 9 is
    the only route. That makes it the one EDOF path here with no second
    opinion, which is exactly why it gets measured rather than trusted.

    **The band is asymmetric on purpose.** An interval that is too wide is
    a defect of efficiency; one that is too narrow is a defect of
    correctness, and only the second can make a conformity statement wrong.
    Table 9 is a fitted approximation and it under-estimates the degrees of
    freedom at the long-tau end, so the measured containment runs above
    nominal there. That is allowed room; falling below is not.
    """
    trials = max(trials, MIN_COVERAGE_TRIALS)
    worst_lo, worst_hi = 1.0, 0.0
    for alpha in (2, 0, -2):
        rng = np.random.default_rng(SEEDS["coverage"] + alpha)
        runs = [theo1(power_law_phase(512, alpha, rate_hz=1.0, rng=rng,
                                      h_alpha=1e-22), taus="octave")
                for _ in range(trials)]
        truth = np.mean([r.var for r in runs], axis=0)
        hits = np.zeros(runs[0].af.size)
        for r in runs:
            c = confidence(r, level=ONE_SIGMA, method="theo1", alpha=alpha)
            hits += (c.var_lo <= truth) & (truth <= c.var_hi)
        cov = hits / trials
        worst_lo = min(worst_lo, float(cov.min()))
        worst_hi = max(worst_hi, float(cov.max()))

    se = float(np.sqrt(ONE_SIGMA * (1.0 - ONE_SIGMA) / trials))
    lo_bound = ONE_SIGMA - 0.10 - 4.0 * se
    hi_bound = 0.95
    ok = lo_bound <= worst_lo and worst_hi <= hi_bound
    return Check(
        "coverage", "Thêo1 intervals from SP 1065 Table 9",
        "Fraction of realizations whose Thêo1 interval contains the "
        "ensemble truth, over white PM, white FM and random-walk FM at "
        "N=512, at every averaging factor Thêo1 offers. Table 9 is the only "
        "degrees-of-freedom route this family has -- there is no "
        "coefficient array for the data-driven estimators to read -- so "
        "there is no second opinion to differ from and the measurement is "
        "the whole of the evidence.",
        f"containment within [{lo_bound:.3f}, {hi_bound:.3f}] against "
        f"nominal {ONE_SIGMA:.4f}. The lower bound is nominal minus 0.10 "
        f"systematic minus four Monte-Carlo standard errors ({se:.3f} at "
        f"{trials} trials); the upper is deliberately looser, because "
        "Table 9 is a fitted approximation that under-estimates the "
        "degrees of freedom at long tau and an interval that is too wide "
        "is a defect of efficiency rather than of correctness.",
        f"{worst_lo:.3f} to {worst_hi:.3f} over three power laws",
        ok,
    )


def _wavelet_level_against_exact_expectation(trials: int) -> Check:
    """End to end: does the estimator return the level the theory says?

    `_wavelet_level_identity` pins the *filters* exactly. What it cannot
    reach is the aggregation on top of them -- the boundary drop, the mean
    of squares, the `tau_j = 2**(j-1) tau0` assignment -- which for Haar is
    tied down by `2 nu**2 = oavar` and for the longer filters by nothing.
    So run the estimator and compare its ensemble mean against
    `E[nu_j**2] = ||a_j * h||**2`, which is exact and needs no simulation of
    its own.

    Scored in units of the ensemble's *own* standard error rather than at a
    fixed percentage. The coarsest levels retain a handful of coefficients,
    so their spread is genuinely large, and a fixed band would either pass
    them vacuously or fail them spuriously.
    """
    tau0 = 1.0
    worst, where = 0.0, ""
    for alpha in (2, 0, -2):
        rng = np.random.default_rng(SEEDS["coverage"] + alpha)
        series = [power_law_freq(512, alpha, rate_hz=1.0, rng=rng)
                  for _ in range(trials)]
        for name in ("haar", "d4", "d6", "d8"):
            runs = [wavelet_variance(s, name=name, keep_coeffs=False)
                    for s in series]
            v = np.array([r.var for r in runs])
            want = np.array([expected_wvar(name, level, float(alpha), tau0)
                             for level in range(1, v.shape[1] + 1)])
            # Relative standard error of the ensemble mean, measured rather
            # than assumed: eta is not known here and is the quantity the
            # wavelet EDOF check exists to estimate.
            se = v.std(axis=0, ddof=1) / (np.sqrt(trials) * v.mean(axis=0))
            z = np.abs(v.mean(axis=0) / want - 1.0) / se
            if z.max() > worst:
                worst, where = float(z.max()), f"{name}, alpha={alpha}"
    return Check(
        "coverage", "wavelet variance against its exact expected value",
        "The ensemble mean of nu**2(tau_j) against ||a_j * h||**2, the "
        "expectation computed exactly from the generator's impulse "
        "response -- no Monte Carlo on the truth side. Over haar, d4, d6 "
        "and d8, at white PM, white FM and random-walk FM, N=512. This is "
        "the only absolute check the non-Haar wavelet variances have: "
        "every other one in the suite is scale-free.",
        "within 4 standard errors of the ensemble mean, measured from the "
        "ensemble itself. 96 level-and-filter comparisons are scored, so a "
        "3-sigma band would be expected to throw a false failure roughly "
        "one run in four.",
        f"worst {worst:.1f} sigma ({where})",
        worst < 4.0,
    )


def _greenhall_against_truth(ensembles, trials: int, truth_se: float) -> list[Check]:
    """The model-based EDOF measured the same way eta1 is, and compared to it.

    Two checks, deliberately separate. The first asks whether Greenhall's
    algorithm predicts the sampling variance it claims to. The second asks
    what its assumption costs against an estimator that makes none -- and
    that is the whole argument for the default this library ships, so it is
    measured rather than asserted in prose.
    """
    worst_model, worst_gap, at_m1 = 0.0, 0.0, {}
    for alpha, (runs, _) in ensembles.items():
        v = np.array([r.var for r in runs])
        eta_true = 2.0 * v.mean(axis=0) ** 2 / v.var(axis=0)
        first = runs[0]
        for i, af in enumerate(first.af.tolist()):
            eta = greenhall_edof(alpha, first.shape.d, af, 513,
                                 first.shape.modified, first.shape.overlapped)
            if af == 1:
                at_m1[alpha] = eta / eta_true[i]
                continue
            if af > 512 // 32:
                continue
            worst_model = max(worst_model, abs(eta / eta_true[i] - 1.0))
            data_driven = float(np.median([eta1(r.coeffs[i], af) for r in runs]))
            worst_gap = max(worst_gap, abs(eta / data_driven - 1.0))

    bound = float(np.hypot(0.20, 4.0 * truth_se))
    span = ", ".join(f"alpha={a}: {r:.2f}" for a, r in sorted(at_m1.items()))
    return [
        Check(
            "coverage", "Greenhall EDOF against Monte-Carlo truth",
            "The same definitional measurement applied to the model-based "
            "algorithm: it is told the power law and the record geometry, "
            "never shown the data, and its answer is compared to "
            "2 sigma^4 / Var(sigma_hat^2) measured over the ensemble. This is "
            "the only correctness evidence for the Stable32 parity path that "
            "does not itself come from Stable32.",
            f"within {bound:.0%} for m = 2..16, being 20% for the model "
            f"combined in quadrature with four standard errors of the "
            f"Monte-Carlo truth ({truth_se:.0%} at {trials} trials). Wider "
            "than eta1's band because two approximations are being compared, "
            "not one estimate against a truth.",
            f"worst deviation {worst_model:.0%} over m = 2..16",
            worst_model < bound,
        ),
        Check(
            "coverage", "what assuming a power law costs, measured",
            "Greenhall against eta1 on the same realizations. They answer "
            "different questions -- one is told the noise type, the other "
            "reads the data -- so this is not a correctness check on either. "
            "It is the honest statement of the trade the user is making, and "
            "the reason `auto` never selects the model-based route.",
            "the two agree within 40% for m = 2..16 across white PM, white FM "
            "and random-walk FM. At m = 1 they diverge sharply and by design: "
            "the paper models phase as a difference of a band-unlimited power "
            "law, so its noise is not flat to Nyquist. The ratios there are "
            "reported rather than bounded.",
            f"worst disagreement {worst_gap:.0%} for m = 2..16; "
            f"greenhall/eta1 at m = 1 is {span}",
            worst_gap < 0.40,
        ),
    ]

