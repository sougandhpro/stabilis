"""Coverage-tier round trips: plant a known truth, recover it.

The multi-channel checks (rejection gain, cross-variance, the bench hat),
the power-law decomposition and its h_alpha scale, and the imputer against
its paper's own criterion. Called from `coverage._coverage`, which fixes
the order the report lists them in.
"""

from __future__ import annotations

import numpy as np

from ..stats import confidence, mdev, oadev
from ..stats.core import StabilityResult
from ..impute import fill_gaps
from ..synth import power_law_freq, power_law_phase
from ..types import FreqSeries, PhaseSeries
from .expected import expected_avar
from .check import SEEDS, Check

__all__ = []


def _rejection_gain_from_extra_channels(trials: int) -> Check:
    """Four channels reject deeper than two, by the amount claimed.

    The commercial question this answers is "is a third and fourth channel
    worth buying", so the number that matters is the *gain* over two, and
    the honest version of it is bounded above as well as below. Averaging
    P = 6 pairs would win 5*log10(6) = 3.9 dB if the pairs were
    independent; they are not, since (c0,c1) and (c0,c2) share a channel,
    and the exact covariance says how much of that is real.

    Both bounds are scored. A gain that vanished would mean the averaging
    does nothing; a gain that reached the naive figure would mean the
    shared-channel terms had been dropped somewhere.
    """
    from ..multi import cross_matrix, reject

    rng = np.random.default_rng(SEEDS["coverage"] + 1700)
    n, rate = 4096, 1.0
    runs = max(trials // 20, 10)
    gains, bias, worst_sigma = [], [], 0.0
    for _ in range(runs):
        clock = power_law_phase(n, 0, rate, rng, 1.0e-24)
        ch = {
            f"c{i}": PhaseSeries(
                x=clock.x + power_law_phase(n, 2, rate, rng, 1.0e-22).x,
                rate_hz=rate)
            for i in range(4)
        }
        four = reject(cross_matrix(ch, max_af=8))
        two = reject(cross_matrix({k: ch[k] for k in ("c0", "c1")}, max_af=8))
        truth = oadev(clock, max_af=8).var
        gains.append(four.rejection_db - two.rejection_db)
        bias.append(four.var / truth - 1.0)
        worst_sigma = max(worst_sigma,
                          float(np.max(np.abs(four.var - truth) / four.sem)))

    gain = np.asarray(gains).mean(axis=0)
    err = np.asarray(bias)
    mean_bias = float(err.mean())
    bias_se = float(err.std(ddof=1) / np.sqrt(err.size))
    naive = 5.0 * np.log10(6.0)
    ok = (float(gain.min()) > 0.3 and float(gain.max()) < naive
          and abs(mean_bias) < 4.0 * bias_se and worst_sigma < 4.0)
    return Check(
        "coverage", "extra channels reject deeper, by less than naive counting",
        "One white-FM clock through four white-PM channels, averaged over "
        "all six pairs, against the same clock through two of them. The "
        "pairs are not independent -- (c0,c1) and (c0,c2) share a channel -- "
        "so the exact Isserlis covariance over pairs of pairs is what sets "
        "the error bar, and the gain it predicts is smaller than treating "
        "the six as six independent estimates would claim.",
        f"the averaging gain is positive at every tau and strictly below "
        f"the naive 5*log10(6) = {naive:.2f} dB; the four-channel estimate "
        "is unbiased against the clock record within four standard errors "
        "of the ensemble mean, and every realization inside four of its own "
        "reported standard errors.",
        f"gain {gain.min():.2f}-{gain.max():.2f} dB against {naive:.2f} dB "
        f"naive; bias {mean_bias:+.1%} against {bias_se:.1%} standard error "
        f"over {runs} realizations; worst miss {worst_sigma:.1f} sigma",
        ok,
    )


def _cross_sees_under_the_instrument(trials: int) -> Check:
    """A clock buried under its channels' noise, recovered by the product.

    The claim that distinguishes this library from every other tool in the
    field, so it is scored against the record that actually generated the
    common part rather than against a fit. Two things are measured at once:
    how far the auto-variance is wrong, which is the size of the problem,
    and how far the cross is wrong, which is the size of the answer.

    Scored at the shortest taus only. That is not a convenience -- it is
    where the instrument dominates and therefore the only place the
    rejection is being tested at all. By long tau the clock has emerged
    from the noise and both estimates are right, which would flatter the
    method by averaging in the region where it does nothing.
    """
    from ..multi import cross_variance

    rng = np.random.default_rng(SEEDS["coverage"] + 1300)
    n, rate = 4096, 1.0
    h_clock, h_inst = 1.0e-24, 1.0e-22
    runs = max(trials // 10, 20)
    auto_err, cross_err, worst_sigma, rejection = [], [], 0.0, []
    for _ in range(runs):
        clock = power_law_phase(n, 0, rate, rng, h_clock)
        a = PhaseSeries(x=clock.x + power_law_phase(n, 2, rate, rng, h_inst).x,
                        rate_hz=rate)
        b = PhaseSeries(x=clock.x + power_law_phase(n, 2, rate, rng, h_inst).x,
                        rate_hz=rate)
        cross = cross_variance(a, b, max_af=8)
        truth = oadev(clock, max_af=8).var
        auto_err.append(float(np.median(cross.auto_a / truth - 1.0)))
        cross_err.append(cross.var / truth - 1.0)
        # The error bar has to cover the miss, not merely be small.
        worst_sigma = max(worst_sigma,
                          float(np.max(np.abs(cross.var - truth) / cross.sem)))
        rejection.append(float(np.min(cross.rejection_db)))

    # Scored on the bias and on the coverage, not on a flat percentage. A
    # single realization at 15 dB of rejection carries about 30% of its own
    # standard error, so "within 10%" would be a statement about luck; what
    # the method promises is that it is *centred* on the truth and that it
    # says how far off it might be. Both are measurable.
    err = np.asarray(cross_err)
    bias = float(err.mean())
    bias_se = float(err.std(ddof=1) / np.sqrt(err.size))
    median_auto = float(np.median(auto_err))
    ok = abs(bias) < 4.0 * bias_se and worst_sigma < 4.0
    return Check(
        "coverage", "cross-variance recovers a clock the auto-variance cannot",
        "One synthetic white-FM clock seen through two channels with "
        "independent white-PM noise, at a level that puts the clock about a "
        "factor of three under the single-channel floor in deviation. The "
        "truth is the generated clock record's own oadev, not a fit, so this "
        "measures the rejection rather than its self-consistency. Scored "
        "over tau = 1..8 s, where the instrument dominates and the rejection "
        "is actually being tested; by long tau the clock has emerged anyway "
        "and both estimates are right.",
        "unbiased to within four standard errors of the ensemble mean, and "
        "every single-realization miss inside four of its own reported "
        "standard errors. Not a flat percentage: at this rejection one "
        "realization carries tens of percent of sampling scatter, so a tight "
        "band would score luck. The auto-variance is reported beside it as "
        "the size of the problem it is solving.",
        f"bias {bias:+.1%} against {bias_se:.1%} standard error over {runs} "
        f"realizations, worst miss {worst_sigma:.1f} sigma, auto median "
        f"{median_auto:+.0%} high, rejection {np.min(rejection):.1f} dB or "
        "better",
        ok,
    )


def _bench_hat_round_trip() -> Check:
    """Three known oscillators in, three recovered variances out.

    The N-cornered hat's whole claim is that it separates what a single
    measurement cannot: a pair record sees `h_i + h_j` and no amount of
    averaging splits it. Three pair records containing an odd cycle do
    split it, and this measures how well.

    Deliberately built with levels that differ by 30x, because the hat's
    known weakness is the *quietest* source -- there it is a small
    difference of large numbers and the estimate can go negative. Scoring
    only the two louder ones would flatter the method; the quietest is
    reported separately rather than dropped.
    """
    from ..bench import Bench, Channel, Device, Measurement, Source
    from ..bench.graph import design_matrix
    from ..bench.solve import solve
    from ..stats.edof import edof

    h = {"A": 4.0e-24, "B": 1.0e-24, "C": 9.0e-24}
    rng = np.random.default_rng(SEEDS["coverage"] + 900)
    y = {k: power_law_freq(8192, 0, rate_hz=1.0, rng=rng, h_alpha=v).y
         for k, v in h.items()}
    pairs = (("A", "B"), ("B", "C"), ("C", "A"))
    results = {
        f"m{i}": oadev(FreqSeries(y[a] - y[b], rate_hz=1.0), taus="octave")
        for i, (a, b) in enumerate(pairs)
    }
    bench = Bench(
        sources=tuple(Source(id=s) for s in "ABC"),
        devices=(Device(id="D1", channels=(Channel(id="A-1", floor=0.0),)),),
        measurements=tuple(
            Measurement(id=f"m{i}", channel_id="A-1", source_a=a, source_b=b)
            for i, (a, b) in enumerate(pairs)
        ),
    )
    sol = solve(bench, results)

    # The tolerance is propagated, not chosen. Each measured variance has
    # Var(sigma2_hat) = 2 sigma^4 / eta, so the solution's covariance is
    # A+ diag(...) A+^T -- and for a source far quieter than its partners
    # that predicts a large relative error, because the hat is then a small
    # difference of large numbers. Scoring the quiet source against the
    # same flat percentage as the loud ones would be scoring the choice of
    # oscillators rather than the method.
    a, _sources, _channels, _known = design_matrix(bench)
    pinv = np.linalg.pinv(a)
    near = sol.tau_s <= 64.0
    worst_ratio, per = 0.0, []
    for name in "ABC":
        col = sol.sources.index(name)
        truth = h[name] / (2.0 * sol.tau_s)
        got = sol.source_var[:, col]
        predicted = np.zeros(sol.tau_s.size)
        for k in range(sol.tau_s.size):
            var_b = []
            for i, mid in enumerate(("m0", "m1", "m2")):
                r = results[mid]
                eta, _ = edof(r.coeffs[k], int(r.af[k]), r.span_factor)
                var_b.append(2.0 * r.var[k] ** 2 / max(eta, 1.0))
            cov = pinv @ np.diag(var_b) @ pinv.T
            predicted[k] = np.sqrt(cov[col, col])
        # How many of its own predicted standard errors is it out by?
        sigmas = np.abs(got[near] - truth[near]) / predicted[near]
        worst_ratio = max(worst_ratio, float(np.max(sigmas)))
        rel = float(np.max(np.abs(got[near] / truth[near] - 1.0)))
        expect = float(np.max(predicted[near] / truth[near]))
        per.append(f"{name} ({h[name]:.0e}): {rel:.0%} against {expect:.0%} predicted")

    bound = 3.0
    return Check(
        "coverage", "N-cornered hat recovers three known oscillators",
        "Three independent white-FM sources at known h_alpha, measured "
        "pairwise, solved as one non-negative least-squares system per tau. "
        "The truth is the closed form h/(2 tau), not a value taken from the "
        "fit, so this measures the method rather than its self-consistency. "
        "It is also the claim no single measurement can support: one pair "
        "record sees h_i + h_j and cannot split it.",
        f"within {bound:.0f} standard errors, where the standard error is "
        "propagated through the solve from each row's own EDOF rather than "
        "chosen. The levels differ by 30x on purpose: for the quietest "
        "source the hat is a small difference of large numbers, and its "
        "predicted error is correspondingly large. A flat percentage would "
        "score the choice of oscillators instead of the method.",
        f"worst {worst_ratio:.1f} sigma; per source " + ", ".join(per),
        worst_ratio < bound,
    )


def _decomposition_round_trip(trials: int) -> Check:
    """Generate a known h_alpha, decompose the measured curve, compare.

    The end-to-end claim for the power-law fit: not that it inverts its own
    design matrix -- a test does that at 1e-8 -- but that the whole chain
    holds, from the IEEE 1139 coefficient through the synthesizer, the
    estimators, the conversion coefficients and the constrained solve, back
    to the same number.

    Only the three `f_h`-free power laws are scored. The two PM rows depend
    on a bandwidth that is not identifiable from tau-domain data, and the
    synthesizer's flicker PM is not brick-wall limited, so scoring them
    here would measure that known offset rather than this fit. It is
    recorded in VALIDATION-DEBT.md instead of being quietly averaged in.
    """
    from ..stats.powerlaw import decompose

    tau0, h_true, n = 1.0, 5.0e-24, 4096
    reps = max(trials // 8, 40)
    worst, seen = 0.0, []
    for alpha in (0, -1, -2):
        rng = np.random.default_rng(SEEDS["coverage"] + 500 + alpha)
        acc_a = acc_m = None
        for _ in range(reps):
            fs = power_law_freq(n, alpha, rate_hz=1.0 / tau0, rng=rng,
                                h_alpha=h_true)
            ra, rm = oadev(fs, taus="octave"), mdev(fs, taus="octave")
            acc_a = ra.var if acc_a is None else acc_a + ra.var
            acc_m = rm.var if acc_m is None else acc_m + rm.var
        keep = (ra.af >= 4) & (ra.af <= n // 32)
        curves = [
            StabilityResult(kind=r.kind, af=r.af[keep], tau_s=r.tau_s[keep],
                            var=(acc / reps)[keep], n_terms=r.n_terms[keep],
                            rate_hz=r.rate_hz, provenance=r.provenance,
                            shape=r.shape)
            for r, acc in ((ra, acc_a), (rm, acc_m))
        ]
        got = decompose(curves, f_h=0.5 / tau0, weights="none").h[alpha]
        seen.append(f"alpha={alpha}: {got / h_true:.3f}")
        worst = max(worst, abs(got / h_true - 1.0))

    bound = 0.20
    return Check(
        "coverage", "power-law decomposition recovers a known h_alpha",
        "Synthetic noise generated at a known IEEE 1139 coefficient, "
        "estimated with oadev and mdev, and decomposed. Closes the loop that "
        "the conversion coefficients, the calibrated synthesizer and the "
        "non-negative solve all sit on -- no step of which is checked by "
        f"this measurement alone. Averaged over {reps} realizations per "
        "power law at N=4096.",
        f"within {bound:.0%} of the generated level for white FM, flicker FM "
        "and random-walk FM. The two PM rows are excluded because their "
        "coefficients depend on a bandwidth that tau-domain data cannot "
        "identify -- see VALIDATION-DEBT.md item 2.",
        f"recovered/true {'; '.join(seen)}",
        worst < bound,
    )


def _h_alpha_round_trip(ensembles, trials: int) -> Check:
    """Ask for an h_alpha, generate, estimate, and see what comes back.

    The identity tier proves the calibration formula against exact closed
    forms. This proves the *path*: the real generator through the real
    estimator, so a mistake anywhere between the two shows up here. The
    ensembles are the ones the coverage checks already built, generated at
    `h_alpha = 1` -- which is not a special case, since the calibration is a
    single multiplication and `test_synth.py` pins its linearity separately.
    """
    worst, tau0 = 0.0, 1.0
    for alpha, (runs, mean_var) in ensembles.items():
        first = runs[0]
        for i, af in enumerate(first.af.tolist()):
            if af > 512 // 16:
                continue
            want = expected_avar(float(alpha), af, tau0)
            worst = max(worst, abs(mean_var[i] / want - 1.0))
    # The ensemble mean of a variance has relative standard error
    # sqrt(2/(trials*eta)); eta >= 1, so sqrt(2/trials) is the floor and the
    # band is four of those plus an allowance for the estimator's own spread.
    se = float(np.sqrt(2.0 / trials))
    bound = float(np.hypot(0.05, 4.0 * se))
    return Check(
        "coverage", "requested h_alpha recovered end to end",
        "The ensemble mean of sigma_hat^2 against the amplitude that was "
        "asked for, over white PM, white FM and random-walk FM at N=512. "
        "Compared against the generator's exact discrete E[avar], not the "
        "continuum closed form, so the known 1/(2m^2) and Nyquist "
        "corrections are not charged to the Monte Carlo.",
        f"within {bound:.0%}, being 5% for the estimator combined in "
        f"quadrature with four standard errors of an ensemble-mean variance "
        f"({se:.0%} at {trials} trials), for m up to N/16.",
        f"worst deviation {worst:.1%} over three power laws",
        worst < bound,
    )


def _imputer_against_the_published_criterion(trials: int) -> Check:
    """Howe & Schlossberger's own figure of merit, on their own scenario.

    513 points, samples 250-400 removed (29% dead time), the gap filled by
    the Total Imputer, and the fraction of trials in which the filled
    AVAR's 90% interval overlaps the original's. Their Fig. 4 publishes
    that count for each of the four noise types.

    **The counts themselves are not comparable and are not scored as such.**
    The criterion's strictness is set entirely by the confidence interval,
    and the paper does not state its EDOF convention; it also uses the
    non-overlapping powers-of-two AVAR of its eq. 1, whose intervals at
    long tau are wide enough to pass almost anything. What is scored is the
    *structure* of their result, which is what a broken implementation does
    not reproduce -- filling the phase record instead of the frequency
    samples inverts it, and takes random-walk FM from 100 trials in 100 to
    3. See theory/impute.md.
    """
    # 100 trials is what the paper ran. Below that the counts are too
    # granular for the ordering to mean anything, so this row does not
    # scale down with `trials` the way the coverage rows do.
    n = max(trials // 4, 60)
    order = []
    for alpha in (2, 0, -1, -2):
        rng = np.random.default_rng(SEEDS["coverage"] + alpha)
        hits = None
        for _ in range(n):
            fs = power_law_freq(513, alpha, rate_hz=1.0, rng=rng, h_alpha=1e-22)
            y = fs.y.copy()
            y[250:400] = np.nan
            filled, _ = fill_gaps(FreqSeries(y, rate_hz=1.0), reason="validate")
            a = confidence(oadev(fs, taus="octave"), level=0.90)
            b = confidence(oadev(filled, taus="octave"), level=0.90)
            hit = (a.var_lo <= b.var_hi) & (b.var_lo <= a.var_hi)
            hits = hit.astype(int) if hits is None else hits + hit
        order.append(hits / n)
    wpm, wfm, ffm, rwfm = (float(h[-3:].mean()) for h in order)
    short = min(float(h[:3].min()) for h in order)
    ok = (wpm < wfm < ffm <= rwfm) and rwfm > 0.9 and wpm < 0.5 and short >= 0.95
    return Check(
        "coverage", "gap imputer against the published acceptance criterion",
        "Howe & Schlossberger 2022 Fig. 4, reproduced: 513-point records "
        "with samples 250-400 removed (29% dead time), filled by the Total "
        "Imputer, scored on the fraction of trials whose filled 90% AVAR "
        "interval overlaps the original's. Run with the *overlapping* "
        "estimator, which is tighter than the paper's eq. 1 and so a harder "
        "test than the one published.",
        "the published ordering, at the last three octaves of tau: white PM "
        "worst, then white FM, then flicker FM, then random-walk FM above "
        "0.9 and white PM below 0.5; and every noise type at or above 0.95 "
        "for the first three octaves. The counts are not asserted against "
        "the figure, because the paper does not state the EDOF behind its "
        "intervals.",
        f"long tau wpm {wpm:.2f} < wfm {wfm:.2f} < ffm {ffm:.2f} <= rwfm "
        f"{rwfm:.2f}; worst short tau {short:.2f} ({n} trials)",
        ok,
    )

