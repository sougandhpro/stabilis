"""The power-law decomposition.

Four routes, in increasing order of how much they can go wrong: the solve
against a curve built from the coefficient table itself, the drift identity,
the degeneracy that motivates stacking modified rows, and finally the real
thing -- synthesize noise of known `h_alpha`, estimate it, and ask whether
the fit returns what went in. That last one is only possible because
`synth` is amplitude-calibrated; before that it could not have been written.
"""

from dataclasses import replace

import numpy as np
import pytest

from stabilis.powerlaw_coeffs import ALPHAS, avar_coeff, drift_coeff, mvar_coeff
from stabilis.provenance import new_provenance
from stabilis.stats.classical import mdev, oadev
from stabilis.stats.core import Shape, StabilityResult
from stabilis.stats.powerlaw import decompose
from stabilis.synth import power_law_freq
from stabilis.types import FreqSeries


def synthetic_curve(h: dict, drift: float, tau, f_h: float, modified: bool):
    """A `StabilityResult` built from the coefficients, with no data behind it.

    Exercises the solve in isolation: if the fit cannot invert a curve that
    the design matrix generated, nothing downstream is worth testing.
    """
    tau = np.asarray(tau, dtype=np.float64)
    var = drift_coeff(tau) * drift**2
    for alpha, level in h.items():
        var = var + level * (mvar_coeff(alpha, tau) if modified
                             else avar_coeff(alpha, tau, f_h))
    return StabilityResult(
        kind="mdev" if modified else "oadev",
        af=np.round(tau / tau[0]).astype(int),
        tau_s=tau,
        var=var,
        n_terms=np.full(tau.size, 100),
        rate_hz=1.0 / tau[0],
        provenance=new_provenance("test"),
        shape=Shape(2, modified, True),
    )


TAU = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
F_H = 100.0


def test_recovers_a_mixture_from_a_noiseless_curve():
    """Exact inversion when the curve is exactly the model."""
    want = {2: 1e-22, 1: 3e-23, 0: 5e-24, -1: 2e-24, -2: 7e-26}
    fit = decompose(
        [synthetic_curve(want, 0.0, TAU, F_H, modified=False),
         synthetic_curve(want, 0.0, TAU, F_H, modified=True)],
        f_h=F_H, weights="none")
    for alpha in ALPHAS:
        assert fit.h[alpha] == pytest.approx(want[alpha], rel=1e-8)
    assert fit.drift == pytest.approx(0.0, abs=1e-16)
    assert fit.residual < 1e-30


def test_the_solution_carries_a_kkt_certificate():
    """An exact solver's answer should be checkable, and is."""
    want = {2: 0.0, 1: 0.0, 0: 4e-24, -1: 0.0, -2: 1e-26}
    fit = decompose(
        [synthetic_curve(want, 0.0, TAU, F_H, modified=False),
         synthetic_curve(want, 0.0, TAU, F_H, modified=True)], f_h=F_H)
    assert fit.kkt < 1e-8
    assert all(v >= 0.0 for v in fit.h.values())


def test_pure_drift_returns_the_drift_and_no_noise():
    """`sigma2 = D**2 tau**2/2` exactly, so the fit must find D and nothing else.

    Reuses M1's drift identity, which is asserted at 1e-13 elsewhere. A
    decomposition that leaks drift into random-walk FM -- the two nearest
    slopes, tau^2 against tau^1 -- would show up here.
    """
    d, n = 2.5e-12, 4096
    fs = FreqSeries(y=d * np.arange(n, dtype=np.float64), rate_hz=1.0)
    fit = decompose([oadev(fs, taus="octave"), mdev(fs, taus="octave")],
                    f_h=0.5, weights="none")
    assert fit.drift == pytest.approx(d, rel=1e-6)
    for alpha in ALPHAS:
        assert fit.h[alpha] == pytest.approx(0.0, abs=1e-30)
    assert fit.dominant is None, "pure drift resolves no stochastic noise"


def test_avar_separates_the_pm_noises_only_through_a_logarithm():
    """The degeneracy is conditioning, not rank -- and that is worth stating.

    On a *noiseless* curve, AVAR alone does recover the pair, because
    `[1.038 + 3 ln(2 pi f_h tau)]/tau^2` and `1/tau^2` are not proportional.
    The columns are linearly independent, so the exact fit is exact.

    What makes AVAR alone useless in practice is that they differ only by a
    slowly varying logarithm, so the sub-block is desperately
    ill-conditioned and ordinary measurement noise swamps the distinction.
    The modified rows separate the same pair by a whole power of tau,
    `tau^-3` against `tau^-2`. Both halves are measured here, because the
    first is the honest qualification on the second.
    """
    want = {2: 0.0, 1: 4e-23, 0: 0.0, -1: 0.0, -2: 0.0}
    avar = synthetic_curve(want, 0.0, TAU, F_H, modified=False)
    mvar = synthetic_curve(want, 0.0, TAU, F_H, modified=True)

    # Noiseless: AVAR alone is enough, and says so.
    clean = decompose(avar, f_h=F_H, weights="none")
    assert clean.h[1] == pytest.approx(want[1], rel=1e-6)

    # The conditioning of the two-column PM sub-block, which is what decides
    # whether that survives contact with data.
    pm_avar = np.column_stack([avar_coeff(a, TAU, F_H) for a in (2, 1)])
    pm_both = np.vstack([pm_avar,
                         np.column_stack([mvar_coeff(a, TAU) for a in (2, 1)])])
    assert np.linalg.cond(pm_avar) > 5.0 * np.linalg.cond(pm_both)

    # And with 2% noise on the curve -- far less than a real sigma(tau)
    # point scatters -- the AVAR-only fit misassigns badly. Measured over
    # several seeds rather than one, because on any single realization it
    # can be right by luck: over seeds 0..5 it returns 0.87, 1.00, 0.81,
    # 0.00, 0.97 and 0.69 times the true h_1, including one where NNLS put
    # the entire flicker-PM power into white PM and pinned h_1 at zero.
    # The stacked fit stays inside 5% on every one of them.
    worst_alone, worst_stacked = 0.0, 0.0
    for seed in range(6):
        kick = 1.0 + 0.02 * np.random.default_rng(seed).standard_normal(TAU.size)
        noisy_a = replace(avar, var=avar.var * kick)
        noisy_m = replace(mvar, var=mvar.var * kick)
        alone = decompose(noisy_a, f_h=F_H, weights="none")
        stacked = decompose([noisy_a, noisy_m], f_h=F_H, weights="none")
        worst_alone = max(worst_alone, abs(alone.h[1] / want[1] - 1.0))
        worst_stacked = max(worst_stacked, abs(stacked.h[1] / want[1] - 1.0))
    assert worst_alone > 0.5
    assert worst_stacked < 0.10


@pytest.mark.parametrize("alpha,h_alpha", [(0, 4e-24), (-1, 2e-24), (-2, 5e-27)])
def test_recovers_h_alpha_from_real_synthesized_noise(alpha, h_alpha):
    """End to end: generate at a known `h_alpha`, estimate, decompose.

    The one test here with actual data in it, and the reason the
    synthesizer had to be calibrated first. Averaged over realizations,
    because a single record's sigma(tau) scatters by tens of percent at long
    tau and the fit inherits that.

    Only the three `f_h`-free power laws are asked for. The two PM rows are
    excluded deliberately: `theory/powerlaw.md` records that the
    Kasdin-Walter flicker-PM process is not brick-wall limited, so its
    measured level differs from the coefficient by a known amount, and
    testing against it here would be testing that discrepancy rather than
    this solver.
    """
    trials, n, tau0 = 60, 4096, 1.0
    rng = np.random.default_rng(31337 + alpha)
    acc_a = acc_m = None
    for _ in range(trials):
        fs = power_law_freq(n, alpha, rate_hz=1.0 / tau0, rng=rng, h_alpha=h_alpha)
        ra, rm = oadev(fs, taus="octave"), mdev(fs, taus="octave")
        acc_a = ra.var if acc_a is None else acc_a + ra.var
        acc_m = rm.var if acc_m is None else acc_m + rm.var

    keep = (ra.af >= 4) & (ra.af <= n // 32)
    curves = []
    for res, acc in ((ra, acc_a), (rm, acc_m)):
        curves.append(StabilityResult(
            kind=res.kind, af=res.af[keep], tau_s=res.tau_s[keep],
            var=(acc / trials)[keep], n_terms=res.n_terms[keep],
            rate_hz=res.rate_hz, provenance=res.provenance, shape=res.shape))
    fit = decompose(curves, f_h=0.5 / tau0, weights="none")

    assert fit.h[alpha] == pytest.approx(h_alpha, rel=0.15)
    assert fit.dominant == alpha
    others = sum(fit.h[a] * float(avar_coeff(a, np.array([4.0]), 0.5)[0])
                 for a in ALPHAS if a != alpha)
    true = fit.h[alpha] * float(avar_coeff(alpha, np.array([4.0]), 0.5)[0])
    assert others < 0.25 * true, "power leaked into the wrong noise types"


# ---------------------------------------------------------------------------
# Refusals and flags -- the parts an ISO-17025 report has to be able to quote
# ---------------------------------------------------------------------------


def test_missing_f_h_is_flagged_not_silently_assumed():
    """`f_h` is not identifiable from the data, so a default must announce itself."""
    want = {0: 4e-24}
    curves = [synthetic_curve(want, 0.0, TAU, 0.5, modified=False),
              synthetic_curve(want, 0.0, TAU, 0.5, modified=True)]
    fit = decompose(curves)
    assert fit.assumed_fh is True
    assert fit.f_h == pytest.approx(0.5)
    actions = [s.action for s in fit.provenance.steps]
    assert "assumed_fh" in actions

    given = decompose(curves, f_h=0.5)
    assert given.assumed_fh is False
    assert "assumed_fh" not in [s.action for s in given.provenance.steps]


def test_rows_outside_the_valid_range_are_dropped_and_counted():
    """`2*pi*f_h*tau >= 10` or the coefficient does not apply."""
    tau = np.array([0.001, 0.01, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    curves = [synthetic_curve({0: 4e-24}, 0.0, tau, 1.0, modified=False),
              synthetic_curve({0: 4e-24}, 0.0, tau, 1.0, modified=True)]
    fit = decompose(curves, f_h=1.0)
    # 2*pi*1*tau >= 10 needs tau >= 1.59, so 0.001, 0.01 and 1.0 go.
    assert fit.dropped_rows == 6
    assert fit.n_rows == 12
    assert fit.tau_s.min() >= 1.59
    assert "dropped_rows" in [s.action for s in fit.provenance.steps]


def test_non_variance_statistics_are_refused():
    """MTIE and TIE rms are not variances; the sum over power laws is false.

    The refusal keys on the result's `is_variance` flag rather than on its
    name, so this uses the real estimators. `test_mtie.py` separately
    proves a disguised plugin result is caught by the same check.
    """
    from stabilis.stats import mtie, tierms
    from stabilis.synth import power_law_phase

    ps = power_law_phase(2048, 0, rate_hz=1.0, rng=np.random.default_rng(21))
    for estimator in (mtie, tierms):
        with pytest.raises(ValueError, match="is not a variance"):
            decompose(estimator(ps, taus="octave"), f_h=F_H)


def test_too_few_rows_names_the_way_out():
    """Six unknowns need six rows, and the error says how to get them."""
    short = synthetic_curve({0: 1e-24}, 0.0, TAU[:3], F_H, modified=False)
    with pytest.raises(ValueError, match="usable rows for 6 unknowns"):
        decompose(short, f_h=F_H)


def test_everything_dropped_is_a_directed_error():
    curves = synthetic_curve({0: 1e-24}, 0.0, TAU, 1e-6, modified=False)
    with pytest.raises(ValueError, match="every row was dropped"):
        decompose(curves, f_h=1e-6)


def test_bad_weights_method_is_refused():
    r = synthetic_curve({0: 1e-24}, 0.0, TAU, F_H, modified=False)
    with pytest.raises(ValueError, match="weights must be one of"):
        decompose(r, f_h=F_H, weights="inverse-vibes")


def test_edof_weighting_runs_on_real_results_and_changes_the_answer():
    """The default path, exercised on results that carry coefficient arrays.

    Weighted and unweighted fits should not be identical -- if they were,
    the weights would be doing nothing and the default would be a lie.
    """
    rng = np.random.default_rng(4)
    fs = power_law_freq(2048, 0, rate_hz=1.0, rng=rng, h_alpha=3e-24)
    curves = [oadev(fs, taus="octave"), mdev(fs, taus="octave")]
    weighted = decompose(curves, f_h=0.5, weights="edof")
    plain = decompose(curves, f_h=0.5, weights="none")
    assert weighted.h[0] == pytest.approx(3e-24, rel=0.4)
    assert weighted.h != plain.h
    assert weighted.weights == "edof"
