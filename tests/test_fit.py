"""The fit driver: uncertainties, goodness of fit, and the refusals.

The algebra of the objective and the optimizer is tested in
tests/test_fit_models.py. What is here needs data: the sandwich against the
naive covariance it must reduce to, the copula draw against the law it
claims to sample, the deviance against its own predicted mean over many
realizations, and every place the driver declines to answer.
"""

from dataclasses import replace

import numpy as np
import pytest

import stabilis
from stabilis.fitting import fit
from stabilis.fitting.models import GAUSS_MARKOV, POWERLAW, QUANTIZATION, sum_of
from stabilis.fitting.uncertainty import (
    copula_draw,
    naive_covariance,
    nearest_correlation,
    residual_hat,
    sandwich_covariance,
)
from stabilis.stats.classical import adev, oadev
from stabilis.stats.mtie import mtie
from stabilis.stats.taucov import tau_covariance
from stabilis.synth import power_law_freq
from stabilis.types import to_phase


def white_fm(n=8192, seed=5, h_alpha=2e-22, alpha=0):
    rng = np.random.default_rng(seed)
    return power_law_freq(n, alpha, rate_hz=1.0, rng=rng, h_alpha=h_alpha)


def curve(n=8192, seed=5, max_af=512, alpha=0, h_alpha=2e-22):
    return oadev(white_fm(n, seed, h_alpha, alpha), taus="octave", max_af=max_af)


def exact_curve(model, params, base=None):
    """A `StabilityResult` whose variances are the model's, exactly.

    The coefficient arrays stay real, so the correlation between taus is a
    measured one; only the variances are replaced. Fitting this must return
    the parameters that generated it.
    """
    base = base or curve()
    return replace(base, var=model.variance(base.tau_s, params))


# ---------------------------------------------------------------------------
# the fit itself
# ---------------------------------------------------------------------------


def test_a_curve_the_model_generated_comes_back_exactly():
    truth = {"sigma_y": 3e-12, "T": 200.0}
    f = fit(exact_curve(GAUSS_MARKOV, truth), GAUSS_MARKOV)
    assert f.converged
    for name, value in truth.items():
        assert f.params[name] == pytest.approx(value, rel=1e-7)
    assert f.gof.deviance < 1e-14


def test_a_model_can_be_named_or_passed():
    r = exact_curve(POWERLAW, {"A": 1e-22, "b": -1.0})
    by_name = fit(r, "powerlaw")
    by_record = fit(r, POWERLAW)
    assert by_name.params == by_record.params


def test_the_fitted_curve_and_the_measured_one_are_both_carried():
    f = fit(curve(), "powerlaw")
    assert f.tau_s.size == f.measured.size == f.fitted.size == f.edof.size
    assert np.allclose(f.residual_ratio, f.measured / f.fitted)


def test_a_white_fm_record_recovers_its_own_power_law():
    """h0/2 and a slope of -1, from data rather than from a formula."""
    f = fit(curve(h_alpha=2e-22), "powerlaw")
    assert f.ok
    assert f.params["b"] == pytest.approx(-1.0, abs=4 * f.stderr["b"])
    assert f.params["A"] == pytest.approx(1e-22, abs=4 * f.stderr["A"])


# ---------------------------------------------------------------------------
# the sandwich
# ---------------------------------------------------------------------------


def test_the_sandwich_reduces_to_the_naive_covariance_on_a_diagonal():
    """Not approximately. The independent case is a special case of it."""
    r = curve()
    tc = tau_covariance(r)
    theta = np.array([1e-22, -1.0])

    def variance_of(t):
        return POWERLAW.variance(r.tau_s, {"A": t[0], "b": t[1]})

    diagonal = np.diag(np.diag(tc.cov))
    a = sandwich_covariance("ml", r.var, variance_of, tc.edof, theta, diagonal)
    b = naive_covariance("ml", r.var, variance_of, tc.edof, theta, tc.cov)
    assert np.array_equal(a, b)
    # ... and the correlated one is genuinely different.
    full = sandwich_covariance("ml", r.var, variance_of, tc.edof, theta, tc.cov)
    assert not np.allclose(full, b, rtol=1e-3)


def test_the_correlation_widens_the_error_bars_it_should_have_widened():
    """The measurement this whole subsystem exists to make."""
    f = fit(curve(), "powerlaw")
    assert f.correlation == "isserlis"
    for name in f.free:
        assert f.understatement[name] > 1.1
        assert f.stderr[name] > f.stderr_naive[name]


def test_an_uncorrelatable_estimator_reports_equal_error_bars_and_says_so():
    """`adev` decimates, so the correlation cannot be measured for it."""
    r = adev(white_fm(), taus="octave", max_af=512)
    f = fit(r, "powerlaw")
    assert f.correlation == "independent"
    assert "decimates" in f.correlation_note
    for name in f.free:
        assert f.understatement[name] == pytest.approx(1.0, rel=1e-9)


def test_the_hat_matrix_is_a_projector_of_the_right_rank():
    r = curve()
    tc = tau_covariance(r)

    def variance_of(t):
        return POWERLAW.variance(r.tau_s, {"A": t[0], "b": t[1]})

    h = residual_hat("ml", r.var, variance_of, tc.edof, np.array([1e-22, -1.0]))
    assert np.allclose(h @ h, h, atol=1e-10)
    assert np.trace(h) == pytest.approx(2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# the bootstrap
# ---------------------------------------------------------------------------


def test_the_copula_draw_has_chi_square_margins_and_the_asked_correlation():
    """Both halves of the claim, measured over draws rather than argued."""
    rng = np.random.default_rng(3)
    s = np.array([4.0, 2.0, 1.0])
    eta = np.array([6.0, 6.0, 6.0])
    corr = np.array([[1.0, 0.8, 0.5], [0.8, 1.0, 0.8], [0.5, 0.8, 1.0]])
    chol = np.linalg.cholesky(corr)
    draws = np.array([copula_draw(s, eta, chol, rng) for _ in range(20000)])

    assert np.allclose(draws.mean(axis=0), s, rtol=0.02)
    # Var(x) = 2 s^2 / eta for a chi-square sampling law.
    assert np.allclose(draws.var(axis=0), 2.0 * s**2 / eta, rtol=0.06)
    got = np.corrcoef(draws.T)
    assert np.allclose(got, corr, atol=0.04)


def test_the_bootstrap_interval_and_the_sandwich_interval_agree_when_they_should():
    """A well-specified, nearly linear fit: the two routes must not disagree."""
    f = fit(curve(), "powerlaw", bootstrap=200, seed=1)
    assert f.n_bootstrap == 200
    assert f.n_bootstrap_failed == 0
    for name in f.free:
        s_lo, s_hi = f.interval[name]
        b_lo, b_hi = f.bootstrap[name]
        width_s, width_b = s_hi - s_lo, b_hi - b_lo
        assert width_b == pytest.approx(width_s, rel=0.35)


def test_the_bootstrap_is_reproducible_from_its_seed():
    a = fit(curve(), "powerlaw", bootstrap=40, seed=7)
    b = fit(curve(), "powerlaw", bootstrap=40, seed=7)
    assert a.bootstrap == b.bootstrap
    c = fit(curve(), "powerlaw", bootstrap=40, seed=8)
    assert c.bootstrap != a.bootstrap


def test_no_bootstrap_is_the_default_and_leaves_the_field_empty():
    f = fit(curve(), "powerlaw")
    assert f.bootstrap == {}
    assert f.n_bootstrap == 0
    assert f.interval  # the sandwich interval is always there


def test_a_non_positive_definite_correlation_is_repaired_and_reported():
    """A truncated lag sum is not a positive operation, so this happens."""
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert np.linalg.eigvalsh(bad).min() < 0.0
    fixed, worst = nearest_correlation(bad)
    assert worst < 0.0
    assert np.linalg.eigvalsh(fixed).min() >= -1e-12
    assert np.allclose(np.diag(fixed), 1.0)
    np.linalg.cholesky(fixed)  # must not raise


def test_an_already_good_correlation_is_left_alone():
    good = np.array([[1.0, 0.5], [0.5, 1.0]])
    fixed, worst = nearest_correlation(good)
    assert worst == 0.0
    assert np.array_equal(fixed, good)


# ---------------------------------------------------------------------------
# calibration of the deviance, which is the claim a report rests on
# ---------------------------------------------------------------------------


def test_the_deviance_matches_its_own_predicted_mean_over_realizations():
    """Fit the right model many times; the deviance must land where it says.

    This is the end-to-end check on the whole chain -- eta from the
    coefficients, the correlation from the pair FFT, the Satterthwaite
    correction and the objective. Getting any one of them wrong moves this
    number, and the projector correction moved it by 27%.
    """
    rng = np.random.default_rng(77)
    devs, predicted, passed = [], [], 0
    for _ in range(40):
        fs = power_law_freq(8192, 0, rate_hz=1.0, rng=rng, h_alpha=2e-22)
        f = fit(oadev(fs, taus="octave", max_af=512), "powerlaw")
        devs.append(f.gof.deviance)
        predicted.append(f.gof.scale * f.gof.dof)
        passed += f.gof.ok
    mean, sem = float(np.mean(devs)), float(np.std(devs) / np.sqrt(len(devs)))
    assert mean == pytest.approx(float(np.mean(predicted)), abs=4 * sem)
    assert passed == 40


# ---------------------------------------------------------------------------
# frozen parameters and bounds
# ---------------------------------------------------------------------------


def test_freezing_a_parameter_holds_it_and_frees_a_degree_of_freedom():
    r = exact_curve(GAUSS_MARKOV, {"sigma_y": 3e-12, "T": 200.0})
    free = fit(r, GAUSS_MARKOV)
    held = fit(r, GAUSS_MARKOV, freeze={"T": 200.0})
    assert held.params["T"] == 200.0
    assert held.free == ("sigma_y",)
    assert held.frozen == {"T": 200.0}
    assert "T" not in held.stderr
    assert held.gof.dof > free.gof.dof
    assert held.params["sigma_y"] == pytest.approx(free.params["sigma_y"], rel=1e-5)


def test_an_inactive_bound_changes_nothing():
    r = curve()
    plain = fit(r, "powerlaw")
    bounded = fit(r, "powerlaw", bounds={"b": (-5.0, 5.0)})
    assert bounded.params["b"] == pytest.approx(plain.params["b"], rel=1e-6)
    assert bounded.stderr["b"] == pytest.approx(plain.stderr["b"], rel=1e-4)


def test_a_supplied_guess_overrides_the_models_own():
    r = exact_curve(POWERLAW, {"A": 1e-22, "b": -1.0})
    f = fit(r, "powerlaw", guess={"A": 1e-20, "b": -2.0})
    assert f.converged
    assert f.params["b"] == pytest.approx(-1.0, rel=1e-6)


def test_a_positive_parameter_stays_positive_through_the_fit():
    """The transformation, not a clamp: no iterate can leave the domain."""
    f = fit(curve(), sum_of(QUANTIZATION, POWERLAW))
    assert f.params["quantization_sigma_x"] > 0.0
    assert f.params["powerlaw_A"] > 0.0


# ---------------------------------------------------------------------------
# the refusals
# ---------------------------------------------------------------------------


def test_a_misspecified_model_is_flagged_and_not_quotable():
    """White FM fitted with a tau^-2 quantization floor cannot describe it."""
    f = fit(curve(), "quantization")
    assert not f.gof.ok
    assert not f.ok
    assert "does not describe" in f.gof.note


def test_a_non_variance_is_refused_by_the_flag_not_by_name():
    r = mtie(to_phase(white_fm(2048)), taus="octave")
    with pytest.raises(ValueError, match="not a variance"):
        fit(r, "powerlaw")


def test_a_non_positive_measured_variance_is_refused():
    r = curve()
    spoiled = replace(r, var=np.where(r.af == r.af[-1], -1.0, r.var))
    with pytest.raises(ValueError, match="not positive"):
        fit(spoiled, "powerlaw")


def test_freezing_everything_is_refused_rather_than_fitted():
    with pytest.raises(ValueError, match="nothing to fit"):
        fit(curve(), "powerlaw", freeze={"A": 1e-22, "b": -1.0})


def test_too_few_taus_for_the_parameters_is_refused():
    r = curve().select_afs([1, 2])
    with pytest.raises(ValueError, match="no degrees of freedom"):
        fit(r, "powerlaw")


def test_an_unknown_parameter_name_is_caught_in_bounds_and_in_freeze():
    r = curve()
    with pytest.raises(ValueError, match="not a parameter"):
        fit(r, "powerlaw", bounds={"sigma_y": (0.0, None)})
    with pytest.raises(ValueError, match="cannot freeze"):
        fit(r, "powerlaw", freeze={"sigma_y": 1.0})


def test_reversed_bounds_are_caught_before_the_optimizer_sees_them():
    with pytest.raises(ValueError, match="not increasing"):
        fit(curve(), "powerlaw", bounds={"b": (2.0, -2.0)})


def test_an_iteration_limit_reports_failure_and_ok_is_false():
    f = fit(exact_curve(GAUSS_MARKOV, {"sigma_y": 3e-12, "T": 200.0}),
            GAUSS_MARKOV, guess={"sigma_y": 1e-9, "T": 1e-3}, max_iter=1)
    assert not f.converged
    assert not f.ok
    assert "without converging" in f.reason


# ---------------------------------------------------------------------------
# provenance and the public API
# ---------------------------------------------------------------------------


def test_the_fit_records_what_it_decided():
    f = fit(curve(), "powerlaw", bootstrap=20, seed=3)
    steps = {s.action: s for s in f.provenance.steps}
    assert "fit" in steps
    params = steps["fit"].params
    assert params["model"] == "powerlaw"
    assert params["objective"] == "ml"
    assert params["correlation"] == "isserlis"
    assert params["bootstrap"] == 20
    assert params["seed"] == 3
    assert "tau_covariance" in steps


def test_an_uncorrelated_fit_warns_that_its_error_bars_are_lower_bounds():
    f = fit(adev(white_fm(), taus="octave", max_af=512), "powerlaw")
    steps = {s.action: s for s in f.provenance.steps}
    assert "uncorrelated" in steps
    assert "lower bounds" in steps["uncorrelated"].outcome


def test_a_rejected_fit_warns_in_provenance_rather_than_only_in_a_flag():
    f = fit(curve(), "quantization")
    steps = {s.action: s for s in f.provenance.steps}
    assert "goodness_of_fit" in steps


def test_the_data_history_survives_into_the_fit():
    """The whole chain, from the source to the fitted parameter."""
    f = fit(curve(), "powerlaw")
    actions = [s.action for s in f.provenance.steps]
    assert "evaluate" in actions
    assert actions.index("evaluate") < actions.index("fit")


@pytest.mark.parametrize("objective", ["ml", "wlsq"])
def test_the_variance_objectives_return_the_truth_on_a_mean_curve(objective):
    """They differ in how they weight scatter, not in where the truth is."""
    r = exact_curve(POWERLAW, {"A": 1e-22, "b": -1.0})
    f = fit(r, "powerlaw", objective=objective)
    assert f.converged
    assert f.params["b"] == pytest.approx(-1.0, rel=1e-6)
    assert f.objective == objective


def test_logspace_is_unbiased_for_a_realization_not_for_a_mean_curve():
    """Which is the point of its bias term, and is worth stating out loud.

    `logspace` compares `ln x` against `ln s + psi(eta/2) - ln(eta/2)`,
    because that -- not `ln s` -- is what the logarithm of a chi-square
    estimate averages to. Hand it a curve sitting exactly at the model
    value and it will correctly conclude that every point came out high,
    and shift the fit. Hand it a curve carrying the log bias and it returns
    the truth. The offset is largest where eta is smallest, so it shows up
    as a tilt rather than a scale.
    """
    from stabilis._specfun import digamma

    truth = {"A": 1e-22, "b": -1.0}
    base = curve()
    tc = tau_covariance(base)
    mean_curve = exact_curve(POWERLAW, truth, base)

    naive = fit(mean_curve, "powerlaw", objective="logspace")
    assert naive.params["b"] != pytest.approx(-1.0, rel=1e-5)

    bias = np.exp(digamma(0.5 * tc.edof) - np.log(0.5 * tc.edof))
    realized = replace(mean_curve, var=mean_curve.var * bias)
    corrected = fit(realized, "powerlaw", objective="logspace")
    assert corrected.params["b"] == pytest.approx(-1.0, rel=1e-6)
    assert corrected.params["A"] == pytest.approx(1e-22, rel=1e-6)


def test_fit_and_FitResult_are_on_the_public_api():
    assert "fit" in stabilis.__all__
    assert "FitResult" in stabilis.__all__
    assert stabilis.fit is fit


def test_the_package_is_reachable_by_its_own_name_not_shadowed():
    """`stabilis.fit` is the function; `stabilis.fitting` is the package.

    Naming the package `fit` too would make `import stabilis.fit` and
    `from stabilis.fit import ...` mean different things.
    """
    import stabilis.fitting

    assert stabilis.fitting.fit is stabilis.fit
    assert callable(stabilis.fit)
