"""The fittable models, the objective, and the optimizer.

Everything here is checked against something that is true independently of
the code: closed-form asymptotics for the models, algebraic identities for
the objective, and exact parameter recovery from a curve the model itself
generated for the optimizer. The statistical claims -- coverage, the
sandwich, the bootstrap -- live in tests/test_fit.py, because they need
data rather than algebra.
"""

import math

import numpy as np
import pytest

from stabilis._specfun import digamma, norm_cdf, trigamma
from stabilis.fitting.lm import (
    dtheta_dz,
    levenberg_marquardt,
    numeric_jacobian,
    to_bounded,
    to_free,
)
from stabilis.fitting.models import (
    BUMP,
    DRIFT,
    GAUSS_MARKOV,
    POWERLAW,
    QUANTIZATION,
    Model,
    available_models,
    model,
    sum_of,
)
from stabilis.fitting.objective import (
    GOF_BAND,
    OBJECTIVES,
    deviance,
    goodness_of_fit,
    residuals,
)
from stabilis.powerlaw_coeffs import drift_coeff

EULER_GAMMA = 0.57721566490153286061


# ---------------------------------------------------------------------------
# the special functions the log-space objective needs
# ---------------------------------------------------------------------------


def test_digamma_at_the_values_with_closed_forms():
    """psi(1) = -gamma, psi(1/2) = -gamma - 2 ln 2, psi(n+1) = -gamma + H_n."""
    assert digamma(1.0) == pytest.approx(-EULER_GAMMA, abs=1e-13)
    assert digamma(0.5) == pytest.approx(-EULER_GAMMA - 2 * math.log(2), abs=1e-13)
    assert digamma(4.0) == pytest.approx(
        -EULER_GAMMA + 1 + 0.5 + 1 / 3, abs=1e-13)


def test_trigamma_at_the_values_with_closed_forms():
    """psi'(1) = pi^2/6, psi'(1/2) = pi^2/2, psi'(2) = pi^2/6 - 1."""
    assert trigamma(1.0) == pytest.approx(math.pi**2 / 6, abs=1e-13)
    assert trigamma(0.5) == pytest.approx(math.pi**2 / 2, abs=1e-13)
    assert trigamma(2.0) == pytest.approx(math.pi**2 / 6 - 1.0, abs=1e-13)


def test_the_recurrences_hold_across_the_series_boundary():
    """psi(x+1) - psi(x) = 1/x exactly, and the same for the trigamma.

    The recurrence is what carries small arguments up to where the
    asymptotic series is valid, so an error at the boundary shows here and
    nowhere else.
    """
    x = np.linspace(0.05, 40.0, 500)
    assert np.allclose(digamma(x + 1) - digamma(x), 1.0 / x, atol=1e-13)
    assert np.allclose(trigamma(x + 1) - trigamma(x), -1.0 / x**2, atol=1e-12)


def test_digamma_duplication_formula():
    """psi(2x) = (psi(x) + psi(x+1/2))/2 + ln 2, an independent identity."""
    x = np.linspace(0.05, 30.0, 300)
    want = 0.5 * (digamma(x) + digamma(x + 0.5)) + math.log(2)
    assert np.allclose(digamma(2 * x), want, atol=1e-13)


def test_trigamma_is_the_derivative_of_digamma():
    x = np.linspace(0.2, 20.0, 200)
    h = 1e-5
    assert np.allclose((digamma(x + h) - digamma(x - h)) / (2 * h),
                       trigamma(x), rtol=1e-7)


def test_norm_cdf_keeps_the_left_tail():
    """(1 + erf)/2 would return 0 here; the copula draw needs it not to."""
    assert norm_cdf(0.0) == 0.5
    assert norm_cdf(-10.0) == pytest.approx(7.61985302416e-24, rel=1e-10)
    assert norm_cdf(-38.0) > 0.0


def test_polygamma_refuses_a_non_positive_argument():
    with pytest.raises(ValueError, match="positive"):
        digamma(0.0)
    with pytest.raises(ValueError, match="positive"):
        trigamma(-1.0)


# ---------------------------------------------------------------------------
# the models, against their own closed forms
# ---------------------------------------------------------------------------


def test_drift_model_is_the_shipped_drift_coefficient_exactly():
    """Two independent expressions of avar(tau) = D^2 tau^2 / 2."""
    tau = np.logspace(-2, 4, 50)
    got = DRIFT.variance(tau, {"D": 3.0})
    assert np.array_equal(got, drift_coeff(tau) * 9.0)


def test_gauss_markov_long_tau_limit_is_white_fm():
    """sigma2 -> 2 sigma_y^2 T / tau, i.e. h0 = 4 sigma_y^2 T.

    The next term is -3/(2 rho) of the leading one, so at rho = 1e4 the
    ratio must sit 1.5e-4 below one -- which is a sharper test than
    asserting "close to".
    """
    t, sy = 1.0, 2.0
    tau = np.array([1e4])
    got = GAUSS_MARKOV.variance(tau, {"sigma_y": sy, "T": t})[0]
    leading = 2.0 * sy**2 * t / tau[0]
    assert got / leading == pytest.approx(1.0 - 1.5e-4, rel=1e-6)


def test_gauss_markov_short_tau_limit_rises_linearly():
    """sigma2 -> (2/3) sigma_y^2 tau / T, with a -3/4 rho correction."""
    t, sy = 1.0, 2.0
    tau = np.array([1e-4])
    got = GAUSS_MARKOV.variance(tau, {"sigma_y": sy, "T": t})[0]
    leading = (2.0 / 3.0) * sy**2 * tau[0] / t
    assert got / leading == pytest.approx(1.0 - 0.75e-4, rel=1e-6)


def test_gauss_markov_series_branch_joins_the_direct_one():
    """The two branches must agree where they meet, or the fit sees a step."""
    t, sy = 3.0, 1e-11
    edge = 1e-2 * t
    lo = GAUSS_MARKOV.variance(np.array([edge * (1 - 1e-9)]), {"sigma_y": sy, "T": t})
    hi = GAUSS_MARKOV.variance(np.array([edge * (1 + 1e-9)]), {"sigma_y": sy, "T": t})
    assert lo[0] == pytest.approx(hi[0], rel=1e-12)


def test_gauss_markov_is_scale_free_in_tau_over_T():
    """Only the ratio tau/T can matter, so doubling both changes nothing."""
    tau = np.logspace(-3, 3, 40)
    a = GAUSS_MARKOV.variance(tau, {"sigma_y": 1.0, "T": 1.0})
    b = GAUSS_MARKOV.variance(2 * tau, {"sigma_y": 1.0, "T": 2.0})
    assert np.allclose(a, b, rtol=1e-14)


def test_quantization_is_the_white_pm_law():
    tau = np.logspace(-1, 3, 20)
    got = QUANTIZATION.variance(tau, {"sigma_x": 5e-11})
    assert np.allclose(np.sqrt(got) * tau, math.sqrt(3) * 5e-11, rtol=1e-14)


def test_bump_peaks_where_it_says_and_is_symmetric_in_log_tau():
    p = {"A": 4.0, "tau_c": 10.0, "w": 0.5}
    assert BUMP.variance(np.array([10.0]), p)[0] == pytest.approx(4.0)
    left = BUMP.variance(np.array([1.0]), p)[0]
    right = BUMP.variance(np.array([100.0]), p)[0]
    assert left == pytest.approx(right, rel=1e-14)


@pytest.mark.parametrize("m", [POWERLAW, DRIFT, GAUSS_MARKOV, QUANTIZATION, BUMP])
def test_every_model_refuses_a_non_positive_tau(m):
    p = m.guess(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="tau must be positive"):
        m.variance(np.array([0.0, 1.0]), p)


@pytest.mark.parametrize("m", [POWERLAW, DRIFT, GAUSS_MARKOV, QUANTIZATION, BUMP])
def test_every_guess_names_exactly_the_models_parameters(m):
    tau = np.logspace(0, 3, 12)
    var = 1e-24 / tau
    assert set(m.guess(tau, var)) == set(m.params)
    assert set(m.positive) <= set(m.params)


def test_a_model_cannot_declare_a_bound_on_a_parameter_it_lacks():
    with pytest.raises(ValueError, match="not parameters"):
        Model("bad", lambda t, p: t, ("A",), lambda t, v: {"A": 1.0},
              "doc", positive=("B",))


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


def test_sum_of_adds_the_variances_and_prefixes_the_parameters():
    tau = np.logspace(0, 3, 15)
    both = sum_of(QUANTIZATION, GAUSS_MARKOV)
    assert both.params == ("quantization_sigma_x", "gauss_markov_sigma_y",
                           "gauss_markov_T")
    p = {"quantization_sigma_x": 2e-11, "gauss_markov_sigma_y": 3e-12,
         "gauss_markov_T": 220.0}
    want = (QUANTIZATION.variance(tau, {"sigma_x": p["quantization_sigma_x"]})
            + GAUSS_MARKOV.variance(tau, {"sigma_y": p["gauss_markov_sigma_y"],
                                          "T": p["gauss_markov_T"]}))
    assert np.array_equal(both.variance(tau, p), want)


def test_sum_of_lets_two_components_share_a_parameter_name():
    """POWERLAW and BUMP both have an `A`; the prefix is what keeps them apart."""
    both = sum_of(POWERLAW, BUMP)
    assert "powerlaw_A" in both.params and "bump_A" in both.params
    assert both.positive == ("powerlaw_A", "bump_A", "bump_tau_c", "bump_w")


def test_sum_of_refuses_a_repeated_model():
    with pytest.raises(ValueError, match="distinct"):
        sum_of(POWERLAW, POWERLAW)


def test_sum_of_refuses_a_single_model():
    with pytest.raises(ValueError, match="at least two"):
        sum_of(POWERLAW)


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------


def test_built_in_models_are_reachable_through_the_registry():
    assert set(available_models()) == {"powerlaw", "drift", "gauss_markov",
                                       "quantization", "bump"}
    assert model("gauss_markov") is GAUSS_MARKOV


def test_an_unknown_model_lists_the_known_ones():
    with pytest.raises(KeyError, match="available"):
        model("phlogiston")


# ---------------------------------------------------------------------------
# the objective
# ---------------------------------------------------------------------------


def test_the_deviance_vanishes_only_at_a_perfect_fit():
    x = np.array([1.0, 2.0, 3.0])
    eta = np.array([5.0, 10.0, 2.0])
    assert deviance(x, x, eta) == 0.0
    assert deviance(x, 1.3 * x, eta) > 0.0
    assert deviance(x, 0.7 * x, eta) > 0.0


def test_the_deviance_is_the_sum_of_the_ml_residuals_squared():
    """The construction that lets a least-squares optimizer minimize it."""
    x = np.array([1.0, 2.0, 3.0, 7.0])
    eta = np.array([5.0, 10.0, 2.0, 40.0])
    s = x * np.array([1.234, 0.6, 3.0, 0.99])
    assert np.sum(residuals("ml", x, s, eta) ** 2) == pytest.approx(
        deviance(x, s, eta), rel=1e-14)


def test_the_ml_residual_is_smooth_through_zero():
    """sign(1-u)*sqrt(...) is continuous at u = 1 rather than kinked.

    The claim is `r -> -sqrt(eta/2)*(u-1)` as the fit improves, with the
    departure falling in proportion to `u - 1`. Asserted as a *trend*: a
    fixed tolerance would either pass a discontinuity at the smallest step
    or fail the honest higher-order term at the largest.
    """
    eta = 8.0
    for sign in (1.0, -1.0):
        ratios = []
        for e in (1e-3, 1e-4, 1e-5, 1e-6):
            s = np.array([1.0 + sign * e])
            r = residuals("ml", np.ones(1), s, np.full(1, eta))[0]
            leading = -np.sqrt(0.5 * eta) * (1.0 / s[0] - 1.0)
            ratios.append(abs(r / leading - 1.0))
        for coarse, fine in zip(ratios, ratios[1:]):
            assert fine < 0.2 * coarse
        assert ratios[-1] < 1e-6


def test_the_divergence_keeps_its_digits_where_it_nearly_vanishes():
    """Against an exact Decimal evaluation of `u - ln u - 1`.

    Written directly in float64 this loses about half its significant
    digits near `u = 1`, which is where an optimizer spends every iteration
    after the first few. The oracle is independent of the implementation:
    50-digit Decimal arithmetic, no series, no branches.
    """
    from decimal import Decimal, getcontext

    getcontext().prec = 50
    eta = 6.0
    for e in (1e-2, 1e-3, 1e-5, 1e-8, 1e-11):
        for sign in (1, -1):
            u = Decimal(1) + Decimal(sign) * Decimal(e)
            exact = u - u.ln() - Decimal(1)
            got = residuals("ml", np.array([float(u)]), np.ones(1),
                            np.full(1, eta))[0] ** 2 / eta
            assert got == pytest.approx(float(exact), rel=1e-12)


def test_ml_and_wlsq_agree_to_first_order_and_differ_beyond_it():
    x = np.array([1.0, 1.0, 1.0])
    eta = np.full(3, 12.0)
    near = np.full(3, 1.0 + 1e-6)
    assert np.sum(residuals("ml", x, near, eta) ** 2) == pytest.approx(
        np.sum(residuals("wlsq", x, near, eta) ** 2), rel=1e-5)
    far = np.full(3, 3.0)
    assert np.sum(residuals("ml", x, far, eta) ** 2) != pytest.approx(
        np.sum(residuals("wlsq", x, far, eta) ** 2), rel=1e-2)


def test_wlsq_weights_by_the_model_not_the_measurement():
    """Two measurements either side of one model value must weigh the same.

    Weighting by `x**2` would let the low one claim a smaller uncertainty
    and drag the fit onto itself -- the Pearson-versus-Neyman trap.
    """
    eta = np.array([10.0])
    s = np.array([1.0])
    hi = residuals("wlsq", np.array([1.5]), s, eta)[0]
    lo = residuals("wlsq", np.array([0.5]), s, eta)[0]
    assert hi == pytest.approx(-lo, rel=1e-14)


def test_logspace_removes_the_log_bias_rather_than_ignoring_it():
    """At eta = 2 the correction is -0.58, not zero.

    A measurement sitting exactly at its own expectation in the log domain
    must give a zero residual; one sitting at the model value must not.
    """
    eta = np.array([2.0])
    s = np.array([1.0])
    unbiased = np.exp(digamma(1.0) - np.log(1.0)) * s
    assert residuals("logspace", unbiased, s, eta)[0] == pytest.approx(0.0, abs=1e-12)
    assert abs(residuals("logspace", s, s, eta)[0]) > 0.4


def test_logspace_variance_is_trigamma_not_two_over_eta():
    eta = np.array([2.0])
    s = np.array([1.0])
    x = np.array([math.e]) * np.exp(digamma(1.0))
    got = residuals("logspace", x, s, eta)[0]
    assert got == pytest.approx(1.0 / math.sqrt(trigamma(1.0)), rel=1e-12)


def test_an_unknown_objective_lists_the_known_ones():
    with pytest.raises(ValueError, match="must be one of"):
        residuals("ols", np.ones(2), np.ones(2), np.ones(2))
    assert OBJECTIVES == ("ml", "wlsq", "logspace")


def test_a_non_positive_model_variance_is_refused_not_squared():
    with pytest.raises(ValueError, match="non-positive variance"):
        residuals("ml", np.ones(2), np.array([1.0, -1.0]), np.ones(2))


def test_logspace_refuses_a_negative_measurement_and_says_why():
    with pytest.raises(ValueError, match="no log"):
        residuals("logspace", np.array([1.0, -1.0]), np.ones(2), np.ones(2))


# ---------------------------------------------------------------------------
# goodness of fit
# ---------------------------------------------------------------------------


def _projector(n, p, seed=0):
    """A rank-p orthogonal projector, standing in for a fit's hat matrix."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, p)))
    return q @ q.T


def test_independent_points_give_the_naive_degrees_of_freedom_exactly():
    """With P = I any rank-p projector must return to `scale = 1`, `n - p`."""
    gof = goodness_of_fit(10.0, 14, 2, corr=np.eye(14), hat=_projector(14, 2))
    assert gof.dof == pytest.approx(12.0, rel=1e-12)
    assert gof.scale == pytest.approx(1.0, rel=1e-12)
    assert goodness_of_fit(10.0, 14, 2).dof == 12.0


def test_correlation_widens_the_band_rather_than_moving_the_deviance():
    """Satterthwaite: correlated points make the deviance more variable."""
    n = 14
    corr = 0.5 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    gof = goodness_of_fit(12.0, n, 2, corr=corr, hat=_projector(n, 2))
    assert gof.deviance == 12.0
    assert gof.scale > 1.0
    assert gof.dof < 12.0
    assert "effective degrees of freedom" in gof.note


def test_the_correlation_and_the_projector_are_required_together():
    """Half the correction is the mistake it exists to prevent."""
    with pytest.raises(ValueError, match="go together"):
        goodness_of_fit(12.0, 14, 2, corr=np.eye(14))
    with pytest.raises(ValueError, match="go together"):
        goodness_of_fit(12.0, 14, 2, hat=_projector(14, 2))


def test_a_hopeless_fit_is_flagged_and_says_which_way():
    bad = goodness_of_fit(500.0, 14, 2)
    assert not bad.ok
    assert "does not describe" in bad.note
    assert bad.p_value < GOF_BAND[0]


def test_a_suspiciously_good_fit_is_flagged_too():
    """A deviance far below expectation is a finding, not a success."""
    good = goodness_of_fit(0.001, 14, 2)
    assert not good.ok
    assert "more freedom" in good.note


def test_no_degrees_of_freedom_is_reported_rather_than_divided_by():
    gof = goodness_of_fit(0.0, 2, 2)
    assert not gof.ok
    assert gof.dof == 0.0
    assert "no degrees of freedom" in gof.note


def test_a_typical_fit_passes():
    gof = goodness_of_fit(12.0, 14, 2)
    assert gof.ok
    assert gof.reduced == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# the transformations that carry bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lo,hi,values",
    [
        (None, None, [-2.0, 0.0, 3.0]),
        (0.0, None, [1e-24, 0.5, 3.0]),
        (None, 5.0, [0.5, 2.0, 4.9]),
        (1.0, 4.0, [1.1, 2.0, 3.9]),
    ],
)
def test_the_bound_transform_round_trips_and_its_derivative_is_right(lo, hi, values):
    for theta in values:
        z = to_free(theta, lo, hi)
        assert to_bounded(z, lo, hi) == pytest.approx(theta, rel=1e-12)
        h = 1e-6
        numeric = (to_bounded(z + h, lo, hi) - to_bounded(z - h, lo, hi)) / (2 * h)
        assert numeric == pytest.approx(dtheta_dz(z, lo, hi), rel=1e-5)


def test_a_two_sided_transform_saturates_instead_of_overflowing():
    """exp(z)/(1+exp(z)) written naively gives inf/inf and a NaN parameter."""
    assert to_bounded(1e5, 1.0, 4.0) == 4.0
    assert to_bounded(-1e5, 1.0, 4.0) == 1.0
    assert np.isfinite(dtheta_dz(1e5, 1.0, 4.0))


def test_starting_on_a_bound_is_an_error_rather_than_an_infinity():
    with pytest.raises(ValueError, match="above its lower bound"):
        to_free(0.0, 0.0, None)
    with pytest.raises(ValueError, match="below its upper bound"):
        to_free(5.0, None, 5.0)
    with pytest.raises(ValueError, match="strictly inside"):
        to_free(4.0, 1.0, 4.0)


# ---------------------------------------------------------------------------
# the optimizer
# ---------------------------------------------------------------------------


def test_numeric_jacobian_matches_an_analytic_one():
    def f(z):
        return np.array([z[0] ** 2 + z[1], np.exp(z[0]) * z[1], 3.0 * z[1]])

    z = np.array([0.7, -1.3])
    analytic = np.array([
        [2 * z[0], 1.0],
        [np.exp(z[0]) * z[1], np.exp(z[0])],
        [0.0, 3.0],
    ])
    assert np.allclose(numeric_jacobian(f, z), analytic, rtol=1e-7, atol=1e-9)


def _fit_positive(m, tau, target, eta, max_iter=500):
    """Minimal driver: everything positive, ML objective, model's own guess."""
    names = m.params
    g = m.guess(tau, target)

    def resid(z):
        p = {k: to_bounded(z[i], 0.0, None) for i, k in enumerate(names)}
        return residuals("ml", target, m.variance(tau, p), eta)

    z0 = np.array([to_free(g[k], 0.0, None) for k in names])
    out = levenberg_marquardt(resid, z0, max_iter=max_iter)
    return {k: to_bounded(out.z[i], 0.0, None) for i, k in enumerate(names)}, out


def test_the_optimizer_recovers_a_curve_the_model_generated():
    """No noise: the answer is exact, so the fit must be too."""
    tau = np.logspace(0, 4, 25)
    truth = {"sigma_y": 3e-12, "T": 220.0}
    eta = np.linspace(500.0, 4.0, tau.size)
    got, out = _fit_positive(GAUSS_MARKOV, tau,
                             GAUSS_MARKOV.variance(tau, truth), eta)
    assert out.converged
    assert got["sigma_y"] == pytest.approx(truth["sigma_y"], rel=1e-8)
    assert got["T"] == pytest.approx(truth["T"], rel=1e-8)
    assert out.cost < 1e-16


def test_the_optimizer_recovers_a_three_parameter_composite():
    """And from a starting guess 200x wrong in the correlation time."""
    tau = np.logspace(0, 4, 25)
    both = sum_of(QUANTIZATION, GAUSS_MARKOV)
    truth = {"quantization_sigma_x": 2e-11, "gauss_markov_sigma_y": 3e-12,
             "gauss_markov_T": 220.0}
    eta = np.linspace(500.0, 4.0, tau.size)
    got, out = _fit_positive(both, tau, both.variance(tau, truth), eta)
    assert out.converged
    for name, value in truth.items():
        assert got[name] == pytest.approx(value, rel=1e-6)


def test_the_iteration_limit_reports_failure_rather_than_an_answer():
    tau = np.logspace(0, 4, 25)
    truth = {"sigma_y": 3e-12, "T": 220.0}
    _, out = _fit_positive(GAUSS_MARKOV, tau,
                           GAUSS_MARKOV.variance(tau, truth),
                           np.linspace(500.0, 4.0, tau.size), max_iter=1)
    assert not out.converged
    assert "without converging" in out.reason


def test_a_parameter_the_residual_ignores_freezes_instead_of_breaking():
    """A zero Jacobian column would make the damped system singular."""
    def resid(z):
        return np.array([z[0] - 2.0, z[0] + 1.0])

    out = levenberg_marquardt(resid, np.array([0.0, 5.0]))
    assert out.converged
    assert out.z[0] == pytest.approx(0.5, rel=1e-8)
    assert out.z[1] == pytest.approx(5.0, rel=1e-12)


def test_the_optimizer_is_invariant_to_rescaling_one_parameter():
    """Marquardt's diag(J.T J) damping, as opposed to lam * I.

    The same problem written with one parameter in different units must
    reach the same answer. With identity damping it does not: the damping
    is then a different fraction of each curvature and the fit crawls.
    """
    target = np.array([3.0, 5.0, 9.0])

    def make(scale):
        def resid(z):
            return np.array([z[0] + z[1] / scale - t
                             for t in (3.0, 5.0, 9.0)]) * np.array([1, 2, 3.0])
        return resid

    a = levenberg_marquardt(make(1.0), np.array([0.0, 0.0]))
    b = levenberg_marquardt(make(1e6), np.array([0.0, 0.0]))
    assert a.cost == pytest.approx(b.cost, rel=1e-8)
    assert target.size == 3


def test_differential_against_scipy_least_squares():
    """Same minimum as an implementation nobody here wrote.

    Skipped without scipy, which is optional by design and is never
    required for a correct answer -- that is the point of this file
    existing at all.
    """
    scipy_opt = pytest.importorskip("scipy.optimize")
    tau = np.logspace(0, 4, 25)
    truth = {"sigma_y": 3e-12, "T": 220.0}
    target = GAUSS_MARKOV.variance(tau, truth)
    eta = np.linspace(500.0, 4.0, tau.size)

    def resid(z):
        p = {"sigma_y": to_bounded(z[0], 0.0, None),
             "T": to_bounded(z[1], 0.0, None)}
        return residuals("ml", target, GAUSS_MARKOV.variance(tau, p), eta)

    g = GAUSS_MARKOV.guess(tau, target)
    z0 = np.array([to_free(g["sigma_y"], 0.0, None), to_free(g["T"], 0.0, None)])
    mine = levenberg_marquardt(resid, z0)
    theirs = scipy_opt.least_squares(resid, z0, method="lm")
    assert to_bounded(mine.z[0], 0.0, None) == pytest.approx(
        to_bounded(theirs.x[0], 0.0, None), rel=1e-6)
    assert to_bounded(mine.z[1], 0.0, None) == pytest.approx(
        to_bounded(theirs.x[1], 0.0, None), rel=1e-6)
