"""Total variance, its reflection, and its bias.

The reflection is the whole estimator, so it is checked against SP 1065's
own index formulae by hand on a five-point record where every value can be
written down. After that the properties that matter are structural: the
term count must *not* fall with tau, and the answer must not depend on
which end of the record you start from.
"""

import numpy as np
import pytest

from stabilis.stats import oadev, totdev
from stabilis.stats.bias import TOTVAR_BIAS_A, totvar_bias
from stabilis.stats.edof import totvar_edof
from stabilis.stats.total import totvar_extend
from stabilis.synth import power_law_phase
from stabilis.types import PhaseSeries


def test_the_reflection_matches_the_published_index_formulae():
    """SP 1065 section 5.2.11, by hand.

        x*_{1-j} = 2 x_1 - x_{1+j}      x*_{N+j} = 2 x_N - x_{N-j}

    for j = 1..N-2, giving a virtual sequence of length 3N-4 with the
    original data in the centre. Five points, so every entry below can be
    checked without a computer:

        x  = [0, 1, 3, 2, 5]
        left  (j=3,2,1) = [2*0-2, 2*0-3, 2*0-1] = [-2, -3, -1]
        right (j=1,2,3) = [2*5-2, 2*5-3, 2*5-1] = [ 8,  7,  9]
    """
    x = np.array([0.0, 1.0, 3.0, 2.0, 5.0])
    ext = totvar_extend(x)
    assert ext.size == 3 * x.size - 4
    assert ext.tolist() == [-2, -3, -1, 0, 1, 3, 2, 5, 8, 7, 9]
    # The original data sits in the centre, untouched.
    assert np.array_equal(ext[x.size - 2:2 * x.size - 2], x)


def test_the_reflection_continues_the_trend_rather_than_mirroring_it():
    """`2*x_1 - x` is a point reflection, not a fold.

    A plain mirror would put a corner at each endpoint and manufacture a
    large second difference exactly where the estimator is most sensitive.
    On a straight ramp the point reflection extends the line, so every
    second difference stays zero.
    """
    ramp = np.arange(9, dtype=np.float64) * 3.0
    ext = totvar_extend(ramp)
    assert np.allclose(np.diff(ext), 3.0), "a ramp extends to a longer ramp"


def test_a_record_too_short_to_reflect_is_refused():
    with pytest.raises(ValueError, match="at least 3"):
        totvar_extend(np.array([1.0, 2.0]))


def test_totdev_equals_oadev_at_the_first_averaging_factor():
    """At m=1 the reflected points never enter any term, so the two
    estimators are the same sum and must agree to the bit."""
    ps = power_law_phase(512, 0, rate_hz=1.0, rng=np.random.default_rng(3))
    a = oadev(ps, taus="octave", keep_coeffs=False)
    t = totdev(ps, taus="octave", keep_coeffs=False)
    assert t.var[0] == pytest.approx(a.var[0], rel=1e-15)


def test_the_term_count_does_not_fall_with_tau():
    """The property TOTVAR exists for.

    The Allan variance runs out of data at long tau -- here it drops from
    2047 terms to about a thousand -- and the estimate collapses. TOTVAR
    keeps N-2 at every tau, because the sum always runs over the original
    interior points however large m gets.
    """
    ps = power_law_phase(2048, 0, rate_hz=1.0, rng=np.random.default_rng(4))
    a = oadev(ps, taus="octave", keep_coeffs=False)
    t = totdev(ps, taus="octave", keep_coeffs=False)
    assert len(set(t.n_terms.tolist())) == 1, "constant at every tau"
    assert t.n_terms[0] == ps.n - 2
    # At the longest tau the Allan family is down to half as many terms
    # as TOTVAR, which is where its confidence interval opens up.
    assert a.n_terms[-1] < 0.6 * t.n_terms[-1]


def test_the_answer_does_not_depend_on_which_end_you_start_from():
    """Time reversal, which is where reflection code goes wrong.

    Both ends are extended by the same rule, so reversing the record must
    give exactly the same variances -- not nearly. A one-sided or
    off-by-one reflection breaks this and almost nothing else.
    """
    ps = power_law_phase(600, -1, rate_hz=1.0, rng=np.random.default_rng(2))
    forward = totdev(ps, taus="octave", keep_coeffs=False)
    backward = totdev(PhaseSeries(ps.x[::-1].copy(), rate_hz=1.0),
                      taus="octave", keep_coeffs=False)
    assert np.allclose(forward.var, backward.var, rtol=1e-14, atol=0.0)


def test_linear_drift_is_slightly_biased_low_and_that_is_the_reflection():
    """TOTVAR does *not* reproduce the exact drift identity, by construction.

    For a quadratic phase the Allan variance is exactly `D^2 tau^2 / 2`.
    The endpoint reflection `2*x_1 - x_{1+j}` continues the *slope* but
    inverts the *curvature*, so the reflected data is not the continuation
    of the parabola and the second differences near the ends are wrong.

    The effect is small and grows with tau -- about 1.4% at tau = T/32 --
    and it is asserted in both directions so that it stays a known property
    rather than becoming a silent error.
    """
    n, drift = 1024, 1e-12
    x = drift * np.arange(n, dtype=np.float64) ** 2 / 2.0
    t = totdev(PhaseSeries(x, rate_hz=1.0), taus="octave", keep_coeffs=False)
    exact = drift * t.tau_s / np.sqrt(2.0)
    ratio = t.dev / exact
    assert ratio[0] == pytest.approx(1.0, rel=1e-9), "exact at tau0"
    assert np.all(ratio <= 1.0 + 1e-9), "biased low, never high"
    assert 0.95 < ratio[5] < 1.0, "and only slightly, at tau = 32 tau0"


# ---------------------------------------------------------------------------
# The bias correction
# ---------------------------------------------------------------------------


def test_white_and_flicker_pm_and_white_fm_need_no_correction():
    """SP 1065: TOTVAR "has the same expected value" as AVAR for these."""
    assert TOTVAR_BIAS_A[2] == TOTVAR_BIAS_A[1] == TOTVAR_BIAS_A[0] == 0.0
    assert np.all(totvar_bias(0, np.array([1.0, 100.0]), 1000.0) == 1.0)


@pytest.mark.parametrize("alpha,a", [(-1, 0.481), (-2, 0.750)])
def test_the_published_coefficients_predict_the_measured_bias(alpha, a):
    """The correction, measured against synthetic noise of known type.

    This is the check the coefficients actually need: generate the noise
    SP 1065's `a` was published for, measure TOTVAR against AVAR over an
    ensemble, and see whether `1 - a*tau/T` describes the ratio. It does,
    to a few parts in a thousand out to tau = T/16.
    """
    assert TOTVAR_BIAS_A[alpha] == a

    n, trials = 4096, 120
    rng = np.random.default_rng(7)
    acc_a = acc_t = None
    for _ in range(trials):
        ps = power_law_phase(n, alpha, rate_hz=1.0, rng=rng, h_alpha=1e-24)
        av = oadev(ps, taus="octave", keep_coeffs=False)
        tv = totdev(ps, taus="octave", keep_coeffs=False)
        acc_a = av.var if acc_a is None else acc_a + av.var
        acc_t = tv.var if acc_t is None else acc_t + tv.var

    keep = tv.tau_s <= n / 16
    measured = (acc_t / acc_a)[keep]
    predicted = 1.0 / totvar_bias(alpha, tv.tau_s[keep], float(n))
    assert np.allclose(measured, predicted, atol=0.02)

    # And applying the correction removes the bias it describes.
    corrected = acc_t[keep] * totvar_bias(alpha, tv.tau_s[keep], float(n))
    assert np.allclose(corrected / acc_a[keep], 1.0, atol=0.02)


def test_a_correction_larger_than_the_quantity_is_refused():
    """Past tau ~ T/a the "correction" exceeds the number being corrected."""
    with pytest.raises(ValueError, match="exceeds a factor of two"):
        totvar_bias(-2, np.array([900.0]), 1000.0)
    with pytest.raises(ValueError, match="no published coefficient"):
        totvar_bias(-3, np.array([1.0]), 1000.0)


# ---------------------------------------------------------------------------
# TOTVAR's own degrees of freedom
# ---------------------------------------------------------------------------


def test_the_published_edof_formula():
    """SP 1065: `b(T/tau) - c`, with b, c per noise type."""
    assert totvar_edof(0, 10.0, 1000.0) == pytest.approx(1.500 * 100)
    assert totvar_edof(-1, 10.0, 1000.0) == pytest.approx(1.168 * 100 - 0.222)
    assert totvar_edof(-2, 10.0, 1000.0) == pytest.approx(0.927 * 100 - 0.358)
    # Never below one: a chi-square with less than one degree of freedom is
    # not a distribution anyone can invert.
    assert totvar_edof(-2, 1000.0, 1000.0) >= 1.0


def test_the_pm_noises_are_defined_against_the_allan_edof():
    """SP 1065 gives those as "the overlapping ADEV edf increased by 2",
    so they cannot be computed here without it, and are not guessed."""
    assert totvar_edof(2, 1.0, 100.0, overlapping_edof=40.0) == 42.0
    with pytest.raises(ValueError, match="pass overlapping_edof"):
        totvar_edof(1, 1.0, 100.0)


@pytest.mark.slow
def test_eta1_beats_the_published_formula_on_totvar():
    """The measurement that reversed a claim made here.

    It looked obvious that a coefficient-based EDOF must overstate TOTVAR's
    confidence, since the coefficient array stays `N-2` long at every tau
    by reusing reflected data. That was written into the theory page and
    into a warning before it was checked, and it is **wrong**.

    `eta1` does not count terms -- it reads the degrees of freedom off the
    coefficients' own autocovariance, which is exactly where the
    reflection-induced correlation lives. Against the definitional truth
    `2*sigma^4/Var(sigma_hat^2)`, it tracks within a few percent and beats
    the published `b(T/tau)-c` at short tau, where that formula returns
    more degrees of freedom than the record has terms.
    """
    from stabilis.stats.edof import eta1

    n, trials = 4096, 300
    rng = np.random.default_rng(11)
    runs = []
    for _ in range(trials):
        ps = power_law_phase(n, 0, rate_hz=1.0, rng=rng)
        runs.append(totdev(ps, taus="octave", keep_coeffs=False).var)
    v = np.array(runs)
    truth = 2.0 * v.mean(axis=0) ** 2 / v.var(axis=0)

    ps = power_law_phase(n, 0, rate_hz=1.0, rng=np.random.default_rng(1))
    t = totdev(ps, taus="octave")

    for i in (0, 4, 8):
        tau, m = float(t.tau_s[i]), int(t.af[i])
        measured = eta1(t.coeffs[i], m)
        published = totvar_edof(0, tau, float(n))
        assert measured == pytest.approx(truth[i], rel=0.25), (
            f"eta1 at tau={tau:g} is {measured:.0f} against a measured "
            f"{truth[i]:.0f}")
        if i == 0:
            # tau = tau0: the published formula gives 1.5*N, which is more
            # degrees of freedom than there are terms.
            assert published > t.n_terms[0]
            assert abs(measured / truth[i] - 1) < abs(published / truth[i] - 1)


def test_the_edof_is_far_below_what_the_term_count_suggests():
    """Why the published formula is needed rather than counting terms.

    TOTVAR reports N-2 terms at every tau because it reuses reflected
    data. A data-driven estimator handed those coefficients would report a
    confidence the record cannot support; the published formula counts what
    is really there, and at long tau that is a small number.
    """
    n = 4096
    ps = power_law_phase(n, 0, rate_hz=1.0, rng=np.random.default_rng(5))
    t = totdev(ps, taus="octave", keep_coeffs=False)
    tau = float(t.tau_s[-1])
    assert totvar_edof(0, tau, float(n)) < t.n_terms[-1] / 100
