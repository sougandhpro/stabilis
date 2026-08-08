"""Noise identification and the deviation bias.

Both are validated against synthesized noise of *known* power law, which
is the only honest reference available: every published noise-ID table is
Stable32's own classification of its own test data.
"""

import math

import numpy as np
import pytest

from stabilis.registry import get
from stabilis.stats import mdev, oadev
from stabilis.stats.bias import deviation_bias_factor, unbias_deviation
from stabilis.stats.noiseid import (
    ALPHA_NAMES,
    identify,
    identify_curve,
    lag1_alpha,
    mdev_slope_alpha,
)
from stabilis.synth import power_law_freq
from stabilis.types import FreqSeries, PhaseSeries

ALPHAS = (2, 1, 0, -1, -2)


def _freq(alpha, n=1024, seed=0):
    return power_law_freq(n, alpha, rate_hz=1.0, rng=np.random.default_rng(seed))


# --- lag-1 ---------------------------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
def test_lag1_recovers_the_power_law(alpha):
    """Median over 40 realizations lands within a tenth of the true alpha.

    Measured spread of the individual estimates at N=4096: sd 0.03 for the
    FM noises, 0.09 at white PM.
    """
    got = [lag1_alpha(_freq(alpha, 4096, s).y)[0] for s in range(40)]
    assert float(np.median(got)) == pytest.approx(alpha, abs=0.1)


def test_lag1_counts_the_differences_it_needed():
    """Steeper noise needs differencing before it decorrelates."""
    _, d_white = lag1_alpha(_freq(0, 2048, 1).y)
    _, d_rw = lag1_alpha(_freq(-2, 2048, 1).y)
    assert d_white == 0
    assert d_rw >= d_white


def test_lag1_needs_enough_points():
    with pytest.raises(ValueError, match="at least 3"):
        lag1_alpha(np.array([1.0, 2.0]))


# --- modified-Allan slope ------------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
def test_mdev_slope_recovers_the_power_law(alpha):
    got = []
    for s in range(40):
        fs = _freq(alpha, 4096, 100 + s)
        m = mdev(fs, taus="octave")
        got.append(mdev_slope_alpha(m.tau_s, m.dev))
    assert float(np.median(got)) == pytest.approx(alpha, abs=0.15)


def test_mdev_separates_white_pm_from_flicker_pm():
    """The reason this method exists beside the lag-1 one.

    ADEV cannot tell alpha=2 from alpha=1 -- both give a tau^-1 slope,
    because AVAR's effective bandwidth is set by the hardware cutoff rather
    than by tau. MDEV averages over m phase positions so its bandwidth
    scales as 1/(m*tau0), and the degeneracy breaks. Asserted by showing
    the MDEV slopes differ while the ADEV slopes do not.
    """
    def slope(result):
        return float(np.polyfit(np.log(result.tau_s), np.log(result.dev), 1)[0])

    wpm, fpm = _freq(2, 8192, 5), _freq(1, 8192, 6)
    mod = abs(slope(mdev(wpm, taus="octave")) - slope(mdev(fpm, taus="octave")))
    plain = abs(slope(oadev(wpm, taus="octave")) - slope(oadev(fpm, taus="octave")))
    assert mod > 0.3, f"MDEV slopes should differ by ~0.5, got {mod:.2f}"
    assert plain < 0.2, f"ADEV slopes should barely differ, got {plain:.2f}"


def test_mdev_slope_needs_two_positive_points():
    with pytest.raises(ValueError, match="at least two"):
        mdev_slope_alpha(np.array([1.0]), np.array([1.0]))


# --- the combined identifier --------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
def test_identify_hits_the_right_noise_type(alpha):
    """Measured hit rate: 95-100% at N=256, 100% at N=1024."""
    hits = 0
    for s in range(40):
        fs = _freq(alpha, 512, 200 + s)
        hits += identify(fs, 1, mdev(fs, taus="octave")).alpha == alpha
    assert hits >= 36, f"alpha={alpha}: {hits}/40"


def test_identify_reports_both_estimates_and_a_confidence():
    fs = _freq(0, 2048, 3)
    nid = identify(fs, 1, mdev(fs, taus="octave"))
    assert nid.alpha == 0 and nid.name == "white FM"
    assert nid.confidence == "high"
    assert abs(nid.alpha_lag1 - nid.alpha_slope) <= 0.5


def test_identify_without_a_cross_check_says_so():
    """One method is not evidence, and the answer has to admit it."""
    nid = identify(_freq(0, 2048, 4), 1)
    assert nid.confidence == "single-method"
    assert math.isnan(nid.alpha_slope)


def test_identify_flags_disagreement_rather_than_hiding_it():
    """Feed the two methods inconsistent evidence and the flag must drop."""
    fs = _freq(0, 2048, 8)
    wrong = mdev(_freq(-2, 2048, 9), taus="octave")
    assert identify(fs, 1, wrong).confidence == "low"


def test_identify_accepts_phase_input():
    """Phase is converted to frequency first; forgetting to is a 2-unit error."""
    fs = _freq(-1, 4096, 11)
    x = PhaseSeries(np.concatenate(([0.0], np.cumsum(fs.y))), rate_hz=1.0)
    assert identify(x, 1).alpha == identify(fs, 1).alpha


def test_identify_tracks_noise_that_changes_with_tau():
    """The commonest way to get a stability analysis wrong is one alpha.

    A white-FM oscillator with a random-walk term is white FM at one second
    and random-walk FM at a thousand. Reporting a single number for the
    record hides exactly the thing the customer is buying.
    """
    rng = np.random.default_rng(3)
    y = (power_law_freq(8192, 0, rate_hz=1.0, rng=rng).y
         + 0.05 * power_law_freq(8192, -2, rate_hz=1.0, rng=rng).y)
    fs = FreqSeries(y, rate_hz=1.0)
    alphas = [n.alpha for n in identify_curve(fs, [1, 4, 64, 256])]
    assert alphas[0] == 0
    assert alphas[-1] == -2
    assert alphas == sorted(alphas, reverse=True)  # monotone, no excursions


def test_identify_is_registered_as_a_plugin_point():
    assert get("noiseid", "lag1") is identify


def test_alpha_names_cover_the_reported_range():
    for a in range(-4, 3):
        assert a in ALPHA_NAMES


def test_identify_refuses_to_average_beyond_the_data():
    with pytest.raises(ValueError, match="too few points"):
        identify(_freq(0, 20, 1), m=10)


# --- deviation bias ------------------------------------------------------


def test_bias_factor_has_the_right_closed_forms_at_small_dof():
    """The two values that can be written down exactly.

    c4(2) = Gamma(3/2) = sqrt(pi)/2 = 0.886227 -- the last ADEV point is
    11.4% low on average. c4(1) = sqrt(2/pi).
    """
    assert deviation_bias_factor(2.0) == pytest.approx(math.sqrt(math.pi) / 2, rel=1e-15)
    assert deviation_bias_factor(1.0) == pytest.approx(math.sqrt(2.0 / math.pi), rel=1e-15)


def test_bias_factor_matches_an_independent_recurrence():
    """Checked against elementary arithmetic, with no special functions.

    Gamma(z+1) = z*Gamma(z) gives the exact two-step recurrence

        c4(eta+2) = c4(eta) * (eta+1) / sqrt(eta*(eta+2))

    so c4 at any even eta can be reached from c4(2) = sqrt(pi)/2 by
    multiplication and square roots alone. That shares no code with the
    lgamma path it checks, which is the point -- a plausible-looking but
    wrong constant transcribed from a table would survive any test that
    only compared the implementation with itself. This one caught exactly
    that: the c4 tables in quality-control texts are indexed by sample
    size n, not by degrees of freedom eta = n-1.
    """
    c4 = math.sqrt(math.pi) / 2.0
    eta = 2.0
    while eta < 200.0:
        assert deviation_bias_factor(eta) == pytest.approx(c4, rel=1e-13), f"eta={eta}"
        c4 *= (eta + 1.0) / math.sqrt(eta * (eta + 2.0))
        eta += 2.0


def test_bias_factor_approaches_one_and_never_exceeds_it():
    eta = np.array([1.0, 2.0, 10.0, 1e3, 1e6])
    c4 = deviation_bias_factor(eta)
    assert np.all((c4 > 0.0) & (c4 <= 1.0))
    assert np.all(np.diff(c4) > 0.0)
    assert c4[-1] == pytest.approx(1.0, abs=1e-6)


def test_bias_factor_matches_its_asymptotic_form():
    """c4 ~ 1 - 1/(4*eta), which is how to sanity-check the lgamma path."""
    for eta in (50.0, 200.0, 1000.0):
        assert deviation_bias_factor(eta) == pytest.approx(1.0 - 0.25 / eta, rel=1e-4)


def test_bias_factor_survives_large_eta():
    """A ratio of gammas would overflow above eta ~ 340; logarithms do not."""
    assert np.isfinite(deviation_bias_factor(1e5))
    assert deviation_bias_factor(1e5) == pytest.approx(1.0, abs=1e-5)


def test_bias_factor_rejects_nonpositive_dof():
    with pytest.raises(ValueError, match="must be positive"):
        deviation_bias_factor(0.0)


def test_unbias_raises_the_deviation():
    assert unbias_deviation(1.0, 2.0) == pytest.approx(1.0 / 0.8862269255, rel=1e-9)
    assert unbias_deviation(1.0, 1e4) == pytest.approx(1.0, abs=1e-4)


def test_measured_bias_matches_the_formula():
    """E[sigma_hat]/sqrt(E[var]) over many realizations, against c4(eta).

    The direct check: the formula predicts a specific shortfall and the
    Monte Carlo either shows it or does not. Uses the measured eta rather
    than an estimated one, so this tests the bias formula alone.
    """
    rng = np.random.default_rng(17)
    runs = [oadev(power_law_freq(512, 0, rate_hz=1.0, rng=rng), taus="octave")
            for _ in range(600)]
    var = np.array([r.var for r in runs])
    dev = np.array([r.dev for r in runs])
    eta_true = 2.0 * var.mean(axis=0) ** 2 / var.var(axis=0)
    measured = dev.mean(axis=0) / np.sqrt(var.mean(axis=0))
    predicted = deviation_bias_factor(eta_true)
    assert np.allclose(measured, predicted, atol=0.02), dict(
        zip(runs[0].af.tolist(), (measured - predicted).tolist())
    )
