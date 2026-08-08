"""Confidence intervals: closed forms, then measured coverage.

The coverage test is the one that matters commercially. No competing tool
publishes measured containment of its own error bars, and an ISO-17025
assessor asking "how do you know the interval is right" has no other
answer available. It is also the test that validates the EDOF estimator and
the chi-square inversion jointly, end to end -- either one being wrong
shows up here and nowhere else.
"""

import math

import numpy as np
import pytest

from stabilis._specfun import chi2_ppf
from stabilis.stats import ONE_SIGMA, TWO_SIGMA, confidence, oadev, ohdev
from stabilis.stats.ci import chi2_interval
from stabilis.synth import power_law_freq
from stabilis.types import FreqSeries

ALPHAS = (2, 1, 0, -1, -2)


def _result(alpha=0, n=512, seed=0):
    rng = np.random.default_rng(seed)
    return oadev(power_law_freq(n, alpha, rate_hz=1.0, rng=rng), taus="octave")


# --- the interval itself -------------------------------------------------


def test_the_interval_brackets_the_estimate():
    """lower <= var <= upper. Catches the crossed-quantile mistake.

    The upper bound on the variance comes from the *lower* chi-square
    quantile. Swapping them gives an inverted interval that still looks
    plausible drawn on a log axis.
    """
    for eta in (1.0, 2.0, 5.0, 50.0, 5000.0):
        lo, hi = chi2_interval(4.0, eta, ONE_SIGMA)
        assert lo <= 4.0 <= hi


def test_interval_at_two_degrees_of_freedom_has_a_closed_form():
    """chi2_ppf(p, 2) = -2*ln(1-p), so the bounds are analytic at eta=2."""
    var, level = 3.0, ONE_SIGMA
    alpha = 1.0 - level
    lo, hi = chi2_interval(var, 2.0, level)
    assert lo == pytest.approx(2.0 * var / (-2.0 * math.log(0.5 * alpha)), rel=1e-13)
    assert hi == pytest.approx(2.0 * var / (-2.0 * math.log1p(-0.5 * alpha)), rel=1e-13)


def test_a_higher_confidence_level_gives_a_wider_interval():
    lo1, hi1 = chi2_interval(1.0, 20.0, ONE_SIGMA)
    lo2, hi2 = chi2_interval(1.0, 20.0, TWO_SIGMA)
    assert lo2 < lo1 <= 1.0 <= hi1 < hi2


def test_more_degrees_of_freedom_give_a_tighter_interval():
    widths = [chi2_interval(1.0, eta, ONE_SIGMA) for eta in (2.0, 10.0, 100.0, 1e4)]
    spans = [hi - lo for lo, hi in widths]
    assert spans == sorted(spans, reverse=True)
    assert spans[-1] < 0.03  # 1e4 dof: the interval has all but closed


def test_the_interval_is_strongly_asymmetric_at_low_dof():
    """At eta=2 the 68.3% interval on the deviation runs 0.737 to 2.406.

    Both bounds follow from the closed form: with alpha = 1 - 0.6827,
    chi2_ppf(1-alpha/2, 2) = 3.682 and chi2_ppf(alpha/2, 2) = 0.3454, so
    the variance bounds are 2/3.682 = 0.543 and 2/0.3454 = 5.790, and the
    deviation bounds are their square roots.

    The upper arm is 5.3 times the lower one. This is why the library never
    offers a `sigma +/- delta`: quoting one at the last few tau -- exactly
    where a customer's spec lives -- misrepresents the distribution by more
    than a factor of two.
    """
    lo, hi = chi2_interval(1.0, 2.0, ONE_SIGMA)
    assert lo == pytest.approx(0.5432, abs=1e-3)
    assert hi == pytest.approx(5.790, abs=1e-2)
    dev_lo, dev_hi = math.sqrt(lo), math.sqrt(hi)
    assert dev_lo == pytest.approx(0.737, abs=1e-3)
    assert dev_hi == pytest.approx(2.406, abs=1e-3)
    assert (dev_hi - 1.0) > 5.0 * (1.0 - dev_lo)


def test_interval_scales_with_the_variance():
    """The bounds are proportional to var, so a unit change cannot break them."""
    lo1, hi1 = chi2_interval(1.0, 30.0, ONE_SIGMA)
    lo2, hi2 = chi2_interval(1e-24, 30.0, ONE_SIGMA)
    assert lo2 == pytest.approx(lo1 * 1e-24, rel=1e-13)
    assert hi2 == pytest.approx(hi1 * 1e-24, rel=1e-13)


def test_bad_level_is_rejected():
    for level in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="level must be in"):
            chi2_interval(1.0, 10.0, level)


# --- the result-level wrapper -------------------------------------------


def test_confidence_covers_every_tau_and_records_how():
    r = _result()
    c = confidence(r)
    assert c.edof.shape == r.var.shape
    assert len(c.method) == r.af.size
    assert np.all(c.var_lo <= r.var) and np.all(r.var <= c.var_hi)
    assert np.allclose(c.dev_lo, np.sqrt(c.var_lo))
    assert np.allclose(c.dev_hi, np.sqrt(c.var_hi))


def test_confidence_is_recorded_in_provenance():
    """What the tool decided on the user's behalf has to be answerable."""
    c = confidence(_result(), level=TWO_SIGMA)
    step = [s for s in c.provenance.steps if s.action == "confidence"][-1]
    assert step.params["level"] == TWO_SIGMA
    assert step.params["method_requested"] == "auto"
    assert set(step.params["methods_used"]) <= {"eta1", "eta3"}


def test_confidence_needs_the_coefficients_and_says_so():
    r = oadev(FreqSeries(np.random.default_rng(1).standard_normal(300), rate_hz=1.0),
              keep_coeffs=False)
    with pytest.raises(ValueError, match="keep_coeffs=False"):
        confidence(r)


def test_confidence_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="method must be one of"):
        confidence(_result(), method="greenhall")


def test_eta3_gives_wider_intervals_than_eta1():
    """The conservative fallback must actually be conservative in effect."""
    r = _result(n=1024)
    a = confidence(r, method="eta1")
    b = confidence(r, method="eta3")
    assert np.all(b.edof <= a.edof * 1.01)
    assert np.all((b.var_hi - b.var_lo) >= (a.var_hi - a.var_lo) * 0.99)


def test_hadamard_intervals_work_too():
    """One CI implementation, every estimator -- the single-path contract."""
    c = confidence(ohdev(power_law_freq(512, -2, rate_hz=1.0,
                                        rng=np.random.default_rng(2))))
    assert np.all(np.isfinite(c.var_lo)) and np.all(np.isfinite(c.var_hi))


# --- coverage ------------------------------------------------------------


def _coverage(alpha, level, n, trials, seed):
    """Fraction of realizations whose interval contains the ensemble truth."""
    rng = np.random.default_rng(seed)
    runs = [oadev(power_law_freq(n, alpha, rate_hz=1.0, rng=rng), taus="octave")
            for _ in range(trials)]
    truth = np.mean([r.var for r in runs], axis=0)
    hits = np.zeros(runs[0].af.size)
    for r in runs:
        c = confidence(r, level=level)
        hits += (c.var_lo <= truth) & (truth <= c.var_hi)
    return runs[0].af, hits / trials


@pytest.mark.parametrize("alpha", [0, -2])
def test_coverage_is_close_to_nominal(alpha):
    """Quick version, in the default run so it cannot silently rot.

    The full sweep over all five power laws and both levels is the `slow`
    test below and is what the validation report quotes.
    """
    af, cov = _coverage(alpha, ONE_SIGMA, n=512, trials=200, seed=21)
    assert np.all(cov > 0.55), dict(zip(af.tolist(), cov.tolist()))
    assert np.all(cov < 0.82), dict(zip(af.tolist(), cov.tolist()))


@pytest.mark.slow
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("level,name", [(ONE_SIGMA, "1-sigma"), (TWO_SIGMA, "2-sigma")])
def test_measured_coverage_across_every_power_law(alpha, level, name):
    """The money test. Measured containment against nominal, published.

    Measured over 800 realizations of N=512: 1-sigma coverage lands between
    0.61 and 0.77 against a nominal 0.6827, and 2-sigma between 0.90 and
    0.99 against 0.9545. The spread is worst at the last octave, where the
    estimate has a handful of degrees of freedom and eta1 is working from
    an autocorrelation it can barely resolve.

    The bounds asserted are deliberately looser than the measurement, so
    this fails on a defect rather than on Monte-Carlo noise.
    """
    af, cov = _coverage(alpha, level, n=512, trials=800, seed=22)
    lo, hi = (0.58, 0.80) if level == ONE_SIGMA else (0.88, 0.995)
    assert np.all(cov > lo), f"{name} alpha={alpha}: {dict(zip(af.tolist(), cov.tolist()))}"
    assert np.all(cov < hi), f"{name} alpha={alpha}: {dict(zip(af.tolist(), cov.tolist()))}"
