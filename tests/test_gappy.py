"""Mondal & Percival's gap-tolerant wavelet variance (stats/gappy.py).

Correctness rests on three things: with no gaps it is the MODWT estimator
to rounding; its filters are the same ones an independent cascade builds;
and on gappy records its mean over realizations lands on the exact
expected value. The published AR(1) table is an interop check of the
filter normalization against the paper's own numbers.
"""

import numpy as np
import pytest

from stabilis import FreqSeries, PhaseSeries, confidence, power_law_freq
from stabilis.stats import gappy
from stabilis.stats.gappy import gappy_wavelet_variance, level_filter
from stabilis.stats.wavelet import FILTERS, wavelet_variance
from stabilis.validate.expected import expected_wvar, wavelet_kernel


@pytest.mark.parametrize("name", sorted(FILTERS))
def test_level_filter_is_the_cascade(name):
    for j in range(1, 7):
        h = level_filter(name, j)
        np.testing.assert_allclose(h, wavelet_kernel(name, j), rtol=0, atol=1e-15)
        assert np.sum(h ** 2) == pytest.approx(2.0 ** -j, rel=1e-12)


@pytest.mark.parametrize("name", ["haar", "d4", "d6"])
@pytest.mark.parametrize("estimator", ["v", "u"])
def test_without_gaps_it_is_the_modwt_estimator(name, estimator):
    rng = np.random.default_rng(1)
    fs = FreqSeries(rng.standard_normal(1500) + 3.0, rate_hz=2.0)
    w = wavelet_variance(fs, name, keep_coeffs=False)
    g = gappy_wavelet_variance(fs, name, estimator=estimator)
    np.testing.assert_array_equal(g.af, w.af)
    np.testing.assert_array_equal(g.n_terms, w.n_terms)
    np.testing.assert_allclose(g.tau_s, w.tau_s)
    np.testing.assert_allclose(g.var, w.var, rtol=1e-11)


def _gappy(y, p, rng):
    y = np.array(y, dtype=np.float64)
    y[rng.random(y.size) > p] = np.nan
    return y


@pytest.mark.parametrize("estimator", ["v", "u"])
def test_blind_to_a_constant_offset(estimator):
    rng = np.random.default_rng(2)
    y = _gappy(rng.standard_normal(800), 0.85, rng)
    a = gappy_wavelet_variance(FreqSeries(y, rate_hz=1.0), estimator=estimator)
    b = gappy_wavelet_variance(FreqSeries(y + 1e-3, rate_hz=1.0),
                               estimator=estimator)
    np.testing.assert_allclose(a.var, b.var, rtol=1e-6)


def _ar1_truth(phi, j):
    """Eq. 6: nu^2 = sum_{l,l'} h_l h_l' phi^|l-l'|, unit-variance AR(1)."""
    h = level_filter("haar", j)
    lag = np.abs(np.subtract.outer(np.arange(h.size), np.arange(h.size)))
    return float(h @ (phi ** lag) @ h)


def test_published_ar1_wavelet_variances():
    """INTEROP. Mondal & Percival (2010) Table 1, row nu^2_X(tau_j).

    AR(1), phi = 0.9, unit variance, Haar. Four printed decimals, so
    asserted at half a unit in the fourth.
    """
    published = [0.0500, 0.0689, 0.1079, 0.1585, 0.1907, 0.1710]
    for j, value in enumerate(published, start=1):
        assert _ar1_truth(0.9, j) == pytest.approx(value, abs=5e-5)


def _ar1(n, phi, rng):
    e = rng.standard_normal(n + 200) * np.sqrt(1 - phi ** 2)
    x = np.empty_like(e)
    x[0] = rng.standard_normal()
    for t in range(1, e.size):
        x[t] = phi * x[t - 1] + e[t]
    return x[200:]


@pytest.mark.parametrize("estimator", ["v", "u"])
def test_unbiased_on_the_papers_ar1_experiment(estimator):
    """Section 5.1 at reduced depth: N = 1024, Pr(observed) = 0.9, Haar.

    The mean over realizations lands within 4 standard errors of the exact
    eq.-6 value at every level -- the paper's unbiasedness claim, measured.
    """
    rng = np.random.default_rng(51)
    trials, est = 300, []
    for _ in range(trials):
        y = _gappy(_ar1(1024, 0.9, rng), 0.9, rng)
        est.append(gappy_wavelet_variance(FreqSeries(y, rate_hz=1.0),
                                          levels=6, estimator=estimator).var)
    est = np.array(est)
    truth = np.array([_ar1_truth(0.9, j) for j in range(1, 7)])
    se = est.std(axis=0, ddof=1) / np.sqrt(trials)
    assert np.all(np.abs(est.mean(axis=0) - truth) < 4 * se), (
        est.mean(axis=0), truth, se)


@pytest.mark.parametrize("alpha,name", [(0, "haar"), (-2, "d4")])
def test_unbiased_on_clock_noise_with_gaps(alpha, name):
    """White FM, and random-walk FM -- nonstationary y, stationary increments,
    which only the semivariogram form `v` is entitled to -- against the
    exact expectation of the generator."""
    rng = np.random.default_rng(80 + alpha)
    trials, n, levels, est = 200, 1024, 5, []
    for _ in range(trials):
        fs = power_law_freq(n, alpha, 1.0, rng)
        y = _gappy(fs.y, 0.9, rng)
        est.append(gappy_wavelet_variance(FreqSeries(y, rate_hz=1.0), name,
                                          levels=levels).var)
    est = np.array(est)
    truth = np.array([expected_wvar(name, j, alpha) for j in range(1, levels + 1)])
    se = est.std(axis=0, ddof=1) / np.sqrt(trials)
    assert np.all(np.abs(est.mean(axis=0) - truth) < 4 * se), (
        est.mean(axis=0) / truth, se / truth)


def test_keeps_the_scales_masking_loses():
    """One gap every 50 samples: every Haar level-7 coefficient (64 taps)
    touches a gap, so masking has nothing left there. Pairs survive."""
    rng = np.random.default_rng(3)
    y = rng.standard_normal(4000)
    y[::50] = np.nan
    fs = FreqSeries(y, rate_hz=1.0)
    masked = wavelet_variance(fs, keep_coeffs=False)
    kept = gappy_wavelet_variance(fs, levels=8)
    assert 64 not in masked.af.tolist()
    assert 64 in kept.af.tolist() and 128 in kept.af.tolist()


def test_too_gappy_is_refused_not_guessed():
    """Every other sample missing: no lag-1 pair is ever observed, and every
    Haar level needs one, so beta_hat is undefined everywhere."""
    y = np.ones(200)
    y[::2] = np.nan
    with pytest.raises(ValueError, match="too gappy"):
        gappy_wavelet_variance(FreqSeries(y, rate_hz=1.0))


def test_phase_input_and_provenance():
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.standard_normal(600))
    x[[100, 300]] = np.nan
    r = gappy_wavelet_variance(PhaseSeries(x, rate_hz=1.0))
    step = r.provenance.steps[-1]
    assert step.action == "gappy_wavelet_variance"
    assert step.params["estimator"] == "v" and step.params["n_gaps"] == 4


def test_refusals():
    fs = FreqSeries(np.random.default_rng(5).standard_normal(300), rate_hz=1.0)
    with pytest.raises(ValueError, match="estimator"):
        gappy_wavelet_variance(fs, estimator="w")
    with pytest.raises(ValueError, match="unknown wavelet"):
        gappy_wavelet_variance(fs, "d20")
    with pytest.raises(ValueError, match="coefficient"):
        confidence(gappy_wavelet_variance(fs))


def test_cost_budget(monkeypatch):
    monkeypatch.setattr(gappy, "MAX_PAIR_OPS", 1e3)
    fs = FreqSeries(np.ones(300), rate_hz=1.0)
    with pytest.raises(MemoryError, match="levels="):
        gappy_wavelet_variance(fs)
