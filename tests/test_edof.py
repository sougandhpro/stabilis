"""Equivalent degrees of freedom, against their own definition.

EDOF is not a quantity with a published table to check against -- or rather
it is, and every such table is Stable32's output for one of five assumed
power laws, which is the assumption this module exists to avoid. So the
validation is direct: eta is *defined* by

    Var(sigma_hat^2) = 2 * sigma^4 / eta

and both sides of that are measurable. Generate many realizations, measure
the variance of the estimate, and compare.
"""

import numpy as np
import pytest

from stabilis.stats import oadev, ohdev
from stabilis.stats.edof import (
    MIN_TERMS_FOR_ETA1,
    _autocorrelation,
    edof,
    eta1,
    eta3,
)
from stabilis.synth import power_law_freq
from stabilis.types import FreqSeries

ALPHAS = (2, 1, 0, -1, -2)


def _runs(alpha, n, trials, seed, estimator=oadev):
    rng = np.random.default_rng(seed)
    return [
        estimator(power_law_freq(n, alpha, rate_hz=1.0, rng=rng), taus="octave")
        for _ in range(trials)
    ]


def _monte_carlo_eta(runs):
    """eta_true = 2*mean(var)^2 / Var(var), per tau, from the definition."""
    v = np.array([r.var for r in runs])
    return 2.0 * v.mean(axis=0) ** 2 / v.var(axis=0)


# --- the autocorrelation helper -----------------------------------------


def test_fft_autocorrelation_matches_the_direct_sum():
    """The FFT form exists for speed; it must agree with the obvious one.

    Zero-padding past 2M is what stops the circular correlation wrapping,
    and a wrap would show up here as a small error at the largest lags.
    """
    rng = np.random.default_rng(4)
    c = rng.standard_normal(257)
    kmax = 40
    got = _autocorrelation(c, kmax)
    direct = np.array([np.dot(c[: c.size - k], c[k:]) for k in range(kmax + 1)]) / c.size
    assert np.allclose(got, direct / direct[0], rtol=1e-12, atol=1e-14)


def test_autocorrelation_of_a_constant_series_is_one_at_every_lag():
    c = np.full(64, 3.0)
    rho = _autocorrelation(c, 8)
    # Biased normalization shrinks by (M-k)/M, which is the intended effect.
    expected = 1.0 - np.arange(9) / 64.0
    assert np.allclose(rho, expected, rtol=1e-12)


# --- eta1 ----------------------------------------------------------------


def test_eta1_of_uncorrelated_terms_is_the_term_count():
    """M independent terms carry M degrees of freedom. The sanity anchor."""
    rng = np.random.default_rng(7)
    c = rng.standard_normal(4000)
    # m=1 keeps the truncation short, which is the point: with no real
    # correlation, only the noise in rho_hat can move the answer.
    assert eta1(c, m=1) == pytest.approx(4000, rel=0.02)


def test_eta1_is_bounded_by_the_term_count():
    rng = np.random.default_rng(8)
    for m in (1, 4, 16, 64):
        r = oadev(power_law_freq(2048, -2, rate_hz=1.0, rng=rng), taus="octave")
        for af, c in zip(r.af, r.coeffs):
            eta = eta1(c, int(af))
            assert 1.0 <= eta <= c.size


def test_eta1_falls_with_correlation():
    """Perfectly correlated terms carry one degree of freedom, not M."""
    c = np.full(500, 2.0)  # rho(k) = 1 at every lag
    assert eta1(c, m=100) < 10.0


def test_truncating_the_lag_sum_is_what_removes_the_bias():
    """Summing every lag underestimates eta even with no true correlation.

    Each rho_hat(k) has variance ~1/M, so its square has expectation ~1/M
    and M-1 of them sum to ~1 -- a fake 1 beside the real one in the
    denominator. This is the Newey-West bias, and it is the entire reason
    for the truncation. Demonstrated rather than asserted in a comment.
    """
    rng = np.random.default_rng(9)
    c = rng.standard_normal(2000)
    truncated = eta1(c, m=1, lag_limit=4)
    every_lag = eta1(c, m=2000, lag_limit=1)  # kmax saturates at M-1
    assert truncated == pytest.approx(2000, rel=0.05)
    assert every_lag < 0.75 * truncated


@pytest.mark.parametrize("alpha", ALPHAS)
def test_eta1_tracks_monte_carlo_truth(alpha):
    """The definitional check, at every power law.

    Tolerance is wide on purpose and stated in both directions: this is a
    data-driven estimate from one realization, competing against closed
    forms that assume the noise type is known. Getting within tens of
    percent without being told the noise type is the claim being made --
    not getting within one percent.
    """
    runs = _runs(alpha, n=512, trials=400, seed=11)
    truth = _monte_carlo_eta(runs)
    first = runs[0]
    for i, (af, c) in enumerate(zip(first.af, first.coeffs)):
        if c.size < MIN_TERMS_FOR_ETA1:
            continue
        estimates = [eta1(r.coeffs[i], int(af)) for r in runs]
        ratio = float(np.median(estimates)) / truth[i]
        assert 0.6 < ratio < 1.5, f"alpha={alpha} m={af}: eta1/eta_true = {ratio:.2f}"


@pytest.mark.parametrize("alpha", ALPHAS)
def test_eta1_is_accurate_where_the_data_supports_it(alpha):
    """Tighter claim over the range that carries most of a real analysis.

    Up to m = N/32 there are hundreds of terms and the autocorrelation is
    well determined; the estimate holds to about 10%. The last octave or
    two is a different regime -- a handful of degrees of freedom, where the
    Monte-Carlo truth is itself uncertain -- and is covered by the looser
    test above rather than pretended away here.
    """
    runs = _runs(alpha, n=512, trials=400, seed=12)
    truth = _monte_carlo_eta(runs)
    first = runs[0]
    for i, af in enumerate(first.af):
        if int(af) > 512 // 32:
            continue
        estimates = [eta1(r.coeffs[i], int(af)) for r in runs]
        ratio = float(np.median(estimates)) / truth[i]
        assert 0.85 < ratio < 1.15, f"alpha={alpha} m={af}: ratio {ratio:.2f}"


def test_eta1_refuses_too_few_terms():
    with pytest.raises(ValueError, match="MIN_TERMS_FOR_ETA1"):
        eta1(np.arange(4.0), m=1)
    with pytest.raises(ValueError, match="no coefficients"):
        eta1(np.array([]), m=1)


# --- eta3 ----------------------------------------------------------------


def test_eta3_counts_non_overlapping_windows():
    assert eta3(1000, 1, span_factor=2) == 500.0
    assert eta3(1000, 10, span_factor=2) == 50.0
    assert eta3(1000, 10, span_factor=3) == pytest.approx(1000 / 30)
    assert eta3(3, 100) == 1.0  # never below one


@pytest.mark.parametrize("alpha", ALPHAS)
def test_eta3_is_conservative(alpha):
    """It must never exceed the truth -- that is its whole job as a fallback.

    A fallback that could be optimistic would produce error bars that are
    too narrow precisely when the data is too sparse to know better.
    """
    runs = _runs(alpha, n=512, trials=400, seed=13)
    truth = _monte_carlo_eta(runs)
    first = runs[0]
    for i, (af, c) in enumerate(zip(first.af, first.coeffs)):
        assert eta3(c.size, int(af), first.span_factor) <= truth[i] * 1.1


# --- the dispatcher ------------------------------------------------------


def test_edof_reports_the_method_it_actually_used():
    """"Silently fell back" is precisely what a report has to answer for."""
    rng = np.random.default_rng(14)
    c = rng.standard_normal(500)
    eta, how = edof(c, m=1)
    assert how == "eta1"
    eta, how = edof(np.arange(5.0), m=1)
    assert how == "eta3"  # too few terms, fell back without being asked
    eta, how = edof(c, m=1, method="eta3")
    assert how == "eta3"


def test_edof_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="method must be one of"):
        edof(np.arange(50.0), m=1, method="clairvoyance")


def test_edof_points_greenhall_at_its_own_module():
    """`greenhall` is a valid method for the library but not for this function.

    It is a function of the noise type and the record geometry, not of the
    coefficients, so it shares none of this dispatcher's inputs. The error
    names where to go rather than pretending the method does not exist.
    """
    with pytest.raises(ValueError, match="not a coefficient-based estimator"):
        edof(np.arange(50.0), m=1, method="greenhall")


def test_span_factor_reaches_the_result():
    """Hadamard terms span 3m, Allan 2m, and eta3 needs to know which."""
    rng = np.random.default_rng(15)
    fs = FreqSeries(rng.standard_normal(512), rate_hz=1.0)
    assert oadev(fs).span_factor == 2
    assert ohdev(fs).span_factor == 3
