"""The cross-tau covariance.

Four routes, in increasing order of how much can go wrong: the overlap
count against a brute-force enumeration, the diagonal against the two
implementations that already exist, the refusals, and finally the real
thing -- the measured correlation against the sample correlation of many
independent realizations.
"""

import numpy as np
import pytest

from stabilis.stats.classical import adev, mdev, oadev, ohdev
from stabilis.stats.edof import eta1
from stabilis.stats.isserlis import isserlis_var, lagged_covariance
from stabilis.stats.mtie import mtie
from stabilis.stats.taucov import (
    TAUCOV_METHODS,
    _isserlis_matrix,
    _overlap_counts,
    tau_covariance,
)
from stabilis.synth import power_law_freq
from stabilis.stats.total import totdev


def white_fm(n=2048, seed=7, alpha=0, h_alpha=1e-22):
    rng = np.random.default_rng(seed)
    return power_law_freq(n, alpha, rate_hz=1.0, rng=rng, h_alpha=h_alpha)


# ---------------------------------------------------------------------------
# the overlap count, which is the only new algebra
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m_i,m_j", [(10, 10), (10, 4), (4, 10), (1, 7), (9, 1)])
def test_overlap_count_matches_brute_force(m_i, m_j):
    """n_ij(k) counts index pairs, so enumerate the pairs and count them."""
    lags = np.arange(-12, 13)
    got = _overlap_counts(m_i, m_j, lags)
    want = [
        sum(1 for t in range(m_i) for s in range(m_j) if t - s == k)
        for k in lags
    ]
    assert got.tolist() == want


def test_equal_lengths_reduce_to_the_bartlett_factor():
    """With M_i = M_j the count is M - |k|, the weight `eta1` already uses."""
    m = 64
    lags = np.arange(-70, 71)
    got = _overlap_counts(m, m, lags)
    assert got.tolist() == np.maximum(0, m - np.abs(lags)).tolist()


# ---------------------------------------------------------------------------
# the diagonal, which two other code paths already compute
# ---------------------------------------------------------------------------


def test_diagonal_equals_isserlis_var_on_the_same_array():
    """Same formula, two implementations: a pair FFT against the auto case.

    `isserlis_var` normalizes by M and weights by 1 - |k|/M; this module
    normalizes by sqrt(M_i M_j) and weights by the pair count. Equal on the
    diagonal is not a tautology -- it is the check that the unequal-length
    generalization did not quietly change the equal-length answer.
    """
    r = oadev(white_fm(), taus="octave")
    raw = _isserlis_matrix(r.coeffs, r.af, r.scale, 4)
    for i, m in enumerate(r.af.tolist()):
        c = r.coeffs[i]
        kmax = min(4 * m, c.size - 1)
        ref = r.scale**2 * isserlis_var(
            lagged_covariance(c[None, :], kmax), c.size, [(0, 0)]
        )
        assert raw[i, i] == pytest.approx(ref, rel=1e-12)


def test_diagonal_equals_two_var_squared_over_eta1():
    """The shipped chi-square error bar is the diagonal of this matrix."""
    r = oadev(white_fm(), taus="octave")
    raw = _isserlis_matrix(r.coeffs, r.af, r.scale, 4)
    for i, m in enumerate(r.af.tolist()):
        eta = eta1(r.coeffs[i], int(m))
        assert raw[i, i] == pytest.approx(2.0 * r.var[i] ** 2 / eta, rel=1e-12)


def test_reported_diagonal_is_exactly_the_edof_it_reports():
    """A fit's weights and its covariance must not disagree about a point."""
    r = oadev(white_fm(), taus="octave")
    tc = tau_covariance(r)
    assert np.allclose(np.diag(tc.cov), 2.0 * r.var**2 / tc.edof, rtol=1e-14)


def test_correlation_matrix_is_a_correlation_matrix():
    r = oadev(white_fm(), taus="octave")
    tc = tau_covariance(r)
    corr = tc.corr
    assert np.allclose(np.diag(corr), 1.0, atol=1e-12)
    assert np.allclose(corr, corr.T, atol=1e-15)
    assert np.all(np.abs(corr) <= 1.0 + 1e-12)
    assert tc.condition > 1.0


@pytest.mark.parametrize("estimator", [oadev, mdev, ohdev, totdev])
def test_every_overlapping_estimator_can_be_measured(estimator):
    """The alignment argument is about the window step, not about the kind."""
    tc = tau_covariance(estimator(white_fm(1024), taus="octave"))
    assert tc.method == "isserlis"
    assert tc.reason == ""
    assert np.all(np.diag(tc.cov) > 0.0)


# ---------------------------------------------------------------------------
# the refusals, each of which is a wrong answer avoided
# ---------------------------------------------------------------------------


def test_a_decimating_estimator_declines_rather_than_misaligning():
    """`adev` advances by 2m samples, so a lag in its index is not a lag."""
    tc = tau_covariance(adev(white_fm(), taus="octave"))
    assert tc.method == "independent"
    assert "decimates" in tc.reason
    off = tc.cov - np.diag(np.diag(tc.cov))
    assert not np.any(off)


def test_no_coefficients_declines_and_says_which_flag_did_it():
    tc = tau_covariance(oadev(white_fm(), taus="octave", keep_coeffs=False))
    assert tc.method == "independent"
    assert "keep_coeffs=False" in tc.reason


def test_a_segmented_record_declines_because_the_index_is_pooled():
    fs = white_fm(1024)
    y = np.array(fs.y, dtype=float)
    y[500:505] = np.nan
    from stabilis.types import FreqSeries

    r = oadev(FreqSeries(y=y, rate_hz=fs.rate_hz), taus="octave")
    tc = tau_covariance(r)
    assert tc.method == "independent"
    assert "runs" in tc.reason


def test_a_masked_gap_declines_because_the_index_is_no_longer_contiguous():
    """One run, but with holes punched in it: the count alone cannot tell.

    `n_terms` falling short is indistinguishable from a shorter record,
    which is why `run_estimator` records the dropped count separately.
    """
    from stabilis.types import PhaseSeries, to_phase

    ps = to_phase(white_fm(1024))
    x = np.array(ps.x, dtype=float)
    x[400] = np.nan
    r = oadev(PhaseSeries(x=x, rate_hz=ps.rate_hz), taus="octave")
    assert any(r.dropped_terms)
    tc = tau_covariance(r)
    assert tc.method == "independent"
    assert "dropped at a gap" in tc.reason


def test_a_clean_record_drops_nothing():
    r = oadev(white_fm(512), taus="octave")
    assert set(r.dropped_terms) == {0}


def test_the_reason_reaches_provenance():
    tc = tau_covariance(adev(white_fm(), taus="octave"))
    steps = [s for s in tc.provenance.steps if s.action == "independent"]
    assert len(steps) == 1
    assert "decimates" in steps[0].params["reason"]
    assert "understates" in steps[0].outcome


def test_a_non_variance_is_refused_by_the_flag_not_by_name():
    from stabilis.types import to_phase

    r = mtie(to_phase(white_fm(512)), taus="octave")
    with pytest.raises(ValueError, match="not a variance"):
        tau_covariance(r)


def test_methods_tuple_is_the_only_source_of_truth():
    r = oadev(white_fm(512), taus="octave")
    assert tau_covariance(r).method in TAUCOV_METHODS
    assert tau_covariance(adev(white_fm(512), taus="octave")).method in TAUCOV_METHODS


def test_lag_limit_must_be_at_least_one():
    with pytest.raises(ValueError, match="lag_limit"):
        tau_covariance(oadev(white_fm(512), taus="octave"), lag_limit=0)


# ---------------------------------------------------------------------------
# against Monte-Carlo truth, which is the claim that matters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alpha,expected", [(0, 0.52), (-2, 0.88)],
    ids=["white-FM", "random-walk-FM"],
)
def test_measured_correlation_matches_the_scatter_of_the_estimates(alpha, expected):
    """Estimate the same curve many times and correlate the estimates.

    The reference here is the data's own behaviour, not a formula: 800
    independent records, the sample correlation of `var` between adjacent
    octave taus, against what a single record's coefficients predict. Both
    the level (0.52 for white FM, 0.88 for random-walk FM) and its
    insensitivity to tau are properties of the estimator, not of this code.
    """
    rng = np.random.default_rng(101)
    reps, n = 800, 1024
    estimates, predicted = [], []
    for _ in range(reps):
        fs = power_law_freq(n, alpha, rate_hz=1.0, rng=rng, h_alpha=1e-22)
        r = oadev(fs, taus="octave", max_af=n // 16)
        estimates.append(r.var)
        predicted.append(tau_covariance(r).corr)

    sample = np.corrcoef(np.array(estimates).T)
    mean_predicted = np.mean(predicted, axis=0)
    adjacent_sample = np.diag(sample, 1)
    adjacent_predicted = np.diag(mean_predicted, 1)

    assert np.allclose(adjacent_predicted, adjacent_sample, atol=0.06)
    assert np.mean(adjacent_predicted) == pytest.approx(expected, abs=0.05)


def test_ignoring_the_correlation_understates_a_summed_uncertainty():
    """The one-line statement of why this module exists.

    The variance of a sum of correlated estimates exceeds the sum of their
    variances. That gap *is* the understatement a fit inherits when it
    treats the taus as independent, and it is measured here rather than
    asserted.
    """
    r = oadev(white_fm(2048, seed=19, alpha=-2), taus="octave")
    tc = tau_covariance(r)
    full = float(np.ones(tc.cov.shape[0]) @ tc.cov @ np.ones(tc.cov.shape[0]))
    diagonal_only = float(np.sum(np.diag(tc.cov)))
    assert full > 1.5 * diagonal_only
