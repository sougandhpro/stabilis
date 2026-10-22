"""Descriptive statistics and gap segmentation."""

import numpy as np
import pytest

from stabilis.preprocess import basic_stats, longest_segment, segments


def test_basic_stats():
    s = basic_stats(np.array([1.0, 2.0, 3.0, 4.0]))
    assert s["n"] == 4 and s["n_finite"] == 4 and s["n_gaps"] == 0
    assert s["min"] == 1.0 and s["max"] == 4.0
    assert s["mean"] == 2.5 and s["median"] == 2.5
    assert s["std"] == pytest.approx(np.sqrt(5.0 / 3.0))


def test_basic_stats_excludes_gaps():
    s = basic_stats(np.array([1.0, np.nan, 3.0]))
    assert s["n"] == 3 and s["n_finite"] == 2 and s["n_gaps"] == 1
    assert s["mean"] == 2.0


def test_basic_stats_json_safe():
    """Values must be builtins, not numpy scalars, before serialization."""
    s = basic_stats(np.array([1.0, 2.0]))
    assert all(isinstance(v, (int, float)) for v in s.values())
    assert not any(isinstance(v, np.generic) for v in s.values())


def test_basic_stats_single_point_std_is_nan():
    assert np.isnan(basic_stats(np.array([7.0]))["std"])


def test_basic_stats_all_gaps_rejected():
    with pytest.raises(ValueError, match="no finite values"):
        basic_stats(np.array([np.nan, np.nan]))


def test_segments_no_gaps():
    assert segments(np.arange(5.0)) == [(0, 5)]


def test_segments_interior_gap():
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    assert segments(a) == [(0, 2), (3, 5)]


def test_segments_leading_and_trailing_gaps():
    a = np.array([np.nan, 1.0, 2.0, np.nan])
    assert segments(a) == [(1, 3)]


def test_segments_min_length_filter():
    a = np.array([1.0, np.nan, 2.0, 3.0, 4.0])
    assert segments(a, min_length=2) == [(2, 5)]


def test_segments_empty_and_all_gap():
    assert segments(np.array([])) == []
    assert segments(np.array([np.nan, np.nan])) == []


def test_longest_segment():
    a = np.array([1.0, np.nan, 2.0, 3.0, 4.0, np.nan, 9.0])
    assert longest_segment(a) == (2, 5)
    with pytest.raises(ValueError, match="no finite values"):
        longest_segment(np.array([np.nan]))


# --- drift removal and decimation -----------------------------------------
#
# Identities first. The fits are checked on noiseless trends, where the
# answer is exact, and against the Hadamard variance, which annihilates a
# quadratic in phase and so must not notice one being subtracted.

from stabilis import ohdev, adev  # noqa: E402
from stabilis.preprocess import (  # noqa: E402
    DRIFT_MODELS, decimate, fit_drift, remove_drift,
)
from stabilis.types import FreqSeries, PhaseSeries, to_frequency, to_phase  # noqa: E402

D = 3.0e-12  # fractional frequency drift, 1/s


def _quadratic_phase(n=1000, rate=2.0):
    t = np.arange(n) / rate
    return PhaseSeries(1e-9 + 2e-10 * t + D * t ** 2 / 2, rate_hz=rate)


@pytest.mark.parametrize("model", ["quadratic", "diff2"])
def test_phase_drift_recovered_exactly(model):
    fit = fit_drift(_quadratic_phase(), model)
    assert fit.drift_per_s == pytest.approx(D, rel=1e-9)
    assert fit.drift_per_day == pytest.approx(D * 86400, rel=1e-9)


def test_quadratic_removal_leaves_rounding():
    ps = _quadratic_phase()
    out = remove_drift(ps)
    assert np.max(np.abs(out.x)) < 1e-12 * np.max(np.abs(ps.x))


@pytest.mark.parametrize("model", ["linear", "bisection"])
def test_frequency_drift_recovered_exactly(model):
    n, rate = 999, 1.0
    y = 5e-11 + D * np.arange(n) / rate
    fit = fit_drift(FreqSeries(y, rate_hz=rate), model)
    assert fit.drift_per_s == pytest.approx(D, rel=1e-9)
    resid = remove_drift(FreqSeries(y, rate_hz=rate), model).y
    assert np.max(np.abs(resid)) < 1e-9 * np.max(np.abs(y))


def test_offset_models_fit_no_drift():
    ps = _quadratic_phase()
    assert fit_drift(ps, "linear").drift_per_s == 0.0
    fs = FreqSeries(np.full(10, 4.0), rate_hz=1.0)
    fit = fit_drift(fs, "mean")
    assert fit.drift_per_s == 0.0 and np.all(fit.trend == 4.0)


def test_hadamard_blind_to_quadratic_removal():
    """ohvar annihilates a phase quadratic, so subtracting one is invisible."""
    rng = np.random.default_rng(1)
    t = np.arange(4096.0)
    ps = PhaseSeries(np.cumsum(rng.standard_normal(4096)) * 1e-9
                     + D * t ** 2 / 2, rate_hz=1.0)
    before = ohdev(ps, keep_coeffs=False).var
    after = ohdev(remove_drift(ps), keep_coeffs=False).var
    np.testing.assert_allclose(after, before, rtol=1e-8)


def test_noisy_drift_recovered_statistically():
    """White FM on a drift: the least-squares D is unbiased."""
    rng = np.random.default_rng(7)
    n = 20000
    y = D * np.arange(n) + 1e-9 * rng.standard_normal(n)
    # sigma_D for white residuals = sigma / sqrt(sum (t - tbar)^2)
    sigma_d = 1e-9 / np.sqrt(n ** 3 / 12)
    fit = fit_drift(FreqSeries(y, rate_hz=1.0))
    assert abs(fit.drift_per_s - D) < 5 * sigma_d


def test_drift_fit_steps_over_gaps():
    ps = _quadratic_phase()
    x = np.array(ps.x)
    x[[10, 11, 500]] = np.nan
    gapped = PhaseSeries(x, rate_hz=ps.rate_hz)
    for model in ("quadratic", "diff2"):
        assert fit_drift(gapped, model).drift_per_s == pytest.approx(D, rel=1e-9)
    out = remove_drift(gapped)
    assert np.isnan(out.x[[10, 11, 500]]).all()
    assert np.isfinite(out.x).sum() == x.size - 3


def test_drift_is_recorded_in_provenance():
    out = remove_drift(_quadratic_phase(), "diff2")
    actions = [s.action for s in out.provenance.steps]
    assert "remove_drift" in actions
    step = next(s for s in out.provenance.steps if s.action == "remove_drift")
    assert step.params["model"] == "diff2"
    assert step.params["drift_per_s"] == pytest.approx(D, rel=1e-9)


def test_drift_model_must_match_kind():
    with pytest.raises(ValueError, match="does not apply"):
        fit_drift(_quadratic_phase(), "bisection")
    with pytest.raises(ValueError, match="does not apply"):
        fit_drift(FreqSeries(np.ones(10), rate_hz=1.0), "quadratic")
    assert DRIFT_MODELS["phase"][0] == "quadratic"


def test_decimated_phase_reproduces_nonoverlapping_allan():
    """adev of x[::m] at af k is the same set of terms as adev at af k*m."""
    rng = np.random.default_rng(3)
    ps = PhaseSeries(np.cumsum(rng.standard_normal(4096)) * 1e-9, rate_hz=10.0)
    m = 4
    full = adev(ps, keep_coeffs=False)
    dec = adev(decimate(ps, m), keep_coeffs=False)
    assert dec.rate_hz == pytest.approx(ps.rate_hz / m)
    for tau, var in zip(dec.tau_s, dec.var):
        i = np.flatnonzero(np.isclose(full.tau_s, tau))
        if i.size:
            assert var == pytest.approx(full.var[i[0]], rel=1e-12)


def test_frequency_averaging_is_phase_decimation():
    """The phase of the averaged frequency is the decimated phase."""
    rng = np.random.default_rng(4)
    fs = FreqSeries(rng.standard_normal(1200) * 1e-11, rate_hz=1.0)
    m = 5
    via_freq = to_phase(decimate(fs, m)).x
    via_phase = decimate(to_phase(fs), m).x
    np.testing.assert_allclose(via_freq, via_phase, rtol=0,
                               atol=1e-12 * np.max(np.abs(via_phase)))


def test_frequency_decimation_drops_partial_block_and_gapped_blocks():
    y = np.arange(11.0)
    y[4] = np.nan
    out = decimate(FreqSeries(y, rate_hz=1.0), 3)
    assert out.y.size == 3
    assert out.y[0] == 1.0 and np.isnan(out.y[1]) and out.y[2] == 7.0


def test_phase_decimation_keeps_gap_only_on_kept_samples():
    x = np.arange(10.0)
    x[3] = np.nan
    x[4] = np.nan
    out = decimate(PhaseSeries(x, rate_hz=1.0), 2)
    np.testing.assert_array_equal(np.isnan(out.x), [0, 0, 1, 0, 0])


def test_decimate_rejects_bad_factor():
    ps = PhaseSeries(np.arange(10.0), rate_hz=1.0)
    for m in (0, 2.5, -1):
        with pytest.raises(ValueError):
            decimate(ps, m)
    with pytest.raises(ValueError):
        decimate(ps, 10)
    assert decimate(ps, 1).x.tolist() == ps.x.tolist()
