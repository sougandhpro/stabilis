"""Container invariants, domain conversion, and the float64 precision guard."""

import numpy as np
import pytest

from stabilis.types import (
    PRECISION_WARN_RATIO,
    FreqSeries,
    PhaseSeries,
    precision_ratio,
    to_frequency,
    to_phase,
)


def test_roundtrip_drops_the_integration_constant():
    """to_phase pins x[0] = 0; the constant is unrecoverable from frequency."""
    x = np.array([7.0, 8.0, 10.0, 9.0])
    back = to_phase(to_frequency(PhaseSeries(x, rate_hz=1.0)))
    assert back.x[0] == 0.0
    assert np.allclose(back.x, x - x[0])


@pytest.mark.parametrize("n", [10, 1000, 100_000])
def test_phase_frequency_roundtrip_is_accurate_to_a_few_ulp(n):
    """x -> y -> x recovers the offset-removed phase to within a few ULP.

    Not bit-exact: a cumulative sum of differences does not telescope
    exactly in floating point. The bound below is derived rather than
    tuned -- measured accumulation is 0.2 ULP at n=10, 1.0 at n=1e3 and
    2.5 at n=1e5, so 8 ULP carries ample headroom while still failing
    loudly if the conversion ever grows a real error term.
    """
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(n)) * 1e-12
    ps = PhaseSeries(x, rate_hz=1.0)
    back = to_phase(to_frequency(ps)).x
    reference = ps.x - ps.x[0]
    peak = np.max(np.abs(reference))
    assert np.max(np.abs(back - reference)) <= 8 * np.spacing(peak)


def test_frequency_values():
    """y[i] = (x[i+1] - x[i]) * rate."""
    ps = PhaseSeries(np.array([0.0, 2.0, 6.0]), rate_hz=4.0)
    assert np.allclose(to_frequency(ps).y, [8.0, 16.0])


def test_arrays_are_read_only():
    """Immutable containers must actually be immutable."""
    ps = PhaseSeries(np.arange(5.0), rate_hz=1.0)
    with pytest.raises(ValueError):
        ps.x[0] = 99.0


def test_lengths_and_spans():
    ps = PhaseSeries(np.arange(11.0), rate_hz=2.0)
    assert ps.n == 11
    assert ps.tau0_s == 0.5
    assert ps.duration_s == 5.0
    fs = to_frequency(ps)
    assert fs.n == 10
    assert fs.duration_s == 5.0


def test_gaps_are_counted_and_propagate():
    """NaN marks a gap and must reach both straddling frequency samples."""
    x = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    ps = PhaseSeries(x, rate_hz=1.0)
    assert ps.n_gaps == 1
    y = to_frequency(ps).y
    assert np.isnan(y[1]) and np.isnan(y[2])
    assert np.isfinite(y[0]) and np.isfinite(y[3])


@pytest.mark.parametrize("bad_rate", [0.0, -1.0])
def test_rate_must_be_positive(bad_rate):
    with pytest.raises(ValueError, match="rate_hz"):
        PhaseSeries(np.arange(5.0), rate_hz=bad_rate)


def test_empty_series_rejected():
    with pytest.raises(ValueError, match="empty"):
        PhaseSeries(np.array([]), rate_hz=1.0)
    with pytest.raises(ValueError, match="empty"):
        FreqSeries(np.array([]), rate_hz=1.0)


def test_precision_ratio_quiet_for_well_conditioned_data():
    """Picosecond steps around zero are nowhere near the float64 floor."""
    x = np.cumsum(np.full(1000, 1e-12))
    assert precision_ratio(x) < PRECISION_WARN_RATIO


def test_precision_ratio_detects_epoch_quantization():
    """Picosecond steps riding on a large absolute offset lose resolution.

    This is the sub-nanosecond failure mode the epoch field exists to
    prevent: at 1e6 s the float64 spacing is ~1.2e-10 s, so 1 ps steps are
    quantized away entirely.
    """
    x = 1e6 + np.cumsum(np.full(1000, 1e-12))
    assert precision_ratio(x) > PRECISION_WARN_RATIO


def test_precision_warning_is_recorded_in_provenance():
    """The warning is not just logged -- it lands in the audit trail."""
    x = 1e6 + np.cumsum(np.full(500, 1e-12))
    ps = PhaseSeries(x, rate_hz=1.0)
    actions = [s.action for s in ps.provenance.steps]
    assert "precision_check" in actions


def test_precision_ratio_degenerate_cases():
    """No answer is better than a wrong answer when there is no step to see."""
    assert precision_ratio(np.array([1.0])) == 0.0
    assert precision_ratio(np.full(10, 3.0)) == 0.0
