"""Power-law noise synthesis.

Two of these are exact algebraic checks on the Kasdin-Walter recursion, which
are worth far more than the statistical slope test: if the impulse response
is right for the two cases where it terminates, the recursion itself is
right.
"""

import numpy as np
import pytest

from stabilis.synth import (
    _kasdin_walter_impulse,
    power_law_freq,
    power_law_noise,
    power_law_phase,
)


def test_impulse_random_walk_is_cumulative_sum():
    """beta = 2 (alpha = -2, RWFM) gives h = [1, 1, 1, ...].

    Integrating white noise gives a random walk -- exact, not approximate.
    """
    h = _kasdin_walter_impulse(16, beta=2.0)
    assert np.array_equal(h, np.ones(16))


def test_impulse_white_pm_is_a_difference():
    """beta = -2 (alpha = +2, WPM) terminates at h = [1, -1, 0, 0, ...].

    Differencing white noise gives f^2 noise. The recursion truncating here
    is the strongest single check that the (k - 1 + beta/2)/k factor is right.
    """
    h = _kasdin_walter_impulse(16, beta=-2.0)
    assert h[0] == 1.0 and h[1] == -1.0
    assert np.array_equal(h[2:], np.zeros(14))


def test_impulse_white_fm_is_an_impulse():
    """beta = 0 (alpha = 0, WFM) is pass-through."""
    h = _kasdin_walter_impulse(16, beta=0.0)
    assert h[0] == 1.0
    assert np.array_equal(h[1:], np.zeros(15))


def test_impulse_flicker_coefficients():
    """beta = 1 (alpha = -1, FFM), the half-integrator: 1, 1/2, 3/8, 5/16."""
    h = _kasdin_walter_impulse(4, beta=1.0)
    assert np.allclose(h, [1.0, 0.5, 0.375, 0.3125])


@pytest.mark.parametrize("alpha", [2, 1, 0, -1, -2])
def test_psd_slope_recovers_alpha(alpha):
    """The generated series has S_y(f) ~ f^alpha.

    Fitted over the middle of the band, away from the DC end where the
    estimate is dominated by a handful of bins and the top octave where the
    finite impulse response rolls off.
    """
    rng = np.random.default_rng(20260807)
    n = 1 << 16
    # Average several periodograms: one realization is chi2_2 per bin, which
    # is far too noisy to fit a slope through.
    power = np.zeros(n // 2 + 1)
    n_avg = 8
    for _ in range(n_avg):
        y = power_law_noise(n, alpha, rng)
        power += np.abs(np.fft.rfft(y)) ** 2
    power /= n_avg

    freq = np.fft.rfftfreq(n)
    band = (freq > 1e-3) & (freq < 0.1)
    slope = np.polyfit(np.log(freq[band]), np.log(power[band]), 1)[0]
    assert abs(slope - alpha) < 0.1, f"alpha={alpha}: fitted slope {slope:.3f}"


def test_same_seed_reproduces_exactly():
    """Validation evidence that cannot be reproduced bit for bit is not evidence."""
    a = power_law_noise(1024, -1, np.random.default_rng(42))
    b = power_law_noise(1024, -1, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    a = power_law_noise(1024, -1, np.random.default_rng(1))
    b = power_law_noise(1024, -1, np.random.default_rng(2))
    assert not np.allclose(a, b)


def test_no_global_rng_is_touched():
    """synth must never consume numpy's global random state."""
    np.random.seed(0)
    before = np.random.get_state()[2]
    power_law_noise(256, 0, np.random.default_rng(7))
    assert np.random.get_state()[2] == before


def test_series_wrappers_carry_provenance():
    rng = np.random.default_rng(3)
    fs = power_law_freq(512, -1, rate_hz=1.0, rng=rng)
    assert fs.n == 512
    actions = [s.action for s in fs.provenance.steps]
    assert "power_law_noise" in actions

    ps = power_law_phase(512, -1, rate_hz=1.0, rng=np.random.default_rng(3))
    assert ps.n == 513, "integrating n frequency points gives n+1 phase points"


def test_rejects_empty():
    with pytest.raises(ValueError, match="n must be"):
        power_law_noise(0, 0, np.random.default_rng(0))
