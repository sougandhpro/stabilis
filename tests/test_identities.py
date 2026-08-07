"""Tier 0: analytic identities, asserted at machine precision.

These need no reference data and no oracle, which is exactly why they carry
the correctness argument. Every published ADEV table in existence traces
back to Stable32, so agreeing with one proves interoperability; agreeing
with algebra proves correctness.
"""

import numpy as np
import pytest

from stabilis.stats import adev, mdev, oadev, ohdev, tdev
from stabilis.types import FreqSeries, PhaseSeries


def _phase(x, rate_hz=1.0):
    return PhaseSeries(np.asarray(x, dtype=np.float64), rate_hz=rate_hz)


def test_linear_frequency_drift_gives_exact_avar_and_zero_hvar():
    """The single most valuable non-fixture test in the suite.

    For y(t) = D*t the phase is x(t) = D*t**2/2, and the second difference
    of a quadratic is the constant D*tau**2, so

        avar(tau) = D**2 * tau**2 / 2      exactly
        hvar(tau) = 0                      exactly

    because the third difference of a quadratic vanishes. One test thereby
    validates AVAR's normalization, HVAR's normalization, and the drift
    convention simultaneously. With D = 1 and tau0 = 1 every intermediate
    value is exactly representable, so these are bit-exact, not approximate.
    """
    n, drift = 1024, 1.0
    i = np.arange(n, dtype=np.float64)
    ps = _phase(drift * i**2 / 2.0)

    a = oadev(ps, taus="octave")
    expected = drift**2 * a.tau_s**2 / 2.0
    assert np.array_equal(a.var, expected)

    h = ohdev(ps, taus="octave")
    assert np.array_equal(h.var, np.zeros_like(h.var))


def test_estimators_coincide_at_averaging_factor_one():
    """At m=1 there is no overlap to exploit, so these are one estimator.

    adev == oadev == mdev at AF 1, and hdev == ohdev at AF 1. This identity
    is what exposed a transcription error in the published NBS14 table --
    see the note in reference.py.
    """
    rng = np.random.default_rng(5)
    fs = FreqSeries(rng.standard_normal(300), rate_hz=1.0)
    a = adev(fs, taus="all").dev_at(1)
    assert oadev(fs, taus="all").dev_at(1) == pytest.approx(a, rel=1e-15)
    assert mdev(fs, taus="all").dev_at(1) == pytest.approx(a, rel=1e-15)


def test_tdev_is_mdev_scaled_by_tau_over_root_three():
    """tdev(tau) = tau/sqrt(3) * mdev(tau), definitionally."""
    rng = np.random.default_rng(6)
    fs = FreqSeries(rng.standard_normal(400), rate_hz=4.0)
    m, t = mdev(fs, taus="octave"), tdev(fs, taus="octave")
    assert np.allclose(t.dev, t.tau_s / np.sqrt(3.0) * m.dev, rtol=1e-15, atol=0)


def test_avar_is_blind_to_a_constant_frequency_offset():
    """Adding a constant to y is a linear phase ramp; the second difference kills it.

    Integers throughout, so the arithmetic is exact and this is bit-exact.
    """
    y = np.array([3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, 6.0, -5.0, 3.0, 5.0, 8.0])
    base = oadev(FreqSeries(y, rate_hz=1.0), taus="all")
    shifted = oadev(FreqSeries(y + 1000.0, rate_hz=1.0), taus="all")
    assert np.array_equal(base.var, shifted.var)


def test_hvar_is_blind_to_linear_frequency_drift():
    """The third difference annihilates drift, so HVAR needs no detrending."""
    rng = np.random.default_rng(7)
    n = 512
    i = np.arange(n, dtype=np.float64)
    noise = np.cumsum(rng.standard_normal(n))
    base = ohdev(_phase(noise), taus="octave")
    drifted = ohdev(_phase(noise + 3.0 * i**2 / 2.0), taus="octave")
    assert np.allclose(base.var, drifted.var, rtol=1e-9, atol=0)


def test_avar_is_not_blind_to_drift():
    """The companion to the test above -- otherwise it could pass vacuously."""
    rng = np.random.default_rng(7)
    n = 512
    i = np.arange(n, dtype=np.float64)
    noise = np.cumsum(rng.standard_normal(n))
    base = oadev(_phase(noise), taus="octave")
    drifted = oadev(_phase(noise + 3.0 * i**2 / 2.0), taus="octave")
    assert not np.allclose(base.var, drifted.var, rtol=1e-3)


@pytest.mark.parametrize("fn", [oadev, mdev, ohdev])
def test_scale_equivariance_is_bit_exact(fn):
    """Scaling phase by 2**k scales the variance by 4**k, with no rounding."""
    rng = np.random.default_rng(8)
    x = np.cumsum(rng.standard_normal(300))
    base = fn(_phase(x), taus="octave")
    scaled = fn(_phase(x * 8.0), taus="octave")
    assert np.array_equal(scaled.var, base.var * 64.0)


@pytest.mark.parametrize("fn", [oadev, mdev, ohdev])
def test_time_reversal_invariance(fn):
    """Reversing the record permutes the terms without changing the multiset."""
    rng = np.random.default_rng(9)
    x = np.cumsum(rng.standard_normal(300))
    forward = fn(_phase(x), taus="octave")
    backward = fn(_phase(x[::-1]), taus="octave")
    assert np.allclose(forward.var, backward.var, rtol=1e-13, atol=0)


@pytest.mark.parametrize("fn", [adev, oadev, mdev, tdev, ohdev])
def test_constant_phase_gives_exactly_zero(fn):
    """No noise means no variance -- exactly zero, not nearly zero."""
    result = fn(_phase(np.full(256, 7.5)), taus="octave")
    assert np.array_equal(result.var, np.zeros_like(result.var))


@pytest.mark.parametrize("fn", [adev, oadev, mdev, tdev, ohdev])
def test_phase_and_frequency_input_agree(fn):
    """The same data in either domain gives the same answer.

    NIST SP 1065 section 12.1 lists this among its recommended internal
    consistency checks for stability software.
    """
    rng = np.random.default_rng(10)
    y = rng.standard_normal(400)
    from_freq = fn(FreqSeries(y, rate_hz=1.0), taus="octave")
    from_phase = fn(_phase(np.concatenate(([0.0], np.cumsum(y)))), taus="octave")
    assert np.allclose(from_freq.var, from_phase.var, rtol=1e-14, atol=0)


def test_white_fm_avar_falls_as_one_over_tau():
    """For white FM, avar(tau) = h0/(2*tau), so the log-log slope is -1.

    A statistical check rather than an identity, so the tolerance is loose;
    it is here to catch a wrong power of m, which no algebraic test above
    would reveal.
    """
    rng = np.random.default_rng(12)
    fs = FreqSeries(rng.standard_normal(1 << 15), rate_hz=1.0)
    r = oadev(fs, taus="octave", max_af=1024)
    slope = np.polyfit(np.log(r.tau_s), np.log(r.var), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.05)
