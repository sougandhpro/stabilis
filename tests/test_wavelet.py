"""The MODWT, its filters, and the identity the library is built around.

Three things get the most attention: that the computed Daubechies filters
really are Daubechies filters (certified by algebra, not by trusting where
the numbers came from), that Haar reproduces the Allan variance exactly,
and that shift-equivariance holds where it should and *not* where it
should not.
"""

import numpy as np
import pytest

from stabilis.stats import oadev
from stabilis.stats.wavelet import (
    FILTERS,
    daubechies,
    modwt,
    wavelet_filters,
    wavelet_variance,
    wdev,
)
from stabilis.synth import power_law_freq
from stabilis.types import FreqSeries


@pytest.mark.parametrize("name", sorted(FILTERS))
def test_the_computed_filters_satisfy_the_defining_algebra(name):
    """Which is what makes computing them safer than transcribing them.

    A Daubechies filter of order K is *defined* by these conditions. A
    coefficient list passing all of them is the right filter however it was
    obtained -- whereas a transcribed list with one wrong digit still looks
    like a filter and fails none of the obvious sanity checks.
    """
    g, h = wavelet_filters(name)
    k = FILTERS[name]
    assert g.size == h.size == 2 * k
    assert np.dot(g, g) == pytest.approx(1.0, abs=1e-14)
    assert g.sum() == pytest.approx(np.sqrt(2.0), abs=1e-12)
    assert h.sum() == pytest.approx(0.0, abs=1e-12)
    assert np.dot(h, h) == pytest.approx(1.0, abs=1e-14)
    for shift in range(1, k):
        overlap = float(np.dot(g[: g.size - 2 * shift], g[2 * shift:]))
        assert overlap == pytest.approx(0.0, abs=1e-12)
    for p in range(k):
        moment = float(np.sum(h * np.arange(h.size, dtype=np.float64) ** p))
        assert moment == pytest.approx(0.0, abs=1e-10), f"moment {p}"


def test_haar_is_the_two_tap_case_exactly():
    """The overall sign of `h` is a convention -- `h_l = (-1)^l g_{L-1-l}`
    gives `[+, -]` here -- and the variance squares it away either way."""
    g, h = wavelet_filters("haar")
    assert g.tolist() == pytest.approx([1 / np.sqrt(2)] * 2)
    assert h.tolist() == pytest.approx([1 / np.sqrt(2), -1 / np.sqrt(2)])
    assert abs(h[0]) == abs(h[1]) and h[0] * h[1] < 0


def test_an_unknown_filter_is_refused_with_the_list():
    with pytest.raises(ValueError, match="unknown wavelet filter"):
        wavelet_filters("morlet")
    with pytest.raises(ValueError, match="k must be"):
        daubechies(0)


def test_the_haar_wavelet_deviation_is_the_allan_deviation():
    """`avar(tau) = 2 nu^2_Haar(tau)`, the claim the library rests on.

    Exact, not statistical: the retained Haar coefficients at scale
    `2^{j-1}` are the overlapping Allan second differences, so the two
    estimators are the same sum written two ways. The project plan asked
    for 1e-12; this holds to 1e-14.
    """
    fs = power_law_freq(4096, 0, rate_hz=1.0, rng=np.random.default_rng(1),
                        h_alpha=1e-22)
    allan = oadev(fs, taus="octave", keep_coeffs=False)
    wave = wdev(fs, "haar")
    compared = 0
    for i, tau in enumerate(wave.tau_s):
        j = int(np.argmin(np.abs(allan.tau_s - tau)))
        if abs(allan.tau_s[j] - tau) > 1e-9:
            continue
        assert wave.dev[i] == pytest.approx(allan.dev[j], rel=1e-14)
        compared += 1
    assert compared >= 10, "the identity should hold over many octaves"


def test_the_identity_holds_from_phase_input_too():
    """Phase in, frequency derived, same answer."""
    from stabilis.types import to_phase

    fs = power_law_freq(2048, -1, rate_hz=1.0, rng=np.random.default_rng(6))
    assert np.allclose(wdev(fs, "haar").var,
                       wdev(to_phase(fs), "haar").var, rtol=1e-12)


@pytest.mark.parametrize("shift", [1, 7, 100])
def test_the_coefficients_are_shift_equivariant_exactly(shift):
    """The property that distinguishes the MODWT from the DWT.

    Rotating the input rotates the coefficients, with no decimation to
    break the alignment. Bit-exact, not approximate -- the decimating DWT
    fails it outright, and a transform that fails it makes a variance
    depend on where the record happened to start.
    """
    y = power_law_freq(1024, -1, rate_hz=1.0, rng=np.random.default_rng(2)).y
    for a, b in zip(modwt(y, "d4"), modwt(np.roll(y, shift), "d4")):
        assert np.array_equal(np.roll(a, shift), b)


@pytest.mark.parametrize("shift", [1, 7, 100])
def test_the_biased_variance_is_shift_invariant_to_machine_precision(shift):
    """It uses every coefficient, so rotating cannot change the mean."""
    fs = power_law_freq(1024, -1, rate_hz=1.0, rng=np.random.default_rng(2))
    base = wavelet_variance(fs, "d4", unbiased=False)
    rolled = wavelet_variance(FreqSeries(np.roll(fs.y, shift), rate_hz=1.0),
                              "d4", unbiased=False)
    assert np.allclose(rolled.var, base.var, rtol=1e-14, atol=0.0)


def test_the_unbiased_variance_is_not_shift_invariant_and_should_not_be():
    """A distinction worth keeping straight rather than calling a bug.

    The unbiased estimator drops the coefficients that wrapped around the
    circular boundary, and that set is *positional*. Rotating the record
    discards different data, so the estimate moves by ordinary sampling
    variation -- a few percent at the coarsest scale, where only a couple
    of hundred coefficients survive.

    Keeping the wrapped coefficients would make it perfectly invariant
    *and* wrong: it treats the end of the record as adjacent to its start.
    """
    fs = power_law_freq(1024, -1, rate_hz=1.0, rng=np.random.default_rng(2))
    base = wavelet_variance(fs, "d4")
    rolled = wavelet_variance(FreqSeries(np.roll(fs.y, 100), rate_hz=1.0), "d4")
    assert not np.allclose(rolled.var, base.var, rtol=1e-6)
    assert np.allclose(rolled.var, base.var, rtol=0.5), "but not wildly"
    assert base.n_terms[-1] < base.n_terms[0], "coarse scales keep fewer"


def test_d4_annihilates_a_linear_frequency_drift_and_haar_does_not():
    """`K` vanishing moments kill any polynomial trend of degree `< K`.

    The concrete payoff of the wavelet framing: removing a drift is
    otherwise a separate, contestable pre-processing step, and with `K=2`
    it is simply not necessary.
    """
    n = 2048
    drift = 1e-14 * np.arange(n, dtype=np.float64)
    noise = power_law_freq(n, 0, rate_hz=1.0, rng=np.random.default_rng(3),
                           h_alpha=1e-26).y
    inflation = {}
    for name in ("haar", "d4", "d6"):
        clean = wavelet_variance(FreqSeries(noise, rate_hz=1.0), name)
        dirty = wavelet_variance(FreqSeries(noise + drift, rate_hz=1.0), name)
        inflation[name] = float(dirty.var[-1] / clean.var[-1])
    assert inflation["haar"] > 1e5, "Haar has one vanishing moment and sees it"
    assert inflation["d4"] == pytest.approx(1.0, abs=0.01), "K=2 annihilates it"
    assert inflation["d6"] == pytest.approx(1.0, abs=0.01)


def test_scales_are_dyadic_and_carry_the_sample_rate():
    fs = power_law_freq(512, 0, rate_hz=4.0, rng=np.random.default_rng(8))
    w = wavelet_variance(fs, "haar")
    assert w.af.tolist() == [1 << j for j in range(w.af.size)]
    assert w.tau_s[0] == pytest.approx(0.25), "tau0 = 1/rate"
    assert w.tau_s[1] == pytest.approx(0.5)


def test_a_record_too_short_for_the_filter_is_refused():
    fs = FreqSeries(np.arange(6, dtype=np.float64), rate_hz=1.0)
    with pytest.raises(ValueError, match="no scale survives|at least 2"):
        wavelet_variance(fs, "d8")
