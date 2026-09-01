"""Power-law noise synthesis, and the amplitude calibration.

Two of these are exact algebraic checks on the Kasdin-Walter recursion, which
are worth far more than the statistical slope test: if the impulse response
is right for the two cases where it terminates, the recursion itself is
right.

The calibration tests follow the same preference. `_unit_h_alpha` is a
*derivation*, so it is not evidence for itself; it is checked against the
closed forms for `sigma_y**2(tau)`, three of which are exact at every `m`
and are asserted at machine precision rather than statistically.
"""

import numpy as np
import pytest

from stabilis.stats.classical import oadev
from stabilis.synth import (
    _kasdin_walter_impulse,
    power_law_freq,
    power_law_noise,
    power_law_phase,
)
from stabilis.validate.expected import expected_avar


# The truncation depth. `expected_variance` defaults to 2**16 terms, which
# is ample for the report's 1e-10 claims; the closed forms below are
# asserted at 1e-14 with m = 256, so they buy four more binary orders of
# `h` rather than widen a tolerance to fit the default.
EXACT_TERMS = 1 << 20


def exact_avar(alpha: float, m: int, tau0: float, h_alpha: float) -> float:
    """`E[avar]` for a series generated with this `tau0` and this `h_alpha`.

    An *independent path* to the number the estimators compute: impulse
    response and exact expectation there, time-domain differencing of a
    realization here. It lives in `validate/expected.py` because the
    wavelet family needs the same machinery -- see that module for why the
    trailing entries of the convolution are dropped.
    """
    return expected_avar(alpha, m, tau0, h_alpha, n=EXACT_TERMS)


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


# ---------------------------------------------------------------------------
# Amplitude calibration.
#
# Four exact closed forms, then one statistical end-to-end check. The exact
# ones come first because they cannot be satisfied by a scale error that
# happens to sit inside a Monte-Carlo error bar.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 3, 7, 16, 64])
@pytest.mark.parametrize("tau0", [1.0, 0.1])
def test_white_fm_is_h0_over_2tau_exactly(m, tau0):
    """alpha = 0: `sigma2 = h_0/(2*tau)` exactly, at every `m`.

    White FM is the one power law where the discrete generator *is* the
    ideal process -- `h = [1, 0, 0, ...]`, so `y` is iid and
    `E[avar(m*tau0)] = var(y)/m` identically. Nothing here is asymptotic,
    which makes it the tightest anchor the calibration has.
    """
    h0 = 3.7e-24
    got = exact_avar(0.0, m, tau0, h0)
    assert got == pytest.approx(h0 / (2.0 * m * tau0), rel=1e-14)


@pytest.mark.parametrize("m", [1, 2, 5, 16])
def test_white_pm_matches_a_brick_wall_at_exactly_nyquist(m):
    """alpha = +2: `sigma2 = 3*h_2/(8*pi**2*tau0*tau**2)` exactly.

    `h = [1, -1]`, so the integrated phase telescopes to `tau0 * w` -- white
    phase of variance `tau0**2` -- and the second difference gives
    `avar = 3*var(x)/tau**2` with no approximation.

    Written against the measurement bandwidth this is `3*f_h*h_2/(4*pi**2*
    tau**2)` at **`f_h = 1/(2*tau0)`, exactly Nyquist**. That is the useful
    part: the `f_h`-dependent white-PM coefficient, which is not
    identifiable from tau-domain data, has one setting here that is known by
    construction rather than assumed. See `theory/powerlaw.md`.
    """
    tau0, h2 = 0.05, 1.1e-22
    tau = m * tau0
    got = exact_avar(2.0, m, tau0, h2)
    assert got == pytest.approx(3.0 * h2 / (8.0 * np.pi**2 * tau0 * tau**2), rel=1e-13)
    f_h = 0.5 / tau0
    assert got == pytest.approx(3.0 * f_h * h2 / (4.0 * np.pi**2 * tau**2), rel=1e-13)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32, 64])
def test_random_walk_fm_carries_an_exact_1_over_2m2_correction(m):
    """alpha = -2: `sigma2 = (2*pi**2*h_-2*tau/3) * (1 + 1/(2*m**2))`.

    The continuum form is the `m -> inf` limit; the discrete process has an
    exact excess of `1/(2*m**2)`, which at `m = 1` is a full 50%. Asserted
    at machine precision at every `m`, in *both* directions -- it is not a
    tolerance to be widened but a closed form to be reproduced.
    """
    tau0, h2 = 1.0, 2.5e-30
    tau = m * tau0
    continuum = 2.0 * np.pi**2 * h2 * tau / 3.0
    got = exact_avar(-2.0, m, tau0, h2)
    assert got == pytest.approx(continuum * (1.0 + 1.0 / (2.0 * m**2)), rel=1e-12)


def test_flicker_fm_at_m_equals_one_is_exactly_2h():
    """alpha = -1, m = 1: `sigma2 = 2*h_-1`, exactly.

    The spectral integral collapses because the kernel's `sin**4/sin**2`
    cancels the generator's own `1/sin`:

        sigma2(tau0) = 2 * int_0^fN (tau0/sin(pi*f*tau0)) * sin(pi*f*tau0)**2 df
                     = (2/pi) * int_0^(pi/2) sin(u) du = 2/pi = 2*h_-1

    since `h_-1 = 1/pi` at unit driving variance and `tau0 = 1`. The ratio
    to the continuum `2*ln(2)*h_-1` is therefore exactly `1/ln(2)`, a 44%
    excess at the first tau. Flicker FM is where the discrete and ideal
    processes differ most at short tau, and pretending otherwise would put
    that 44% into the calibration.
    """
    h1 = 6.6e-27
    got = exact_avar(-1.0, 1, 1.0, h1)
    assert got == pytest.approx(2.0 * h1, rel=1e-9)
    continuum = 2.0 * np.log(2.0) * h1
    assert got / continuum == pytest.approx(1.0 / np.log(2.0), rel=1e-9)


@pytest.mark.parametrize("alpha", [0, -1, -2])
def test_the_continuum_closed_forms_are_recovered_at_large_m(alpha):
    """The `f_h`-free closed forms hold once the discrete correction dies.

    This is the check that the calibration is *the IEEE 1139 one* and not
    merely self-consistent: the three forms below are published, contain no
    free parameter, and involve `ln(2)` and `pi**2` factors that no scale
    error can reproduce by accident.
    """
    tau0, h = 1.0, 4.2e-25
    m = 256
    tau = m * tau0
    continuum = {0: h / (2.0 * tau),
                 -1: 2.0 * np.log(2.0) * h,
                 -2: 2.0 * np.pi**2 * h * tau / 3.0}[alpha]
    assert exact_avar(alpha, m, tau0, h) == pytest.approx(continuum, rel=2e-4)


@pytest.mark.parametrize("alpha", [0, -1, -2])
@pytest.mark.parametrize("tau0", [1.0, 0.02])
def test_the_generator_actually_delivers_the_requested_h_alpha(alpha, tau0):
    """End to end: generate, estimate, recover `h_alpha`.

    Everything above is expectation algebra. This one runs the real
    generator through the real estimator and asks whether the number that
    comes back is the number that went in. Compared against `exact_avar`
    rather than the continuum form, so the discrete correction is not
    charged to the Monte Carlo.
    """
    h_alpha, trials, n = 8.0e-24, 200, 4096
    rng = np.random.default_rng(20260808 + int(alpha))
    total = None
    for _ in range(trials):
        r = oadev(power_law_freq(n, alpha, rate_hz=1.0 / tau0, rng=rng,
                                 h_alpha=h_alpha), taus="octave")
        total = r.var if total is None else total + r.var
    mean = total / trials

    # Only the taus where the estimate has enough independent terms to be
    # worth comparing: at m > n/16 the ensemble mean is itself uncertain at
    # the several-percent level and the check would be vacuous.
    keep = r.af <= n // 16
    want = np.array([exact_avar(alpha, int(m), tau0, h_alpha) for m in r.af[keep]])
    assert np.allclose(mean[keep], want, rtol=0.06)


def test_h_alpha_scales_the_variance_linearly():
    """Doubling `h_alpha` doubles every variance, bit for bit."""
    kw = dict(n=512, alpha=-1, rate_hz=1.0)
    a = oadev(power_law_freq(**kw, rng=np.random.default_rng(5), h_alpha=1.0))
    b = oadev(power_law_freq(**kw, rng=np.random.default_rng(5), h_alpha=2.0))
    assert np.allclose(b.var, 2.0 * a.var, rtol=1e-13)


def test_h_alpha_is_recorded_in_provenance():
    """A report has to be able to say what was generated."""
    fs = power_law_freq(64, 0, rate_hz=1.0, rng=np.random.default_rng(0),
                        h_alpha=1e-22)
    step = next(s for s in fs.provenance.steps if s.action == "power_law_noise")
    assert step.params["h_alpha"] == 1e-22
    assert "scale" in step.params


def test_rejects_negative_h_alpha():
    """`h_alpha` is a noise power. A negative one is a caller error."""
    with pytest.raises(ValueError, match="h_alpha"):
        power_law_freq(64, 0, rate_hz=1.0, rng=np.random.default_rng(0),
                       h_alpha=-1.0)
