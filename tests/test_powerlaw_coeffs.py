"""Power-law conversion coefficients, against the integrals they come from.

The Allan rows are transcribed from NIST SP 1065 section 7.1; the modified
rows are derived here, because the handbook tabulates none. Both are checked
the same way: by integrating the handbook's *own* transfer-function
expressions, eq. (65) and eq. (66) on p. 73, which are a level below the
power-law formulae and independent of them.

One row -- flicker PM, unmodified -- is printed two inconsistent ways in the
source. That is settled here by measurement rather than by preference, and
`test_the_flicker_pm_row_settles_a_contradiction_in_the_source` is the
test that does it.
"""

import numpy as np
import pytest

from stabilis.powerlaw_coeffs import (
    ALPHAS,
    MVAR_FLICKER_FM_COEFF,
    avar_coeff,
    drift_coeff,
    mvar_coeff,
)
from stabilis.stats.classical import mdev, oadev
from stabilis.synth import power_law_freq
from stabilis.types import FreqSeries

_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)


def _integrate(fn, a: float, b: float, panels: int) -> float:
    """Panelled Gauss-Legendre. The integrands oscillate as sin^k(pi tau f),
    so panels are aligned to that scale and the rule is applied in each."""
    edges = np.linspace(a, b, panels + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    return float(np.sum(half * _GL_W[None, :] * fn(mid + half * _GL_X[None, :])))


def avar_from_psd(alpha: int, m: int, tau0: float, f_h: float) -> float:
    """SP 1065 eq. (65): `2 int_0^fh S_y(f) sin^4(pi tau f)/(pi tau f)^2 df`."""
    tau = m * tau0
    fn = lambda f: np.where(  # noqa: E731
        f > 0, f**alpha * np.sin(np.pi * tau * f) ** 4
        / (np.pi * tau * np.where(f > 0, f, 1.0)) ** 2, 0.0)
    return 2.0 * _integrate(fn, 0.0, f_h, max(int(16 * f_h * tau), 800))


def mvar_from_psd(alpha: int, m: int, tau0: float, f_h: float) -> float:
    """SP 1065 eq. (66), with N = m."""
    tau = m * tau0

    def fn(f):
        num = f**alpha * np.sin(np.pi * tau * f) ** 6
        den = f**2 * np.sin(np.pi * tau0 * f) ** 2
        return np.where(f > 0, num / np.where(den > 0, den, 1.0), 0.0)

    val = _integrate(fn, 0.0, f_h, max(int(16 * f_h * tau), 800))
    return 2.0 / (m**4 * np.pi**2 * tau0**2) * val


# ---------------------------------------------------------------------------
# The transcribed Allan rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
def test_avar_rows_reproduce_the_handbooks_own_integral(alpha):
    """Each SP 1065 section 7.1 row against SP 1065 eq. (65).

    The power-law formulae are an approximation to the integral, and the
    parameter they are asymptotic in is `2*pi*f_h*tau` -- so that is what
    sets the tolerance here, rather than a round number chosen to pass.
    White FM is the row that exercises it: at `2*pi*f_h*tau = 201` the
    closed form `h_0/(2*tau)` is 0.47% high, because the sinc tail beyond
    `f_h` is real signal the closed form has integrated in. Asserting 1e-3
    flat would fail that legitimately-approximate row while telling us
    nothing about the four exact ones.
    """
    tau0, f_h = 1.0, 0.5
    for m in (64, 256):
        tau = m * tau0
        got = avar_from_psd(alpha, m, tau0, f_h)
        want = float(avar_coeff(alpha, np.array([tau]), f_h)[0])
        assert got == pytest.approx(want, rel=3.0 / (2.0 * np.pi * f_h * tau))


def test_the_flicker_pm_row_settles_a_contradiction_in_the_source():
    """SP 1065 prints the flicker-PM conversion two ways, and they differ.

    * eq. (67), p. 73:  `[1.038 + 3*ln(2*pi*f_h*tau)] / (2*pi**2*tau**2)`
    * the table, p. 74: `[1.038 + 3*ln(2*pi*f_h*tau0)] / (4*pi**2)` / tau**2

    The denominators differ by a factor of two, and the logarithm's argument
    carries `tau` in one and `tau0` in the other. Integrating the handbook's
    own eq. (65) picks the third combination -- the table's `4*pi**2` with
    the equation's `tau` -- and rejects both printed forms outright: eq.
    (67) by exactly a factor of two, the table by a factor that *drifts with
    m*, which is the signature of the wrong variable inside the log.

    Recorded rather than silently corrected, per `CLAUDE.md`. The measured
    numbers are in `theory/powerlaw.md`.
    """
    tau0, f_h = 1.0, 0.5
    for m in (64, 256):
        tau = m * tau0
        quad = avar_from_psd(1, m, tau0, f_h)
        log_tau = 1.038 + 3.0 * np.log(2.0 * np.pi * f_h * tau)
        log_tau0 = 1.038 + 3.0 * np.log(2.0 * np.pi * f_h * tau0)

        as_shipped = log_tau / (4.0 * np.pi**2 * tau**2)
        assert quad == pytest.approx(as_shipped, rel=1e-3)

        as_eq67 = log_tau / (2.0 * np.pi**2 * tau**2)
        assert quad / as_eq67 == pytest.approx(0.5, rel=1e-3)

        as_table = log_tau0 / (4.0 * np.pi**2 * tau**2)
        assert quad / as_table > 3.0


def test_white_pm_row_is_exact_against_a_generator_of_known_bandwidth():
    """The `f_h` row checked at an `f_h` that is known by construction.

    `f_h` cannot be recovered from tau-domain data, so an end-to-end test of
    this coefficient needs a process whose bandwidth is known a priori. The
    synthesizer's alpha = +2 output is exactly that: white phase of variance
    `tau0**2`, whose Allan variance is `3*tau0**2/tau**2` exactly, which is
    this row at exactly `f_h = 1/(2*tau0)`. See `theory/noise-types.md`.
    """
    tau0, h2 = 0.05, 4.0e-23
    f_h = 0.5 / tau0
    for m in (1, 4, 32):
        tau = m * tau0
        want = 3.0 * h2 / (8.0 * np.pi**2 * tau0 * tau**2)
        got = float(avar_coeff(2, np.array([tau]), f_h)[0]) * h2
        assert got == pytest.approx(want, rel=1e-14)


# ---------------------------------------------------------------------------
# The derived modified rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("tau0", [1.0, 0.05])
def test_mvar_rows_reproduce_the_handbooks_own_integral(alpha, tau0):
    """The derivation's evidence. Two `tau0`, because a missing `tau0`
    factor is invisible at `tau0 = 1` and these rows have none."""
    f_h = 0.5 / tau0
    for m in (64, 256):
        got = mvar_from_psd(alpha, m, tau0, f_h)
        want = float(mvar_coeff(alpha, np.array([m * tau0]))[0])
        assert got == pytest.approx(want, rel=2e-3)


def test_modified_white_pm_falls_as_tau_cubed_and_breaks_the_degeneracy():
    """Why the fit stacks MVAR rows instead of fitting AVAR alone.

    Unmodified, white PM and flicker PM both give `sigma2 ~ tau**-2` and no
    amount of data separates them -- SP 1065 says so in its own introduction
    to section 7. Modified, white PM falls as `tau**-3` while flicker PM
    stays at `tau**-2`, so one octave of tau resolves them.
    """
    tau = np.array([1.0, 2.0, 4.0, 8.0])
    a2, a1 = avar_coeff(2, tau, 100.0), avar_coeff(1, tau, 100.0)
    # Unmodified: both are tau^-2 up to the log, so their ratio is nearly
    # constant -- that near-constancy *is* the degeneracy.
    ratio = a2 / a1
    assert ratio.max() / ratio.min() < 2.0

    m2, m1 = mvar_coeff(2, tau), mvar_coeff(1, tau)
    slope2 = np.polyfit(np.log(tau), np.log(m2), 1)[0]
    slope1 = np.polyfit(np.log(tau), np.log(m1), 1)[0]
    assert slope2 == pytest.approx(-3.0, abs=1e-12)
    assert slope1 == pytest.approx(-2.0, abs=1e-12)


def test_the_flicker_fm_modified_constant_is_not_the_pretty_closed_form():
    """Guards the one derived number against being "tidied" into 27*ln2/20.

    That form is within 0.06% and is wrong: quadrature converges to
    0.9352277 as 1/m**2, four figures away from 0.9357487, and a search for
    `a*ln2 + b*ln3` with small rational coefficients found nothing at 2e-9.
    A number with a derivation beats a closed form without one.
    """
    assert mvar_from_psd(-1, 512, 1.0, 0.5) == pytest.approx(
        MVAR_FLICKER_FM_COEFF, rel=1e-5)
    assert MVAR_FLICKER_FM_COEFF != pytest.approx(27 * np.log(2) / 20, rel=1e-4)


# ---------------------------------------------------------------------------
# Against the shipped estimators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
def test_coefficients_describe_the_real_estimators(alpha):
    """Ensemble mean of the shipped estimators against both coefficient sets.

    Everything above compares one integral to another. This runs the
    calibrated generator through `oadev` and `mdev` and asks whether the
    coefficients predict what actually comes out.

    Unmodified flicker PM is excluded, and that exclusion is the honest
    part: the coefficient assumes a brick-wall `f_h`, and the generator's
    flicker PM is not brick-wall limited -- it rises to Nyquist and stops.
    Its measured constant is 1.764 where the brick-wall model says 1.038,
    equivalent to an effective `f_h` about 27% above Nyquist. The white-PM
    row has no such gap because that process *is* brick-wall at Nyquist
    exactly, and the modified flicker-PM row has none because it carries no
    `f_h` at all. See `theory/powerlaw.md`.
    """
    tau0, h, trials, n = 1.0, 3.0e-24, 300, 8192
    f_h = 0.5 / tau0
    rng = np.random.default_rng(777 + alpha)
    totals = {"a": None, "m": None}
    for _ in range(trials):
        fs = power_law_freq(n, alpha, rate_hz=1.0 / tau0, rng=rng, h_alpha=h)
        ra, rm = oadev(fs, taus="octave"), mdev(fs, taus="octave")
        totals["a"] = ra.var if totals["a"] is None else totals["a"] + ra.var
        totals["m"] = rm.var if totals["m"] is None else totals["m"] + rm.var

    keep = (ra.af >= 16) & (ra.af <= n // 32)
    tau = ra.tau_s[keep]
    if alpha != 1:
        got = totals["a"][keep] / trials
        assert np.allclose(got, avar_coeff(alpha, tau, f_h) * h, rtol=0.03)
    got_m = totals["m"][keep] / trials
    assert np.allclose(got_m, mvar_coeff(alpha, tau) * h, rtol=0.03)


def test_drift_coefficient_is_exact_for_both_variances():
    """`sigma2_y(tau) = D**2 tau**2/2`, and the modified variance agrees.

    Exact rather than asymptotic, which is what lets drift join the design
    matrix as a `D**2` column without making the model nonlinear.
    """
    d, n = 3.0e-12, 4096
    fs = FreqSeries(y=d * np.arange(n, dtype=np.float64), rate_hz=1.0)
    ra, rm = oadev(fs, taus="octave"), mdev(fs, taus="octave")
    want = drift_coeff(ra.tau_s) * d**2
    assert np.allclose(ra.var, want, rtol=1e-12)
    assert np.allclose(rm.var, drift_coeff(rm.tau_s) * d**2, rtol=1e-12)


def test_unknown_alpha_is_refused():
    for fn in (lambda: avar_coeff(3, np.array([1.0]), 10.0),
               lambda: mvar_coeff(-3, np.array([1.0]))):
        with pytest.raises(ValueError, match="alpha must be"):
            fn()
