"""Thêo1: reach, the definition's own tau, and a missing 0.75.

The interesting test is the last one. SP 1065 eq. (30) as printed gives an
estimator that is uniformly 0.75 times AVAR, when Thêo1 is documented as
unbiased against AVAR at large m. Measurement is what settled it.
"""

import numpy as np
import pytest

from stabilis.stats import oadev, theo1
from stabilis.stats.theo1 import MIN_AF, theo1_afs, theo1_taus
from stabilis.synth import power_law_phase
from stabilis.types import FreqSeries


def test_averaging_factors_are_even_and_at_least_ten():
    """SP 1065: "for m even, and 10 <= m <= N-1".

    An odd m is not a poor choice for Thêo1, it is outside the definition:
    the inner sum runs to m/2 - 1 and needs m/2 to be a whole number.
    """
    afs = theo1_afs(4096)
    assert np.all(afs % 2 == 0)
    assert afs[0] >= MIN_AF
    assert afs[-1] <= 4095
    assert theo1_afs(8).size == 0, "too short to have any valid m"


def test_tau_is_three_quarters_of_m_minus_one():
    """The definition's own averaging time, and the usual place to slip.

    Thêo1 is the one estimator here whose tau is not `m * tau0`. Getting
    this wrong shifts every point along the axis and the curve still looks
    entirely plausible.
    """
    assert theo1_taus([10, 100], 1.0).tolist() == [6.75, 74.25]
    assert theo1_taus([10], 10.0).tolist() == [0.675], "scales with tau0"


def test_it_reaches_three_quarters_of_the_record():
    """The whole point: AVAR stops at half, Thêo1 goes to three quarters."""
    n = 4096
    ps = power_law_phase(n, 0, rate_hz=1.0, rng=np.random.default_rng(1))
    t = theo1(ps)
    a = oadev(ps, taus="octave", keep_coeffs=False)
    assert t.tau_s[-1] > 0.7 * n
    assert t.tau_s[-1] > 2.5 * a.tau_s[-1]


def test_frequency_input_is_refused():
    with pytest.raises(TypeError, match="needs phase data"):
        theo1(FreqSeries(np.zeros(100), rate_hz=1.0))


def test_too_short_a_record_is_refused_with_the_reason():
    ps = power_law_phase(8, 0, rate_hz=1.0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="gap-free run of at least"):
        theo1(ps)


@pytest.mark.slow
def test_the_missing_0_75_in_sp1065_equation_30():
    """The correction, and the measurement that justifies it.

    Eq. (30) as printed has `1/[(N-m)(m tau0)^2]`. Thêo1 is documented as
    an unbiased estimator of AVAR at large m, and as printed it is not --
    it lands at 0.75 of AVAR across the whole range, which is exactly the
    factor the printed equation is missing. With the 0.75 restored it
    converges to 1.

    This is the second printing slip found in SP 1065 by measurement; the
    first was the flicker-PM conversion row in theory/powerlaw.md.
    """
    n, trials = 8192, 40
    rng = np.random.default_rng(5)
    acc_t = acc_a = None
    for _ in range(trials):
        ps = power_law_phase(n, 0, rate_hz=1.0, rng=rng, h_alpha=1e-24)
        t = theo1(ps)
        a = oadev(ps, taus="all", keep_coeffs=False)
        allan_here = np.interp(t.tau_s, a.tau_s, a.var)
        acc_t = t.var if acc_t is None else acc_t + t.var
        acc_a = allan_here if acc_a is None else acc_a + allan_here

    # Only where AVAR is actually defined. Theo1 reaches 0.75*N while the
    # Allan grid stops at N/4, and `np.interp` *clamps* past its last
    # point rather than extrapolating -- so comparing out there measures
    # the clamp, not the estimator.
    ratio = acc_t / acc_a
    large = (t.af >= 100) & (t.tau_s <= a.tau_s[-1])
    assert large.sum() >= 5, "not enough overlap to test"
    assert np.allclose(ratio[large], 1.0, atol=0.06), (
        "Theo1 must be unbiased against AVAR for white FM at large m; "
        f"got {ratio[large]}")
    # As printed -- without the 0.75 -- it would sit here instead.
    assert not np.allclose(ratio[large] * 0.75, 1.0, atol=0.06)


@pytest.mark.slow
@pytest.mark.parametrize("alpha,expected", [(-1, 0.55), (-2, 0.37)])
def test_the_fm_biases_are_what_theobr_exists_to_remove(alpha, expected):
    """Thêo1 is biased for the FM noises even with the 0.75 restored.

    Recorded rather than corrected: TheoBR removes this by averaging
    against AVAR over the overlapping range, and is not implemented. A
    number divided by a ratio measured on synthetic noise is not the same
    thing as one corrected by a published estimator, and the difference is
    worth keeping visible.
    """
    from stabilis.stats.bias import THEO1_BIAS_MEASURED

    n, trials = 8192, 30
    rng = np.random.default_rng(5)
    acc_t = acc_a = None
    for _ in range(trials):
        ps = power_law_phase(n, alpha, rate_hz=1.0, rng=rng, h_alpha=1e-24)
        t = theo1(ps)
        a = oadev(ps, taus="all", keep_coeffs=False)
        allan_here = np.interp(t.tau_s, a.tau_s, a.var)
        acc_t = t.var if acc_t is None else acc_t + t.var
        acc_a = allan_here if acc_a is None else acc_a + allan_here

    ratio = (acc_t / acc_a)[(t.af >= 100) & (t.tau_s <= a.tau_s[-1])]
    assert THEO1_BIAS_MEASURED[alpha] == expected
    assert np.median(ratio) == pytest.approx(expected, abs=0.07)
