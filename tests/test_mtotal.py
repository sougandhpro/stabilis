"""MTOT, TTOT and HTOT: the total variances built on 3m subsequences.

The weight here is carried by two things. First, a second implementation:
`allantools` reads the same published equations independently, and agreeing
with it to machine precision is the only interop evidence available for
this family -- no Stable32 fixture for it exists in the repository. Second,
the alpha -> alpha+2 shift between the MTOT and HTOT bias tables, which is
a cross-check on both estimators at once: a normalization error in either
would break it.
"""

import numpy as np
import pytest

from stabilis.stats import (htotdev, mdev, mtotdev, ohdev, oadev, tdev,
                            total_subsequence_terms, ttotdev)
from stabilis.stats.bias import (HTOT_BIAS_MEASURED, MTOT_BIAS_MEASURED,
                                 MTOT_BIAS_PUBLISHED, htot_bias, mtot_bias)
from stabilis.stats.classical import _coeffs_ohvar
from stabilis.synth import power_law_phase
from stabilis.types import PhaseSeries, to_frequency

TAUS = [1, 2, 4, 8, 16, 32]


def record(alpha=0, n=512, seed=5):
    return power_law_phase(n, alpha, rate_hz=1.0,
                           rng=np.random.default_rng(seed), h_alpha=1e-22)


# ---------------------------------------------------------------------------
# the second implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [2, 0, -2])
@pytest.mark.parametrize("name", ["mtotdev", "ttotdev", "htotdev"])
def test_against_allantools(name, alpha):
    """Two independent readings of SP 1065 eq. (27)-(29) agreeing.

    `allantools` is a numeric oracle only -- no code read, nothing copied,
    per the rule in `references.md`. It is not Stable32, so this shows the
    equations were read the same way twice; it does not show that Stable32
    computes the same thing. See theory/mtotal.md for the bias
    disagreement that distinction leaves open.
    """
    at = pytest.importorskip("allantools")
    ps = record(alpha)
    ours = {"mtotdev": mtotdev, "ttotdev": ttotdev, "htotdev": htotdev}[name]
    mine = ours(ps, taus=TAUS, keep_coeffs=False)

    if name == "htotdev":
        got = at.htotdev(to_frequency(ps).y, rate=1.0, data_type="freq",
                         taus=[float(t) for t in TAUS])
    else:
        got = getattr(at, name)(ps.x, rate=1.0, data_type="phase",
                                taus=[float(t) for t in TAUS])
    assert np.allclose(mine.dev, got[1], rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# the algorithm, in pieces
# ---------------------------------------------------------------------------


def test_the_subsequence_block_has_the_published_shape():
    """N-3m+1 subsequences of 6m second differences each, SP 1065 eq. (27)."""
    a = np.arange(100.0)
    for m in (1, 2, 5, 13):
        z = total_subsequence_terms(a, m)
        assert z.shape == (100 - 3 * m + 1, 6 * m)


def test_a_record_shorter_than_one_subsequence_gives_nothing():
    assert total_subsequence_terms(np.arange(8.0), 3).shape == (0, 18)


def test_the_detrend_annihilates_a_linear_ramp_exactly():
    """The subsequence's own frequency offset is removed before anything
    else happens, so a pure ramp must leave second differences of exactly
    zero -- not almost zero. That is the whole first step, isolated."""
    a = 3.0 + 0.25 * np.arange(200.0)
    for m in (2, 3, 7, 16):
        assert np.max(np.abs(total_subsequence_terms(a, m))) < 1e-9


def test_the_extension_is_an_uninverted_mirror():
    """TOTVAR continues the trend across the boundary with `2x1 - x`; this
    family mirrors instead, because it has already removed the trend. The
    difference is visible in the terms: a mirrored, detrended constant-
    curvature block is even about each end, so reversing the record leaves
    every subsequence's set of second differences unchanged."""
    a = np.cumsum(np.cumsum(np.random.default_rng(3).standard_normal(120)))
    for m in (2, 5):
        forward = np.sort(np.abs(total_subsequence_terms(a, m)).ravel())
        backward = np.sort(np.abs(total_subsequence_terms(a[::-1], m)).ravel())
        assert np.allclose(forward, backward, rtol=1e-9)


@pytest.mark.parametrize("fn", [mtotdev, ttotdev, htotdev])
def test_the_estimator_contract_holds(fn):
    """`variance = scale * mean(c**2)`, re-derived from the returned array.

    The published prefactor is `1/(2 (m tau0)**2 (N-3m+1) 6m)`; flattening
    the block and taking the mean of squares reproduces it with nothing
    left over, which is what this asserts.
    """
    r = fn(record(), taus=TAUS)
    for i in range(r.af.size):
        assert r.var[i] == pytest.approx(
            r.scale * float(np.mean(r.coeffs[i] ** 2)), rel=1e-13)


def test_time_total_is_modified_total_rescaled():
    """SP 1065 eq. (28): sigma2_x,total = (tau**2/3) * Mod Tot var."""
    ps = record()
    m_, t_ = mtotdev(ps, taus=TAUS), ttotdev(ps, taus=TAUS)
    assert np.allclose(t_.var, m_.tau_s ** 2 * m_.var / 3.0, rtol=1e-13)


def test_hadamard_total_at_m_equals_one_is_the_overlapping_hadamard():
    """The construction degenerates at m=1 -- three points with one degree
    of freedom removed -- so the convention is OHVAR, and it is an identity
    rather than a substitution: the second difference of frequency *is* the
    third difference of phase there."""
    ps = record()
    assert htotdev(ps, taus=[1]).at(1) == pytest.approx(
        ohdev(ps, taus=[1]).at(1), rel=1e-15)
    assert np.array_equal(htotdev(ps, taus=[1]).coeffs[0],
                          _coeffs_ohvar(ps.x, 1, ps.tau0_s))


def test_frequency_input_gives_the_same_answer_as_phase():
    """One record, two representations. `to_phase` is exact, so the answers
    must agree to rounding, not to a tolerance chosen to make them."""
    ps = record()
    fs = to_frequency(ps)
    for fn in (mtotdev, ttotdev, htotdev):
        a = fn(ps, taus=TAUS, keep_coeffs=False)
        b = fn(fs, taus=TAUS, keep_coeffs=False)
        assert np.allclose(a.var, b.var, rtol=1e-12), fn.__name__


def test_the_tau_grid_and_the_algebraic_limit():
    """m <= N/3: a subsequence is 3m points and there must be one."""
    ps = record(n=300)
    r = mtotdev(ps, taus="all")
    assert int(r.af.max()) <= 300 // 3
    assert r.n_terms[0] == (301 - 3 + 1) * 6   # (N-3m+1) * 6m at m=1


def test_a_record_too_short_is_refused():
    """Three phase points leave no room for even m=1: a subsequence is 3m
    and the driver's algebraic limit is (N-1)//3."""
    ps = PhaseSeries(np.arange(3, dtype=np.float64), rate_hz=1.0)
    with pytest.raises(ValueError, match="no valid averaging factor"):
        mtotdev(ps, taus="all", max_af=100)


def test_the_quadratic_working_set_is_guarded_with_a_number():
    """This family is quadratic in N near m = N/6, unlike everything else
    here, so the driver's own byte budget is a large under-estimate."""
    import stabilis.stats.mtotal as mt
    with pytest.raises(MemoryError, match="quadratic in the record length"):
        mt._guard_size(2_000_000, 300_000, "mtotdev")


# ---------------------------------------------------------------------------
# no interval, deliberately
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [mtotdev, ttotdev, htotdev])
def test_the_confidence_interval_is_refused_with_the_reason(fn):
    """The coefficients are blocks from different subsequences, mostly from
    their reflections, so an autocorrelation over them means nothing. A
    fallback would be worse: eta3 would over-count by orders of magnitude
    in the optimistic direction."""
    from stabilis.stats import confidence
    r = fn(record(), taus=TAUS)
    assert r.coeffs_are_ordered is False
    with pytest.raises(ValueError, match="not one time-ordered sequence"):
        confidence(r)


def test_the_cross_tau_covariance_is_refused_too():
    from stabilis.stats import tau_covariance
    with pytest.raises(ValueError, match="not one time-ordered sequence"):
        tau_covariance(mtotdev(record(), taus=TAUS))


def test_the_ordinary_estimators_are_unaffected():
    """The flag defaults true, so nothing else changed."""
    for fn in (oadev, mdev, tdev, ohdev):
        assert fn(record(), taus=TAUS).coeffs_are_ordered is True


# ---------------------------------------------------------------------------
# bias
# ---------------------------------------------------------------------------


def test_the_htot_bias_table_is_the_mtot_table_shifted_by_two_in_alpha():
    """HTOT applies to the frequency record the operator MTOT applies to the
    phase, and S_x = S_y/(2 pi f)^2, so a y-noise of exponent alpha presents
    to it as an x-noise of alpha+2. A normalization error in either
    estimator would break this, which is why it is worth asserting on the
    tables rather than only measuring each.
    """
    for alpha in (0, -1, -2):
        assert HTOT_BIAS_MEASURED[alpha] == pytest.approx(
            MTOT_BIAS_MEASURED[alpha + 2], rel=2e-3)


@pytest.mark.parametrize("alpha", [2, 1, 0, -1, -2])
def test_the_measured_mtot_bias_is_reproduced(alpha):
    """The table in bias.py, re-measured at a fraction of the trials. The
    band is loose because this is a check that the transcribed number is the
    right one, not a re-derivation of it."""
    got, want = [], []
    for t in range(40):
        ps = power_law_phase(1024, alpha, rate_hz=1.0,
                             rng=np.random.default_rng(7000 + t), h_alpha=1e-22)
        got.append(mtotdev(ps, taus=[16], keep_coeffs=False).var)
        want.append(mdev(ps, taus=[16], keep_coeffs=False).var)
    ratio = float(np.mean(want) / np.mean(got))
    assert ratio == pytest.approx(MTOT_BIAS_MEASURED[alpha], rel=0.05)


def test_the_published_and_measured_tables_are_kept_apart():
    """`CLAUDE.md`: a published value and a measured one mean different
    things and are labelled differently. They disagree by up to 12% here and
    the disagreement is recorded rather than reconciled."""
    assert mtot_bias(0) == MTOT_BIAS_PUBLISHED[0]
    assert mtot_bias(0, published=False) == MTOT_BIAS_MEASURED[0]
    worst = max(abs(MTOT_BIAS_MEASURED[a] / MTOT_BIAS_PUBLISHED[a] - 1.0)
                for a in MTOT_BIAS_PUBLISHED)
    assert 0.10 < worst < 0.15, "the recorded disagreement has moved"
    assert MTOT_BIAS_MEASURED[1] == pytest.approx(MTOT_BIAS_PUBLISHED[1], rel=2e-3)


def test_an_unknown_noise_type_is_refused_by_both_tables():
    with pytest.raises(ValueError, match="no published factor"):
        mtot_bias(-3)
    with pytest.raises(ValueError, match="no measured factor"):
        mtot_bias(-3, published=False)
    with pytest.raises(ValueError, match="measured for alpha in"):
        htot_bias(-3)
