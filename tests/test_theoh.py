"""ThêoBR and ThêoH.

The bias ratio is the whole content of ThêoBR, so most of this is about
that number: that its terms are paired at equal tau, that it recovers the
published Table 2 asymptotes over an ensemble, and that it is applied as a
single common factor. ThêoH is then a splice, and the tests check it is
exactly a splice rather than something recomputed.
"""

import importlib

import numpy as np
import pytest

from stabilis.stats import confidence, oadev, theo1, theobr, theoh
from stabilis.stats.bias import THEO1_BIAS_PARAMS, theo1_bias
from stabilis.stats.theo1 import theo1_taus
from stabilis.stats.theoh import bias_ratio, bias_terms
from stabilis.synth import power_law_phase
from stabilis.types import FreqSeries


def test_the_bias_terms_are_paired_at_equal_tau():
    """`0.75(12+4i) = 9+3i`, which is why eq. (33) is a ratio at all.

    A bias factor is `AVAR/Thêo1` *at the same averaging time*. Under the
    other tau reading SP 1065 prints, `0.75(m-1)tau0`, the Thêo1 term would
    land at `8.25+3i` and pair with nothing -- see theory/theo1.md.
    """
    allan, theo = bias_terms(600)
    assert allan.tolist() == [9 + 3 * i for i in range(allan.size)]
    assert theo.tolist() == [12 + 4 * i for i in range(theo.size)]
    assert allan.size == 600 // 6 - 3 + 1
    assert np.array_equal(theo1_taus(theo, 1.0), allan.astype(float))
    assert all(m % 2 == 0 for m in theo), "Thêo1 needs even m"
    assert theo[0] >= 10, "and at least ten"


def test_too_short_a_record_is_refused_by_the_bias_average():
    with pytest.raises(ValueError, match="bias average"):
        bias_terms(12)


@pytest.mark.slow
@pytest.mark.parametrize("alpha", [0, -1])
def test_the_bias_ratio_recovers_the_published_asymptote(alpha):
    """Eq. (33) measured from data against Table 2's `a`, over an ensemble.

    `a` is the large-m limit of `AVAR/Thêo1`, so the eq. (33) average --
    which is dominated by its large-m terms -- should land near it without
    ever being told the noise type. Single records will not: see
    `test_the_bias_ratio_is_noisy_on_one_record`.

    Random-walk FM is left out on purpose. Its measured mean was 2.30
    against a published 2.70, inside the scatter of a dozen draws of that
    noise but not a check that would fail for the right reason.
    """
    published = THEO1_BIAS_PARAMS[alpha][0]
    ratios = [bias_ratio(power_law_phase(1024, alpha, rate_hz=1.0,
                                         rng=np.random.default_rng(900 + s)))[0]
              for s in range(12)]
    assert np.mean(ratios) == pytest.approx(published, rel=0.15)


@pytest.mark.slow
def test_the_bias_ratio_is_noisy_on_one_record():
    """The caveat that matters more than the mean.

    The average is dominated by AVAR terms at `m` up to `N/2`, which have a
    handful of degrees of freedom however long the record is. So a single
    record's ratio is a wide draw around the truth -- and ThêoBR multiplies
    every tau by that one number, making the error a common vertical shift
    of the whole curve rather than scatter between points.
    """
    ratios = [bias_ratio(power_law_phase(1024, 0, rate_hz=1.0,
                                         rng=np.random.default_rng(900 + s)))[0]
              for s in range(12)]
    assert min(ratios) < 0.7 and max(ratios) > 1.4, "the spread is real"
    assert np.mean(ratios) == pytest.approx(1.0, rel=0.15), "but centred"


def test_theobr_is_theo1_times_the_ratio_and_records_it():
    """Nothing else about the estimator changes -- same taus, same terms."""
    ps = power_law_phase(600, -1, rate_hz=1.0, rng=np.random.default_rng(3))
    ratio, terms = bias_ratio(ps)
    base, corrected = theo1(ps), theobr(ps)

    assert np.array_equal(base.af, corrected.af)
    assert np.array_equal(base.tau_s, corrected.tau_s)
    assert np.array_equal(base.n_terms, corrected.n_terms)
    assert np.allclose(corrected.var, base.var * ratio, rtol=1e-15, atol=0.0)

    step = next(s for s in corrected.provenance.steps if s.action == "theobr")
    assert step.params["bias_ratio"] == pytest.approx(ratio)
    assert step.params["bias_terms"] == terms


def test_theoh_is_avar_below_the_crossover_and_theobr_above_it():
    """Eq. (34) is a splice, so it must be exactly the two halves.

    Recomputing either half differently -- a different tau grid, a
    different gap policy -- would make ThêoH a third estimator that happens
    to resemble both, which is not what the equation says.
    """
    ps = power_law_phase(1024, -1, rate_hz=1.0, rng=np.random.default_rng(4))
    h = theoh(ps)
    allan, upper = oadev(ps, keep_coeffs=False), theobr(ps)

    step = next(s for s in h.provenance.steps if s.action == "theoh")
    k = step.params["crossover_s"]
    record_s = (ps.n - 1) * ps.tau0_s
    assert k <= 0.2 * record_s, "the join is at or below 20% of the record"
    assert allan.tau_s[allan.tau_s <= 0.2 * record_s][-1] == k, "the largest such"

    low, high = h.tau_s <= k, h.tau_s > k
    assert np.array_equal(h.var[low], allan.var[allan.tau_s <= k])
    assert np.array_equal(h.var[high], upper.var[upper.tau_s > k])
    assert step.params["allan_points"] == int(low.sum())
    assert step.params["theobr_points"] == int(high.sum())


def test_theoh_is_one_rising_tau_axis_reaching_three_quarters():
    """The point of the hybrid: a single curve over the whole range."""
    ps = power_law_phase(1024, -1, rate_hz=1.0, rng=np.random.default_rng(5))
    h = theoh(ps)
    record_s = (ps.n - 1) * ps.tau0_s
    assert np.all(np.diff(h.tau_s) > 0), "no backwards step at the join"
    assert h.tau_s[0] == pytest.approx(ps.tau0_s)
    assert h.tau_s[-1] == pytest.approx(0.75 * record_s)
    assert h.tau_s[-1] > 2.5 * oadev(ps, keep_coeffs=False).tau_s[-1]


@pytest.mark.parametrize("fn", [theobr, theoh])
def test_frequency_input_is_refused(fn):
    with pytest.raises(TypeError, match="needs phase data"):
        fn(FreqSeries(np.zeros(400), rate_hz=1.0))


def test_the_cost_guard_fires_before_a_long_record_does(monkeypatch):
    """It grows as N^3, so the guard has to come before the work, not after.

    Forced with a low ceiling rather than a huge record, because a test
    that actually triggered it at N = 6500 would take the half minute the
    guard exists to prevent.
    """
    # By module object, not by name: `stats/__init__` binds the *function*
    # `theoh` over the submodule, exactly as it does for `theo1`.
    mod = importlib.import_module("stabilis.stats.theoh")

    monkeypatch.setattr(mod, "MAX_BIAS_OPS", 1e3)
    ps = power_law_phase(600, 0, rate_hz=1.0, rng=np.random.default_rng(6))
    with pytest.raises(ValueError, match="past the .* limit"):
        mod.theobr(ps)


def test_theo1_bias_matches_table_2():
    """The published route, for when the noise type *is* known."""
    assert theo1_bias(0, 10) == 1.0 and theo1_bias(0, 10_000) == 1.0
    # a + b/m^c, read straight off the table
    assert theo1_bias(-1, 100) == pytest.approx(1.87 - 1.05 / 100**0.79)
    assert theo1_bias(-2, 50) == pytest.approx(2.70 - 1.53 / 50**0.85)
    # and it approaches `a` from below for the FM noises
    for alpha in (-1, -2):
        a = THEO1_BIAS_PARAMS[alpha][0]
        assert theo1_bias(alpha, 10) < theo1_bias(alpha, 1000) < a
    assert np.allclose(theo1_bias(1, np.array([10.0, 100.0])),
                       0.14 + 0.82 / np.array([10.0, 100.0]) ** 0.30)
    with pytest.raises(ValueError, match="Table 2 covers"):
        theo1_bias(-3, 10)


# ---------------------------------------------------------------------------
# SP 1065 Table 9: the confidence interval the family did not have
# ---------------------------------------------------------------------------


def _phase(n, alpha, seed):
    return power_law_phase(n, alpha, rate_hz=1.0,
                           rng=np.random.default_rng(seed), h_alpha=1e-22)


def test_the_edf_falls_with_averaging_factor_and_rises_with_record_length():
    """The two monotonicities a degrees-of-freedom formula must have, and
    the one place Table 9 does not have the first.

    Every row is a falling term times a roll-off `t^p/(t^p+c)`. In the
    white-PM row the falling term is nearly constant while `t << nx`, so
    the roll-off wins and the edf *rises* until `t` approaches `nx` -- at
    N = 4001 it peaks near m = 200. That is the published formula, not a
    transcription slip, so it is asserted rather than worked around.
    """
    from stabilis.stats.edof import theo1_edof
    for alpha in (1, 0, -1, -2):
        by_m = [theo1_edof(alpha, m, 4001) for m in (10, 40, 160, 640)]
        assert by_m == sorted(by_m, reverse=True), alpha

    white_pm = [theo1_edof(2, m, 4001) for m in (10, 40, 160, 640)]
    assert white_pm[0] < white_pm[1] < white_pm[2], "the roll-off dominates"
    assert white_pm[3] < white_pm[2], "and gives way once t approaches nx"

    for alpha in (2, 1, 0, -1, -2):
        by_n = [theo1_edof(alpha, 100, n) for n in (301, 1001, 4001)]
        assert by_n == sorted(by_n), alpha


def test_the_edf_is_ordered_by_noise_type():
    """Whiter noise decorrelates faster, so it carries more independent
    information at the same tau. White PM highest, random-walk FM lowest --
    the same ordering every EDOF model in this library shows."""
    from stabilis.stats.edof import theo1_edof
    got = [theo1_edof(a, 100, 1001) for a in (2, 1, 0, -1, -2)]
    assert got == sorted(got, reverse=True)


def test_the_edf_is_floored_at_one_rather_than_going_negative():
    """Table 9's white-PM and flicker-PM rows have `nx - t` in a
    denominator, and Thêo1 runs out to t = 0.75(N-1). A fitted
    approximation is allowed to be absurd there; the library is not."""
    from stabilis.stats.edof import theo1_edof
    for alpha in (2, 1, 0, -1, -2):
        assert theo1_edof(alpha, 26, 30) >= 1.0
        assert theo1_edof(alpha, 1000, 30) == 1.0


def test_an_unlisted_noise_type_is_refused():
    from stabilis.stats.edof import theo1_edof
    with pytest.raises(ValueError, match="Table 9 covers alpha in"):
        theo1_edof(-3, 10, 1001)


@pytest.mark.parametrize("fn", [theo1, theobr, theoh])
def test_the_theo_family_now_has_an_interval(fn):
    """It had none at all until Table 9 was implemented: Thêo1 is a sum over
    a triangular set of lag pairs, not a mean of one coefficient per term,
    so there is no array for the data-driven path to read."""
    ps = _phase(1024, 0, seed=11)
    r = fn(ps, taus="octave")
    c = confidence(r, method="theo1", alpha=0)
    assert np.all(c.var_lo <= r.var) and np.all(r.var <= c.var_hi)
    assert np.all(c.edof >= 1.0)
    assert np.all(np.diff(c.edof[-8:]) < 0), "edf must fall at the long end"


def test_theoh_uses_two_models_and_records_which():
    """ThêoH is two estimators spliced, and one formula would be wrong on
    half the curve: Greenhall's overlapping-Allan model below the crossover,
    Table 9 above it."""
    ps = _phase(1024, 0, seed=12)
    r = theoh(ps, taus="octave")
    c = confidence(r, method="theo1", alpha=0)
    crossover = next(s.params["crossover_s"] for s in r.provenance.steps
                     if s.action == "theoh")
    for tau, how in zip(r.tau_s, c.method):
        if tau <= crossover:
            assert how in ("greenhall", "eta3"), tau
        else:
            assert how == "theo1", tau
    assert "theo1" in set(c.method) and "greenhall" in set(c.method)


def test_theo1_and_theobr_get_the_same_degrees_of_freedom():
    """SP 1065: the table is for "Thêo1, hence, ThêoBR and ThêoH". ThêoBR
    rescales Thêo1 by a measured ratio; a deterministic rescaling does not
    change how much data the estimate rests on."""
    ps = _phase(1024, -1, seed=13)
    a = confidence(theo1(ps, taus="octave"), method="theo1", alpha=-1)
    b = confidence(theobr(ps, taus="octave"), method="theo1", alpha=-1)
    assert np.array_equal(a.edof, b.edof)


def test_the_interval_beats_the_allan_one_at_the_same_tau():
    """The entire point of Thêo1 is more confidence at long tau. If the
    interval did not come out tighter than the overlapping Allan one at the
    same averaging time, the statistic would have no reason to exist."""
    ps = _phase(2048, 0, seed=14)
    t = theo1(ps, taus="octave")
    ct = confidence(t, method="theo1", alpha=0)
    a = oadev(ps, taus="all", max_af=700)
    ca = confidence(a, method="greenhall", alpha=0)

    # Compare at the longest tau Thêo1 reports that AVAR also reaches.
    shared = t.tau_s[t.tau_s <= a.tau_s.max()]
    tau = float(shared[-1])
    i = int(np.argmin(np.abs(t.tau_s - tau)))
    j = int(np.argmin(np.abs(a.tau_s - tau)))
    assert ct.edof[i] > ca.edof[j], (ct.edof[i], ca.edof[j])
    assert ct.var_hi[i] / ct.var_lo[i] < ca.var_hi[j] / ca.var_lo[j]


def test_the_noise_type_is_not_optional_here():
    """Table 9 is the only route and every row of it needs an alpha, so
    `confidence` must ask rather than assume one."""
    ps = _phase(512, 0, seed=15)
    with pytest.raises(ValueError, match="assumes a power law"):
        confidence(theo1(ps, taus="octave"), method="theo1")
