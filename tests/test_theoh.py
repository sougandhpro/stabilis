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

from stabilis.stats import oadev, theo1, theobr, theoh
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
