"""The N x N cross matrix, and averaging it to reject instrument noise.

The load-bearing check here is not a number, it is an *identity*: two
channels through `cross_matrix` + `reject` must give bit-for-bit what
`cross_variance` gives on the same pair. They are the same computation
seen from two API shapes, and the M-channel error bar is `isserlis_var`
with more pairs passed in -- not a second derivation that could drift.

After that: the averaging must actually buy something (and less than the
naive `5 log10 P`, because the pairs share channels), the dataset must
refuse what it cannot align, and the error bar must again be the measured
scatter of the estimate.
"""

import numpy as np
import pytest

from stabilis.multi import (CrossMatrix, cross_matrix, cross_variance,
                            dataset, reject)
from stabilis.stats import oadev
from stabilis.synth import power_law_phase
from stabilis.types import PhaseSeries

RATE = 1.0
H_CLOCK = 1e-24
H_INSTRUMENT = 1e-22


def channels(rng, n, m_ch, h_clock=H_CLOCK, h_inst=H_INSTRUMENT, alpha_inst=2):
    """One clock through `m_ch` channels with independent noise, plus the clock."""
    clock = power_law_phase(n, 0, RATE, rng, h_clock)
    ch = {
        f"c{i}": PhaseSeries(
            x=clock.x + power_law_phase(n, alpha_inst, RATE, rng, h_inst).x,
            rate_hz=RATE)
        for i in range(m_ch)
    }
    return ch, clock


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------

def test_two_channels_reject_to_exactly_the_pairwise_cross():
    """One pair is `cross_variance`, to the last bit, including the error bar.

    This is the test that keeps the two entry points one implementation. If
    it ever fails by a little, the M-channel path has grown its own
    derivation of something the pairwise path already knew.
    """
    rng = np.random.default_rng(1)
    ch, _ = channels(rng, 1 << 11, 2)
    both = reject(cross_matrix(ch, max_af=64))
    pair = cross_variance(ch["c0"], ch["c1"], max_af=64)
    assert np.array_equal(both.var, pair.var)
    assert np.array_equal(both.sem, pair.sem)
    assert np.allclose(both.eta, pair.eta, rtol=1e-12)
    assert both.n_pairs == 1
    assert np.allclose(both.floor, np.sqrt(pair.auto_a * pair.auto_b))


def test_the_matrix_diagonal_is_each_channel_own_variance():
    """`mean(c*c)` is `mean(c**2)`; no tolerance is needed or allowed."""
    rng = np.random.default_rng(2)
    ch, _ = channels(rng, 1 << 10, 3)
    mx = cross_matrix(ch, max_af=32)
    for i, name in enumerate(mx.names):
        assert np.array_equal(mx.auto[:, i],
                              oadev(ch[name], max_af=32).var)
    assert np.allclose(np.diagonal(mx.correlation, axis1=1, axis2=2), 1.0)


def test_the_matrix_is_symmetric_and_named():
    rng = np.random.default_rng(3)
    ch, _ = channels(rng, 1 << 10, 3)
    mx = cross_matrix(ch, max_af=32)
    assert mx.names == ("c0", "c1", "c2")
    assert np.array_equal(mx.var, np.transpose(mx.var, (0, 2, 1)))
    assert np.array_equal(mx.pair("c0", "c2"), mx.pair("c2", "c0"))
    assert np.array_equal(mx.pair("c0", "c1"),
                          cross_variance(ch["c0"], ch["c1"], max_af=32).var)
    with pytest.raises(KeyError, match="no channel named"):
        mx.pair("c0", "nope")


def test_every_pair_of_the_matrix_is_its_own_cross_variance():
    """The matrix is not a separate estimator, it is the pairwise one N times."""
    rng = np.random.default_rng(4)
    ch, _ = channels(rng, 1 << 10, 4)
    mx = cross_matrix(ch, max_af=16)
    for i, a in enumerate(mx.names):
        for b in mx.names[i + 1:]:
            assert np.allclose(mx.pair(a, b),
                               cross_variance(ch[a], ch[b], max_af=16).var,
                               rtol=1e-14)


# ---------------------------------------------------------------------------
# What the extra channels buy
# ---------------------------------------------------------------------------

def test_more_channels_reject_deeper_but_not_as_deep_as_naive_counting():
    """The averaging gain is real, and smaller than `5 log10 P`.

    Six pairs from four channels would be 3.9 dB if the pairs were
    independent. They are not -- (0,1) and (0,2) share channel 0 -- and the
    exact covariance says so. Both halves are asserted: a gain that
    vanished would mean the averaging does nothing, and a gain that reached
    the naive figure would mean the shared-channel terms had been dropped.
    """
    rng = np.random.default_rng(5)
    ch, _ = channels(rng, 1 << 13, 4)
    four = reject(cross_matrix(ch, max_af=8))
    two = reject(cross_matrix({k: ch[k] for k in ("c0", "c1")}, max_af=8))
    gain = four.rejection_db - two.rejection_db
    naive = 5.0 * np.log10(four.n_pairs)
    assert four.n_pairs == 6
    assert np.all(gain > 0.5), f"averaging bought nothing: {gain}"
    assert np.all(gain < naive), f"gain {gain} reached the naive {naive:.2f} dB"


def test_rejection_recovers_a_clock_buried_under_four_channels():
    rng = np.random.default_rng(6)
    ch, clock = channels(rng, 1 << 13, 4)
    r = reject(cross_matrix(ch, max_af=8))
    truth = oadev(clock, max_af=8).var
    assert np.all(r.floor > 1.5 * truth)
    assert np.allclose(r.var, truth, rtol=0.15)
    assert np.all(r.rejection_db > 12.0)


def test_independent_channels_reject_to_nothing_and_go_negative():
    rng = np.random.default_rng(7)
    ch = {f"c{i}": power_law_phase(1 << 12, 0, RATE, rng, H_CLOCK)
          for i in range(3)}
    r = reject(cross_matrix(ch))
    assert r.negative.any()
    assert np.all(np.isnan(r.dev[r.negative]))
    assert np.all(np.abs(r.var) < 3.0 * r.sem)


def test_a_loud_channel_is_flagged_rather_than_silently_averaged_in():
    """Unweighted averaging is a choice, so the case where it costs is named."""
    rng = np.random.default_rng(8)
    clock = power_law_phase(1 << 12, 0, RATE, rng, H_CLOCK)
    ch = {
        "quiet1": PhaseSeries(x=clock.x + power_law_phase(1 << 12, 2, RATE, rng, 1e-23).x, rate_hz=RATE),
        "quiet2": PhaseSeries(x=clock.x + power_law_phase(1 << 12, 2, RATE, rng, 1e-23).x, rate_hz=RATE),
        "loud": PhaseSeries(x=clock.x + power_law_phase(1 << 12, 2, RATE, rng, 1e-19).x, rate_hz=RATE),
    }
    r = reject(cross_matrix(ch, max_af=16))
    assert any("differ by up to" in n for n in r.notes)
    assert any("loud" in n for n in r.notes)


def test_two_channels_are_told_there_is_no_averaging_gain():
    rng = np.random.default_rng(9)
    ch, _ = channels(rng, 1 << 10, 2)
    r = reject(cross_matrix(ch, max_af=16))
    assert any("no averaging gain" in n for n in r.notes)


def test_the_independence_assumption_is_recorded_and_names_the_bench():
    """M channels of one clock are usually M channels of one instrument."""
    rng = np.random.default_rng(10)
    ch, _ = channels(rng, 1 << 10, 3)
    r = reject(cross_matrix(ch, max_af=16))
    steps = {s.action: s.outcome for s in r.provenance.steps}
    assert "independence_assumed" in steps
    assert "common_mode" in steps["independence_assumed"]


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------

def test_a_dataset_crops_and_unions_gaps_across_every_member():
    """The union is over all members, not pairwise. That is the whole point."""
    x = np.arange(64.0)
    a, b, c = (np.array(x), np.array(x), np.array(x))
    a[10] = np.nan
    b[20] = np.nan
    c[30] = np.nan
    ds = dataset({
        "a": PhaseSeries(x=a, rate_hz=RATE),
        "b": PhaseSeries(x=b, rate_hz=RATE),
        "c": PhaseSeries(x=c, rate_hz=RATE),
    })
    missing = ~np.isfinite(ds["a"].x)
    assert list(np.flatnonzero(missing)) == [10, 20, 30]
    for name in ds.names:
        assert np.array_equal(~np.isfinite(ds[name].x), missing)
    assert ds.n == 64 and ds.names == ("a", "b", "c") and len(ds) == 3


def test_a_dataset_preserves_insertion_order():
    """The matrix axes are reported by name; they must not move between runs."""
    x = np.arange(32.0)
    names = ("zulu", "alpha", "mike")
    ds = dataset({n: PhaseSeries(x=x, rate_hz=RATE) for n in names})
    assert ds.names == names
    assert cross_matrix(ds, max_af=4).names == names


def test_a_dataset_crops_to_the_common_window():
    x = np.arange(100.0)
    ds = dataset({
        "early": PhaseSeries(x=x, rate_hz=RATE, epoch=0.0),
        "late": PhaseSeries(x=x, rate_hz=RATE, epoch=10.0),
    })
    assert ds.n == 90 and ds.epoch == 10.0
    assert ds.alignment.offsets == (10, 0)
    assert np.array_equal(ds["early"].x, x[10:])


def test_a_dataset_needs_two_records():
    with pytest.raises(ValueError, match="at least two records"):
        dataset({"only": PhaseSeries(x=np.arange(16.0), rate_hz=RATE)})


def test_a_dataset_refuses_what_alignment_refuses():
    x = np.arange(64.0)
    with pytest.raises(ValueError, match="off the common grid"):
        dataset({"a": PhaseSeries(x=x, rate_hz=RATE, epoch=0.0),
                 "b": PhaseSeries(x=x, rate_hz=RATE, epoch=2.5)})


def test_the_dataset_carries_what_the_cropping_cost():
    """A user asking why a six-channel solve used half their record."""
    x = np.arange(200.0)
    ds = dataset({"a": PhaseSeries(x=x, rate_hz=RATE, epoch=0.0),
                  "b": PhaseSeries(x=x[:150], rate_hz=RATE, epoch=50.0)})
    assert ds.alignment.offsets == (50, 0)
    assert ds.alignment.trimmed == (0, 0)
    assert ds.n == 150


def test_a_matrix_without_coefficients_refuses_to_be_rejected():
    """`reject` needs the coefficient arrays; it will not guess an error bar."""
    rng = np.random.default_rng(11)
    ch, _ = channels(rng, 1 << 9, 2)
    mx = cross_matrix(ch, max_af=8)
    from dataclasses import replace
    with pytest.raises(ValueError, match="no coefficient arrays"):
        reject(replace(mx, coeffs=()))


def test_a_statistic_that_is_not_a_variance_has_no_matrix():
    rng = np.random.default_rng(12)
    ch, _ = channels(rng, 1 << 9, 2)
    with pytest.raises(ValueError, match="no cross form"):
        cross_matrix(ch, "mtie", max_af=8)


def test_a_short_record_falls_back_and_says_which():
    rng = np.random.default_rng(13)
    ch, _ = channels(rng, 40, 3)
    r = reject(cross_matrix(ch, taus="all", max_af=20))
    assert set(r.edof_method) <= {"isserlis", "eta3"}
    assert r.edof_method[0] == "isserlis"
    assert r.edof_method[-1] == "eta3"
    assert any("fell back" in n or "conservative" in n for n in r.notes)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("m_ch", [3, 4])
def test_the_rejection_error_bar_is_the_scatter_of_the_estimate(m_ch):
    """`sem` against Monte-Carlo truth, over the whole tau range.

    The band is the same +/-20% the pairwise calibration uses, and for the
    same reason: 250 realizations determine a standard deviation to about
    4.5%, so the band is several times that and still tight enough to catch
    a factor. It is the exact Isserlis sum being checked here -- the
    equal-shape closed form ran 0.81-0.88 in the middle of this range,
    which is what the exact one was adopted to fix.
    """
    rng = np.random.default_rng(100 + m_ch)
    var, sem = [], []
    for _ in range(250):
        ch, _ = channels(rng, 1 << 12, m_ch)
        r = reject(cross_matrix(ch, max_af=256))
        var.append(r.var)
        sem.append(r.sem)
    ratio = np.array(sem).mean(axis=0) / np.array(var).std(axis=0, ddof=1)
    for m, x in zip(r.af, ratio):
        assert 0.8 < x < 1.25, f"M={m_ch}, af={m}: sem/scatter = {x:.3f}"
