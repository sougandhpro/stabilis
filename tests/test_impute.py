"""The Total Imputer, and the confidence interval that refuses to believe it.

Two of these carry most of the weight. The first is the paper's own figure
of merit -- the fraction of trials in which the filled AVAR's 90% interval
overlaps the original's -- which is what settled the phase-versus-frequency
ambiguity in Howe & Schlossberger's description: on phase, random-walk FM
comes back 74x high and the criterion collapses to 3 trials in 100. The
second is that filling a record can never narrow its error bar, which is
the whole reason the imputed mask is returned rather than discarded.
"""

import numpy as np
import pytest

from stabilis.impute import _lowpass, fill_gaps, live_terms
from stabilis.stats import confidence, oadev
from stabilis.synth import power_law_freq
from stabilis.types import FreqSeries, PhaseSeries


def gapped(n=513, alpha=0, seed=7, lo=250, hi=400, h_alpha=1e-22):
    """The paper's scenario: 513 points, 250..400 removed, 29% dead time."""
    fs = power_law_freq(n, alpha, rate_hz=1.0,
                        rng=np.random.default_rng(seed), h_alpha=h_alpha)
    y = fs.y.copy()
    y[lo:hi] = np.nan
    return fs, FreqSeries(y, rate_hz=1.0)


# ---------------------------------------------------------------------------
# the identities, which cost nothing and would catch a lot
# ---------------------------------------------------------------------------


def test_a_record_with_no_gaps_comes_back_bit_identical():
    """And with an empty mask, so nothing downstream has to special-case it."""
    fs = power_law_freq(256, 0, rate_hz=1.0, rng=np.random.default_rng(1))
    filled, mask = fill_gaps(fs, reason="nothing to do")
    assert np.array_equal(filled.y, fs.y)
    assert not mask.any()


def test_the_fill_is_invariant_to_tau0_as_bit_equality():
    """The paper's eq. 3 reads `f` as a Fourier frequency and `T` as a
    duration, which would make the endpoint filter depend on whether the
    user called the sample spacing seconds or milliseconds. It is
    implemented with `f` a bin index instead, so it does not -- and the
    invariance is asserted as bit equality, not a tolerance."""
    _, g = gapped()
    a, _ = fill_gaps(g, reason="rate a")
    b, _ = fill_gaps(FreqSeries(g.y.copy(), rate_hz=1e6), reason="rate b")
    assert np.array_equal(a.y, b.y)


def test_live_samples_are_never_modified():
    """A fill that touched a measurement would not be a fill."""
    fs, g = gapped()
    filled, mask = fill_gaps(g, reason="check")
    assert np.array_equal(filled.y[~mask], fs.y[~mask])
    assert np.isfinite(filled.y).all()
    assert mask.sum() == 150


def test_the_mask_marks_exactly_what_was_missing():
    fs, g = gapped()
    _, mask = fill_gaps(g, reason="check")
    assert np.array_equal(mask, ~np.isfinite(g.y))


def test_the_lowpass_preserves_the_mean_and_only_removes_the_top_end():
    """Bin 0 is untouched, so the block's level survives; the Nyquist bin is
    down by exp(-4). A filter that moved the mean would bias every match."""
    rng = np.random.default_rng(2)
    block = rng.standard_normal(128) + 5.0
    out = _lowpass(block)
    assert out.mean() == pytest.approx(block.mean(), rel=1e-12)
    assert out.std() < 0.5 * block.std()


def test_multiple_gaps_are_all_filled_and_the_order_is_outward():
    """Every source block must be live measurement rather than something
    imputed at one remove, which is what the longest-run ordering buys."""
    fs = power_law_freq(600, 0, rate_hz=1.0, rng=np.random.default_rng(4))
    y = fs.y.copy()
    for lo, hi in ((40, 60), (120, 150), (400, 430), (520, 530)):
        y[lo:hi] = np.nan
    filled, mask = fill_gaps(FreqSeries(y, rate_hz=1.0), reason="four gaps")
    assert np.isfinite(filled.y).all()
    assert mask.sum() == 20 + 30 + 30 + 10
    assert np.array_equal(filled.y[~mask], fs.y[~mask])


def test_the_reason_and_the_imputed_fraction_reach_provenance():
    """The paper asks that gaps and their cause be stated with the results,
    so a fill that did not say so would not be usable evidence."""
    _, g = gapped()
    filled, _ = fill_gaps(g, reason="receiver reboot 04:12")
    step = next(s for s in filled.provenance.steps if s.action == "fill_gaps")
    assert step.params["reason"] == "receiver reboot 04:12"
    assert step.params["imputed"] == 150
    assert step.params["imputed_fraction"] == pytest.approx(150 / 513)
    assert "imputed" in step.outcome and "not measured" in step.outcome


# ---------------------------------------------------------------------------
# the refusals
# ---------------------------------------------------------------------------


def test_a_phase_record_is_refused_with_the_reason():
    """The phase across a gap is not in the data even once the frequency
    samples are: a counter that stopped carries an unknown phase step."""
    x = np.arange(20.0)
    x[5:8] = np.nan
    with pytest.raises(TypeError, match="fills frequency samples"):
        fill_gaps(PhaseSeries(x, rate_hz=1.0), reason="no")


def test_a_gap_at_either_end_is_refused_rather_than_extrapolated():
    """It has live data on one side only, and it bridges nothing. Trimming
    loses no live data, which extrapolating into it cannot claim."""
    fs = power_law_freq(200, 0, rate_hz=1.0, rng=np.random.default_rng(5))
    for lo, hi in ((0, 10), (190, 200)):
        y = fs.y.copy()
        y[lo:hi] = np.nan
        with pytest.raises(ValueError, match="reaches an end of the record"):
            fill_gaps(FreqSeries(y, rate_hz=1.0), reason="end gap")


def test_a_gap_with_too_little_before_it_is_filled_from_the_other_side():
    """The paper's "or the following n measurements if there is not enough
    data before the gap" is not a special case here -- it falls out of the
    outward ordering. The longest run is to the *right* of this gap, so the
    reversed pass reaches it with 150 live samples in hand."""
    fs = power_law_freq(300, 0, rate_hz=1.0, rng=np.random.default_rng(6))
    y = fs.y.copy()
    y[20:150] = np.nan          # 130 wide, only 20 live samples to its left
    filled, mask = fill_gaps(FreqSeries(y, rate_hz=1.0), reason="from the right")
    assert np.isfinite(filled.y).all()
    assert np.array_equal(filled.y[~mask], fs.y[~mask])


def test_a_gap_wider_than_the_live_data_on_either_side_is_refused():
    """The method extends live data into a gap, so it needs at least as much
    of it as the gap is wide -- and here neither flank has enough. The
    paper's remedy, imputing from both sides and matching averages in the
    middle, is in a conference paper we do not have, so it is refused
    rather than improvised."""
    fs = power_law_freq(300, 0, rate_hz=1.0, rng=np.random.default_rng(6))
    y = fs.y.copy()
    y[100:220] = np.nan         # 120 wide, with only 100 left and 80 right
    with pytest.raises(ValueError, match="live samples precede it"):
        fill_gaps(FreqSeries(y, rate_hz=1.0), reason="too wide")


def test_a_blank_reason_is_refused():
    _, g = gapped()
    with pytest.raises(ValueError, match="reason is required"):
        fill_gaps(g, reason="")


# ---------------------------------------------------------------------------
# the confidence interval
# ---------------------------------------------------------------------------


def test_filling_never_narrows_the_interval():
    """Imputation raises the naive EDOF -- the paper measures up to 6x for
    white noise -- and an interval built on that is too narrow in the
    optimistic direction. This is what `live_terms` exists to stop."""
    _, g = gapped(alpha=0)
    filled, mask = fill_gaps(g, reason="check")
    r = oadev(filled, taus="octave")
    naive = confidence(r)
    honest = confidence(r, live_terms=live_terms(r, mask))

    finite = np.isfinite(honest.var_lo)
    assert finite.any()
    assert np.all(honest.edof[finite] <= naive.edof[finite])
    widened = honest.var_hi[finite] / honest.var_lo[finite]
    assert np.all(widened >= naive.var_hi[finite] / naive.var_lo[finite])


def test_no_interval_where_every_term_rests_on_imputed_data():
    """A chi-square quantile at zero degrees of freedom is not a wide
    interval, it is no interval. The point estimate survives; the error bar
    does not, and provenance names the averaging factors."""
    _, g = gapped()
    filled, mask = fill_gaps(g, reason="check")
    r = oadev(filled, taus="octave")
    counts = live_terms(r, mask)
    assert counts[-1] == 0, "the coarsest tau should be entirely imputed here"

    c = confidence(r, live_terms=counts)
    assert np.isnan(c.var_lo[-1]) and np.isnan(c.var_hi[-1])
    assert np.isfinite(c.var_lo[:-1]).all()
    step = next(s for s in c.provenance.steps
                if s.params.get("no_live_terms_at_af"))
    assert int(r.af[-1]) in step.params["no_live_terms_at_af"]


def test_live_terms_counts_windows_free_of_imputed_samples():
    """Directly, against a hand-countable case: span_factor*m samples per
    term, so on a 20-point record with one imputed sample at index 10 the
    m=1 Allan term at offset t is live unless t is 9 or 10."""
    fs = power_law_freq(20, 0, rate_hz=1.0, rng=np.random.default_rng(8))
    r = oadev(fs, taus="all")
    mask = np.zeros(20, bool)
    mask[10] = True
    counts = live_terms(r, mask)
    assert r.span_factor == 2
    assert counts[0] == int(r.n_terms[0]) - 2


def test_live_terms_of_an_unfilled_record_changes_nothing():
    """An all-false mask must give back the term counts unchanged, so the
    argument can be passed unconditionally by a caller that does not know
    whether the record was filled."""
    fs = power_law_freq(256, 0, rate_hz=1.0, rng=np.random.default_rng(9))
    r = oadev(fs, taus="octave")
    counts = live_terms(r, np.zeros(fs.n, bool))
    assert np.array_equal(counts, r.n_terms)
    a = confidence(r)
    b = confidence(r, live_terms=counts)
    assert np.array_equal(a.edof, b.edof)
    assert np.array_equal(a.var_lo, b.var_lo)


def test_a_mismatched_live_count_is_refused():
    fs = power_law_freq(256, 0, rate_hz=1.0, rng=np.random.default_rng(10))
    r = oadev(fs, taus="octave")
    with pytest.raises(ValueError, match="entries but the result has"):
        confidence(r, live_terms=np.ones(3))
    with pytest.raises(ValueError, match="exceeds n_terms"):
        confidence(r, live_terms=r.n_terms + 1)


# ---------------------------------------------------------------------------
# the paper's acceptance test
# ---------------------------------------------------------------------------


# Howe & Schlossberger Fig. 4: trials in 100 whose filled 90% AVAR interval
# overlapped the original's, at tau/tau0 = 1..128, for a 513-point record
# with points 250-400 removed. Read off the published figure.
PUBLISHED_OVERLAP = {
    2: (96, 98, 96, 94, 84, 34, 11, 20),
    0: (95, 93, 96, 97, 94, 84, 59, 63),
    -1: (94, 97, 97, 97, 98, 95, 84, 87),
    -2: (92, 99, 97, 96, 94, 98, 100, 98),
}


def _overlap_counts(alpha, trials=40):
    """Trials whose filled 90% interval meets the original's, per tau."""
    rng = np.random.default_rng(50 + alpha)
    hits = None
    for _ in range(trials):
        fs = power_law_freq(513, alpha, rate_hz=1.0, rng=rng, h_alpha=1e-22)
        y = fs.y.copy()
        y[250:400] = np.nan
        filled, _ = fill_gaps(FreqSeries(y, rate_hz=1.0), reason="acceptance")
        a = confidence(oadev(fs, taus="octave"), level=0.90)
        b = confidence(oadev(filled, taus="octave"), level=0.90)
        hit = (a.var_lo <= b.var_hi) & (b.var_lo <= a.var_hi)
        hits = hit.astype(int) if hits is None else hits + hit
    return hits / trials


@pytest.mark.parametrize("alpha", [2, 0, -1, -2])
def test_the_papers_own_acceptance_criterion_at_short_tau(alpha):
    """At tau up to 4, where the gap is wide compared with the term, every
    noise type must agree essentially always -- the published counts are
    93..99 there and ours are 100. This is the column that inverts if the
    imputer is built on phase instead of frequency: random-walk FM drops to
    3 in 100."""
    assert np.all(_overlap_counts(alpha)[:3] >= 0.95)


def test_the_published_ranking_of_noise_types_is_reproduced():
    """The paper's structural result, and the one a broken implementation
    does not produce: random-walk FM is the best case and white PM the
    worst, with the failure concentrated at long tau where the terms are
    mostly imputed. The *counts* are not comparable -- the criterion's
    strictness is set by the confidence interval, and the paper does not
    state its EDOF convention -- so the ordering is what is asserted."""
    long_tau = {a: _overlap_counts(a)[-3:].mean() for a in (2, 0, -1, -2)}
    assert long_tau[2] < long_tau[0] < long_tau[-1] <= long_tau[-2]
    assert long_tau[-2] > 0.9, "random-walk FM should survive the whole range"
    assert long_tau[2] < 0.5, "white PM at long tau is the published failure"

    for alpha, published in PUBLISHED_OVERLAP.items():
        assert len(published) == 8, "one published count per octave of tau"
