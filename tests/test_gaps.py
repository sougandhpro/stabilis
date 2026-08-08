"""Gap handling: what a missing sample is allowed to cost.

The defect these tests pin was silent and expensive. Integrating frequency
to phase with a cumulative sum propagates a NaN to the end of the record,
so a single missing sample halfway through a 200-point series discarded
half the terms and moved ADEV by 3.5% -- a plausible number, no warning,
no symptom. The correct cost of one gap is the 2m (or 3m) terms whose
window spans it, and nothing else.
"""

import numpy as np
import pytest

from stabilis.stats import adev, mdev, oadev, ohdev, tdev
from stabilis.stats.core import phase_runs, run_estimator
from stabilis.stats.classical import SCALE_ALLAN, _coeffs_oavar
from stabilis.synth import power_law_freq
from stabilis.types import FreqSeries, PhaseSeries

ESTIMATORS = [("adev", adev), ("oadev", oadev), ("mdev", mdev),
              ("tdev", tdev), ("ohdev", ohdev)]


def _series(n=512, alpha=0, seed=7):
    """A reproducible frequency record. alpha=0 is white FM."""
    return power_law_freq(n, alpha, rate_hz=1.0, rng=np.random.default_rng(seed))


def _with_gap(fs, i):
    y = fs.y.copy()
    y[i] = np.nan
    return FreqSeries(y, rate_hz=fs.rate_hz)


def _paired(clean, gapped):
    """Yield (m, clean_index, gapped_index) over the shared tau grid.

    The grids legitimately differ: the gapped record's algebraic limit
    follows its longest run, so it reports fewer long taus. Comparing term
    counts means comparing where both reported something.
    """
    for gi, m in enumerate(gapped.af):
        ci = np.flatnonzero(clean.af == m)
        if ci.size:
            yield int(m), int(ci[0]), gi


# --- the defect itself -------------------------------------------------


def test_one_frequency_gap_costs_2m_terms_not_half_the_record():
    """The whole point. Overlapping AVAR loses exactly the spanning terms.

    A gap at y[k] kills phase sample x[k+1] onward as an *absolute* value,
    but the second difference never needs absolute phase. Splitting into
    two runs of k and n-k-1 frequency points gives (k+1) and (n-k) phase
    samples, hence (k+1-2m) + (n-k-2m) surviving terms against the
    ungapped (n+1-2m) -- a loss of exactly 2m.
    """
    fs = _series(n=200)
    clean = oadev(fs, taus="all")
    gapped = oadev(_with_gap(fs, 100), taus="all")

    losses = {m: int(clean.n_terms[ci] - gapped.n_terms[gi])
              for m, ci, gi in _paired(clean, gapped)}
    assert losses == {m: 2 * m for m in losses}


def test_one_hadamard_gap_costs_3m_terms():
    """The third difference spans 3m, so that is what it costs."""
    fs = _series(n=200)
    clean = ohdev(fs, taus="all")
    gapped = ohdev(_with_gap(fs, 100), taus="all")

    losses = {m: int(clean.n_terms[ci] - gapped.n_terms[gi])
              for m, ci, gi in _paired(clean, gapped)}
    assert losses == {m: 3 * m for m in losses}


def test_one_gap_barely_moves_the_estimate():
    """Where the gap costs few terms, it must cost little accuracy.

    Before the fix, ADEV moved 3.5% at m=1 on a 200-point record, because
    the surviving terms were not a random subset -- they were the first
    half of the record. Now the shift is 0.04% there. The claim asserted is
    the general one: lose under 5% of the terms, move under 1%.
    """
    fs = _series(n=1024)
    clean = oadev(fs, taus="octave")
    gapped = oadev(_with_gap(fs, 512), taus="octave")

    checked = 0
    for m, ci, gi in _paired(clean, gapped):
        if gapped.n_terms[gi] < 0.95 * clean.n_terms[ci]:
            continue
        rel = abs(gapped.dev[gi] - clean.dev[ci]) / clean.dev[ci]
        assert rel < 0.01, f"m={m}: {rel:.4f}"
        checked += 1
    assert checked >= 4


@pytest.mark.parametrize("name,fn", ESTIMATORS)
def test_no_estimator_loses_more_than_the_spanning_terms(name, fn):
    """The generic bound, across the whole family.

    No estimator here has a window wider than 3m phase samples, and
    decimation (in non-overlapping `adev`) can waste at most a couple more.
    So `3m + 2` bounds what one gap may cost, whatever the record length --
    which is exactly the property the cumulative-sum defect violated, since
    it cost everything downstream.
    """
    fs = _series(n=512)
    clean = fn(fs, taus="octave")
    gapped = fn(_with_gap(fs, 256), taus="octave")

    for m, ci, gi in _paired(clean, gapped):
        lost = int(clean.n_terms[ci] - gapped.n_terms[gi])
        assert 0 <= lost <= 3 * m + 2, f"{name} lost {lost} terms at m={m}"


# --- the policy --------------------------------------------------------


def test_ungapped_data_is_bit_identical_under_both_policies():
    """Segmenting a record with no gaps must change nothing at all.

    One run is one run; if this is not bit-exact then the pooling path has
    introduced an arithmetic difference where there is no mathematical one.
    """
    x = np.cumsum(_series(n=300).y)
    ps = PhaseSeries(x, rate_hz=1.0)
    masked = oadev(ps, taus="all", gap_policy="mask")
    segmented = oadev(ps, taus="all", gap_policy="segment")

    assert np.array_equal(masked.var, segmented.var)
    assert np.array_equal(masked.n_terms, segmented.n_terms)


def test_frequency_input_refuses_the_mask_policy_when_gapped():
    """Masking gapped frequency data is the bug, so it is not offered."""
    with pytest.raises(ValueError, match="propagates NaN"):
        oadev(_with_gap(_series(n=100), 50), gap_policy="mask")


def test_phase_defaults_to_mask_and_frequency_to_segment():
    """`auto` resolves by domain, and the choice reaches provenance."""
    fs = _series(n=100)
    ps = PhaseSeries(np.cumsum(fs.y), rate_hz=1.0)

    assert phase_runs(fs)[1] == "segment"
    assert phase_runs(ps)[1] == "mask"

    step = [s for s in oadev(fs).provenance.steps if s.action == "evaluate"][-1]
    assert step.params["gap_policy"] == "segment"


def test_unknown_gap_policy_is_rejected():
    with pytest.raises(ValueError, match="gap_policy must be one of"):
        oadev(_series(n=100), gap_policy="fill")


def test_segment_policy_splits_phase_at_gaps():
    """Phase after a counter reset needs its own origin, not a masked term."""
    x = np.arange(21.0)
    x[10] = np.nan
    runs, policy = phase_runs(PhaseSeries(x, rate_hz=1.0), "segment")

    assert policy == "segment"
    assert [r.size for r in runs] == [10, 10]


# --- pooling is exact --------------------------------------------------


def test_pooled_variance_is_the_mean_over_all_runs_terms():
    """`variance = scale * mean(c**2)` still holds across pooled runs.

    The single-CI-path contract from core.py, checked where it is most at
    risk: pooling the wrong way (averaging per-run variances rather than
    per-run terms) would weight a 5-point run equally with a 500-point one.
    """
    fs = _with_gap(_series(n=400), 137)
    res = oadev(fs, taus="octave")
    runs, _ = phase_runs(fs)

    for m, var, n_terms, coeffs in zip(res.af, res.var, res.n_terms, res.coeffs):
        c = np.concatenate([_coeffs_oavar(r, int(m), 1.0) for r in runs])
        c = c[np.isfinite(c)]
        assert c.size == n_terms
        assert np.array_equal(c, coeffs)
        assert var == SCALE_ALLAN * float(np.mean(c**2))


def test_gap_at_the_edge_costs_less_than_a_gap_in_the_middle():
    """A gap in the first sample removes no term that would have existed."""
    fs = _series(n=300)
    edge = oadev(_with_gap(fs, 0), taus="all")
    middle = oadev(_with_gap(fs, 150), taus="all")

    assert np.all(edge.n_terms >= middle.n_terms)


def test_many_gaps_still_report_every_surviving_term():
    """Ten scattered gaps must leave the estimator working, not empty."""
    y = _series(n=1000).y.copy()
    y[np.arange(50, 1000, 100)] = np.nan
    res = oadev(FreqSeries(y, rate_hz=1.0), taus="octave")

    runs, _ = phase_runs(FreqSeries(y, rate_hz=1.0))
    assert len(runs) == 11
    assert res.n_terms.min() > 0
    step = [s for s in res.provenance.steps if s.action == "evaluate"][-1]
    assert step.params["runs"] == 11


def test_averaging_factor_limit_follows_the_longest_run():
    """With every run short, no long tau may be reported.

    The algebraic limit m <= (N-1)/2 binds on the longest gap-free run, not
    on the record: a term cannot straddle a gap however long the file is.
    """
    y = _series(n=1000).y.copy()
    y[np.arange(20, 1000, 20)] = np.nan  # no frequency run longer than 20
    fs = FreqSeries(y, rate_hz=1.0)
    runs, _ = phase_runs(fs)
    longest = max(r.size for r in runs)

    res = oadev(fs, taus="all")
    assert longest == 21  # 20 frequency samples integrate to 21 phase samples
    assert res.af.max() == (longest - 1) // 2


def test_a_record_of_nothing_but_gaps_raises():
    y = np.full(50, np.nan)
    with pytest.raises(ValueError, match="no valid averaging factor"):
        oadev(FreqSeries(y, rate_hz=1.0))
