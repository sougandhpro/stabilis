"""MTIE and TIE rms.

The important test is the first one: the fast doubling algorithm against a
brute-force sliding window, over every window length on random data. MTIE
is the one estimator here where the *algorithm* is the risk rather than the
definition -- the naive form is obviously right and unusably slow, so the
fast one has to be shown to agree with it exactly.
"""

import time

import numpy as np
import pytest

from stabilis.stats import confidence, mtie, oadev, tierms
from stabilis.stats.mtie import running_extremes
from stabilis.synth import power_law_phase
from stabilis.types import FreqSeries, PhaseSeries


def brute_mtie(x, m):
    """The definition, directly. O(N*m), correct by inspection."""
    return max(float(x[k:k + m + 1].max() - x[k:k + m + 1].min())
               for k in range(x.size - m))


def test_the_fast_algorithm_equals_brute_force_at_every_window():
    """Exact agreement, not approximate: both take a max of the same set.

    Every window length from 1 to 40 on a record with plateaus and repeats,
    because the doubling covers a window with two *overlapping* blocks and
    the overlap is only harmless because max is idempotent. Ties and equal
    runs are where an off-by-one in the tail index would hide.
    """
    rng = np.random.default_rng(7)
    x = np.round(rng.standard_normal(200) * 3.0)  # many repeats, many ties
    for length in range(1, 41):
        hi, lo, _lvl, _tbl = running_extremes(x, length)
        assert hi.size == x.size - length + 1
        for k in range(0, x.size - length + 1, 17):
            assert hi[k] == x[k:k + length].max()
            assert lo[k] == x[k:k + length].min()
        assert float((hi - lo).max()) == brute_mtie(x, length - 1)


def test_reusing_the_table_across_growing_windows_gives_the_same_answer():
    """The incremental path is what makes the octave grid O(N log N)."""
    rng = np.random.default_rng(8)
    x = rng.standard_normal(500)
    level, table = 0, None
    for length in (2, 3, 5, 9, 17, 33, 65, 129):
        hi, lo, level, table = running_extremes(x, length, level, table)
        fresh_hi, fresh_lo, _, _ = running_extremes(x, length)
        assert np.array_equal(hi, fresh_hi)
        assert np.array_equal(lo, fresh_lo)


def test_mtie_matches_the_definition_on_a_real_record():
    ps = power_law_phase(600, 0, rate_hz=1.0, rng=np.random.default_rng(3),
                         h_alpha=1e-22)
    result = mtie(ps, taus="octave")
    assert result.kind == "mtie"
    for af, value in zip(result.af.tolist(), result.dev.tolist()):
        assert value == pytest.approx(brute_mtie(ps.x, af), rel=1e-12)


def test_mtie_is_monotone_in_tau():
    """A longer window contains every shorter one, so its spread cannot fall.

    True of the quantity itself, not merely of an estimate -- which makes it
    a check no amount of noise can make flaky.
    """
    ps = power_law_phase(1024, -2, rate_hz=1.0, rng=np.random.default_rng(4))
    dev = mtie(ps, taus="all", max_af=60).dev
    assert np.all(np.diff(dev) >= 0.0)


def test_mtie_of_a_pure_ramp_is_the_ramp_over_the_window():
    """x = a*t gives MTIE(tau) = a*tau exactly, with no statistics in it."""
    a, n = 3.0e-12, 400
    ps = PhaseSeries(a * np.arange(n, dtype=np.float64), rate_hz=1.0)
    result = mtie(ps, taus="octave", max_af=64)
    assert np.allclose(result.dev, a * result.tau_s, rtol=1e-13)


def test_mtie_is_fast_enough_to_use():
    """The whole reason for the doubling algorithm.

    A quarter-million points over the full octave grid. The naive form is
    O(N*m) per tau and would take minutes here; this asserts seconds, which
    is a loose bound deliberately -- the point is the complexity class, not
    the constant, and a machine-speed assertion that fails on a slow runner
    is worse than none.
    """
    ps = power_law_phase(250_000, -1, rate_hz=1.0,
                         rng=np.random.default_rng(5))
    start = time.monotonic()
    result = mtie(ps, taus="octave")
    elapsed = time.monotonic() - start
    assert result.af.size > 10
    assert elapsed < 10.0, f"octave-grid MTIE on 250k points took {elapsed:.1f}s"


def test_tierms_matches_its_definition():
    ps = power_law_phase(500, 0, rate_hz=1.0, rng=np.random.default_rng(6))
    result = tierms(ps, taus="octave")
    for af, value in zip(result.af.tolist(), result.dev.tolist()):
        want = np.sqrt(np.mean((ps.x[af:] - ps.x[:-af]) ** 2))
        assert value == pytest.approx(want, rel=1e-12)


def test_tierms_grows_linearly_with_a_frequency_offset():
    """It does not remove a constant offset, and that is the definition.

    A clock fast by a fixed amount contributes `offset * tau`, which comes
    to dominate at long tau. The Allan family removes exactly this; TIE rms
    keeps it, which is why timing-distribution specs are written against
    it.
    """
    offset, n = 2.0e-12, 2048
    ps = PhaseSeries(offset * np.arange(n, dtype=np.float64), rate_hz=1.0)
    result = tierms(ps, taus="octave", max_af=256)
    assert np.allclose(result.dev, offset * result.tau_s, rtol=1e-13)


def test_frequency_input_is_refused_rather_than_integrated():
    """MTIE is a spread of time errors; integrating invents the constant."""
    fs = FreqSeries(np.zeros(100), rate_hz=1.0)
    with pytest.raises(TypeError, match="needs phase"):
        mtie(fs)


# ---------------------------------------------------------------------------
# Not a variance -- and the three places that has to be enforced
# ---------------------------------------------------------------------------


def test_neither_statistic_claims_to_be_a_variance():
    ps = power_law_phase(400, 0, rate_hz=1.0, rng=np.random.default_rng(9))
    assert mtie(ps, taus="octave").is_variance is False
    assert tierms(ps, taus="octave").is_variance is False
    assert oadev(ps, taus="octave").is_variance is True


def test_a_chi_square_interval_on_mtie_is_refused():
    """An extreme value has no chi-square sampling distribution.

    Its spread grows with record length rather than shrinking, which is the
    opposite of what an error bar implies -- so no interval at all beats a
    plausible wrong one.
    """
    ps = power_law_phase(400, 0, rate_hz=1.0, rng=np.random.default_rng(10))
    with pytest.raises(ValueError, match="no chi-square sampling"):
        confidence(mtie(ps, taus="octave"))


def test_the_power_law_fit_refuses_a_non_variance():
    from stabilis.stats.powerlaw import decompose

    ps = power_law_phase(2048, 0, rate_hz=1.0, rng=np.random.default_rng(11))
    with pytest.raises(ValueError, match="is not a variance"):
        decompose(mtie(ps, taus="octave"), f_h=0.5)


def test_the_refusals_key_on_the_flag_not_on_a_name():
    """A plugin estimator must not slip past a blacklist of known names."""
    from dataclasses import replace

    from stabilis.stats.powerlaw import decompose

    ps = power_law_phase(2048, 0, rate_hz=1.0, rng=np.random.default_rng(12))
    disguised = replace(oadev(ps, taus="octave"), kind="my-plugin-statistic",
                        is_variance=False)
    with pytest.raises(ValueError, match="is not a variance"):
        decompose(disguised, f_h=0.5)
    with pytest.raises(ValueError, match="no chi-square sampling"):
        confidence(disguised)


def test_the_bench_solver_refuses_both_and_says_why_for_each():
    from stabilis.bench import Bench, Channel, Device, Measurement, Source
    from stabilis.bench.solve import solve

    ps = power_law_phase(2048, 0, rate_hz=1.0, rng=np.random.default_rng(13))
    results = {f"m{i}": mtie(ps, taus="octave") for i in range(3)}
    pairs = (("A", "B"), ("B", "C"), ("C", "A"))
    bench = Bench(
        sources=tuple(Source(id=s) for s in "ABC"),
        devices=(Device(id="D1", channels=(Channel(id="A-1", floor=0.0),)),),
        measurements=tuple(
            Measurement(id=f"m{i}", channel_id="A-1", source_a=a, source_b=b)
            for i, (a, b) in enumerate(pairs)),
    )
    with pytest.raises(ValueError) as exc:
        solve(bench, results)
    assert "maximum of a sum" in str(exc.value)
    assert "frequency offset" in str(exc.value)


def test_gapped_records_drop_the_spoiled_windows():
    """A window straddling a gap is dropped, not silently shortened."""
    x = np.arange(200, dtype=np.float64) * 1e-12
    x[100] = np.nan
    ps = PhaseSeries(x, rate_hz=1.0)
    result = mtie(ps, taus="octave", max_af=32, gap_policy="mask")
    assert result.af.size > 0
    assert np.all(np.isfinite(result.var))
    # Windows touching the gap are gone, so fewer than the ungapped count.
    clean = mtie(PhaseSeries(np.arange(200, dtype=np.float64) * 1e-12,
                             rate_hz=1.0), taus="octave", max_af=32)
    assert np.all(result.n_terms <= clean.n_terms)
