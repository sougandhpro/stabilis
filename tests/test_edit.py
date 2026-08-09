"""The preparation layer: range selection, outliers, and timebases.

Two properties get the most attention here because getting them wrong is
undetectable downstream: that removing a point never re-times the ones
after it, and that a timestamp format is never guessed when the evidence
does not single one out.
"""

import numpy as np
import pytest

from stabilis.edit import MJD_EPOCH_UNIX, drop_outliers, select_range, summary
from stabilis.io.timebase import (
    AmbiguousTimebase,
    infer_rate,
    parse_times,
)
from stabilis.types import FreqSeries, PhaseSeries


def phase(n=100, rate_hz=1.0, epoch=0.0):
    x = np.arange(n, dtype=np.float64) * 1e-12
    return PhaseSeries(x, rate_hz=rate_hz, epoch=epoch)


# ---------------------------------------------------------------------------
# select_range
# ---------------------------------------------------------------------------


def test_select_by_index_keeps_exactly_that_slice():
    out = select_range(phase(100), 10, 20, reason="warm-up")
    assert out.n == 10
    assert out.x[0] == pytest.approx(10e-12)


def test_select_by_seconds_uses_the_sample_rate():
    out = select_range(phase(1000, rate_hz=10.0), 5.0, 15.0,
                       by="seconds", reason="settling")
    assert out.n == 100, "10 s at 10 Hz"


def test_the_epoch_moves_with_the_window():
    """So a second selection by absolute time behaves like the first.

    If the epoch stayed put, selecting 10-20 s and then 5-10 s of *that*
    would silently mean something else.
    """
    out = select_range(phase(1000, rate_hz=10.0, epoch=1_700_000_000.0),
                       100, 200, reason="first pass")
    assert out.epoch == pytest.approx(1_700_000_000.0 + 10.0)


def test_select_by_mjd_and_unix_agree():
    epoch = 1_700_000_000.0
    series = phase(10_000, rate_hz=1.0, epoch=epoch)
    start_unix = epoch + 100.0
    start_mjd = MJD_EPOCH_UNIX + start_unix / 86400.0
    by_unix = select_range(series, start_unix, None, by="unix", reason="r")
    by_mjd = select_range(series, start_mjd, None, by="mjd", reason="r")
    assert by_unix.n == by_mjd.n == 9900


def test_absolute_selection_without_an_epoch_is_refused():
    """Guessing that the epoch is zero would put the window in 1970."""
    with pytest.raises(ValueError, match="needs the series to carry an epoch"):
        select_range(phase(100), 60000.0, None, by="mjd", reason="r")


def test_a_reason_is_required():
    """An exclusion that cannot be defended later is not evidence."""
    with pytest.raises(ValueError, match="`reason` is required"):
        select_range(phase(100), 10, 20)
    with pytest.raises(ValueError, match="`reason` is required"):
        drop_outliers(phase(100))


def test_the_reason_reaches_provenance():
    out = select_range(phase(100), 10, 20, reason="cable touched at 14:05")
    step = next(s for s in out.provenance.steps if s.action == "select_range")
    assert step.params["reason"] == "cable touched at 14:05"
    assert step.params["dropped"] == 90


def test_an_empty_or_single_point_window_is_refused():
    with pytest.raises(ValueError, match="need at least 2"):
        select_range(phase(100), 50, 51, reason="r")


# ---------------------------------------------------------------------------
# drop_outliers
# ---------------------------------------------------------------------------


def test_outliers_become_gaps_and_never_shift_later_samples():
    """The property that makes this safe.

    Deleting a spike would shorten the record and re-time everything after
    it, so a point removed at index 500 silently moves sample 501 into its
    slot. Marking it NaN keeps the timebase intact, which is why the length
    is asserted here and not just the count.
    """
    rng = np.random.default_rng(0)
    y = rng.standard_normal(500) * 1e-12
    y[123] = 5e-9                                  # a spike
    series = FreqSeries(y, rate_hz=1.0)

    out = drop_outliers(series, threshold=5.0, reason="counter glitch")
    assert out.n == series.n, "length preserved; nothing deleted"
    assert np.isnan(out.y[123])
    assert np.isfinite(out.y[122]) and np.isfinite(out.y[124])
    assert out.y[200] == series.y[200], "untouched points identical"


def test_mad_is_not_fooled_by_the_spike_it_is_looking_for():
    """The reason it is not standard deviation.

    One spike large enough inflates the standard deviation until it is no
    longer an outlier by its own test -- the classic masking failure. The
    MAD is unmoved by it.
    """
    rng = np.random.default_rng(5)
    y = rng.standard_normal(201)
    y[100] = 1000.0
    series = FreqSeries(y, rate_hz=1.0)

    sd = np.std(y)
    assert abs(y[100]) < 15 * sd, "by sigma the spike hides inside 15 sigma"

    out = drop_outliers(series, threshold=5.0, reason="test")
    assert np.isnan(out.y[100]), "by MAD it is caught"
    assert int(np.isnan(out.y).sum()) == 1


def test_phase_outliers_are_judged_on_the_difference_not_the_level():
    """A phase record wanders, so distance from the median means nothing.

    Testing the level would flag the whole end of any drifting record --
    which is most of them.
    """
    x = np.arange(500, dtype=np.float64) * 1e-9      # a strong ramp
    x[250] += 1e-6                                    # one step out of place
    out = drop_outliers(PhaseSeries(x, rate_hz=1.0), threshold=6.0,
                        reason="glitch")
    assert np.isnan(out.x[250])
    assert int(np.isnan(out.x).sum()) <= 2, "the ramp itself is not an outlier"


def test_a_perfectly_uniform_record_has_no_outliers_and_says_so():
    """The other half of the pair above, and the reason for the sigma floor.

    On a clean ramp more than half the differences are bit-identical, so
    the MAD is float dust. Declining outright would be wrong -- the test
    above shows a glitch on such a ramp must still be caught -- so the
    floor is used *as* sigma. Here the deviations are dust too, and nothing
    is flagged.
    """
    x = np.arange(100, dtype=np.float64) * 1e-9
    out = drop_outliers(PhaseSeries(x, rate_hz=1.0), reason="r")
    assert not np.isnan(out.x).any()
    step = next(s for s in out.provenance.steps if s.action == "drop_outliers")
    assert step.params["at_floor"] is True
    assert "floating-point rounding" in step.outcome


def test_summary_reports_the_fraction_not_only_the_count():
    y = np.ones(100)
    y[10:20] = np.nan
    s = summary(FreqSeries(y, rate_hz=2.0))
    assert s["n_gaps"] == 10
    assert s["gap_fraction"] == pytest.approx(0.1)
    assert s["runs"] == 2 and s["longest_run"] == 80
    assert s["duration_s"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Timebases
# ---------------------------------------------------------------------------


def test_iso_timestamps_parse_and_set_an_epoch():
    texts = ["2026-01-01T00:00:00", "2026-01-01T00:00:01", "2026-01-01T00:00:02"]
    seconds, epoch, fmt = parse_times(texts)
    assert fmt == "iso"
    assert seconds.tolist() == [0.0, 1.0, 2.0]
    assert epoch == pytest.approx(
        __import__("datetime").datetime(2026, 1, 1,
                                        tzinfo=__import__("datetime").timezone.utc
                                        ).timestamp())


def test_a_trailing_z_and_a_space_separator_are_both_accepted():
    seconds, _e, fmt = parse_times(["2026-01-01 00:00:00Z", "2026-01-01 00:00:05Z"])
    assert fmt == "iso" and seconds[1] == 5.0


def test_unix_seconds_are_recognised_but_mjd_must_be_named():
    """Unix seconds are unmistakable; MJD is not, and that asymmetry is the
    honest one.

    MJD 40000-70000 sits inside any plausible range of elapsed seconds, so
    no amount of looking at the values separates them. Guessing wrong is
    not a small error: MJD is converted by 86400, so a 1 Hz record read as
    MJD becomes one sample per day and every tau is out by that factor.
    Unix seconds start above 3e8, far outside elapsed's range, so they can
    be told apart safely.
    """
    _s, epoch, fmt = parse_times(np.array([1.7e9, 1.7e9 + 1.0]))
    assert fmt == "unix" and epoch == pytest.approx(1.7e9)

    with pytest.raises(AmbiguousTimebase, match="mjd"):
        parse_times(np.array([60000.0, 60000.0 + 1 / 86400]))

    mjd, _e, fmt = parse_times(np.array([60000.0, 60000.0 + 1 / 86400]),
                               fmt="mjd")
    assert fmt == "mjd" and mjd[1] == pytest.approx(1.0, rel=1e-6)


def test_an_ambiguous_time_column_is_refused_rather_than_guessed():
    """A wrong timebase rescales every tau, and nothing downstream notices.

    Values in 40000-70000 are a credible MJD *and* a credible count of
    elapsed seconds, so the caller has to say which.
    """
    values = np.array([60000.0, 60001.0, 60002.0])
    with pytest.raises(AmbiguousTimebase, match="could be any of"):
        parse_times(values)
    # ...and naming it resolves it, both ways.
    assert parse_times(values, fmt="mjd")[2] == "mjd"
    assert parse_times(values, fmt="elapsed")[2] == "elapsed"


def test_elapsed_seconds_are_relative_to_the_first_sample():
    seconds, epoch, fmt = parse_times(np.array([0.0, 0.5, 1.0]))
    assert fmt == "elapsed" and epoch == 0.0 and seconds[-1] == 1.0


def test_the_rate_comes_from_the_median_interval():
    """One duplicated or missing stamp must not move the rate."""
    t = np.arange(100, dtype=np.float64) * 0.1
    t[50] = t[49]                       # a duplicate
    rate, jitter = infer_rate(t)
    assert rate == pytest.approx(10.0)
    assert jitter == pytest.approx(0.0, abs=1e-9)


def test_ragged_timestamps_report_their_jitter():
    rng = np.random.default_rng(3)
    t = np.arange(500, dtype=np.float64) + rng.normal(0, 0.02, 500)
    rate, jitter = infer_rate(np.sort(t))
    assert rate == pytest.approx(1.0, rel=0.05)
    assert jitter > 1e-3, "a ragged grid must not be reported as uniform"


def test_non_increasing_timestamps_are_refused():
    with pytest.raises(ValueError, match="do not increase"):
        infer_rate(np.array([5.0, 4.0, 3.0]))


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------


def table():
    labels = ["time", "VSL", "NCS", "NRB"]
    rows = np.array([[0.0, 10.0, 3.0, 1.0],
                     [1.0, 20.0, 5.0, 2.0],
                     [2.0, 30.0, 9.0, 4.0]])
    return labels, rows


def test_a_difference_of_two_named_columns():
    from stabilis.edit import combine, parse_expression

    labels, rows = table()
    assert parse_expression("NCS - VSL", labels) == ((1, "NCS"), (-1, "VSL"))
    assert combine("NCS - VSL", labels, rows).tolist() == [-7.0, -15.0, -21.0]


def test_a_leading_minus_negates_a_single_column():
    """`-VSL` is how "the other side of the difference" gets expressed."""
    from stabilis.edit import combine

    labels, rows = table()
    assert combine("-VSL", labels, rows).tolist() == [-10.0, -20.0, -30.0]


def test_columns_may_be_named_or_numbered():
    """A file with no header is still usable."""
    from stabilis.edit import combine

    labels, rows = table()
    assert np.array_equal(combine("2-1", labels, rows),
                          combine("NCS - VSL", labels, rows))


def test_several_terms_accumulate():
    from stabilis.edit import combine

    labels, rows = table()
    assert combine("NCS + NRB - VSL", labels, rows).tolist() == [-6.0, -13.0, -17.0]


def test_a_gap_in_either_term_makes_the_result_a_gap():
    """A difference is unknown wherever either side is."""
    from stabilis.edit import combine

    labels, rows = table()
    rows = rows.copy()
    rows[1, 1] = np.nan
    out = combine("NCS - VSL", labels, rows)
    assert out[0] == -7.0 and np.isnan(out[1]) and out[2] == -21.0


def test_an_unknown_column_is_refused_with_the_list_of_real_ones():
    """A silent zero column would be a wrong answer that looks plausible."""
    from stabilis.edit import parse_expression

    labels, _rows = table()
    with pytest.raises(ValueError, match="unknown column 'NOPE'"):
        parse_expression("NOPE - VSL", labels)
    with pytest.raises(ValueError, match="out of range"):
        parse_expression("99", labels)
    with pytest.raises(ValueError, match="empty column expression"):
        parse_expression("   ", labels)


def test_arithmetic_beyond_plus_and_minus_is_refused_not_evaluated():
    """There is no evaluator here, deliberately.

    This library refuses pickle because a file from someone else's rig is
    untrusted input; an expression box wired to a Python evaluator would be
    a far shorter path to running their code. Only whole columns, added and
    subtracted.
    """
    from stabilis.edit import parse_expression

    labels, _rows = table()
    for hostile in ("NCS * VSL", "__import__('os').system('x')", "NCS / 2",
                    "2 ** 40"):
        with pytest.raises(ValueError):
            parse_expression(hostile, labels)
