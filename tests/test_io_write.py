"""Writing a series back out, and reading it back unchanged.

Almost everything here is a **self-consistency** check: it shows that this
library's writer and this library's reader agree with each other. That is
worth having -- it is what makes an export usable as an intermediate -- but
it is not evidence that Stable32 or TimeLab can read what we write. Nothing
in this file claims otherwise; see `theory/export.md` and item 8 of
`VALIDATION-DEBT.md`.
"""

import numpy as np
import pytest

from stabilis import (
    FreqSeries,
    PhaseSeries,
    load_frequency,
    load_phase,
    save_frequency,
    save_phase,
)
from stabilis.io.text import read_header_meta, read_text
from stabilis.io.write import write_text


def _series(n=64, seed=3, gaps=()):
    x = np.random.default_rng(seed).normal(0.0, 1e-12, n)
    for i in gaps:
        x[i] = np.nan
    return PhaseSeries(x=x, rate_hz=2.0, epoch=1.7e9)


# --------------------------------------------------------------------------
# The round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["stabilis", "csv"])
def test_a_written_series_reads_back_bit_for_bit(tmp_path, fmt):
    """Self-consistency. Not 'close': the same doubles, bit for bit.

    Seventeen significant digits is the width at which that is guaranteed,
    which is why it is the default and not a taste.
    """
    s = _series()
    save_phase(s, tmp_path / "a.dat", format=fmt)
    back = load_phase(tmp_path / "a.dat")
    assert np.array_equal(s.x, back.x)
    assert back.rate_hz == s.rate_hz


def test_gaps_survive_the_round_trip_as_gaps(tmp_path):
    """Self-consistency. A gap must not come back as a number."""
    s = _series(gaps=(0, 7, 63))
    save_phase(s, tmp_path / "a.dat")
    back = load_phase(tmp_path / "a.dat")
    assert np.array_equal(s.x, back.x, equal_nan=True)
    assert back.n_gaps == 3


def test_a_written_series_reads_back_without_reported_defects(tmp_path):
    """A NaN we wrote on purpose is not damage the reader had to tolerate.

    Guards a real hazard: `text.py`'s tolerant path counts 'nan' as an
    unreadable field, so if a written gap ever stopped going through the
    fast `np.loadtxt` path, every export would come back carrying a
    spurious defect warning.
    """
    s = _series(gaps=(4,))
    save_phase(s, tmp_path / "a.dat")
    _, _, defects = read_text(tmp_path / "a.dat")
    assert defects.clean, defects.describe()


def test_the_hardest_doubles_survive(tmp_path):
    """Signed zero, a subnormal, the top of the range, and NaN."""
    y = np.array([-0.0, 0.0, 5e-324, 1e-300, -1.5e308, np.nan, 1.0 / 3.0])
    save_frequency(FreqSeries(y=y, rate_hz=1.0), tmp_path / "a.dat")
    back = load_frequency(tmp_path / "a.dat")
    assert np.array_equal(y, back.y, equal_nan=True)
    # -0.0 == 0.0 compares equal, so the sign bit needs its own assertion.
    assert np.signbit(back.y[0]) and not np.signbit(back.y[1])


def test_fifteen_digits_would_not_have_been_enough(tmp_path):
    """The measurement behind the default, rather than an assertion of it.

    `%.15e` is a common choice and it is wrong: a large fraction of
    float64 values do not survive it. This test exists so that the day
    somebody lowers the default, the suite says why they should not.
    """
    v = np.random.default_rng(11).normal(0.0, 1e-12, 20_000)
    lossy = sum(1 for x in v if float("%.15e" % x) != x)
    exact = sum(1 for x in v if float("%.17g" % x) != x)
    assert lossy > v.size // 10       # measured ~42%
    assert exact == 0


# --------------------------------------------------------------------------
# What the header carries
# --------------------------------------------------------------------------


def test_the_rate_comes_back_without_being_supplied_again(tmp_path):
    """The whole point of the header: the file knows its own sample rate.

    Without it an export is a bag of numbers, and a caller who has to
    re-supply the rate by hand is a caller who can supply the wrong one.
    """
    save_phase(_series(), tmp_path / "a.dat")
    back = load_phase(tmp_path / "a.dat")           # no rate_hz argument
    assert back.rate_hz == 2.0
    assert any(step.action == "header_rate" for step in back.provenance.steps)


def test_the_epoch_comes_back_too(tmp_path):
    """Offsets without their epoch are unanchored.

    Phase is stored as an offset precisely so float64 can resolve it; the
    epoch is the other half of that pair and dropping it on export would
    undo the arrangement `theory/precision.md` exists to defend.
    """
    save_phase(_series(), tmp_path / "a.dat")
    assert load_phase(tmp_path / "a.dat").epoch == 1.7e9


def test_an_explicit_rate_beats_the_header(tmp_path):
    """The caller is the authority. The header is a convenience."""
    save_phase(_series(), tmp_path / "a.dat")
    assert load_phase(tmp_path / "a.dat", rate_hz=10.0).rate_hz == 10.0


def test_the_header_names_the_columns_for_whoever_opens_it(tmp_path):
    """The label line is first because that is the only line `_header` reads."""
    save_phase(_series(), tmp_path / "a.dat", format="csv",
               time_column="elapsed")
    assert tmp_path.joinpath("a.dat").read_text().splitlines()[0] == \
        "# time_s,phase_s"


def test_an_unreadable_header_line_is_ignored_not_fatal(tmp_path):
    """A comment block is a courtesy and must never stop a file being read."""
    path = tmp_path / "a.dat"
    path.write_text("# value\n# rate_hz = not-a-number\n# nonsense = 1\n"
                    "# epoch_unix = 5\n1.0\n2.0\n")
    assert read_header_meta(path) == {"epoch_unix": 5.0}
    assert load_phase(path, rate_hz=1.0).epoch == 5.0


def test_metadata_after_the_data_has_started_is_not_read(tmp_path):
    """Only the leading block counts; a comment further down is a comment."""
    path = tmp_path / "a.dat"
    path.write_text("1.0\n2.0\n# rate_hz = 99\n")
    assert read_header_meta(path) == {}


# --------------------------------------------------------------------------
# Gaps, and the format that cannot spell one
# --------------------------------------------------------------------------


def test_stable32_format_refuses_a_gapped_record(tmp_path):
    """Refuse rather than substitute a zero.

    Writing 0 is what Stable32's own convention would suggest, and it is
    exactly the trap `text.py` refuses on the reading side: a genuine
    zero-offset measurement and a missing sample become the same bytes.
    """
    with pytest.raises(ValueError, match="no spelling for a missing sample"):
        save_phase(_series(gaps=(3,)), tmp_path / "a.dat", format="stable32")


def test_a_gap_sentinel_that_occurs_in_the_data_is_refused(tmp_path):
    """A sentinel the data can take would delete real samples on read-back."""
    x = np.array([1e-12, 0.0, np.nan, 2e-12])
    s = PhaseSeries(x=x, rate_hz=1.0)
    with pytest.raises(ValueError, match="as a real measurement"):
        save_phase(s, tmp_path / "a.dat", format="stable32", gap_value=0.0)


def test_a_gap_sentinel_round_trips_when_the_reader_is_told(tmp_path):
    """Self-consistency, and it only works because the reader is told."""
    s = _series(gaps=(2, 5))
    save_phase(s, tmp_path / "a.dat", format="stable32", gap_value=-9.99e9)
    back = load_phase(tmp_path / "a.dat", rate_hz=2.0, gap_value=-9.99e9)
    assert np.array_equal(s.x, back.x, equal_nan=True)


def test_stable32_format_writes_bare_numbers(tmp_path):
    """No comment lines at all -- Stable32 was not asked to tolerate ours."""
    save_phase(_series(), tmp_path / "a.dat", format="stable32")
    text = tmp_path.joinpath("a.dat").read_text()
    assert "#" not in text
    assert len(text.splitlines()) == 64


# --------------------------------------------------------------------------
# Timetag columns
# --------------------------------------------------------------------------


def test_timetags_that_cannot_tell_two_samples_apart_are_refused(tmp_path):
    """MJD is days; at a present-day epoch float64 quantizes at ~0.6 us.

    On a nanosecond grid consecutive tags print identically, which would
    make our own reader infer an infinite rate from our own file.
    """
    s = PhaseSeries(x=np.zeros(16), rate_hz=1e9, epoch=1.7e9)
    with pytest.raises(ValueError, match="print as the same value"):
        save_phase(s, tmp_path / "a.dat", time_column="mjd")


def test_coarse_timetags_are_written_but_recorded_as_coarse(tmp_path):
    """Lossy is not the same as wrong, so this warns rather than refusing."""
    s = PhaseSeries(x=np.zeros(16), rate_hz=1e6, epoch=1.7e9)
    written = save_phase(s, tmp_path / "a.dat", time_column="mjd")
    assert any(step.action == "timetag_resolution"
               for step in written.provenance.steps)


def test_elapsed_timetags_stay_exact_at_any_epoch(tmp_path):
    """Counting from zero is why 'elapsed' is the one recommended."""
    s = PhaseSeries(x=np.zeros(16), rate_hz=1e9, epoch=1.7e9)
    written = save_phase(s, tmp_path / "a.dat", time_column="elapsed")
    assert not any(step.action == "timetag_resolution"
                   for step in written.provenance.steps)


def test_a_timetag_column_is_off_by_default(tmp_path):
    """The rate in the header is exact and smaller than any timetag column."""
    save_phase(_series(), tmp_path / "a.dat")
    first_data = [ln for ln in tmp_path.joinpath("a.dat").read_text()
                  .splitlines() if not ln.startswith("#")][0]
    assert len(first_data.split()) == 1


# --------------------------------------------------------------------------
# What the writer hands back
# --------------------------------------------------------------------------


def test_the_written_record_hashes_what_is_on_disk(tmp_path):
    """One hashing path, so the bundle manifest cannot disagree with it."""
    from stabilis.provenance import file_sha256

    written = save_phase(_series(), tmp_path / "a.dat")
    assert written.sha256 == file_sha256(tmp_path / "a.dat")
    assert written.n_bytes == tmp_path.joinpath("a.dat").stat().st_size


def test_the_write_appears_in_provenance(tmp_path):
    """Provenance is not optional, and an export is a step like any other."""
    written = save_phase(_series(), tmp_path / "a.dat", format="csv")
    step = [s for s in written.provenance.steps if s.action == "write_text"][-1]
    assert step.module == "io.write"
    assert step.params["format"] == "csv"
    assert step.params["digits"] == 17


def test_the_line_endings_are_lf_whichever_platform_wrote_them(tmp_path):
    """The manifest hashes bytes, so the newline cannot be the platform's.

    `open(..., newline=None)` would translate to CRLF on Windows and give
    the same series two different hashes on two machines.
    """
    save_phase(_series(), tmp_path / "a.dat")
    assert b"\r\n" not in tmp_path.joinpath("a.dat").read_bytes()


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_an_unknown_format_is_refused_and_the_choices_named(tmp_path):
    with pytest.raises(ValueError, match="unknown format"):
        write_text(tmp_path / "a.dat", np.zeros(4), 1.0, format="parquet")


def test_an_unknown_time_column_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown time_column"):
        write_text(tmp_path / "a.dat", np.zeros(4), 1.0, time_column="gps")


def test_an_empty_series_is_refused(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_text(tmp_path / "a.dat", np.zeros(0), 1.0)


def test_a_two_dimensional_array_is_refused(tmp_path):
    with pytest.raises(ValueError, match="1-D"):
        write_text(tmp_path / "a.dat", np.zeros((4, 2)), 1.0)


# --------------------------------------------------------------------------
# The registry, and the command line
# --------------------------------------------------------------------------


def test_the_writer_is_reachable_through_the_registry():
    """Built-ins register through the same decorator a plugin would."""
    from stabilis.registry import available, get

    assert "text" in available("writer")
    assert get("writer", "text") is write_text


def test_cli_export_data_round_trips_through_the_file(tmp_path, capsys):
    from stabilis.cli import main

    src = tmp_path / "in.dat"
    src.write_text("1e-12\n2e-12\n3e-12\n4e-12\n")
    out = tmp_path / "out.dat"
    assert main(["export", "data", str(src), "--rate", "5", "--out", str(out)]) == 0
    assert "wrote" in capsys.readouterr().out

    # Read back with no --rate: the file carries it.
    assert main(["stats", str(out), "--json"]) == 0
    import json
    assert json.loads(capsys.readouterr().out)["rate_hz"] == 5.0
