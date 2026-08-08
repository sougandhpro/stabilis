"""Generic delimited-text reading, dialect sniffing and rate inference."""

import numpy as np
import pytest

from stabilis.io import load_frequency, load_phase, read_text


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_whitespace_single_column(tmp_path):
    p = _write(tmp_path, "a.txt", "1.0\n2.0\n3.0\n")
    values, times, _ = read_text(p)
    assert np.allclose(values, [1, 2, 3])
    assert times is None


@pytest.mark.parametrize(
    "sep,name", [(",", "csv"), (";", "semi"), ("\t", "tab"), (" ", "space")]
)
def test_delimiter_sniffing(tmp_path, sep, name):
    body = "\n".join(sep.join(["1.5", "2.5"]) for _ in range(3))
    p = _write(tmp_path, f"{name}.txt", body + "\n")
    values, _, _ = read_text(p, column=1)
    assert np.allclose(values, [2.5, 2.5, 2.5])


def test_comments_and_blank_lines_are_stripped(tmp_path):
    p = _write(
        tmp_path,
        "c.txt",
        "# STATISTICS FOR FILE: PHASE.DAT\n"
        "% another comment style\n"
        "\n"
        "1.0\n"
        "2.0   // trailing note\n"
        "3.0\n",
    )
    values, _, _ = read_text(p)
    assert np.allclose(values, [1, 2, 3])


def test_timetag_column_and_rate_inference(tmp_path):
    p = _write(tmp_path, "t.txt", "0.0 10.0\n0.5 11.0\n1.0 12.0\n1.5 13.0\n")
    fs = load_frequency(p, column=1, time_column=0)
    assert fs.rate_hz == pytest.approx(2.0)
    assert np.allclose(fs.y, [10, 11, 12, 13])
    outcomes = [s.outcome for s in fs.provenance.steps]
    assert any("tau0=" in o for o in outcomes), "inferred rate must be recorded"


def test_jittery_timetags_are_flagged(tmp_path):
    """A non-uniform grid still loads, but the record says so."""
    p = _write(tmp_path, "j.txt", "0.0 1\n1.0 2\n2.3 3\n2.9 4\n4.4 5\n")
    fs = load_frequency(p, column=1, time_column=0)
    outcomes = [s.outcome for s in fs.provenance.steps]
    assert any("jitter=" in o for o in outcomes)


def test_gap_sentinel_becomes_nan_only_when_asked(tmp_path):
    """Stable32 treats 0 as a gap; we refuse to assume it."""
    p = _write(tmp_path, "g.txt", "1.0\n0.0\n3.0\n")
    kept, _, _ = read_text(p)
    assert np.count_nonzero(np.isnan(kept)) == 0
    dropped, _, _ = read_text(p, gap_value=0.0)
    assert np.isnan(dropped[1])
    assert np.isfinite(dropped[0]) and np.isfinite(dropped[2])


def test_provenance_records_source_hash(tmp_path):
    p = _write(tmp_path, "h.txt", "1.0\n2.0\n3.0\n")
    ps = load_phase(p, rate_hz=1.0)
    assert ps.provenance.source_sha256 is not None
    assert len(ps.provenance.source_sha256) == 64
    assert ps.provenance.source == str(p)


def test_ragged_rows_are_padded_and_reported_not_rejected(tmp_path):
    """Real exports are not rectangular, and one short line must not lose
    the record.

    The data column is the reference: a row that is missing its later
    fields still contributes its first one, and the absent fields become
    gaps -- which is already this library's NaN representation, so the
    gap machinery handles the consequences and nothing is invented.
    """
    p = _write(tmp_path, "r.txt", "1.0 2.0\n3.0\n5.0 6.0\n")
    values, _times, defects = read_text(p)
    assert values.tolist() == [1.0, 3.0, 5.0]
    assert defects.short_rows == 1
    assert defects.lines == (2,)
    assert not defects.clean
    assert "short rows" in defects.describe()

    # The padded field is a gap, not a zero.
    second, _t, _d = read_text(p, column=1)
    assert second[0] == 2.0 and np.isnan(second[1]) and second[2] == 6.0


def test_strict_still_refuses_a_ragged_file(tmp_path):
    """For a report pipeline, where a malformed input should stop the run."""
    p = _write(tmp_path, "r.txt", "1.0 2.0\n3.0\n")
    with pytest.raises(ValueError, match="ragged"):
        read_text(p, strict=True)


def test_unreadable_fields_become_gaps_and_are_counted(tmp_path):
    """Blank, NaN, NA and dash all mean "no reading here"."""
    p = _write(tmp_path, "n.txt", "1.0,2.0\n3.0,NaN\n5.0,-\n7.0,\n9.0,NA\n")
    values, _t, defects = read_text(p, column=1)
    assert values[0] == 2.0
    assert np.isnan(values[1:]).all()
    assert defects.bad_fields == 4
    assert "gaps" in defects.describe()

    with pytest.raises(ValueError, match="not numbers"):
        read_text(p, column=1, strict=True)


def test_the_fast_and_tolerant_paths_agree_exactly(tmp_path):
    """A fast path that disagrees with the careful one is worse than none.

    `np.loadtxt` handles every well-formed file and is about five times
    quicker; the tolerant parser is the fallback. On a file both can read
    they must be bit-identical.
    """
    from stabilis.io.text import _parse_rows, _tolerant

    rng = np.random.default_rng(0)
    body = "\n".join(f"{a:.15e} {b:.15e}" for a, b in rng.standard_normal((200, 2)))
    p = _write(tmp_path, "clean.txt", body + "\n")

    fast, fast_defects = _parse_rows(p, "#%", None)
    assert fast_defects.clean

    lines = [ln for ln in body.splitlines() if ln.strip()]
    split = [ln.split() for ln in lines]
    slow, _ = _tolerant(split, list(range(1, len(split) + 1)), 2, p)
    assert np.array_equal(fast, slow)


def test_too_many_columns_rejected(tmp_path):
    p = _write(tmp_path, "w.txt", " ".join("1.0" for _ in range(9)) + "\n")
    with pytest.raises(ValueError, match="column limit|exceeds"):
        read_text(p)


def test_missing_column_rejected(tmp_path):
    p = _write(tmp_path, "m.txt", "1.0 2.0\n3.0 4.0\n")
    with pytest.raises(ValueError, match="requested column"):
        read_text(p, column=5)


def test_empty_file_rejected(tmp_path):
    p = _write(tmp_path, "e.txt", "# only a comment\n")
    with pytest.raises(ValueError, match="no data rows"):
        read_text(p)


def test_rate_required_without_timetags(tmp_path):
    p = _write(tmp_path, "n.txt", "1.0\n2.0\n")
    with pytest.raises(ValueError, match="rate_hz"):
        load_phase(p)
