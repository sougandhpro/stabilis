"""The stability curve as CSV and as JSON.

Self-consistency throughout -- these show that what this library writes,
this library reads. The interop assertions live in
`tests/test_export_stable32.py`, which has somebody else's bytes to compare
against.
"""

import json

import numpy as np
import pytest

from stabilis import PhaseSeries, confidence, oadev, tierms
from stabilis.export import COLUMNS, to_csv, to_dict, to_json
from stabilis.io.text import MAX_COLUMNS, read_table


def _curve(n=512, seed=5):
    x = np.random.default_rng(seed).normal(0.0, 1e-12, n)
    series = PhaseSeries(x=x, rate_hz=1.0)
    result = oadev(series, taus="octave", keep_coeffs=True)
    return series, result, confidence(result, series=series)


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------


def test_the_csv_stays_inside_the_width_our_own_reader_accepts():
    """Nine columns would be a file this library could not read back.

    `io.text` refuses anything wider than MAX_COLUMNS, so the column set is
    a hard budget rather than a preference.
    """
    assert len(COLUMNS) <= MAX_COLUMNS
    _, result, ci = _curve()
    header = to_csv(result, ci).splitlines()[0]
    assert len(header.lstrip("# ").split(",")) <= MAX_COLUMNS


def test_the_csv_reads_back_through_our_own_reader(tmp_path):
    """Self-consistency, and the reader recovers the column names."""
    _, result, ci = _curve()
    path = tmp_path / "curve.csv"
    path.write_text(to_csv(result, ci))
    labels, rows, defects = read_table(path)
    assert defects.clean, defects.describe()
    assert labels[:3] == ["af", "tau_s", "n_terms"]
    assert rows.shape[0] == result.af.size


def test_the_csv_keeps_every_digit_of_the_curve(tmp_path):
    """Self-consistency. A curve exported and re-read is the same curve."""
    _, result, ci = _curve()
    path = tmp_path / "curve.csv"
    path.write_text(to_csv(result, ci))
    _, rows, _ = read_table(path)
    names = to_csv(result, ci).splitlines()[0].lstrip("# ").split(",")
    assert np.array_equal(rows[:, names.index("dev")], result.dev)
    assert np.array_equal(rows[:, names.index("dev_lo")], ci.dev_lo)


def test_a_curve_without_an_interval_writes_four_columns():
    """No interval, no interval columns -- not columns of filler.

    An empty cell reads back as a gap and a zero reads back as a
    measurement, so a column with nothing to say is better absent.
    """
    _, result, _ = _curve()
    header = to_csv(result).splitlines()[0].lstrip("# ").split(",")
    assert header == ["af", "tau_s", "n_terms", "dev"]


def test_the_alpha_column_appears_only_when_a_power_law_was_assumed():
    """Data-driven EDOF leaves alpha None, and that absence is data."""
    series, result, _ = _curve()
    driven = confidence(result, method="eta1", series=series)
    assumed = confidence(result, method="greenhall", alpha=0, series=series)
    assert "alpha" not in to_csv(result, driven).splitlines()[0]
    assert "alpha" in to_csv(result, assumed).splitlines()[0]


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


def test_the_json_carries_the_variance_beside_the_deviation():
    """`var` is what is stored and `dev` is derived; both are exported.

    Shipping only the deviation would force every consumer to square it
    back, which is the round trip the code style forbids internally.
    """
    _, result, ci = _curve()
    payload = to_dict(result, ci)
    assert np.allclose(np.sqrt(payload["var"]), payload["dev"], rtol=0, atol=0)


def test_the_json_provenance_is_the_intervals_not_the_results():
    """Computing an interval appends steps; the shorter record loses them.

    Dropping to the result's provenance would silently remove the EDOF
    method from the audit trail of a number that depends on it.
    """
    _, result, ci = _curve()
    payload = to_dict(result, ci)
    assert len(payload["provenance"]["steps"]) == len(ci.provenance.steps)
    assert len(ci.provenance.steps) > len(result.provenance.steps)


def test_a_result_with_no_interval_omits_the_block_rather_than_faking_it():
    _, result, _ = _curve()
    assert "confidence" not in to_dict(result)


def test_the_json_says_whether_the_number_is_a_variance():
    """MTIE and TIE rms ride in the same container and are not variances.

    A consumer that squares or sums them is wrong, and the flag is the only
    thing in the file that says so.
    """
    series, _, _ = _curve()
    assert to_dict(tierms(series, taus="octave"))["is_variance"] is False


def test_the_reason_a_point_declined_survives_into_the_json():
    """The CSV cannot carry it, so the JSON must."""
    _, result, ci = _curve()
    assert "declined" in to_dict(result, ci)["confidence"]
    assert "edof_method" in to_dict(result, ci)["confidence"]


# --------------------------------------------------------------------------
# The command line, which no longer keeps its own copy of this schema
# --------------------------------------------------------------------------


def test_the_command_line_json_is_the_export_json(tmp_path, capsys):
    """The proof the duplication is gone.

    `cli.estimators._print_result` used to build this payload by hand. Two hand-built
    schemas for one object drift, and the one nobody looks at drifts first.
    """
    from stabilis.cli import main

    path = tmp_path / "in.dat"
    x = np.random.default_rng(9).normal(0.0, 1e-12, 256)
    path.write_text("\n".join("%.17g" % v for v in x) + "\n")

    assert main(["oadev", str(path), "--rate", "1", "--json"]) == 0
    from_cli = json.loads(capsys.readouterr().out)

    series = PhaseSeries(x=x, rate_hz=1.0)
    result = oadev(series, taus="octave", keep_coeffs=True)
    from_lib = to_dict(result, confidence(result, series=series))

    assert from_cli["af"] == from_lib["af"]
    assert from_cli["var"] == from_lib["var"]
    assert from_cli["confidence"]["edof"] == from_lib["confidence"]["edof"]
    assert set(from_cli) == set(from_lib)


def test_the_command_line_json_still_carries_what_it_always_did(tmp_path, capsys):
    """Adding keys is safe; losing one breaks somebody's script."""
    from stabilis.cli import main

    path = tmp_path / "in.dat"
    path.write_text("\n".join("%.17g" % v for v in
                              np.random.default_rng(2).normal(0, 1e-12, 128)) + "\n")
    assert main(["oadev", str(path), "--rate", "1", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    for key in ("kind", "af", "tau_s", "var", "dev", "n_terms", "provenance"):
        assert key in payload
    for key in ("level", "edof", "edof_method", "dev_lo", "dev_hi",
                "alpha", "declined"):
        assert key in payload["confidence"]


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_cli_export_table_writes_a_readable_file(tmp_path, capsys, fmt):
    from stabilis.cli import main

    path = tmp_path / "in.dat"
    path.write_text("\n".join("%.17g" % v for v in
                              np.random.default_rng(4).normal(0, 1e-12, 256)) + "\n")
    out = tmp_path / f"curve.{fmt}"
    assert main(["export", "table", str(path), "--rate", "1",
                 "--format", fmt, "--out", str(out)]) == 0
    assert "wrote" in capsys.readouterr().out
    if fmt == "json":
        assert json.loads(out.read_text())["kind"] == "oadev"
    else:
        assert out.read_text().startswith("# af,")
