"""Writing the block Stable32 prints, checked against Stable32's own bytes.

Three assertions here are **interop**, and they are the only ones in the
export layer that are: the reference is a file Stable32 1.62 wrote, not a
file this library wrote. They are labelled individually because they
establish different things, and the difference matters --

* the *layout* reproduces the committed bytes exactly;
* the *preamble* reproduces them exactly;
* the *numbers*, computed here from the same input, agree to the last digit
  the export actually carries.

What none of them shows is that Stable32 can read what we write. That is
`VALIDATION-DEBT.md` item 8 and it is not falsifiable in CI.
"""

import glob
import re

import numpy as np
import pytest

from stabilis import PhaseSeries, confidence, load_phase, oadev, to_frequency
from stabilis.export.stable32 import (
    ABSENT,
    HEADERS,
    ROW,
    SIGMA_LABELS,
    to_stable32,
)
from stabilis.validate.report import STABLE32_PRINT_RTOL
from stabilis.validate.stable32 import SIGMA_TYPES, parse

OUTPUTS = sorted(glob.glob("tests/data/stable32/outputs/*.txt"))
INPUTS = "tests/data/stable32/inputs"
_DATA_ROW = re.compile(r"^\s*\d+\s")


def _committed_rows(path):
    """The literal data lines of a committed export, CRLF stripped."""
    raw = open(path, newline="").read()
    return [ln for ln in raw.split("\r\n")
            if _DATA_ROW.match(ln) and len(ln.split()) == 7]


def _curve(n=256, seed=6):
    x = np.random.default_rng(seed).normal(0.0, 1e-12, n)
    series = PhaseSeries(x=x, rate_hz=1.0)
    result = oadev(series, taus="octave", keep_coeffs=True)
    return series, result, confidence(result, method="greenhall", alpha=0,
                                      series=series)


# --------------------------------------------------------------------------
# INTEROP: the format itself
# --------------------------------------------------------------------------


@pytest.mark.skipif(not OUTPUTS, reason="no Stable32 session has been run yet")
def test_the_row_layout_reproduces_every_committed_stable32_row():
    """INTEROP. Every data row Stable32 wrote, rebuilt character for character.

    The values come from our parser, so this tests the *formatter* against
    somebody else's bytes and not our arithmetic -- field widths, decimal
    places, the exponent form, and the absent-bound convention. That is
    exactly the half a round trip through our own code cannot check.
    """
    total = 0
    for path in OUTPUTS:
        expected = _committed_rows(path)
        rebuilt = []
        for table in parse(open(path).read(), path):
            for i in range(table.af.size):
                lo, hi = table.sigma_lo[i], table.sigma_hi[i]
                rebuilt.append(ROW.format(
                    af=int(table.af[i]), tau=float(table.tau_s[i]),
                    n=int(table.n_terms[i]), alpha=int(table.alpha[i]),
                    lo=ABSENT if np.isnan(lo) else float(lo),
                    sig=float(table.sigma[i]),
                    hi=ABSENT if np.isnan(hi) else float(hi)))
        assert rebuilt == expected, path
        total += len(expected)
    assert total > 6000, f"only {total} rows scored"


@pytest.mark.skipif(not OUTPUTS, reason="no Stable32 session has been run yet")
def test_the_preamble_and_header_reproduce_the_committed_bytes():
    """INTEROP. The eight preamble lines and the column header, exactly.

    Maximum, Minimum and Average are `preprocess.basic_stats` at six
    decimals -- so this also confirms that our descriptive statistics agree
    with Stable32's on the same file, which nothing else checks.
    """
    path = "tests/data/stable32/outputs/STATISTICS FOR FILE alphap0.dat.txt"
    committed = open(path, newline="").read().split("\r\n")[:10]

    series = load_phase(f"{INPUTS}/alphap0.dat", rate_hz=1.0)
    result = oadev(series, taus="octave", keep_coeffs=True)
    ci = confidence(result, method="greenhall", alpha=0, series=series)
    ours = to_stable32(series, result, ci,
                       source="alphap0.dat").split("\r\n")[:10]
    assert ours == committed


@pytest.mark.skipif(not OUTPUTS, reason="no Stable32 session has been run yet")
def test_our_sigma_column_agrees_with_stable32_to_its_last_printed_digit():
    """INTEROP, numbers and formatting together -- the strongest one here.

    Everything is ours: the input file is read, the estimator run, the
    block formatted. Only the reference is Stable32's. Tolerance is the
    half-ulp of the fifth significant figure, which is the last digit the
    export carries; asserting tighter would assert against digits Stable32
    never wrote.
    """
    scored = 0
    for name in ("alphap0", "alpham1", "alpham2", "drift"):
        series = load_phase(f"{INPUTS}/{name}.dat", rate_hz=1.0)
        result = oadev(series, taus="octave")
        ours = dict(zip(result.af.tolist(), result.dev.tolist()))

        for table in parse(open(f"tests/data/stable32/outputs/"
                                f"STATISTICS FOR FILE {name}.dat.txt").read()):
            if table.estimator != "oadev" or table.tau_mode != "octave":
                continue
            for af, sigma in zip(table.af.tolist(), table.sigma.tolist()):
                if af not in ours:
                    continue
                rel = abs(ours[af] - sigma) / sigma
                assert rel < STABLE32_PRINT_RTOL, (name, af, ours[af], sigma)
                # And the formatted field itself, which is what a user diffs.
                assert "%14.4e" % ours[af] == "%14.4e" % sigma, (name, af)
                scored += 1
    assert scored > 30, f"only {scored} points scored"


def test_our_labels_are_the_readers_map_inverted():
    """One format, two modules, and they must name the statistics alike.

    `export/` deliberately does not import `validate/` -- that would pull
    the whole evidence-report chain into an export -- so the agreement is
    asserted here instead of enforced by a shared constant.
    """
    assert {v: k for k, v in SIGMA_TYPES.items()} == SIGMA_LABELS
    assert set(HEADERS) == set(SIGMA_LABELS.values())


# --------------------------------------------------------------------------
# The absent-bound convention, both directions
# --------------------------------------------------------------------------


def test_a_curve_with_no_interval_writes_the_zeros_that_mean_absent(tmp_path):
    """Round trip of a convention, not of a number.

    Stable32 prints 0.0000e+00 for a bound it did not compute, and our
    reader turns that back into NaN because it is an absence. This is the
    other half: we write the same zeros, and our own reader recovers the
    absence rather than a measurement of zero.
    """
    series, result, _ = _curve()
    text = to_stable32(series, result, ci=None, alpha=0, source="x.dat")
    assert " 0.0000e+00" in text

    table = parse(text)[0]
    assert np.isnan(table.sigma_lo).all()
    assert np.isnan(table.sigma_hi).all()
    assert not table.has_bounds


def test_the_table_reads_back_through_our_own_parser(tmp_path):
    """Self-consistency, labelled as such: our writer against our reader.

    Useful, and not evidence about Stable32. Sigma is compared at the four
    decimals the format carries, because that is all it carries.
    """
    series, result, ci = _curve()
    table = parse(to_stable32(series, result, ci, source="x.dat"))[0]
    assert table.estimator == "oadev"
    assert table.af.tolist() == result.af.tolist()
    assert np.allclose(table.sigma, result.dev, rtol=1e-4)


def test_the_line_endings_are_crlf_because_stable32s_are():
    """Byte-comparison against its own output is the only interop we have."""
    series, result, ci = _curve()
    assert to_stable32(series, result, ci, source="x.dat").endswith("\r\n")
    assert "\r\n" in to_stable32(series, result, ci, source="x.dat")


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_an_estimator_stable32_has_no_column_for_is_refused():
    """TOTDEV, Thêo1, MTIE and the wavelets have no Sigma Type here."""
    from stabilis import totdev

    series, _, _ = _curve()
    result = totdev(series, taus="octave")
    with pytest.raises(ValueError, match="no column for"):
        to_stable32(series, result, alpha=0)


def test_a_frequency_series_is_refused_because_no_export_of_one_was_captured():
    """Every committed export says 'Phase Data Points'. Guessing the other
    wording would be inventing a format from nothing."""
    series, result, ci = _curve()
    with pytest.raises(ValueError, match="Phase Data Points"):
        to_stable32(to_frequency(series), result, ci)


def test_a_result_with_no_alpha_is_refused_rather_than_given_one():
    """Writing 0 in the Alpha column asserts white FM.

    The data-driven EDOF methods leave alpha None *because* not assuming a
    noise type is the point of them; filling it in here would undo that
    where nobody would look.
    """
    series, result, _ = _curve()
    driven = confidence(result, method="eta1", series=series)
    assert driven.alpha is None
    with pytest.raises(ValueError, match="no way to say 'unknown'"):
        to_stable32(series, result, driven)


def test_a_gapped_record_is_refused_because_the_preamble_states_a_range():
    """'Points 1 thru N of N' is a contiguity claim, and gaps break it."""
    x = np.random.default_rng(8).normal(0.0, 1e-12, 256)
    x[10] = np.nan
    series = PhaseSeries(x=x, rate_hz=1.0)
    result = oadev(series, taus="octave")
    with pytest.raises(ValueError, match="contiguous point range"):
        to_stable32(series, result, alpha=0)


def test_cli_export_table_stable32_matches_the_library(tmp_path, capsys):
    from stabilis.cli import main

    out = tmp_path / "curve.txt"
    assert main(["export", "table", f"{INPUTS}/alphap0.dat", "--rate", "1",
                 "--edof-method", "greenhall", "--alpha", "0",
                 "--format", "stable32", "--out", str(out)]) == 0
    assert "wrote" in capsys.readouterr().out

    series = load_phase(f"{INPUTS}/alphap0.dat", rate_hz=1.0)
    result = oadev(series, taus="octave", keep_coeffs=True)
    ci = confidence(result, method="greenhall", alpha=0, series=series)
    with open(out, newline="") as fh:          # newline="" keeps the CRLF
        assert fh.read() == to_stable32(series, result, ci, alpha=0,
                                        source="alphap0.dat")
