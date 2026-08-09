"""The measurement report, which is the deliverable and so is tested.

The report's one contentious act is turning a curve and a datasheet limit
into a verdict, so that is what most of this file is about. The rule is
guarded acceptance on the confidence interval rather than simple
acceptance on the point estimate; see `theory/report.md` for why.
"""

import json
import re

import numpy as np
import pytest

from stabilis import confidence, oadev, power_law_phase
from stabilis.cli import main
from stabilis.report import (
    SPEC_TAU_RTOL,
    build,
    spec_afs,
    to_html,
    to_json,
)
from stabilis.stats.bias import deviation_bias_factor


@pytest.fixture(scope="module")
def measurement():
    """One white-FM record and its curve, shared by the module."""
    series = power_law_phase(
        n=4096, alpha=0, rate_hz=1.0,
        rng=np.random.default_rng(11), h_alpha=1e-24,
    )
    result = oadev(series, taus=np.array([1, 2, 4, 8, 16, 32, 64, 128]))
    return series, result, confidence(result)


def _report(measurement, spec=(), ci_given=True, **kw):
    series, result, ci = measurement
    return build(series, result, ci if ci_given else None, spec=spec, **kw)


# --------------------------------------------------------------------------
# the decision rule
# --------------------------------------------------------------------------


def test_a_limit_at_the_upper_bound_passes(measurement):
    """The boundary is closed on the pass side: sigma_hi <= limit."""
    _, result, ci = measurement
    tau, hi = float(result.tau_s[3]), float(ci.dev_hi[3])
    check = _report(measurement, spec=((tau, hi),)).specs[0]
    assert check.evaluated
    assert check.verdict == "pass"
    assert check.margin == pytest.approx(1.0, rel=1e-12)


def test_a_limit_inside_the_interval_is_indeterminate(measurement):
    """Straddling the limit is not a pass and not a failure.

    The record cannot decide, and saying so is the whole difference
    between this and simple acceptance.
    """
    _, result, ci = measurement
    tau = float(result.tau_s[3])
    limit = float(ci.dev_hi[3]) * (1.0 - 1e-9)
    check = _report(measurement, spec=((tau, limit),)).specs[0]
    assert check.verdict == "indeterminate"
    assert check.dev_lo <= limit <= check.dev_hi


def test_a_limit_below_the_lower_bound_fails(measurement):
    _, result, ci = measurement
    tau = float(result.tau_s[3])
    limit = float(ci.dev_lo[3]) * 0.999
    check = _report(measurement, spec=((tau, limit),)).specs[0]
    assert check.verdict == "fail"
    assert check.margin < 1.0


def test_the_guarded_margin_is_never_looser_than_the_point_margin(measurement):
    """sigma_hi >= sigma_hat, so the guarded margin is the conservative one.

    Asserted across the whole curve rather than at one tau, because the
    gap between the two grows with tau and a single-point test would miss
    a sign error at the long end.
    """
    _, result, _ = measurement
    spec = tuple((float(t), float(d)) for t, d in zip(result.tau_s, result.dev))
    for check in _report(measurement, spec=spec).specs:
        assert check.evaluated
        assert check.margin <= check.margin_point
        # A limit set to the point estimate itself can never pass.
        assert check.verdict in ("indeterminate", "fail")


def test_the_overall_verdict_is_the_worst_one(measurement):
    _, result, ci = measurement
    generous = (float(result.tau_s[0]), float(ci.dev_hi[0]) * 10.0)
    hopeless = (float(result.tau_s[5]), float(ci.dev_lo[5]) * 0.5)
    report = _report(measurement, spec=(generous, hopeless))
    assert report.verdict == "fail"
    assert report.worst.tau_spec_s == hopeless[0]


def test_the_worst_point_is_named_even_when_everything_passes(measurement):
    """A customer reads the tightest margin before they read the verdict."""
    _, result, ci = measurement
    spec = ((float(result.tau_s[0]), float(ci.dev_hi[0]) * 10.0),
            (float(result.tau_s[4]), float(ci.dev_hi[4]) * 1.5))
    report = _report(measurement, spec=spec)
    assert report.verdict == "pass"
    assert report.worst.tau_spec_s == result.tau_s[4]
    assert report.worst.margin == pytest.approx(1.5, rel=1e-9)


# --------------------------------------------------------------------------
# where a limit may be evaluated at all
# --------------------------------------------------------------------------


def test_spec_afs_lands_the_grid_on_every_spec_tau():
    series = power_law_phase(
        n=4096, alpha=0, rate_hz=10.0,
        rng=np.random.default_rng(3), h_alpha=1e-24,
    )
    spec = ((1.0, 1e-12), (10.0, 1e-13), (100.0, 1e-14))
    afs = spec_afs(spec, series.tau0_s)
    assert list(afs) == [10, 100, 1000]
    result = oadev(series, taus=afs)
    report = build(series, result, confidence(result), spec=spec)
    for check, (tau, _) in zip(report.specs, spec):
        assert check.evaluated, check.reason
        assert check.tau_used_s == pytest.approx(tau, rel=1e-12)


def test_a_limit_at_an_unmeasured_tau_is_refused_not_interpolated(measurement):
    """A datasheet states a limit at a tau. Between them is a drawing."""
    _, result, _ = measurement
    off = float(result.tau_s[3]) * (1.0 + 4.0 * SPEC_TAU_RTOL)
    check = _report(measurement, spec=((off, 1e-12),)).specs[0]
    assert not check.evaluated
    assert check.verdict == "not evaluated"
    assert "not interpolated" in check.reason
    assert not np.isfinite(check.margin)


def test_a_spec_within_tolerance_is_evaluated_and_says_which_tau(measurement):
    _, result, _ = measurement
    near = float(result.tau_s[3]) * (1.0 + 0.5 * SPEC_TAU_RTOL)
    check = _report(measurement, spec=((near, 1e-9),)).specs[0]
    assert check.evaluated
    assert check.tau_spec_s == near
    assert check.tau_used_s == pytest.approx(float(result.tau_s[3]))


def test_without_an_interval_there_is_no_verdict(measurement):
    """The honest outcome for MTIE, and for any estimator with no CI.

    Falling back to comparing the point estimate would be simple
    acceptance with a 50 % false-accept risk, silently.
    """
    _, result, _ = measurement
    report = _report(measurement, spec=((float(result.tau_s[2]), 1e-9),),
                     ci_given=False)
    assert report.verdict == "none"
    assert report.worst is None
    check = report.specs[0]
    assert not check.evaluated
    assert "false-accept" in check.reason
    assert not report.has_intervals


# --------------------------------------------------------------------------
# the rows
# --------------------------------------------------------------------------


def test_rows_carry_the_interval_the_bias_factor_and_the_method(measurement):
    _, result, ci = measurement
    report = _report(measurement)
    assert len(report.rows) == result.tau_s.size
    for i, row in enumerate(report.rows):
        assert row.dev_lo <= row.dev <= row.dev_hi
        assert row.edof == pytest.approx(float(ci.edof[i]))
        assert row.bias == pytest.approx(deviation_bias_factor(row.edof))
        assert 0.0 < row.bias <= 1.0
        assert row.method


def test_the_noise_type_is_identified_per_tau(measurement):
    """White FM is what was synthesized; alpha = 0 in the h_alpha convention."""
    report = _report(measurement)
    identified = [r.alpha for r in report.rows if r.alpha is not None]
    assert identified, "no tau got a noise identification"
    assert identified.count(0) >= len(identified) - 1


def test_noise_identification_can_be_switched_off(measurement):
    report = _report(measurement, identify_noise=False)
    assert all(r.alpha is None and r.alpha_name == "" for r in report.rows)


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------


def test_the_digest_ignores_who_ran_it(measurement):
    """Two copies of a report are compared by sixteen characters."""
    a = _report(measurement, title="A", operator="alice", subject="x")
    b = _report(measurement, title="B", operator="bob", subject="y")
    assert a.digest == b.digest
    assert a.generated  # and the timestamp is still recorded


def test_the_digest_moves_when_a_limit_moves(measurement):
    _, result, _ = measurement
    tau = float(result.tau_s[2])
    a = _report(measurement, spec=((tau, 1e-12),))
    b = _report(measurement, spec=((tau, 1.1e-12),))
    assert a.digest != b.digest


def test_the_digest_moves_when_the_measurement_moves(measurement):
    series, result, ci = measurement
    other = power_law_phase(
        n=4096, alpha=0, rate_hz=1.0,
        rng=np.random.default_rng(12), h_alpha=1e-24,
    )
    other_result = oadev(other, taus=result.af)
    a = build(series, result, ci)
    b = build(other, other_result, confidence(other_result))
    assert a.digest != b.digest


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def test_html_is_self_contained(measurement):
    """It gets emailed to an assessor and opened three years later."""
    html = to_html(_report(measurement))
    for external in ("http://", "https://", "<script", "<link", "url("):
        assert external not in html


def test_html_escapes_what_it_is_given(measurement):
    html = to_html(_report(measurement, subject="<script>x</script>",
                           notes="a & b"))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "a &amp; b" in html


def test_html_states_the_decision_rule_and_the_two_source_caveat(measurement):
    """Both are load-bearing, and a report that omits them misleads."""
    _, result, ci = measurement
    html = to_html(_report(measurement,
                           spec=((float(result.tau_s[1]), float(ci.dev_hi[1])),)))
    assert "Decision rule" in html
    assert "indeterminate" in html
    assert "bounds the device under test from above" in html
    assert "not a calibration certificate" in html


def test_html_prints_no_bare_nan(measurement):
    """An em dash where there is no number. 'nan' in a cell is a bug report.

    The report with no interval is the hard case: every bound, every
    degree of freedom and every margin is absent at once.
    """
    _, result, _ = measurement
    off = float(result.tau_s[3]) * 1.5
    html = to_html(_report(measurement, spec=((off, 1e-12),), ci_given=False))
    bare = re.findall(r">\s*[-+]?nan\s*<", html, flags=re.IGNORECASE)
    assert not bare, bare


def test_json_carries_every_row_and_the_verdict(measurement):
    _, result, ci = measurement
    report = _report(measurement,
                     spec=((float(result.tau_s[1]), float(ci.dev_hi[1])),))
    data = json.loads(to_json(report))
    assert len(data["rows"]) == len(report.rows)
    assert data["verdict"] == report.verdict
    assert data["digest"] == report.digest
    assert data["provenance"]["steps"]


def test_the_figure_is_embedded_not_referenced(measurement):
    """Evidence in a sibling file arrives with the evidence missing."""
    matplotlib = pytest.importorskip("matplotlib")
    del matplotlib
    from stabilis.report import figure_uri

    _, result, ci = measurement
    uri = figure_uri(result, ci, spec=((1.0, 1e-12),), spec_label="datasheet")
    assert uri.startswith("data:image/png;base64,")
    html = to_html(_report(measurement), uri)
    assert uri in html


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------


def test_cli_writes_a_report_and_states_the_verdict(tmp_path, capsys):
    series = power_law_phase(
        n=2048, alpha=0, rate_hz=1.0,
        rng=np.random.default_rng(5), h_alpha=1e-24,
    )
    data = tmp_path / "dut.dat"
    np.savetxt(data, series.x, fmt="%.17e")
    out = tmp_path / "report.html"
    code = main([
        "report", str(data), "--rate", "1", "--taus", "octave",
        "--spec", "1:1e-9", "--spec", "10:1e-15", "--no-figure",
        "--subject", "unit 42 vs reference", "--out", str(out),
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "FAIL" in printed
    assert "worst margin" in printed
    html = out.read_text(encoding="utf-8")
    assert "unit 42 vs reference" in html
    # The 10 s limit is not on the octave grid; the grid was extended to it.
    assert "<td class='num'>10</td>" in html


def test_cli_json_is_parseable(tmp_path, capsys):
    series = power_law_phase(
        n=1024, alpha=0, rate_hz=1.0,
        rng=np.random.default_rng(6), h_alpha=1e-24,
    )
    data = tmp_path / "dut.dat"
    np.savetxt(data, series.x, fmt="%.17e")
    assert main(["report", str(data), "--rate", "1", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "none"
    assert payload["rows"]


def test_cli_spec_needs_a_colon(tmp_path):
    series = power_law_phase(
        n=512, alpha=0, rate_hz=1.0,
        rng=np.random.default_rng(7), h_alpha=1e-24,
    )
    data = tmp_path / "dut.dat"
    np.savetxt(data, series.x, fmt="%.17e")
    assert main(["report", str(data), "--rate", "1", "--spec", "1e-12"]) == 1
