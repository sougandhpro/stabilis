"""Limit masks, and the verdict a unit gets against one.

The milestone criterion for M8 is that "a unit measured against a mask
yields the same verdict as hand-evaluation at every tau, and the report
states the margin at the worst point rather than only pass/fail". The first
test here is literally that: limits worked out by hand from the text of
ITU-T G.811, and every verdict checked against them.

The transcription itself is checked by arithmetic rather than by eye. Both
of G.811's masks are continuous at their breakpoints, and the two forms are
written differently on either side -- so agreement there is a real check
that the digits went in correctly, and a typo shows up as a step.
"""

from typing import NamedTuple

import numpy as np
import pytest

from stabilis import PhaseSeries, confidence, oadev, power_law_phase
from stabilis.mask import (
    CATALOG,
    Mask,
    Segment,
    available,
    evaluate,
    limit_at,
)
from stabilis.provenance import new_provenance
from stabilis.stats.core import StabilityResult

MTIE = CATALOG["itu-t-g.811-mtie"]
TDEV = CATALOG["itu-t-g.811-tdev"]


class _Record(NamedTuple):
    """Stands in for a series: the two attributes a condition reads.

    A real record long enough to satisfy `12*tau` at tau = 5000 s would be
    1.8 million samples at G.811's sampling rate, which is a lot of memory
    to prove some arithmetic about record lengths. The CLI test below runs
    the real thing end to end.
    """

    tau0_s: float
    duration_s: float


def _curve(kind, taus, devs, is_variance=True):
    """A result with hand-chosen values, so the expected verdicts are known."""
    taus = np.asarray(taus, dtype=np.float64)
    devs = np.asarray(devs, dtype=np.float64)
    return StabilityResult(
        kind=kind, af=np.arange(1, taus.size + 1, dtype=np.int64),
        tau_s=taus, var=devs ** 2,
        n_terms=np.full(taus.size, 100, dtype=np.int64), rate_hz=30.0,
        provenance=new_provenance("test"), is_variance=is_variance,
    )


# --------------------------------------------------------------------------
# The transcription
# --------------------------------------------------------------------------


def test_the_g811_masks_are_continuous_at_their_breakpoints():
    """Written differently on either side, so agreement checks the digits.

    G.811's MTIE changes from `0.275e-3*tau + 0.025 us` to
    `1e-5*tau + 0.29 us` at 1000 s; both give 0.3 us there. TDEV changes
    from a 3 ns floor to `0.03*tau ns` at 100 s and back to a 30 ns ceiling
    at 1000 s. A mistyped coefficient shows up here as a step.
    """
    assert limit_at(MTIE, 1000.0) == pytest.approx(0.3e-6, rel=1e-12)
    assert limit_at(MTIE, 1000.0 + 1e-9) == pytest.approx(0.3e-6, rel=1e-9)
    assert limit_at(TDEV, 100.0) == pytest.approx(3e-9, rel=1e-12)
    assert limit_at(TDEV, 100.0 + 1e-9) == pytest.approx(3e-9, rel=1e-9)
    assert limit_at(TDEV, 1000.0) == pytest.approx(30e-9, rel=1e-12)
    assert limit_at(TDEV, 1000.0 + 1e-9) == pytest.approx(30e-9, rel=1e-9)


@pytest.mark.parametrize("tau,want", [
    (1.0, 0.025275e-6),        # 0.275e-3 * 1 + 0.025 us
    (100.0, 0.0525e-6),        # 0.275e-3 * 100 + 0.025 us
    (1000.0, 0.3e-6),          # the breakpoint
    (10_000.0, 0.39e-6),       # 1e-5 * 10000 + 0.29 us
])
def test_the_mtie_limit_matches_hand_arithmetic(tau, want):
    assert limit_at(MTIE, tau) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("tau,want", [
    (1.0, 3e-9), (50.0, 3e-9), (100.0, 3e-9),
    (200.0, 6e-9),             # 0.03 * 200 ns
    (1000.0, 30e-9), (5000.0, 30e-9),
])
def test_the_tdev_limit_matches_hand_arithmetic(tau, want):
    assert limit_at(TDEV, tau) == pytest.approx(want, rel=1e-12)


def test_a_mask_is_never_extrapolated_past_its_stated_range():
    """G.811's TDEV mask ends at 10 000 s. Beyond it the standard states no
    limit, so neither does this -- continuing the last segment would be
    inventing a requirement."""
    assert limit_at(TDEV, 20_000.0) is None
    assert limit_at(TDEV, 0.05) is None
    assert limit_at(MTIE, 0.05) is None


def test_every_catalogued_mask_cites_its_source():
    for name in available():
        mask = CATALOG[name]
        assert "ITU-T" in mask.source and mask.source.strip()
        assert mask.tau_min > 0


# --------------------------------------------------------------------------
# Hand evaluation -- the milestone criterion
# --------------------------------------------------------------------------


def test_every_verdict_matches_hand_evaluation():
    """The M8 criterion, worked by hand from the text of G.811.

    Five taus, chosen to sit either side of both breakpoints, with measured
    TDEV values placed deliberately above and below limits computed here in
    the test rather than by the code under test.
    """
    taus = [1.0, 200.0, 1000.0, 5000.0, 20_000.0]
    limits = [3e-9, 6e-9, 30e-9, 30e-9, None]
    measured = [1e-9,        # under 3 ns    -> pass
                7e-9,        # over 6 ns     -> fail
                30e-9,       # exactly at 30 ns -> pass, the limit is inclusive
                1e-9,        # well under    -> pass
                1e-3]        # no limit here -> not_evaluated
    want = ["pass", "fail", "pass", "pass", "not_evaluated"]

    record = _Record(tau0_s=1.0 / 30.0, duration_s=12 * 20_000.0)
    result = evaluate(_curve("tdev", taus, measured), TDEV, ci=None,
                      series=record)

    for check, tau, limit, verdict in zip(result.checks, taus, limits, want):
        assert check.tau_s == tau
        assert check.limit == (None if limit is None
                               else pytest.approx(limit, rel=1e-12))
        assert check.verdict == verdict, tau
    assert result.verdict == "fail"          # the worst present


def test_the_worst_margin_is_reported_not_only_the_verdict():
    """"A unit that passes by 1% and one that passes by 40% are not the same
    unit" -- the second half of the milestone criterion."""
    record = _Record(tau0_s=1.0 / 30.0, duration_s=1e9)
    result = evaluate(_curve("tdev", [1.0, 200.0], [1e-9, 5.9e-9]), TDEV,
                      series=record)
    assert result.verdict == "pass"
    assert result.worst.tau_s == 200.0
    assert result.worst.margin == pytest.approx(6e-9 / 5.9e-9, rel=1e-12)


def test_the_worst_margin_ignores_points_that_were_not_evaluated():
    """A tau with no limit has no margin, and must not become the worst one."""
    record = _Record(tau0_s=1.0 / 30.0, duration_s=1e9)
    result = evaluate(_curve("tdev", [20_000.0, 1.0], [1e-3, 2.9e-9]), TDEV,
                      series=record)
    assert result.worst.tau_s == 1.0


# --------------------------------------------------------------------------
# A maximum and a variance are not judged the same way
# --------------------------------------------------------------------------


def test_exceeding_an_mtie_limit_fails_without_needing_an_interval():
    """The excursion was observed. No statistics are required to believe it."""
    record = _Record(tau0_s=1.0 / 30.0, duration_s=1e9)
    result = evaluate(_curve("mtie", [1.0], [1e-6], is_variance=False),
                      MTIE, series=record)
    assert result.verdict == "fail"
    assert result.checks[0].limit == pytest.approx(0.025275e-6, rel=1e-12)


def test_staying_under_an_mtie_limit_passes_only_over_the_window_measured():
    """A longer record can only find a larger maximum, never a smaller one,
    so the window is part of the verdict rather than a footnote."""
    record = _Record(tau0_s=1.0 / 30.0, duration_s=1e9)
    result = evaluate(_curve("mtie", [1.0], [1e-9], is_variance=False),
                      MTIE, series=record)
    assert result.verdict == "pass"
    assert "longer record" in result.checks[0].reason


def test_a_variance_is_judged_on_its_interval_not_its_point():
    """The rule already settled in theory/report.md, applied to masks.

    A point below the limit whose upper bound is above it is
    `indeterminate`: the record is too short to decide, and calling it a
    pass would be simple acceptance with the error running towards passing.
    """
    series = power_law_phase(4096, 0.0, 30.0, np.random.default_rng(3),
                             h_alpha=1e-24)
    result = oadev(series, taus="octave", keep_coeffs=True)
    ci = confidence(result, series=series)

    # A mask placed exactly at each measured point: the interval straddles
    # it everywhere by construction.
    straddling = Mask(
        name="straddle", quantity="oadev", source="ITU-T (test fixture)",
        segments=tuple(
            Segment(tau_lo=t * 0.99, tau_hi=t * 1.01, offset=float(d))
            for t, d in zip(result.tau_s, result.dev)),
    )
    verdicts = {c.verdict for c in evaluate(result, straddling, ci).checks}
    assert verdicts == {"indeterminate"}

    # The same curve against a limit far above every interval passes.
    generous = Mask(
        name="generous", quantity="oadev", source="ITU-T (test fixture)",
        segments=(Segment(tau_lo=0.0, tau_hi=np.inf, offset=1.0),))
    assert evaluate(result, generous, ci).verdict == "pass"


def test_without_an_interval_a_variance_is_judged_on_the_point():
    """Explicitly asked for, and it is simple acceptance -- so it is not the
    default anywhere."""
    result = _curve("tdev", [1.0], [2.9e-9])
    record = _Record(tau0_s=1.0 / 30.0, duration_s=1e9)
    assert evaluate(result, TDEV, ci=None, series=record).verdict == "pass"


# --------------------------------------------------------------------------
# The standard's measurement conditions are part of its limit
# --------------------------------------------------------------------------


def test_a_record_sampled_too_coarsely_is_not_compared_at_all():
    """G.811 requires tau0 no coarser than 1/30 s. A 1 Hz record has not
    been measured under the standard's conditions, whatever the numbers
    say, so the answer is 'not evaluated' and not 'pass'."""
    result = evaluate(_curve("tdev", [1.0], [1e-12]), TDEV,
                      series=_Record(tau0_s=1.0, duration_s=1e9))
    assert result.verdict == "not_evaluated"
    assert "coarser" in result.checks[0].reason


def test_a_record_shorter_than_twelve_tau_is_not_compared_at_that_tau():
    """"The minimum measurement period for TDEV is twelve times the
    integration period (T = 12 tau)" -- G.811 clause 6."""
    record = _Record(tau0_s=1.0 / 30.0, duration_s=1000.0)
    checks = evaluate(_curve("tdev", [10.0, 500.0], [1e-12, 1e-12]), TDEV,
                      series=record).checks
    assert checks[0].verdict == "pass"            # 12*10 = 120 s, satisfied
    assert checks[1].verdict == "not_evaluated"   # 12*500 = 6000 s, not
    assert "12*tau" in checks[1].reason


def test_conditions_that_cannot_be_checked_are_not_assumed_met():
    """No series, no check -- and absence of evidence is not a pass."""
    result = evaluate(_curve("tdev", [1.0], [1e-12]), TDEV, series=None)
    assert result.verdict == "not_evaluated"
    assert "no series was supplied" in result.checks[0].reason


def test_a_mask_with_no_conditions_needs_no_series():
    plain = Mask(name="plain", quantity="tdev", source="ITU-T (test fixture)",
                 segments=(Segment(tau_lo=0.0, tau_hi=np.inf, offset=1e-9),))
    assert evaluate(_curve("tdev", [1.0], [1e-12]), plain).verdict == "pass"


def test_nothing_evaluated_is_recorded_as_a_warning_not_a_pass():
    result = evaluate(_curve("tdev", [1.0], [1e-12]), TDEV, series=None)
    assert any(s.action == "nothing_evaluated" for s in result.provenance.steps)


# --------------------------------------------------------------------------
# Refusals and bookkeeping
# --------------------------------------------------------------------------


def test_a_mask_refuses_a_curve_of_a_different_statistic():
    """A limit on TDEV says nothing about MTIE."""
    with pytest.raises(ValueError, match="limits 'tdev'"):
        evaluate(_curve("mtie", [1.0], [1e-9], is_variance=False), TDEV,
                 series=_Record(1 / 30, 1e9))


def test_the_overall_verdict_is_the_worst_one_present():
    record = _Record(tau0_s=1.0 / 30.0, duration_s=1e9)
    passing = [1e-12, 1e-12]
    assert evaluate(_curve("tdev", [1.0, 2.0], passing), TDEV,
                    series=record).verdict == "pass"
    assert evaluate(_curve("tdev", [1.0, 2.0], [1e-12, 9e-9]), TDEV,
                    series=record).verdict == "fail"


def test_the_evaluation_is_recorded_in_provenance():
    record = _Record(tau0_s=1.0 / 30.0, duration_s=1e9)
    result = evaluate(_curve("tdev", [1.0], [1e-12]), TDEV, series=record)
    step = [s for s in result.provenance.steps if s.action == "evaluate"][-1]
    assert step.params["mask"] == "itu-t-g.811-tdev"
    assert step.outcome == "pass"


# --------------------------------------------------------------------------
# The command line, end to end on a real record
# --------------------------------------------------------------------------


def test_cli_mask_judges_a_real_record_and_exits_nonzero_on_failure(
        tmp_path, capsys):
    from stabilis import save_phase
    from stabilis.cli import main

    path = tmp_path / "bad.dat"
    save_phase(power_law_phase(2**14, 0.0, 30.0, np.random.default_rng(2),
                               h_alpha=1e-14), path)
    assert main(["mask", str(path), "--mask", "itu-t-g.811-tdev"]) == 1
    out = capsys.readouterr().out
    assert "VERDICT: FAIL" in out
    assert "worst margin" in out
    assert "G.811" in out                       # the citation is on the page


def test_cli_mask_lists_the_catalogue(capsys):
    from stabilis.cli import main

    assert main(["mask", "--list-masks"]) == 0
    out = capsys.readouterr().out
    assert "itu-t-g.811-mtie" in out and "itu-t-g.811-tdev" in out


def test_cli_mask_json_carries_the_source_and_every_check(tmp_path, capsys):
    import json

    from stabilis import save_phase
    from stabilis.cli import main

    path = tmp_path / "x.dat"
    save_phase(power_law_phase(2**13, 0.0, 30.0, np.random.default_rng(2),
                               h_alpha=1e-27), path)
    assert main(["mask", str(path), "--mask", "itu-t-g.811-tdev",
                 "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "G.811" in payload["source"]
    assert payload["notes"]
    assert payload["checks"] and "verdict" in payload["checks"][0]


def test_cli_mask_refuses_an_unknown_name(tmp_path):
    from stabilis.cli import main

    path = tmp_path / "x.dat"
    path.write_text("1e-12\n2e-12\n3e-12\n4e-12\n")
    assert main(["mask", str(path), "--rate", "30", "--mask", "g.999"]) == 1
