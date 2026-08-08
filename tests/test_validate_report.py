"""The validation report itself, which is a deliverable and so is tested.

The two properties that make it evidence rather than decoration are
structural, and therefore checkable: every claim states a tolerance, and
the whole thing is capable of failing. Both are asserted here.
"""

import json

import pytest

from stabilis.cli import main
from stabilis.validate import run, to_html, to_json
from stabilis.validate.report import MIN_COVERAGE_TRIALS, TIERS


@pytest.fixture(scope="module")
def report():
    """One run shared by the module -- the Monte-Carlo tier is not cheap."""
    return run(trials=MIN_COVERAGE_TRIALS)


def test_the_report_passes(report):
    assert report.passed, [c.name for c in report.failures]


def test_every_tier_is_represented(report):
    for tier in TIERS:
        if tier == "self-test":
            continue
        assert report.by_tier(tier), f"no checks in tier {tier}"


def test_every_check_states_a_tolerance_and_a_measurement(report):
    """A pass at an unstated tolerance says nothing.

    Enforced structurally rather than by review, so a check added later
    cannot skip it. The length floor is crude on purpose: it is there to
    stop 'ok' or '1e-9' appearing with no reason attached.
    """
    for c in report.checks:
        assert len(c.claim) > 40, f"{c.name}: claim is too thin"
        assert len(c.tolerance) > 20, f"{c.name}: tolerance without a reason"
        assert c.measured, f"{c.name}: no measured value"
        assert isinstance(c.passed, bool)  # not np.bool_, which JSON refuses
        # Single-line: these land in table cells and in a JSON payload, and
        # an embedded newline from a mis-joined string literal renders as
        # ragged whitespace that nobody notices until a customer reads it.
        for field in (c.claim, c.tolerance, c.measured):
            assert "\n" not in field, f"{c.name}: embedded newline"


def test_the_report_can_fail():
    """The single most important property. A report that cannot fail is worthless."""
    r = run(trials=MIN_COVERAGE_TRIALS, self_test=True)
    assert not r.passed
    assert [c.name for c in r.failures] == ["deliberate failure"]


def test_the_digest_is_reproducible(report):
    """Same seeds, same library, same digest -- the reproducibility claim."""
    again = run(trials=MIN_COVERAGE_TRIALS)
    assert again.digest == report.digest
    assert again.generated != "" and report.generated != ""


def test_the_digest_covers_the_evidence_not_the_timestamp(report):
    """Two runs differ in `generated` yet agree on `digest`, by construction."""
    payload = json.loads(to_json(report))
    assert payload["digest"] == report.digest
    assert "generated" in payload and payload["generated"] not in payload["digest"]


def test_the_digest_changes_when_a_check_changes(report):
    """Otherwise it is decoration rather than a seal."""
    tampered = run(trials=MIN_COVERAGE_TRIALS, self_test=True)
    assert tampered.digest != report.digest


def test_low_trial_counts_are_raised_rather_than_believed():
    """A coverage number from 20 realizations would look measured and not be."""
    r = run(trials=20)
    assert r.environment["monte_carlo_trials"] == MIN_COVERAGE_TRIALS


def test_json_is_valid_and_complete(report):
    payload = json.loads(to_json(report))
    assert payload["passed"] is True
    assert len(payload["checks"]) == len(report.checks)
    assert payload["environment"]["numpy"]
    assert payload["seeds"]  # published, so the Monte Carlo can be repeated


def test_html_carries_the_verdict_and_every_check(report):
    html = to_html(report)
    assert "PASS" in html and "stabilis validation report" in html
    assert report.digest[:16] in html
    for c in report.checks:
        assert c.name.replace("<", "&lt;").replace(">", "&gt;") in html
    # The ISO clause is the reason the document exists; it must survive edits.
    assert "7.11.2" in html


def test_html_renders_a_failure_as_a_failure():
    html = to_html(run(trials=MIN_COVERAGE_TRIALS, self_test=True))
    assert "class='verdict bad'" in html
    assert "deliberate failure" in html


def test_interop_checks_say_they_are_interop(report):
    """The epistemic caveat has to travel with the claim, not sit in a README."""
    for c in report.by_tier("interop"):
        assert "INTEROP" in c.claim or "interop" in c.claim.lower()
        assert "Stable32" in c.claim


def test_cli_validate_exit_status_is_the_verdict(tmp_path, capsys):
    out = tmp_path / "report.html"
    assert main(["validate", "--trials", "300", "--out", str(out)]) == 0
    assert out.exists() and "stabilis validation report" in out.read_text()
    assert "PASS: " in capsys.readouterr().out


def test_cli_validate_fails_loudly_on_the_self_test(capsys):
    """Exit 1 makes the report usable as a gate, not only as something to read."""
    assert main(["validate", "--trials", "300", "--self-test"]) == 1
    out = capsys.readouterr().out
    assert "[FAIL]" in out and "deliberate failure" in out


def test_cli_validate_json(capsys):
    assert main(["validate", "--trials", "300", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["digest"]
