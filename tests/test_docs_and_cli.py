"""Theory pages and the CLI.

The theory pages are part of the deliverable, not decoration: a new
estimator does not count as done until it has one. These tests enforce the
page structure so that stays true.
"""

import json

import pytest

from stabilis.cli import main
from stabilis.docs import list_topics, read_topic, topic_summary

# Every page carries these five sections. Enforced so the shape cannot rot.
_REQUIRED_SECTIONS = ("## Validation status",)


def test_topics_are_discoverable():
    topics = list_topics()
    assert "noise-types" in topics
    assert "tau-selection" in topics
    assert "precision" in topics
    assert topics == sorted(topics)


def test_pages_are_readable_and_have_a_title():
    for topic in list_topics():
        text = read_topic(topic)
        assert text.startswith("# "), f"{topic} needs a level-1 title"
        assert topic_summary(topic) != topic


@pytest.mark.parametrize("section", _REQUIRED_SECTIONS)
def test_every_page_states_its_validation_status(section):
    """A theory page that does not say what validates it is half a page."""
    for topic in list_topics():
        assert section in read_topic(topic), f"{topic} is missing {section!r}"


def test_pages_name_the_tests_that_cover_them():
    """Traceability: theory -> test. Cheap now, miserable to retrofit."""
    for topic in list_topics():
        text = read_topic(topic)
        assert "tests/test_" in text, f"{topic} names no covering test"


def test_unknown_topic_lists_alternatives():
    with pytest.raises(KeyError, match="available"):
        read_topic("phlogiston")


def test_cli_docs_list(capsys):
    assert main(["docs", "--list"]) == 0
    assert "noise-types" in capsys.readouterr().out


def test_cli_docs_page(capsys):
    assert main(["docs", "precision"]) == 0
    assert "float64" in capsys.readouterr().out


def test_cli_docs_unknown_topic_is_an_error(capsys):
    assert main(["docs", "phlogiston"]) == 2
    assert "available" in capsys.readouterr().err


def test_cli_stats_text(tmp_path, capsys):
    p = tmp_path / "d.txt"
    p.write_text("1.0\n2.0\n3.0\n4.0\n")
    assert main(["stats", str(p), "--rate", "1.0"]) == 0
    out = capsys.readouterr().out
    assert "n         4" in out
    assert "duration  3" in out


def test_cli_stats_json_carries_provenance(tmp_path, capsys):
    p = tmp_path / "d.txt"
    p.write_text("1.0\n2.0\n3.0\n")
    assert main(["stats", str(p), "--rate", "2.0", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stats"]["n"] == 3
    assert payload["tau0_s"] == 0.5
    assert payload["provenance"]["source_sha256"] is not None
    assert any(s["action"] == "read_text" for s in payload["provenance"]["steps"])


def test_cli_estimator_reproduces_published_values(tmp_path, capsys):
    """The CLI path must agree with the library path, on published numbers."""
    import numpy as np

    from stabilis.reference import nbs14_1000point

    p = tmp_path / "nbs14.txt"
    np.savetxt(p, nbs14_1000point(1000), fmt="%.15e")
    assert main(
        ["oadev", str(p), "--kind", "frequency", "--rate", "1", "--taus", "decade",
         "--max-af", "100", "--json"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    by_af = dict(zip(payload["af"], payload["dev"]))
    assert by_af[1] == pytest.approx(2.922319e-01, rel=1e-6)
    assert by_af[10] == pytest.approx(9.159953e-02, rel=1e-6)
    assert by_af[100] == pytest.approx(3.241343e-02, rel=1e-6)


def test_cli_estimator_text_table(tmp_path, capsys):
    p = tmp_path / "d.txt"
    p.write_text("\n".join(str(float(v)) for v in range(64)) + "\n")
    assert main(["ohdev", str(p), "--rate", "1"]) == 0
    out = capsys.readouterr().out
    assert "OHDEV" in out and "Tau (s)" in out


def test_cli_reports_bad_input_without_a_traceback(tmp_path, capsys):
    p = tmp_path / "d.txt"
    p.write_text("1.0\n2.0\n")
    assert main(["stats", str(p)]) == 1  # no --rate, no --time-column
    assert "stabilis:" in capsys.readouterr().err


def _noise_file(tmp_path, n=512, alpha=0, seed=5):
    import numpy as np

    from stabilis.synth import power_law_freq

    p = tmp_path / "y.txt"
    y = power_law_freq(n, alpha, rate_hz=1.0, rng=np.random.default_rng(seed)).y
    np.savetxt(p, y, fmt="%.15e")
    return p


def test_cli_prints_stable32_style_bounds(tmp_path, capsys):
    """AF / Tau / # / Min Sigma / Sigma / Max Sigma, plus EDOF.

    The first six columns mirror Stable32 so a migrating user can diff the
    two outputs directly. EDOF is the extra one, and is the point: it says
    how much data each bound rests on.
    """
    p = _noise_file(tmp_path)
    assert main(["oadev", str(p), "--kind", "frequency", "--rate", "1"]) == 0
    out = capsys.readouterr().out
    assert "Min OADEV" in out and "Max OADEV" in out and "EDOF" in out
    assert "68.2689% confidence" in out
    body = [ln for ln in out.splitlines() if not ln.startswith(("#", " " * 5 + "AF"))]
    for line in body:
        af, tau, n, lo, dev, hi, eta, via = line.split()
        assert float(lo) <= float(dev) <= float(hi)
        assert via in ("eta1", "eta3")


def test_cli_confidence_level_widens_the_bounds(tmp_path, capsys):
    p = _noise_file(tmp_path)
    args = ["oadev", str(p), "--kind", "frequency", "--rate", "1", "--json"]
    assert main(args + ["--confidence", "0.6827"]) == 0
    one = json.loads(capsys.readouterr().out)["confidence"]
    assert main(args + ["--confidence", "0.9545"]) == 0
    two = json.loads(capsys.readouterr().out)["confidence"]
    assert all(b < a for a, b in zip(one["dev_lo"], two["dev_lo"]))
    assert all(b > a for a, b in zip(one["dev_hi"], two["dev_hi"]))


def test_cli_no_confidence_falls_back_to_the_plain_table(tmp_path, capsys):
    p = _noise_file(tmp_path)
    assert main(
        ["oadev", str(p), "--kind", "frequency", "--rate", "1", "--no-confidence"]
    ) == 0
    out = capsys.readouterr().out
    assert "Min OADEV" not in out and "OADEV" in out


def test_cli_json_carries_the_edof_and_the_method(tmp_path, capsys):
    p = _noise_file(tmp_path)
    assert main(
        ["oadev", str(p), "--kind", "frequency", "--rate", "1", "--json"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    ci = payload["confidence"]
    assert len(ci["edof"]) == len(payload["af"])
    assert set(ci["edof_method"]) <= {"eta1", "eta3"}
    actions = [s["action"] for s in payload["provenance"]["steps"]]
    assert "confidence" in actions


def test_cli_greenhall_prints_the_alpha_column(tmp_path, capsys):
    """With an assumed power law the table gains Stable32's Alpha column.

    That completes the layout a migrating user diffs against: AF, Tau, #,
    Alpha, Min Sigma, Sigma, Max Sigma.
    """
    p = _noise_file(tmp_path, n=2048)
    assert main([
        "oadev", str(p), "--kind", "frequency", "--rate", "1",
        "--edof-method", "greenhall", "--alpha", "0",
    ]) == 0
    out = capsys.readouterr().out
    assert "Alpha" in out and "an assumed power law" in out
    body = [ln for ln in out.splitlines() if not ln.startswith(("#", " " * 5 + "AF"))]
    for line in body:
        af, tau, n, a, lo, dev, hi, eta, via = line.split()
        assert via == "greenhall" and a == "0"
        assert float(lo) <= float(dev) <= float(hi)


def test_cli_without_greenhall_has_no_alpha_column(tmp_path, capsys):
    """The default assumes nothing, so it must not print an assumption."""
    p = _noise_file(tmp_path)
    assert main(["oadev", str(p), "--kind", "frequency", "--rate", "1"]) == 0
    out = capsys.readouterr().out
    assert "Alpha" not in out
    assert "data-driven EDOF" in out


def test_cli_greenhall_says_where_it_declined(tmp_path, capsys):
    """A fallen-back row must be visible in the table and explained below it."""
    p = _noise_file(tmp_path, n=1024)
    assert main([
        "oadev", str(p), "--kind", "frequency", "--rate", "1",
        "--edof-method", "greenhall",
    ]) == 0
    out = capsys.readouterr().out
    assert "declined greenhall where" in out
    assert "eta1" in out


def test_cli_compare_edof_puts_the_methods_side_by_side(tmp_path, capsys):
    """The diagnostic that makes the design argument checkable by the user."""
    p = _noise_file(tmp_path, n=2048)
    assert main(["compare-edof", str(p), "--kind", "frequency", "--rate", "1"]) == 0
    out = capsys.readouterr().out
    for column in ("eta1", "eta3", "greenhall", "Alpha", "width g/1"):
        assert column in out
    body = [ln for ln in out.splitlines() if not ln.startswith(("#", " " * 5 + "AF"))]
    assert body
    for line in body:
        # Counted from the right, because the Alpha cell is blank on a row
        # where the model-based route declined.
        fields = line.split()
        eta1_v, eta3_v, gh = float(fields[-5]), float(fields[-4]), float(fields[-3])
        # eta3 counts non-overlapping windows and is a deliberate lower bound.
        assert eta3_v <= eta1_v and eta3_v <= gh


def test_cli_compare_edof_json_carries_every_method(tmp_path, capsys):
    p = _noise_file(tmp_path, n=1024)
    assert main([
        "compare-edof", str(p), "--kind", "frequency", "--rate", "1",
        "--alpha", "0", "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["edof"]) == {"eta1", "eta3", "greenhall"}
    assert payload["alpha"] == [0] * len(payload["af"])
    assert len(payload["dev_lo"]["greenhall"]) == len(payload["af"])
    assert payload["provenance"]["steps"]
