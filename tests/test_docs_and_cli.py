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
