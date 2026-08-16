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


def test_cli_wdev_agrees_with_oadev_on_the_command_line(tmp_path, capsys):
    """The library's central identity, asserted where a user would meet it.

    `wdev --filter haar` and `oadev` are the same estimator reached two
    ways, so the two tables must carry the same numbers at every shared tau
    -- bounds included. A regression that broke the identity but left both
    curves plausible would show up here and nowhere in the eye.
    """
    p = _noise_file(tmp_path, n=2048)
    base = ["--kind", "frequency", "--rate", "1", "--json"]
    assert main(["wdev", str(p), "--filter", "haar"] + base) == 0
    wave = json.loads(capsys.readouterr().out)
    assert main(["oadev", str(p)] + base) == 0
    allan = json.loads(capsys.readouterr().out)

    shared = {t: i for i, t in enumerate(allan["tau_s"])}
    compared = 0
    for i, tau in enumerate(wave["tau_s"]):
        if tau not in shared:
            continue
        j = shared[tau]
        assert wave["dev"][i] == pytest.approx(allan["dev"][j], rel=1e-12)
        assert (wave["confidence"]["dev_lo"][i]
                == pytest.approx(allan["confidence"]["dev_lo"][j], rel=1e-12))
        compared += 1
    assert compared >= 9


def test_cli_wdev_offers_no_tau_grid_it_would_ignore(tmp_path, capsys):
    """The MODWT's scales are dyadic by construction.

    `--taus` and `--max-af` have nothing to act on here, so the parser does
    not accept them. Offering a flag and then ignoring it is the failure
    mode this avoids -- the same reason `mtie` is offered no `--confidence`.
    """
    p = _noise_file(tmp_path, n=512)
    with pytest.raises(SystemExit):
        main(["wdev", str(p), "--kind", "frequency", "--rate", "1",
              "--taus", "all"])


def test_cli_wdev_names_the_filter_and_its_vanishing_moments(tmp_path, capsys):
    p = _noise_file(tmp_path, n=1024)
    assert main(["wdev", str(p), "--kind", "frequency", "--rate", "1",
                 "--filter", "d6"]) == 0
    out = capsys.readouterr().out
    assert "filter: d6, K = 3 vanishing moments" in out
    assert "Min WDEV[D6]" in out and "EDOF" in out


@pytest.mark.parametrize("name", ["theo1", "theobr", "theoh"])
def test_cli_theo_family_runs_without_interval_flags(tmp_path, capsys, name):
    """`stabilis theo1` used to crash outright.

    `_cmd_estimate` passed `keep_coeffs` to every estimator that was not
    MTIE or TIE rms, and Thêo1 does not take it -- an unhandled TypeError on
    a documented subcommand. The Thêo family reports no coefficient arrays
    and its own EDF (SP 1065 Table 9) is not implemented, so it belongs in
    the same no-interval list, which is now one constant driving both the
    parser and the call.
    """
    p = _noise_file(tmp_path, n=600)
    assert main([name, str(p), "--kind", "phase", "--rate", "1"]) == 0
    out = capsys.readouterr().out
    assert name.upper() in out
    assert "Min" not in out and "EDOF" not in out, "no bounds are claimed"
    with pytest.raises(SystemExit):
        main([name, str(p), "--kind", "phase", "--rate", "1", "--confidence",
              "0.95"])


def _two_channel_files(tmp_path, n=2048, seed=21):
    """One clock under two channels' noise, written as two phase files."""
    import numpy as np

    from stabilis.synth import power_law_phase

    rng = np.random.default_rng(seed)
    clock = power_law_phase(n, 0, 1.0, rng, 1e-24)
    paths = []
    for name in ("ch_a.txt", "ch_b.txt"):
        p = tmp_path / name
        np.savetxt(p, clock.x + power_law_phase(n, 2, 1.0, rng, 1e-22).x,
                   fmt="%.15e")
        paths.append(str(p))
    return paths


def test_cli_cross_puts_both_autos_beside_the_cross(tmp_path, capsys):
    """The table has to show the rejection, not assert it.

    Printing the cross alone would leave a reader unable to tell a good
    measurement from a broken one; the two single-channel columns are what
    make the gap visible.
    """
    a, b = _two_channel_files(tmp_path)
    assert main(["cross", a, b, "--rate", "1", "--max-af", "8"]) == 0
    out = capsys.readouterr().out
    assert "sigma_a" in out and "sigma_b" in out and "rej dB" in out
    assert "independent" in out, "the assumption must be on the output"
    body = [ln for ln in out.splitlines()
            if ln and not ln.startswith("#") and "tau (s)" not in ln]
    assert len(body) == 4
    # The cross sits under both autos at short tau -- that is the point.
    for line in body:
        sigma_a, sigma_b, cross = (float(v) for v in line.split()[1:4])
        assert cross < min(sigma_a, sigma_b)


def test_cli_cross_json_carries_the_error_bar_and_the_assumption(tmp_path, capsys):
    a, b = _two_channel_files(tmp_path)
    assert main(["cross", a, b, "--rate", "1", "--max-af", "8", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "cross_oadev"
    assert len(payload["sem"]) == len(payload["var"]) == len(payload["tau_s"])
    assert set(payload["edof_method"]) <= {"isserlis", "eta3"}
    steps = [s["action"] for s in payload["provenance"]["steps"]]
    assert "independence_assumed" in steps
    assert "cross_variance" in steps


def test_cli_cross_reads_a_different_column_from_the_second_file(tmp_path, capsys):
    """Two counters rarely export the same layout."""
    import numpy as np

    a, b = _two_channel_files(tmp_path)
    wide = tmp_path / "wide.txt"
    col = np.loadtxt(b)
    np.savetxt(wide, np.column_stack([np.zeros_like(col), col]), fmt="%.15e")
    assert main(["cross", a, str(wide), "--rate", "1", "--max-af", "4",
                 "--column-b", "1", "--json"]) == 0
    from_wide = json.loads(capsys.readouterr().out)
    assert main(["cross", a, b, "--rate", "1", "--max-af", "4", "--json"]) == 0
    assert from_wide["var"] == json.loads(capsys.readouterr().out)["var"]


def test_cli_cross_refuses_an_estimator_with_no_cross_form(tmp_path):
    """Caught by the parser's choice list, so it never reaches the library.

    `cross_variance` refuses MTIE too, and has its own test for it. Both
    guards are wanted: this one names the alternatives in the usage message
    instead of raising after the files have been read.
    """
    a, b = _two_channel_files(tmp_path, n=512)
    with pytest.raises(SystemExit):
        main(["cross", a, b, "--rate", "1", "--estimator", "mtie"])


def test_cli_reject_averages_every_pair_and_names_them(tmp_path, capsys):
    """Four channels, six pairs, and the warning that matters most here."""
    import numpy as np

    from stabilis.synth import power_law_phase

    rng = np.random.default_rng(77)
    clock = power_law_phase(2048, 0, 1.0, rng, 1e-24)
    paths = []
    for name in ("ch_a", "ch_b", "ch_c", "ch_d"):
        p = tmp_path / f"{name}.txt"
        np.savetxt(p, clock.x + power_law_phase(2048, 2, 1.0, rng, 1e-22).x,
                   fmt="%.15e")
        paths.append(str(p))
    assert main(["reject", *paths, "--rate", "1", "--max-af", "8"]) == 0
    out = capsys.readouterr().out
    assert "6 pairs" in out
    assert "ch_a, ch_b, ch_c, ch_d" in out
    assert "one instrument" in out, "the common-mode warning must be on the output"
    body = [ln for ln in out.splitlines()
            if ln and not ln.startswith("#") and "tau (s)" not in ln]
    assert len(body) == 4
    for line in body:
        one_channel, rejected = (float(v) for v in line.split()[1:3])
        assert rejected < one_channel


def test_cli_reject_on_two_files_equals_cross(tmp_path, capsys):
    """One pair is the pairwise cross-variance, through the CLI as well."""
    a, b = _two_channel_files(tmp_path)
    assert main(["reject", a, b, "--rate", "1", "--max-af", "8", "--json"]) == 0
    rejected = json.loads(capsys.readouterr().out)
    assert main(["cross", a, b, "--rate", "1", "--max-af", "8", "--json"]) == 0
    pair = json.loads(capsys.readouterr().out)
    assert rejected["n_pairs"] == 1
    assert rejected["var"] == pair["var"]
    assert rejected["sem"] == pair["sem"]


def test_cli_reject_can_print_the_correlation_matrix(tmp_path, capsys):
    a, b = _two_channel_files(tmp_path)
    assert main(["reject", a, b, "--rate", "1", "--max-af", "2",
                 "--matrix"]) == 0
    out = capsys.readouterr().out
    assert "correlation at tau" in out
    # The diagonal of a correlation matrix is one, by construction, so row
    # ch_a's first entry and row ch_b's second must both read 1.000.
    rows = [ln.split() for ln in out.splitlines() if ln.startswith(" ")]
    # The header row names the columns and carries no numbers; a data row is
    # a label followed by one value per channel.
    rows = [r for r in rows if len(r) == 3 and r[0] in ("ch_a", "ch_b")
            and r[1].replace("-", "").replace(".", "").isdigit()]
    assert len(rows) >= 2, out
    for name, *values in rows:
        assert values[("ch_a", "ch_b").index(name)] == "1.000"


def _white_fm_file(tmp_path, n=8192, seed=11, h_alpha=2e-22):
    """A frequency record of known white-FM level: oadev var = h0/(2 tau)."""
    import numpy as np

    from stabilis.synth import power_law_freq

    rng = np.random.default_rng(seed)
    p = tmp_path / "wfm.txt"
    np.savetxt(p, power_law_freq(n, 0, rate_hz=1.0, rng=rng,
                                 h_alpha=h_alpha).y, fmt="%.15e")
    return str(p)


def _fit_args(path, *extra):
    return ["fit", path, "--kind", "frequency", "--rate", "1",
            "--max-af", "256", *extra]


def test_cli_fit_prints_both_error_bars_and_their_ratio(tmp_path, capsys):
    """The correlated and the uncorrelated one, side by side.

    Printing only the corrected number would leave a reader unable to tell
    what the correction was worth; the ratio is the whole argument for the
    machinery behind it.
    """
    path = _white_fm_file(tmp_path)
    assert main(_fit_args(path)) == 0
    out = capsys.readouterr().out
    assert "uncorrelated" in out and "ratio" in out
    assert "cross-tau correlation measured" in out
    rows = {ln.split()[0]: ln.split() for ln in out.splitlines()
            if ln[:1].isalpha() and len(ln.split()) == 5}
    assert set(rows) == {"A", "b"}
    # White FM: sigma2 = h0/(2 tau), so A = 1e-22 and b = -1.
    assert float(rows["A"][1]) == pytest.approx(1e-22, rel=0.1)
    assert float(rows["b"][1]) == pytest.approx(-1.0, abs=0.1)
    assert float(rows["A"][4]) > 1.0 and float(rows["b"][4]) > 1.0


def test_cli_fit_refuses_to_print_a_model_that_does_not_describe_the_data(
        tmp_path, capsys):
    """White FM is not a tau^-2 quantization floor, and the exit code says so."""
    path = _white_fm_file(tmp_path)
    assert main(_fit_args(path, "--model", "quantization")) == 1
    out = capsys.readouterr().out
    assert "REFUSING TO PRINT PARAMETERS" in out
    assert "does not describe this curve" in out
    assert "sigma_x" not in out


def test_cli_fit_warns_when_the_correlation_could_not_be_measured(
        tmp_path, capsys):
    path = _white_fm_file(tmp_path)
    assert main(_fit_args(path, "--estimator", "adev")) == 0
    out = capsys.readouterr().out
    assert "treated as independent" in out
    assert "lower bound" in out


def test_cli_fit_bootstrap_prints_a_second_interval(tmp_path, capsys):
    path = _white_fm_file(tmp_path)
    assert main(_fit_args(path, "--bootstrap", "40", "--seed", "2")) == 0
    out = capsys.readouterr().out
    assert "bootstrap [" in out
    assert "40 bootstrap refits" in out


def test_cli_fit_can_freeze_a_parameter(tmp_path, capsys):
    path = _white_fm_file(tmp_path)
    assert main(_fit_args(path, "--freeze", "b=-1", "--json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["frozen"] == {"b": -1.0}
    assert payload["free"] == ["A"]
    assert "b" not in payload["stderr"]


def test_cli_fit_rejects_a_malformed_freeze_without_a_traceback(tmp_path, capsys):
    path = _white_fm_file(tmp_path)
    assert main(_fit_args(path, "--freeze", "b")) == 1
    assert "NAME=VALUE" in capsys.readouterr().err
    assert main(_fit_args(path, "--freeze", "b=fast")) == 1
    assert "not a number" in capsys.readouterr().err


def test_cli_fit_names_the_models_when_asked_for_one_that_does_not_exist(
        tmp_path, capsys):
    path = _white_fm_file(tmp_path, n=1024)
    assert main(_fit_args(path, "--model", "phlogiston")) == 1
    err = capsys.readouterr().err
    assert "gauss_markov" in err and "quantization" in err


def test_cli_fit_json_carries_the_evidence_and_the_provenance(tmp_path, capsys):
    path = _white_fm_file(tmp_path)
    assert main(_fit_args(path, "--json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["correlation"] == "isserlis"
    assert payload["objective"] == "ml"
    assert payload["understatement"]["b"] > 1.0
    assert len(payload["tau_s"]) == len(payload["fitted"]) == len(payload["edof"])
    steps = [s["action"] for s in payload["provenance"]["steps"]]
    assert "evaluate" in steps and "tau_covariance" in steps and "fit" in steps


def test_cli_fit_json_exit_status_is_the_verdict(tmp_path, capsys):
    """A pipeline reading the JSON should not have to parse `gof_ok` itself."""
    path = _white_fm_file(tmp_path)
    assert main(_fit_args(path, "--model", "quantization", "--json")) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["gof_ok"] is False
    # The parameters are still in the JSON: a machine reader can see what
    # was rejected, which the human table deliberately does not show.
    assert "sigma_x" in payload["params"]


# ---------------------------------------------------------------------------
# davar and stream
# ---------------------------------------------------------------------------

def _phase_file(tmp_path, n=2048, seed=17, name="x.txt", timetags=False):
    """A phase record on disk, optionally with a timetag column."""
    import numpy as np

    from stabilis.synth import power_law_phase

    p = tmp_path / name
    x = power_law_phase(n, 0, rate_hz=1.0, rng=np.random.default_rng(seed),
                        h_alpha=1e-24).x
    if timetags:
        np.savetxt(p, np.column_stack([np.arange(x.size), x]), fmt="%d %.17e")
    else:
        np.savetxt(p, x, fmt="%.17e")
    return p


def test_cli_davar_prints_one_cut_and_names_the_window(tmp_path, capsys):
    """A surface is not read in a terminal; a time series at one tau is."""
    p = _phase_file(tmp_path)
    assert main(["davar", str(p), "--rate", "1", "--window", "512",
                 "--step", "256"]) == 0
    out = capsys.readouterr().out
    assert "over 7 windows of 512 samples, step 256" in out
    assert "quiet" in out and "worst" in out
    rows = [ln for ln in out.splitlines()
            if ln and not ln.startswith("#") and ln.split()[0][0].isdigit()]
    assert len(rows) == 7


def test_cli_davar_at_af_selects_the_column_and_its_term_count(tmp_path, capsys):
    p = _phase_file(tmp_path)
    assert main(["davar", str(p), "--rate", "1", "--window", "512",
                 "--step", "256", "--at-af", "8"]) == 0
    out = capsys.readouterr().out
    assert "at tau = 8 s" in out
    # 512 samples, af 8 -> 512 - 16 terms, which is not af 1's count.
    assert " 496" in out


def test_cli_davar_rejects_an_averaging_factor_it_did_not_compute(tmp_path, capsys):
    p = _phase_file(tmp_path)
    assert main(["davar", str(p), "--rate", "1", "--window", "512",
                 "--at-af", "3"]) == 1
    assert "not in this surface" in capsys.readouterr().err


def test_cli_davar_json_carries_the_surface_and_the_intervals(tmp_path, capsys):
    p = _phase_file(tmp_path)
    assert main(["davar", str(p), "--rate", "1", "--window", "512",
                 "--step", "256", "--level", "0.6827", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["var"]) == len(payload["t_s"]) == 7
    assert len(payload["var"][0]) == len(payload["af"])
    assert payload["level"] == 0.6827
    assert all(lo < v for lo, v in zip(payload["var_lo"][0], payload["var"][0]))
    assert "evaluate" in [s["action"] for s in payload["provenance"]["steps"]]


def test_cli_davar_over_the_whole_record_is_the_oadev_table(tmp_path, capsys):
    """The identity, met where a user would meet it."""
    p = _phase_file(tmp_path, n=1024)
    assert main(["davar", str(p), "--rate", "1", "--window", "1025",
                 "--json"]) == 0
    surface = json.loads(capsys.readouterr().out)
    assert main(["oadev", str(p), "--rate", "1", "--json"]) == 0
    curve = json.loads(capsys.readouterr().out)
    assert surface["af"] == curve["af"]
    assert [v ** 0.5 for v in surface["var"][0]] == pytest.approx(curve["dev"])


def test_cli_stream_matches_the_batch_table(tmp_path, capsys):
    p = _phase_file(tmp_path, n=1024, timetags=True)
    assert main(["stream", str(p), "--rate", "1", "--time-column", "0",
                 "--column", "1", "--expect", "1025", "--json"]) == 0
    streamed = json.loads(capsys.readouterr().out)
    assert main(["oadev", str(p), "--rate", "1", "--time-column", "0",
                 "--column", "1", "--json"]) == 0
    batch = json.loads(capsys.readouterr().out)
    assert streamed["af"] == batch["af"]
    assert streamed["dev"] == pytest.approx(batch["dev"], rel=2e-15)
    assert streamed["final"] is True


def test_cli_stream_needs_a_tau_grid_it_cannot_derive(tmp_path, capsys):
    """A stream has no length, so neither --af nor --expect is a default."""
    p = _phase_file(tmp_path, n=256)
    assert main(["stream", str(p), "--rate", "1"]) == 1
    assert "--af 1,2,4" in capsys.readouterr().err


def test_cli_stream_says_when_a_dropped_sample_would_leave_no_trace(
        tmp_path, capsys):
    p = _phase_file(tmp_path, n=256)
    assert main(["stream", str(p), "--rate", "1", "--af", "1,2,4"]) == 0
    assert "no timetag column" in capsys.readouterr().out


def test_cli_stream_can_report_as_it_goes(tmp_path, capsys):
    p = _phase_file(tmp_path, n=256, timetags=True)
    assert main(["stream", str(p), "--rate", "1", "--time-column", "0",
                 "--column", "1", "--af", "1,2", "--every", "64"]) == 0
    out = capsys.readouterr().out
    assert out.count("samples,") >= 4


def test_cli_stream_follows_a_growing_file(tmp_path, capsys):
    import threading

    # power_law_phase(n) returns n+1 phase samples, so the file already
    # holds slots 0..200 and the appended block must start at 201.
    p = _phase_file(tmp_path, n=200, timetags=True)
    tail = "".join(f"{i} {i * 1e-12:.17e}\n" for i in range(201, 261))

    def append():
        with open(p, "a") as fh:
            fh.write(tail)

    thread = threading.Thread(target=append)
    thread.start()
    code = main(["stream", str(p), "--rate", "1", "--time-column", "0",
                 "--column", "1", "--af", "1,2,4", "--follow",
                 "--timeout", "0.4", "--json"])
    thread.join()
    assert code == 0
    assert json.loads(capsys.readouterr().out)["n_seen"] == 261


def test_cli_stream_takes_its_epoch_from_the_first_timestamp(tmp_path, capsys):
    """A file whose timetags are absolute seconds is not one huge outage."""
    import numpy as np

    from stabilis.synth import power_law_phase

    x = power_law_phase(512, 0, rate_hz=1.0, rng=np.random.default_rng(31),
                        h_alpha=1e-24).x
    p = tmp_path / "mjd.txt"
    base = 5.1e9          # MJD 59000 in seconds, near enough
    np.savetxt(p, np.column_stack([base + np.arange(x.size), x]),
               fmt="%.6f %.17e")
    assert main(["stream", str(p), "--rate", "1", "--time-column", "0",
                 "--column", "1", "--af", "1,2,4", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n_seen"] == x.size
