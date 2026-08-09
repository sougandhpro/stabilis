"""The local GUI: routing, the path guard, and the data path behind it.

Tested through real HTTP against a real server on a free port wherever the
handler is involved, because the handler's job *is* the HTTP behaviour --
calling the functions directly would skip routing, JSON encoding and error
mapping, which is most of what could break.

Two groups matter more than the rest.

**The path-escape tests.** This serves local files to a browser, so
`safe_path` is the whole security model.

**The expression tests.** The page has a text box that turns into
arithmetic over a customer's file. `gui.expr` is a closed grammar with no
evaluator behind it, and the tests throw code at it to keep it that way.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from stabilis.gui import Handler, cache_clear, estimators, safe_path, serve
from stabilis.gui import expr, files, prepare
from stabilis.gui.server import ASSETS


@pytest.fixture(autouse=True)
def _fresh_cache():
    """No test may see a table another test loaded."""
    cache_clear()
    yield
    cache_clear()


@pytest.fixture
def tree(tmp_path):
    """A root with a headed CSV, a bare column, and a non-data file."""
    data = tmp_path / "sub"
    data.mkdir()
    rng = np.random.default_rng(0)
    a = np.cumsum(rng.standard_normal(600)) * 1e-12 + 0.35
    b = np.cumsum(rng.standard_normal(600)) * 1e-12 + 0.35
    with open(data / "run.csv", "w") as fh:
        fh.write("t,NCS,VSL\n")
        for i in range(600):
            fh.write(f"{i},{a[i]:.15e},{b[i]:.15e}\n")
    np.savetxt(tmp_path / "top.dat", a, fmt="%.15e")
    (tmp_path / "notes.md").write_text("not data", encoding="utf-8")
    return tmp_path


@pytest.fixture
def server(tree):
    """A live server on an OS-assigned port, rooted at the temp tree."""
    Handler.root = tree.resolve()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def get(base, path):
    """GET returning `(status, body)`; a refusal is a 400 with a message."""
    try:
        with urllib.request.urlopen(base + path) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def post(base, path, payload):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ---------------------------------------------------------------------------
# serving the page
# ---------------------------------------------------------------------------

def test_the_page_and_its_two_scripts_are_served(server):
    for path, needle in [("/", "stabilis"), ("/app.js", "createPlot"),
                         ("/plot.js", "function createPlot"),
                         ("/bench.js", "function benchEnter"),
                         ("/benchdata.js", "function renderDataEditor"),
                         ("/app.css", "--ink")]:
        with urllib.request.urlopen(server + path) as r:
            assert r.status == 200
            assert needle in r.read().decode()


def test_the_page_asks_the_library_what_it_can_do(server):
    """The dropdowns are filled from the engine, not from a copy in the page.

    A hard-coded list is how a GUI ends up quietly older than its library:
    when this was changed, `stats/` had grown three estimators the page did
    not offer and nothing anywhere said so.
    """
    code, body = get(server, "/api/capabilities")
    assert code == 200
    assert set(body["estimators"]) == set(estimators())
    assert "oadev" in body["estimators"] and len(body["estimators"]) >= 7
    assert "octave" in body["tau_modes"]
    assert body["units"]["phase"] == ["s", "ms", "us", "ns", "ps"]
    with urllib.request.urlopen(server + "/app.js") as r:
        page = r.read().decode()
    assert "/api/capabilities" in page


@pytest.mark.parametrize("name", estimators())
def test_every_registered_estimator_runs_through_the_page(server, name):
    """Whatever the library registers, the page can actually run.

    This is the test that catches a new estimator needing different handling
    -- a non-variance one that takes no `keep_coeffs`, say -- at the moment
    it is added rather than the first time someone selects it.
    """
    code, body = post(server, "/api/analyze", {
        "path": "sub/run.csv", "series": [{"expr": "NCS"}],
        "rate": 1.0, "estimator": name, "reason": "coverage"})
    assert code == 200, body
    assert len(body["series"][0]["rows"]["dev"]) > 0


def test_the_csp_allows_the_pages_own_files_and_nothing_remote(server):
    with urllib.request.urlopen(server + "/") as r:
        policy = r.headers["Content-Security-Policy"]
    # 'self' rather than 'unsafe-inline': the scripts are files on disk now,
    # so the page needs no inline execution at all.
    assert "script-src 'self'" in policy
    assert "unsafe-inline" not in policy
    # Without connect-src the page's own fetch() calls are blocked by
    # default-src 'none' and everything renders empty, saying nothing.
    assert "connect-src 'self'" in policy


def test_an_asset_path_cannot_climb_out_of_the_assets_directory(server):
    for attempt in ("/../server.py", "/sub/app.js", "/..%2fserver.py"):
        try:
            with urllib.request.urlopen(server + attempt) as r:
                body = r.read()
            assert b"BaseHTTPRequestHandler" not in body, attempt
        except urllib.error.HTTPError as exc:
            assert exc.code == 404, attempt


def test_unknown_routes_are_404(server):
    for path in ("/nope", "/api/nope"):
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(server + path)
        assert caught.value.code == 404


def test_serve_reports_a_busy_port_rather_than_crashing(tmp_path):
    busy = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = busy.server_port
    try:
        with pytest.raises(OSError, match="already be running"):
            serve(port=port, root=tmp_path, open_browser=False)
    finally:
        busy.server_close()


def test_a_bad_request_becomes_a_message_rather_than_a_traceback(server):
    code, body = post(server, "/api/open", {"path": "does-not-exist.dat"})
    assert code == 400
    assert "error" in body and "Traceback" not in body["error"]


# ---------------------------------------------------------------------------
# the path guard -- the whole security model
# ---------------------------------------------------------------------------

def test_a_path_escaping_the_root_is_refused_not_rebased(server):
    code, body = get(server, "/api/files?dir=..")
    assert code == 400
    assert "outside" in body["error"]


def test_safe_path_refuses_absolute_paths_outside_the_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret.dat").write_text("1\n2\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        safe_path(root, "../secret.dat")
    with pytest.raises(PermissionError):
        safe_path(root, "/etc/passwd")
    assert safe_path(root, "") == root


def test_listing_shows_directories_and_data_files(server):
    code, body = get(server, "/api/files?dir=")
    assert code == 200
    assert "sub" in body["dirs"]
    names = [f["name"] for f in body["files"]]
    assert "top.dat" in names and "notes.md" not in names


# ---------------------------------------------------------------------------
# opening a file
# ---------------------------------------------------------------------------

def test_a_header_names_the_columns_and_is_not_counted_as_a_sample(server):
    """The reader leaves an uncommented header as a row of NaN; we do not.

    `np.loadtxt` declines a file whose first line is words, so the tolerant
    path takes it and the header arrives as a row of unreadable fields. Left
    alone that is a phantom first sample and three phantom defects.
    """
    code, body = post(server, "/api/open", {"path": "sub/run.csv"})
    assert code == 200
    assert body["has_header"] is True
    assert body["labels"] == ["t", "NCS", "VSL"]
    assert body["n"] == 600, "the header line is not a data row"
    assert body["defects"] == "", "and it is not damage either"


def test_a_headerless_file_gets_positional_names_and_says_so(server):
    code, body = post(server, "/api/open", {"path": "top.dat"})
    assert code == 200
    assert body["has_header"] is False
    assert body["labels"] == ["column 0"]


def test_a_ticking_column_is_offered_as_the_timebase(server):
    _code, body = post(server, "/api/open", {"path": "sub/run.csv"})
    assert body["time_candidates"] == [0]
    facts = {f["label"]: f for f in body["columns"]}
    assert facts["t"]["monotonic"] and not facts["NCS"]["monotonic"]
    assert facts["NCS"]["n_finite"] == 600


def test_the_file_is_read_once_and_then_reused(tree, monkeypatch):
    """The cache is what makes the page reactive; assert it actually hits."""
    reads = []
    original = files._read
    monkeypatch.setattr(files, "_read",
                        lambda p, d: (reads.append(p), original(p, d))[1])
    path = tree / "sub" / "run.csv"
    first = files.load_table(path)
    second = files.load_table(path)
    assert len(reads) == 1
    assert second is first


def test_a_file_that_changed_on_disk_is_read_again(tree):
    path = tree / "top.dat"
    first = files.load_table(path)
    time.sleep(0.01)
    np.savetxt(path, np.arange(50.0), fmt="%.15e")
    second = files.load_table(path)
    assert second is not first and second.n == 50


# ---------------------------------------------------------------------------
# the timebase
# ---------------------------------------------------------------------------

@pytest.fixture
def mjd_file(tmp_path):
    """A 1 Hz record timetagged in MJD -- days, not seconds."""
    rows = 400
    with open(tmp_path / "mjd.csv", "w") as fh:
        fh.write("mjd,x\n")
        for i in range(rows):
            fh.write(f"{60000 + i / 86400:.9f},{i * 1e-12:.15e}\n")
    Handler.root = tmp_path.resolve()
    return tmp_path


def test_an_mjd_column_is_not_mistaken_for_seconds(mjd_file):
    """The failure this guards against is a factor of 86400, silently.

    MJD ticks by 1/86400 per sample at 1 Hz. Handing the column straight to
    `infer_rate` returns 86400 Hz and every tau in the report is wrong by
    that factor while looking entirely plausible.
    """
    from stabilis.io.timebase import AmbiguousTimebase

    table = files.load_table(mjd_file / "mjd.csv")
    rate, _jitter, note = files.timebase(table, 0, "mjd")
    assert rate == pytest.approx(1.0, rel=1e-4)
    assert "mjd" in note

    # And with nothing said, it refuses rather than picking one.
    with pytest.raises(AmbiguousTimebase):
        files.timebase(table, 0, None)


def test_an_ambiguous_timebase_asks_rather_than_guesses(mjd_file):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    try:
        code, body = post(base, "/api/analyze", {
            "path": "mjd.csv", "series": [{"expr": "x"}], "time_column": 0,
            "reason": "x"})
        assert code == 400 and "Pass fmt=" in body["error"]

        code, body = post(base, "/api/analyze", {
            "path": "mjd.csv", "series": [{"expr": "x"}], "time_column": 0,
            "time_format": "mjd", "reason": "x"})
        assert code == 200
        assert body["rate_hz"] == pytest.approx(1.0, rel=1e-4)
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# column arithmetic -- there is no evaluator behind the box
# ---------------------------------------------------------------------------

LABELS = ("t", "NCS", "VSL")
ROWS = np.array([[0.0, 10.0, 4.0], [1.0, 20.0, 5.0], [2.0, 30.0, np.nan]])


@pytest.mark.parametrize("attack", [
    "__import__('os').system('ls')",
    "open('/etc/passwd').read()",
    "NCS.__class__",
    "[c for c in ()]",
    "eval('1')",
    "NCS; import os",
    "lambda: 1",
    "NCS if 1 else 2",
])
def test_code_is_refused_rather_than_run(attack):
    with pytest.raises(expr.ExpressionError):
        expr.evaluate(attack, LABELS, ROWS)


def test_an_expression_naming_no_column_is_refused():
    """`2 - 1` is arithmetic here, not "column 2 minus column 1".

    The library's `edit.parse_expression` reads a bare number as a column
    index. This grammar has numeric literals in it, so it cannot, and
    quietly returning the scalar 1.0 would be a wrong answer that looks
    plausible. The error names the `#N` form that does mean a column.
    """
    with pytest.raises(expr.ExpressionError, match="#2"):
        expr.evaluate("2 - 1", LABELS, ROWS)


@pytest.mark.parametrize("text,want", [
    ("NCS", [10.0, 20.0, 30.0]),
    ("NCS - VSL", [6.0, 15.0, np.nan]),
    ("-VSL", [-4.0, -5.0, np.nan]),
    ("NCS / 10", [1.0, 2.0, 3.0]),
    ("NCS * 2 - VSL", [16.0, 35.0, np.nan]),
    ("(NCS - VSL) / 2", [3.0, 7.5, np.nan]),
    ("#1 - #2", [6.0, 15.0, np.nan]),
    ("ncs - vsl", [6.0, 15.0, np.nan]),
    ("NCS - 1e1", [0.0, 10.0, 20.0]),
])
def test_the_grammar_covers_what_the_page_promises(text, want):
    got, _ = expr.evaluate(text, LABELS, ROWS)
    np.testing.assert_allclose(got, want)


def test_multiplication_binds_tighter_than_addition():
    got, _ = expr.evaluate("NCS + VSL * 2", LABELS, ROWS)
    np.testing.assert_allclose(got[:2], [18.0, 30.0])


def test_a_label_containing_a_space_is_still_one_column():
    labels = ("column 0", "column 1")
    rows = np.array([[1.0, 2.0], [3.0, 4.0]])
    got, _ = expr.evaluate("column 1 - column 0", labels, rows)
    np.testing.assert_allclose(got, [1.0, 1.0])


def test_a_zero_divisor_becomes_a_gap_and_is_counted():
    labels, rows = ("a", "b"), np.array([[1.0, 2.0], [1.0, 0.0]])
    got, blown = expr.evaluate("a / b", labels, rows)
    assert blown == 1
    assert np.isnan(got[1]) and got[0] == 0.5


def test_an_unknown_column_names_the_ones_that_exist():
    with pytest.raises(expr.ExpressionError, match="NCS"):
        expr.evaluate("MASER", LABELS, ROWS)


def test_a_column_index_out_of_range_is_refused():
    with pytest.raises(expr.ExpressionError, match="out of range"):
        expr.evaluate("#9", LABELS, ROWS)


def test_the_page_can_rename_columns_and_expressions_follow(server):
    """The user's names override the file's; that is what makes them useful."""
    code, reply = post(server, "/api/traces",
                       {"path": "sub/run.csv",
                        "labels": ["time", "maser", "ref"],
                        "series": [{"expr": "maser - ref"}], "target": 100})
    assert code == 200
    assert reply["series"][0]["stats"]["n"] == 600


def test_a_label_override_of_the_wrong_length_is_ignored_not_shifted(server):
    """Half a rename would silently analyse a different column."""
    code, reply = post(server, "/api/traces",
                       {"path": "sub/run.csv", "labels": ["only-one"],
                        "series": [{"expr": "NCS"}], "target": 50})
    assert code == 200 and reply["series"][0]["stats"]["n"] == 600


# ---------------------------------------------------------------------------
# drawing two million points into nine hundred pixels
# ---------------------------------------------------------------------------

def test_decimation_keeps_spikes_that_stride_sampling_would_lose():
    values = np.zeros(100_000)
    values[54_321] = 7.0
    values[54_322] = -7.0
    drawn = prepare.envelope(values, 0, values.size, target=1000)
    assert drawn["kind"] == "envelope"
    assert np.nanmax(drawn["hi"]) == 7.0
    assert np.nanmin(drawn["lo"]) == -7.0


def test_decimation_is_a_no_op_on_a_short_record():
    values = np.arange(50.0)
    drawn = prepare.envelope(values, 0, 50, target=1000)
    assert drawn["kind"] == "points"
    np.testing.assert_array_equal(drawn["lo"], values)
    np.testing.assert_array_equal(drawn["i0"], np.arange(50))


def test_a_bucket_of_pure_gap_stays_a_gap():
    """A break in the record must stay a break, not become a zero."""
    values = np.arange(10_000.0)
    values[2000:4000] = np.nan
    drawn = prepare.envelope(values, 0, values.size, target=200)
    assert np.isnan(np.asarray(drawn["lo"], dtype=float)).any()


def test_a_bucket_holding_one_good_sample_reports_it():
    values = np.full(10_000, np.nan)
    values[5555] = 3.0
    drawn = prepare.envelope(values, 0, values.size, target=200)
    assert np.nanmax(drawn["hi"]) == 3.0


def test_zooming_in_returns_detail_the_zoomed_out_view_averaged_away(server):
    """Level of detail, from the cached array -- not a magnified picture."""
    wide = post(server, "/api/traces", {"path": "sub/run.csv",
                                        "series": [{"expr": "NCS"}],
                                        "view": [0, 600], "target": 40})[1]
    near = post(server, "/api/traces", {"path": "sub/run.csv",
                                        "series": [{"expr": "NCS"}],
                                        "view": [100, 140], "target": 40})[1]
    assert wide["series"][0]["kind"] == "envelope"
    assert near["series"][0]["kind"] == "points"
    assert near["series"][0]["i0"][0] == 100
    assert len(near["series"][0]["i0"]) == 40


def test_stage_one_shows_the_file_and_stage_two_shows_the_library_input(server):
    """The second plot is captioned "value handed to the library"; be that.

    These columns carry a ~0.35 s pedestal. Drawn with it, the trace is a
    flat line with the whole signal invisible inside the width of it -- and
    it would be showing something the estimators never see, since the
    pedestal is removed before construction.
    """
    def first(**extra):
        body = {"path": "sub/run.csv", "series": [{"expr": "NCS"}],
                "target": 40}
        return post(server, "/api/traces", dict(body, **extra))[1]["series"][0]

    assert first()["hi"][0] == pytest.approx(0.35, abs=0.01)
    assert abs(first(scaled=True, unit="s")["hi"][0]) < 1e-9


def test_the_unit_scales_the_preview_exactly_too(server):
    seconds = post(server, "/api/traces", {
        "path": "sub/run.csv", "series": [{"expr": "NCS"}], "target": 40,
        "scaled": True, "unit": "s"})[1]["series"][0]["hi"]
    nanos = post(server, "/api/traces", {
        "path": "sub/run.csv", "series": [{"expr": "NCS"}], "target": 40,
        "scaled": True, "unit": "ns"})[1]["series"][0]["hi"]
    np.testing.assert_allclose(nanos, np.asarray(seconds) * 1e-9, rtol=1e-12)


def test_gaps_reach_the_page_as_null_not_as_invalid_json(server, tree):
    """`json.dumps` writes a bare NaN, which `JSON.parse` refuses outright."""
    values = np.arange(100.0)
    values[10] = np.nan
    np.savetxt(tree / "holed.dat", values, fmt="%.15e")
    code, body = post(server, "/api/traces",
                      {"path": "holed.dat", "series": [{"expr": "#0"}],
                       "target": 500})
    assert code == 200
    assert body["series"][0]["lo"][10] is None


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------

def test_overlapping_windows_are_merged_rather_than_double_counted():
    assert prepare.clamp_windows([[0, 100], [50, 200]], 500) == [(0, 200)]
    assert prepare.clamp_windows([[300, 400], [0, 100]], 500) == [(0, 100),
                                                                  (300, 400)]


def test_a_window_running_past_the_record_is_clamped():
    assert prepare.clamp_windows([[-10, 10_000]], 500) == [(0, 500)]


def test_a_window_too_short_to_analyse_is_dropped():
    assert prepare.clamp_windows([[10, 11]], 500) == []


def test_each_window_becomes_its_own_result_by_default(server):
    code, body = post(server, "/api/analyze", {
        "path": "sub/run.csv", "series": [{"expr": "NCS - VSL", "label": "A"}],
        "windows": [[0, 200], [300, 600]], "rate": 1.0, "reason": "warm-up"})
    assert code == 200
    assert [s["label"] for s in body["series"]] == ["A [0:200]", "A [300:600]"]
    assert [s["n"] for s in body["series"]] == [200, 300]
    assert all(s["summary"]["n_gaps"] == 0 for s in body["series"])


def test_joined_windows_become_one_record_with_the_rest_as_gaps(server):
    code, body = post(server, "/api/analyze", {
        "path": "sub/run.csv", "series": [{"expr": "NCS - VSL", "label": "A"}],
        "windows": [[0, 200], [300, 600]], "join": True, "rate": 1.0,
        "reason": "warm-up"})
    assert code == 200
    assert len(body["series"]) == 1
    summary = body["series"][0]["summary"]
    assert summary["n"] == 600 and summary["n_gaps"] == 100
    assert summary["runs"] == 2


def test_joining_windows_never_interpolates_and_records_that_it_did_not():
    """Gap filling is a later milestone; the page must not improvise one."""
    from stabilis.types import PhaseSeries

    series = PhaseSeries(np.arange(100.0) * 1e-12, rate_hz=1.0)
    masked = prepare.exclude_between(series, [(0, 20), (60, 100)], "why")
    assert np.isnan(masked.x[20:60]).all()
    step = masked.provenance.steps[-1]
    assert step.action == "exclude_between_windows"
    assert "nothing interpolated" in step.outcome
    assert step.params["dropped"] == 40


def test_the_step_log_reaches_the_page(server):
    """Provenance that nobody can read is barely better than none."""
    _code, body = post(server, "/api/analyze", {
        "path": "sub/run.csv", "series": [{"expr": "NCS"}],
        "windows": [[0, 300]], "rate": 1.0, "reason": "after warm-up"})
    steps = body["series"][0]["steps"]
    assert "select_range" in [s["action"] for s in steps]
    assert any("after warm-up" in str(s["params"]) for s in steps)


# ---------------------------------------------------------------------------
# units and analysis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unit,factor", list(prepare.PHASE_UNITS.items()))
def test_units_scale_the_answer_exactly(server, unit, factor):
    def run(with_unit):
        return post(server, "/api/analyze", {
            "path": "sub/run.csv", "series": [{"expr": "NCS"}],
            "rate": 1.0, "unit": with_unit, "estimator": "oadev",
            "confidence": False, "reason": "units",
        })[1]["series"][0]["rows"]["dev"]

    np.testing.assert_allclose(run(unit), np.asarray(run("s")) * factor,
                               rtol=1e-12)


def test_an_unknown_unit_is_refused_with_the_valid_list(server):
    code, body = post(server, "/api/analyze", {
        "path": "sub/run.csv", "series": [{"expr": "NCS"}],
        "rate": 1.0, "unit": "furlongs", "reason": "x"})
    assert code == 400
    assert "furlongs" in body["error"] and "'ns'" in body["error"]


def test_an_unknown_estimator_is_refused_with_the_valid_list(server):
    code, body = post(server, "/api/analyze", {
        "path": "sub/run.csv", "series": [{"expr": "NCS"}],
        "rate": 1.0, "estimator": "banana", "reason": "x"})
    assert code == 400 and "oadev" in body["error"]


def test_a_result_carries_the_table_the_plot_was_drawn_from(server):
    _code, body = post(server, "/api/analyze", {
        "path": "sub/run.csv", "series": [{"expr": "NCS - VSL"}],
        "rate": 1.0, "estimator": "oadev", "reason": "x"})
    rows = body["series"][0]["rows"]
    assert len(rows["af"]) == len(rows["tau"]) == len(rows["dev"])
    assert body["series"][0]["has_ci"] and len(rows["lo"]) == len(rows["dev"])
    assert all(lo <= d <= hi for lo, d, hi
               in zip(rows["lo"], rows["dev"], rows["hi"]))


def test_a_non_variance_estimator_returns_no_interval_and_says_why(server):
    _code, body = post(server, "/api/analyze", {
        "path": "sub/run.csv", "series": [{"expr": "NCS"}],
        "rate": 1.0, "estimator": "mtie", "reason": "x"})
    assert body["series"][0]["has_ci"] is False
    assert any("mtie" in note for note in body["notes"])


def test_the_gui_matches_the_library_called_by_hand(server, tree):
    """The page must not be able to disagree with the engine.

    The whole design rests on the GUI calling the same functions the CLI
    does, so this runs the page's own printed recipe by hand and requires
    bit-identical numbers back.
    """
    import stabilis
    from stabilis import edit
    from stabilis.io.text import read_table

    _code, body = post(server, "/api/analyze", {
        "path": "sub/run.csv", "series": [{"expr": "NCS"}],
        "windows": [[50, 500]], "rate": 1.0, "estimator": "oadev",
        "confidence": False, "reason": "by hand"})

    labels, rows, _defects = read_table(tree / "sub" / "run.csv")
    values = np.asarray(rows[1:, list(labels).index("NCS")])  # skip the header
    series = stabilis.PhaseSeries(values - values[0], rate_hz=1.0)
    series = edit.select_range(series, 50, 500, by="index", reason="by hand")
    result = stabilis.oadev(series, taus="octave")

    np.testing.assert_array_equal(body["series"][0]["rows"]["dev"], result.dev)


# ---------------------------------------------------------------------------
# the page's own files
# ---------------------------------------------------------------------------

def test_every_asset_the_page_asks_for_exists():
    """A missing file here is a blank page with nothing saying why."""
    page = (ASSETS / "index.html").read_text(encoding="utf-8")
    for name in ("app.css", "plot.js", "app.js", "bench.js", "benchdata.js"):
        assert name in page
        assert (ASSETS / name).is_file()


def test_the_gui_needs_no_matplotlib(server):
    """`stabilis gui` is now as numpy-only as the core, and stays that way.

    The page draws on a canvas from numbers, so the server never imports a
    plotting library. `tests/test_core_deps.py` makes the same assertion for
    the package as a whole; this one is about the endpoint that used to be
    the exception.
    """
    import sys

    before = set(sys.modules)
    post(server, "/api/analyze", {"path": "sub/run.csv",
                                  "series": [{"expr": "NCS"}], "rate": 1.0,
                                  "reason": "x"})
    for module in set(sys.modules) - before:
        assert not module.startswith("matplotlib"), module
