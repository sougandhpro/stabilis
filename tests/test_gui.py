"""The local GUI server.

Tested through real HTTP against a real server on a free port, because the
handler's job *is* the HTTP behaviour -- calling the functions directly
would skip routing, JSON encoding and error mapping, which is most of what
could break.

The path-escape test is the one that matters. This serves local files to a
browser, so `_safe` is the whole security model.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from stabilis.gui import ESTIMATORS, _Handler, _safe, serve  # noqa: E402


@pytest.fixture
def server(tmp_path):
    """A live server on an OS-assigned port, rooted at a temp tree."""
    data = tmp_path / "sub"
    data.mkdir()
    x = np.cumsum(np.random.default_rng(0).standard_normal(600)) * 1e-12
    np.savetxt(data / "run.dat", x, fmt="%.15e")
    np.savetxt(tmp_path / "top.dat", x, fmt="%.15e")
    (tmp_path / "notes.md").write_text("not data", encoding="utf-8")

    _Handler.root = tmp_path.resolve()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
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


def test_the_page_loads_and_offers_every_estimator(server):
    with urllib.request.urlopen(server + "/") as r:
        page = r.read().decode()
    assert r.status == 200
    for name in ESTIMATORS:
        assert name in page
    # The page must load nothing from the network -- it is served to one
    # local browser and a CDN reference would be a silent dependency.
    assert "http://" not in page.replace("http://127.0.0.1", "")
    assert "https://" not in page


def test_listing_shows_directories_and_data_files(server):
    status, top = get(server, "/api/files")
    assert status == 200
    assert top["at_root"] is True
    assert "sub" in top["dirs"]
    assert [f["name"] for f in top["files"]] == ["top.dat"], \
        "notes.md is not a data file and should not clutter the picker"

    _status, inner = get(server, "/api/files?dir=sub")
    assert inner["at_root"] is False
    assert inner["here"] == "sub"
    assert [f["name"] for f in inner["files"]] == ["run.dat"]


def test_a_path_escaping_the_root_is_refused_not_rebased(server):
    """The whole security model of a page that reads local files.

    `..` is resolved *before* the check, so climbing out is refused rather
    than silently normalized back onto the root -- the difference between
    an error and quietly serving the wrong file.
    """
    status, body = get(server, "/api/files?dir=../../../etc")
    assert status == 400
    assert "outside" in body["error"]

    status, body = post(server, "/api/analyze",
                        {"path": "../../../etc/passwd", "rate": "1.0"})
    assert status == 400
    assert "outside" in body["error"] or "PermissionError" in body["error"]


def test_safe_refuses_absolute_paths_outside_the_root(tmp_path):
    root = (tmp_path / "root")
    root.mkdir()
    (root / "ok.dat").write_text("1.0\n", encoding="ascii")
    assert _safe(root, "ok.dat").name == "ok.dat"
    with pytest.raises(PermissionError):
        _safe(root, "/etc/passwd")
    with pytest.raises(PermissionError):
        _safe(root, "../outside.dat")


def test_analyse_returns_a_plot_and_the_numbers_behind_it(server):
    status, body = post(server, "/api/analyze", {
        "path": "sub/run.dat", "data_kind": "phase", "rate": "1.0",
        "estimator": "oadev", "taus": "octave", "confidence": True,
        "slopes": True})
    assert status == 200, body
    assert body["kind"] == "oadev"
    assert body["n"] == 600
    assert body["has_ci"] is True
    assert len(body["png"]) > 1000
    row = body["rows"][0]
    assert row["af"] == 1 and row["dev"] > 0.0
    assert row["lo"] < row["dev"] < row["hi"]
    assert row["via"] in ("eta1", "eta3")


def test_a_non_variance_estimator_returns_no_interval_and_says_why(server):
    """MTIE has no chi-square interval; the page must not imply one."""
    status, body = post(server, "/api/analyze", {
        "path": "sub/run.dat", "rate": "1.0", "estimator": "mtie",
        "taus": "octave", "confidence": True})
    assert status == 200, body
    assert body["has_ci"] is False
    assert any("not a variance" in n for n in body["notes"])
    assert "lo" not in body["rows"][0]


def test_a_bad_request_becomes_a_message_rather_than_a_traceback(server):
    status, body = post(server, "/api/analyze",
                        {"path": "sub/run.dat", "rate": "1.0",
                         "estimator": "not-an-estimator"})
    assert status == 400
    assert "not-an-estimator" in body["error"]

    status, body = post(server, "/api/analyze",
                        {"path": "does-not-exist.dat", "rate": "1.0"})
    assert status == 400
    assert body["error"]


def test_unknown_routes_are_404(server):
    for path in ("/api/nope", "/secrets"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(server + path)
        assert exc.value.code == 404


def test_serve_reports_a_busy_port_rather_than_crashing(tmp_path):
    """Running it twice is a normal mistake and deserves a real message."""
    _Handler.root = tmp_path.resolve()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = httpd.server_port
    try:
        with pytest.raises(OSError, match="already be running"):
            serve(port=port, root=tmp_path, open_browser=False)
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# The page as a browser sees it -- where two real bugs got through
# ---------------------------------------------------------------------------


def test_the_csp_permits_the_pages_own_fetch_calls(server):
    """`connect-src 'self'` is load-bearing, not decoration.

    Without it, `default-src 'none'` blocks every fetch the page makes and
    the file browser renders empty with nothing saying why. The API tests
    above use urllib, which never sees a CSP, so only this catches it.
    """
    with urllib.request.urlopen(server + "/") as r:
        csp = r.headers["Content-Security-Policy"]
    assert "connect-src 'self'" in csp
    assert "default-src 'none'" in csp


def test_the_page_has_no_real_newline_inside_a_javascript_string():
    """The JS lives in a Python string, and `\\n` written for JS is a trap.

    In a non-raw string Python turns it into an actual newline, which lands
    inside a JS string literal and takes the *whole script* down -- the
    file browser and the estimator list both render empty. `PAGE` is a raw
    string now; this guards it staying one.
    """
    from stabilis.gui import PAGE

    for lineno, line in enumerate(PAGE.splitlines(), 1):
        # Ignore HTML lines; only the script body matters, and an odd
        # number of quotes there means a literal spans a newline.
        if "<" in line or "//" in line:
            continue
        assert line.count('"') % 2 == 0, f"unbalanced quote at page line {lineno}"


def test_the_page_offers_column_and_unit_controls(server):
    """Asked for directly: pick which column, and say what units it is in."""
    with urllib.request.urlopen(server + "/") as r:
        page = r.read().decode()
    for element in ('id="column"', 'id="time_column"', 'id="unit"',
                    'id="dirpath"', 'id="godir"'):
        assert element in page, element
    assert '"ns"' in page and '"ps"' in page


# ---------------------------------------------------------------------------
# Columns and units
# ---------------------------------------------------------------------------


@pytest.fixture
def three_column(tmp_path):
    """time, a junk column, then phase -- the shape real exports come in."""
    x = np.cumsum(np.random.default_rng(1).standard_normal(2000)) * 1e-3
    t = np.arange(2000, dtype=float)
    np.savetxt(tmp_path / "three.dat",
               np.column_stack([t, np.full_like(x, 7.0), x]), fmt="%.9f")
    return tmp_path


def test_the_chosen_column_is_the_one_analysed(three_column, server):
    _Handler.root = three_column.resolve()
    status, body = post(server, "/api/analyze", {
        "path": "three.dat", "data_kind": "phase", "column": "2",
        "time_column": "0", "unit": "s", "taus": "decade"})
    assert status == 200, body
    assert body["column"] == 2
    assert body["rate_hz"] == 1.0, "rate inferred from the time column"

    # Column 1 is a constant, so its ADEV is zero -- proof the selector
    # actually selects rather than defaulting to column 0.
    status, flat = post(server, "/api/analyze", {
        "path": "three.dat", "column": "1", "time_column": "0",
        "unit": "s", "taus": "decade"})
    assert status == 200, flat
    assert flat["rows"][0]["dev"] == 0.0


@pytest.mark.parametrize("unit,factor", [("s", 1.0), ("ns", 1e-9), ("ps", 1e-12)])
def test_units_scale_the_answer_exactly(three_column, server, unit, factor):
    """A file in ns analysed as s is wrong by a billion and looks plausible."""
    _Handler.root = three_column.resolve()
    base = post(server, "/api/analyze", {
        "path": "three.dat", "column": "2", "time_column": "0",
        "unit": "s", "taus": "decade"})[1]["rows"][0]["dev"]
    got = post(server, "/api/analyze", {
        "path": "three.dat", "column": "2", "time_column": "0",
        "unit": unit, "taus": "decade"})[1]["rows"][0]["dev"]
    assert got == pytest.approx(base * factor, rel=1e-12)


def test_an_unknown_unit_is_refused_with_the_valid_list(three_column, server):
    _Handler.root = three_column.resolve()
    status, body = post(server, "/api/analyze", {
        "path": "three.dat", "column": "2", "unit": "furlongs", "rate": "1.0"})
    assert status == 400
    assert "furlongs" in body["error"] and "'ns'" in body["error"]


def test_a_unit_conversion_is_recorded_rather_than_folded_in_silently(tmp_path):
    """A rescaled input must be visible in provenance, not just in the number."""
    from stabilis.gui import _rescale
    from stabilis.types import PhaseSeries

    series = PhaseSeries(np.arange(10.0), rate_hz=1.0)
    scaled = _rescale(series, "ns", is_phase=True)
    assert scaled.x[1] == pytest.approx(1e-9)
    actions = [s.action for s in scaled.provenance.steps]
    assert "unit_conversion" in actions

    # A no-op conversion adds no step, so the log stays honest about what
    # actually happened.
    same = _rescale(series, "s", is_phase=True)
    assert "unit_conversion" not in [s.action for s in same.provenance.steps]


# ---------------------------------------------------------------------------
# The preparation stage
# ---------------------------------------------------------------------------


def test_decimation_keeps_spikes_that_stride_sampling_would_lose():
    """The property the preview exists for.

    A preview is for spotting the glitch you are about to exclude. Taking
    every k-th point gives a one-sample spike a (k-1)/k chance of vanishing
    -- 96% at this reduction. Min/max per bucket keeps the envelope, so the
    spike is still there.
    """
    from stabilis.gui import _decimate

    # Every spike position, not one lucky index: 20 000 happens to be a
    # multiple of the stride, so stride sampling would have kept *that* one
    # and the comparison would have proved nothing.
    kept_by_decimation, kept_by_stride = 0, 0
    step = 50_000 // 1000
    for where in range(19_990, 20_030):
        values = np.zeros(50_000)
        values[where] = 8.0

        idx, shown = _decimate(values, target=2000)
        assert shown.size <= 2000
        if shown.max() == 8.0:
            kept_by_decimation += 1
            assert idx[np.argmax(shown)] == where, "kept its original index"
        if values[::step].max() == 8.0:
            kept_by_stride += 1

    assert kept_by_decimation == 40, "min/max keeps every spike"
    assert kept_by_stride <= 2, "stride sampling loses almost all of them"


def test_decimation_is_a_no_op_on_a_short_record():
    from stabilis.gui import _decimate

    values = np.arange(50.0)
    idx, shown = _decimate(values, target=2000)
    assert np.array_equal(shown, values)
    assert np.array_equal(idx, np.arange(50))


def test_preview_returns_a_plot_and_a_suggested_grid(server):
    status, body = post(server, "/api/preview", {
        "path": "sub/run.dat", "data_kind": "phase", "column": "0",
        "rate": "1.0", "unit": "s"})
    assert status == 200, body
    assert body["n"] == 600
    assert len(body["png"]) > 1000
    assert body["shown"] <= 2000
    assert body["suggest"]["mode"] in ("all", "octave")
    assert body["suggest"]["why"]


def test_preview_reports_file_defects_rather_than_hiding_them(server, tmp_path):
    _Handler.root = tmp_path.resolve()
    (tmp_path / "messy.dat").write_text(
        "1.0 2.0\n3.0\n5.0 NaN\n7.0 8.0\n", encoding="ascii")
    status, body = post(server, "/api/preview",
                        {"path": "messy.dat", "column": "0", "rate": "1.0"})
    assert status == 200, body
    assert body["defects"] and "short rows" in body["defects"]


def test_a_selected_range_is_applied_and_its_equivalent_call_printed(server):
    """An exclusion that exists only as a mouse gesture is not evidence."""
    status, body = post(server, "/api/analyze", {
        "path": "sub/run.dat", "rate": "1.0", "taus": "octave",
        "start": "100", "stop": "400", "reason": "after warm-up"})
    assert status == 200, body
    assert body["prep"]["summary"]["n"] == 300, "only the selection analysed"

    calls = body["prep"]["calls"]
    assert len(calls) == 1
    assert "edit.select_range(series, 100, 400" in calls[0]
    assert "after warm-up" in calls[0], "the reason travels into the code shown"


def test_outlier_rejection_from_the_page_marks_gaps_and_is_printed(server,
                                                                   tmp_path):
    _Handler.root = tmp_path.resolve()
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(2000)) * 1e-12
    x[1000] += 5e-9
    np.savetxt(tmp_path / "spike.dat", x, fmt="%.15e")

    status, body = post(server, "/api/analyze", {
        "path": "spike.dat", "rate": "1.0", "taus": "octave",
        "outlier_threshold": "6", "reason": "counter glitch"})
    assert status == 200, body
    summary = body["prep"]["summary"]
    assert summary["n"] == 2000, "nothing deleted; the record keeps its length"
    assert summary["n_gaps"] >= 1, "the spike became a gap"
    assert any("drop_outliers" in c for c in body["prep"]["calls"])


def test_no_preparation_means_no_calls_and_no_claim_of_any(server):
    status, body = post(server, "/api/analyze", {
        "path": "sub/run.dat", "rate": "1.0", "taus": "octave"})
    assert status == 200, body
    assert body["prep"]["calls"] == []
    assert body["prep"]["summary"]["n"] == 600
