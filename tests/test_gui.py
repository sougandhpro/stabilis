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
