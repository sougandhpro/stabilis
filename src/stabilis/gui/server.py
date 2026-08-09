"""The HTTP layer: routing, static assets, and the bind.

**It binds to 127.0.0.1 and nothing else.** The page reads files from the
machine it runs on, so a wider bind would be handing a stranger a file
reader. `--root` narrows it further, to one directory tree, and paths that
try to escape that root are refused rather than normalized into it.

**Why a browser page rather than a desktop toolkit.** tkinter is not in this
environment's Python (Debian splits it into `python3-tk`, which needs root),
and PySide is a hundred-megabyte download to draw a form. `http.server` is
in the standard library, every machine already has a browser, and it works
the same from WSL, Linux and Windows.

The page's own files live in `assets/` and are served from disk rather than
embedded in a Python string. The old version kept the whole page in one raw
literal, which meant a `\\n` written for JavaScript had to survive Python's
escaping to reach the browser as two characters -- it did not, once, and the
symptom was a page that rendered blank with nothing saying why. Files on
disk cannot have that failure, and they let the Content-Security-Policy drop
`unsafe-inline` for both scripts and styles.
"""

from __future__ import annotations

import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .api import GET_ROUTES, POST_ROUTES

__all__ = ["serve", "DEFAULT_PORT", "Handler"]

DEFAULT_PORT = 8765
ASSETS = Path(__file__).parent / "assets"

# A request body is a small JSON object of settings. Anything larger is a
# mistake or an attempt, and reading it into memory first to find out would
# be the wrong order to discover that in.
MAX_BODY = 1 << 20

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}

# Served to one local browser, loading nothing remote, so the strictest
# policy that still works is the right one. `connect-src 'self'` is
# load-bearing rather than decoration: without it `default-src 'none'` blocks
# the page's own fetch() calls and everything renders empty.
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; "
       "img-src 'self' data:; connect-src 'self'; base-uri 'none'; "
       "form-action 'none'")


class Handler(BaseHTTPRequestHandler):
    """Routes to `api`, serves `assets/`, and turns exceptions into messages."""

    root = Path.cwd()
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # noqa: D401
        """Silence per-request logging; the console is for real messages."""

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        # allow_nan=False is deliberate: `api._floats` has already turned
        # every non-finite value into null, and a NaN that slipped past it
        # would serialize to a bare `NaN` that `JSON.parse` rejects, taking
        # the page down for a reason nothing on screen would explain. Better
        # to fail here, where the traceback names the field.
        body = json.dumps(payload, default=float, allow_nan=False).encode()
        self._send(code, body, "application/json")

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        parsed = urlparse(self.path)
        route = GET_ROUTES.get(parsed.path)
        if route is not None:
            self._run(route, parse_qs(parsed.query))
            return
        asset = _asset_for(parsed.path)
        if asset is None:
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, asset.read_bytes(),
                   _CONTENT_TYPES.get(asset.suffix, "application/octet-stream"))

    def do_POST(self) -> None:  # noqa: N802
        route = POST_ROUTES.get(urlparse(self.path).path)
        if route is None:
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            self._json(413, {"error": f"request body over {MAX_BODY} bytes"})
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError as exc:
            self._json(400, {"error": f"malformed JSON: {exc}"})
            return
        self._run(route, payload)

    def _run(self, route, payload) -> None:
        """Call a handler, and surface a failure in the page rather than hide it.

        The message keeps the exception's type as well as its text. "column
        7 is out of range" and "PermissionError" want different reactions
        from whoever is reading, and stripping the type down to a bare
        sentence loses that.
        """
        try:
            self._json(200, route(self.root, payload))
        except Exception as exc:  # surfaced in the page, not swallowed
            self._json(400, {"error": f"{type(exc).__name__}: {exc}"})


def _asset_for(path: str) -> Path | None:
    """The file in `assets/` this URL names, or None.

    One flat directory, matched by name after stripping the leading slash,
    and the result must still be a direct child of `assets/`. No traversal
    reaches here in practice -- browsers normalize -- but the check is two
    lines and this process can read the user's disk.
    """
    name = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
    if "/" in name or not name:
        return None
    candidate = (ASSETS / name).resolve()
    if candidate.parent != ASSETS.resolve() or not candidate.is_file():
        return None
    return candidate


def serve(port: int = DEFAULT_PORT, root: Path | None = None,
          open_browser: bool = True) -> None:
    """Run the local page until interrupted."""
    Handler.root = Path(root or Path.cwd()).resolve()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        raise OSError(
            f"cannot bind 127.0.0.1:{port} ({exc}). Another stabilis gui may "
            f"already be running -- try --port {port + 1}"
        ) from exc

    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"stabilis gui: {url}")
    print(f"  serving files under {Handler.root}")
    print("  bound to localhost only. Ctrl-C to stop.")
    if open_browser:
        # On WSL the Linux `webbrowser` module often has nothing to open, so
        # failure here is expected and harmless -- the URL is printed above.
        threading.Timer(0.4, lambda: _try_open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def _try_open(url: str) -> None:
    try:
        if webbrowser.open(url):
            return
    except Exception:
        pass
    # WSL: hand it to Windows, which definitely has a browser.
    import shutil
    import subprocess
    for opener in ("wslview", "explorer.exe", "xdg-open"):
        if shutil.which(opener):
            subprocess.run([opener, url], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return


def free_port(preferred: int) -> int:
    """`preferred` if it is free, otherwise whatever the OS hands out."""
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            return 0
