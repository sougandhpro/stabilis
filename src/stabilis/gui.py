"""A small local GUI, so routine work does not need CLI flags.

Deliberately modest. This is a convenience wrapper over the same library
calls the CLI makes -- **it computes nothing itself**, exactly like the
CLI, so it cannot disagree with either. It is *not* the workbench described
in the project plan; that is a separate product built on top of this
engine, and this file is not it.

**Why a browser page rather than a desktop toolkit.** tkinter is not in
this environment's Python (Debian splits it into `python3-tk`, which needs
root), and PySide is a hundred-megabyte download to draw four form fields.
`http.server` is in the standard library, every machine already has a
browser, and it works the same from WSL, Linux and Windows. The cost is
some HTML; the benefit is that nothing needs installing.

**It binds to 127.0.0.1 and nothing else.** The page reads files from the
machine it runs on, so a wider bind would be handing a stranger a file
reader. `--root` narrows it further, to one directory tree, and paths that
try to escape that root are refused rather than normalized into it.

    stabilis gui                    # opens http://127.0.0.1:8765
    stabilis gui --root ~/data      # confine the file browser
"""

from __future__ import annotations

import base64
import html
import io
import json
import socket
import threading
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

__all__ = ["serve", "DEFAULT_PORT"]

DEFAULT_PORT = 8765

# Extensions offered in the file browser. Not a security control -- `_safe`
# is -- just a way to keep the listing readable in a directory full of
# unrelated files.
DATA_SUFFIXES = (".dat", ".txt", ".csv", ".tsv", ".npy", ".phase", ".freq", "")

ESTIMATORS = ("oadev", "adev", "mdev", "tdev", "ohdev", "mtie", "tierms")

# What the numbers in the file are in, and the factor that turns them into
# what the library wants. Phase is seconds; fractional frequency is
# dimensionless. Recorded in provenance when it is not 1, because a
# silently rescaled input is a wrong answer that looks right.
PHASE_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}
FREQ_UNITS = {"fractional": 1.0, "ppm": 1e-6, "ppb": 1e-9, "ppt": 1e-15}


def _safe(root: Path, candidate: str) -> Path:
    """Resolve `candidate` under `root`, or refuse.

    `Path.resolve()` collapses `..` *before* the check, so a path that
    climbs out is caught rather than quietly rebased onto the root -- the
    difference between refusing an escape and silently serving the wrong
    file.
    """
    target = (root / candidate).resolve() if candidate else root
    if target != root and root not in target.parents:
        raise PermissionError(f"{candidate!r} is outside {root}")
    return target


def _listing(root: Path, rel: str) -> dict:
    """Directories and plausible data files at `rel`, for the picker."""
    here = _safe(root, rel)
    if not here.is_dir():
        raise NotADirectoryError(str(here))
    dirs, files = [], []
    for entry in sorted(here.iterdir(), key=lambda p: p.name.lower()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            dirs.append(entry.name)
        elif entry.suffix.lower() in DATA_SUFFIXES:
            files.append({"name": entry.name, "size": entry.stat().st_size})
    parent = "" if here == root else str(here.parent.relative_to(root))
    return {"here": "" if here == root else str(here.relative_to(root)),
            "parent": parent, "at_root": here == root,
            "root": str(root), "dirs": dirs, "files": files}


def _analyze(root: Path, req: dict) -> dict:
    """Run one estimator and return a PNG plus the numbers behind it.

    The table is returned alongside the image on purpose: a picture is not
    a result, and anything worth pasting into a notebook is in the table.
    """
    import matplotlib
    matplotlib.use("Agg")

    from . import plot as plotting
    from .io import load_frequency, load_phase
    from .registry import get
    from .stats import confidence

    path = _safe(root, req["path"])
    rate = float(req["rate"]) if req.get("rate") else None
    is_phase = req.get("data_kind", "phase") == "phase"
    loader = load_phase if is_phase else load_frequency
    time_column = int(req["time_column"]) if req.get("time_column") else None
    series = loader(path, rate_hz=rate,
                    column=int(req.get("column") or 0),
                    time_column=time_column)
    series = _rescale(series, req.get("unit"), is_phase)

    name = req.get("estimator", "oadev")
    if name not in ESTIMATORS:
        raise ValueError(f"unknown estimator {name!r}")
    kwargs = {"taus": req.get("taus", "octave")}
    if req.get("max_af"):
        kwargs["max_af"] = int(req["max_af"])
    # Coefficient arrays exist only for the confidence interval, and they
    # cost one array of order n per tau. Asking for them when the bounds
    # are switched off is how a large record turns into gigabytes for
    # nothing. MTIE and TIE rms take no such argument.
    want_ci = bool(req.get("confidence", True))
    if name not in ("mtie", "tierms"):
        kwargs["keep_coeffs"] = want_ci
    result = get("estimator", name)(series, **kwargs)

    ci = None
    if want_ci and result.is_variance:
        ci = confidence(result, level=float(req.get("level", 0.6827)))

    ax = plotting.sigma_tau(result, confidence=ci, label=path.name)
    if req.get("slopes"):
        plotting.slope_guides(ax)
    ax.set_title(f"{name} — {path.name}")
    buf = io.BytesIO()
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)

    rows = []
    for i, af in enumerate(result.af.tolist()):
        row = {"af": af, "tau": result.tau_s[i], "n": int(result.n_terms[i]),
               "dev": result.dev[i]}
        if ci is not None:
            row["lo"], row["hi"] = ci.dev_lo[i], ci.dev_hi[i]
            row["edof"], row["via"] = ci.edof[i], ci.method[i]
        rows.append(row)

    notes = []
    if series.n_gaps:
        notes.append(f"{series.n_gaps} gaps in the record; "
                     "see `stabilis docs gaps` for how they were handled.")
    if ci is None and not result.is_variance:
        notes.append(f"{name} is not a variance, so it has no chi-square "
                     "interval and none is shown. See `stabilis docs mtie`.")
    return {
        "png": base64.b64encode(buf.getvalue()).decode("ascii"),
        "kind": result.kind, "n": series.n, "rate_hz": series.rate_hz,
        "rows": rows, "notes": notes,
        "has_ci": ci is not None,
        "read_as": "phase" if is_phase else "frequency",
        "unit": req.get("unit") or ("s" if is_phase else "fractional"),
        "column": int(req.get("column") or 0),
    }


def _rescale(series, unit, is_phase: bool):
    """Convert the file's units into the library's, and say so.

    The library works in seconds of phase and dimensionless fractional
    frequency. A file in nanoseconds analysed as if it were seconds gives
    a curve that is wrong by a billion and looks entirely plausible on a
    log axis, so the conversion is explicit and lands in provenance rather
    than being folded silently into the numbers.
    """
    from .log import record

    table = PHASE_UNITS if is_phase else FREQ_UNITS
    if not unit or unit not in table:
        if unit:
            raise ValueError(
                f"unknown unit {unit!r} for "
                f"{'phase' if is_phase else 'frequency'} data; "
                f"expected one of {sorted(table)}")
        return series
    factor = table[unit]
    if factor == 1.0:
        return series
    values = (series.x if is_phase else series.y) * factor
    prov = record(series.provenance, "gui", "unit_conversion",
                  {"unit": unit, "factor": factor},
                  "seconds" if is_phase else "fractional frequency")
    return (replace(series, x=values, provenance=prov) if is_phase
            else replace(series, y=values, provenance=prov))


class _Handler(BaseHTTPRequestHandler):
    root = Path.cwd()

    def log_message(self, *args) -> None:  # noqa: D401 - quieten the server
        """Silence per-request logging; the console is for real messages."""

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # This page is served to one local browser and loads nothing
        # remote, so the strictest policy that still *works* is right --
        # and `connect-src 'self'` is load-bearing, not decoration. Without
        # it `default-src 'none'` blocks the page's own fetch() calls and
        # the file browser silently renders empty, which is exactly what
        # happened. Tested through a real browser now, because urllib
        # never sees a CSP.
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; img-src data:; "
                         "style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                         "connect-src 'self'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload, default=float).encode(),
                   "application/json")

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        if self.path.startswith("/api/files"):
            rel = ""
            if "?" in self.path:
                from urllib.parse import parse_qs, urlparse, unquote
                rel = unquote(parse_qs(urlparse(self.path).query).get("dir", [""])[0])
            try:
                self._json(200, _listing(self.root, rel))
            except (PermissionError, NotADirectoryError, OSError) as exc:
                self._json(400, {"error": str(exc)})
            return
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/analyze":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
            self._json(200, _analyze(self.root, req))
        except Exception as exc:  # surfaced in the page, not swallowed
            self._json(400, {"error": f"{type(exc).__name__}: {exc}"})


def serve(port: int = DEFAULT_PORT, root: Path | None = None,
          open_browser: bool = True) -> None:
    """Run the local page until interrupted."""
    _Handler.root = Path(root or Path.cwd()).resolve()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    except OSError as exc:
        raise OSError(
            f"cannot bind 127.0.0.1:{port} ({exc}). Another stabilis gui may "
            f"already be running -- try --port {port + 1}"
        ) from exc

    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"stabilis gui: {url}")
    print(f"  serving files under {_Handler.root}")
    print("  bound to localhost only. Ctrl-C to stop.")
    if open_browser:
        # On WSL the Linux `webbrowser` module often has nothing to open,
        # so failure here is expected and harmless -- the URL is printed
        # above either way.
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


def _free_port(preferred: int) -> int:
    """`preferred` if it is free, otherwise whatever the OS hands out."""
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            return 0


# Raw, deliberately: this is JavaScript, and a `\n` written for JS must
# reach the browser as two characters. In a plain string Python turns it
# into a real newline, which lands inside a JS string literal and takes the
# whole script down with it -- the estimator list and the file browser both
# render empty and nothing says why.
PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>stabilis</title><style>
:root { color-scheme: light dark; }
body { font: 14px/1.5 system-ui, sans-serif; margin: 0; display: flex;
       height: 100vh; }
#side { width: 320px; padding: 16px; overflow-y: auto; box-sizing: border-box;
        border-right: 1px solid rgba(128,128,128,.35); flex: none; }
#main { flex: 1; padding: 16px; overflow-y: auto; }
h1 { font-size: 16px; margin: 0 0 12px; }
label { display: block; margin: 10px 0 3px; font-size: 12px; opacity: .75; }
select, input, button { width: 100%; padding: 6px; box-sizing: border-box;
        font: inherit; }
button { margin-top: 14px; padding: 9px; cursor: pointer; font-weight: 600; }
.row { display: flex; gap: 8px; } .row > * { flex: 1; }
.chk { display: flex; align-items: center; gap: 6px; margin-top: 8px; }
.chk input { width: auto; }
#files { border: 1px solid rgba(128,128,128,.35); border-radius: 4px;
         max-height: 230px; overflow-y: auto; margin-top: 4px; }
#files div { padding: 4px 8px; cursor: pointer; font-size: 13px;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#files div:hover { background: rgba(128,128,128,.18); }
#files div.sel { background: rgba(90,140,255,.35); font-weight: 600; }
.dir { opacity: .8; }
.tiny { font-size: 11px; margin: 4px 0 0; word-break: break-all; }
button.inline { width: auto; margin-top: 0; padding: 6px 14px; flex: none; }
#files .empty { opacity: .55; cursor: default; font-style: italic; }
#files .empty:hover { background: none; }
img { max-width: 100%; border-radius: 4px; }
table { border-collapse: collapse; font-size: 12px; margin-top: 14px; }
th, td { padding: 3px 10px 3px 0; text-align: right; white-space: nowrap; }
th { border-bottom: 1px solid rgba(128,128,128,.5); text-align: right; }
td.via { text-align: left; opacity: .7; }
.err { color: #c0392b; white-space: pre-wrap; font-family: monospace;
       font-size: 12px; }
.note { opacity: .75; font-size: 12px; margin: 6px 0; }
.muted { opacity: .55; }
</style></head><body>
<div id="side">
  <h1>stabilis</h1>
  <label>folder</label>
  <div class="row">
    <input id="dirpath" placeholder="type a subfolder, or blank for root">
    <button id="godir" class="inline">Go</button>
  </div>
  <div id="root" class="muted tiny"></div>
  <div id="hint" class="muted tiny"></div>
  <div id="files"></div>
  <div class="row">
    <div><label>read as</label>
      <select id="data_kind"><option value="phase">phase</option>
      <option value="frequency">frequency</option></select></div>
    <div><label>units in file</label><select id="unit"></select></div>
  </div>
  <div class="row">
    <div><label>data column</label><input id="column" value="0"></div>
    <div><label>time column</label><input id="time_column" placeholder="none"></div>
  </div>
  <div class="row">
    <div><label>rate (Hz)</label><input id="rate" value="1.0"></div>
    <div><label>max avg factor</label><input id="max_af" placeholder="auto"></div>
  </div>
  <div class="row">
    <div><label>estimator</label><select id="estimator"></select></div>
    <div><label>tau grid</label><select id="taus">
      <option>octave</option><option>decade</option><option>all</option>
      </select></div>
  </div>
  <div class="chk"><input type="checkbox" id="confidence" checked>
    <label for="confidence" style="margin:0">error bars</label></div>
  <div class="chk"><input type="checkbox" id="slopes">
    <label for="slopes" style="margin:0">power-law slope guides</label></div>
  <button id="go">Analyse</button>
  <div id="err" class="err"></div>
</div>
<div id="main"><p class="muted">Pick a file on the left, then Analyse.</p></div>
<script>
const $ = id => document.getElementById(id);
let dir = "", picked = null;
for (const e of %ESTIMATORS%) {
  const o = document.createElement("option"); o.textContent = e;
  $("estimator").appendChild(o);
}
const UNITS = %UNITS%;
function fillUnits() {
  const kind = $("data_kind").value;
  const list = UNITS[kind];
  $("unit").innerHTML = "";
  for (const u of list) {
    const o = document.createElement("option"); o.textContent = u;
    $("unit").appendChild(o);
  }
}
$("data_kind").onchange = fillUnits;
fillUnits();
async function browse(d) {
  $("err").textContent = "";
  let j;
  try {
    const r = await fetch("/api/files?dir=" + encodeURIComponent(d));
    j = await r.json();
  } catch (e) { $("err").textContent = "cannot reach the server: " + e; return; }
  if (j.error) { $("err").textContent = j.error; return; }
  dir = j.here;
  $("dirpath").value = j.here;
  $("root").textContent = "root: " + j.root + (j.here ? "  /  " + j.here : "");
  $("hint").textContent = "start stabilis gui --root DIR to browse elsewhere";
  const box = $("files"); box.innerHTML = "";
  if (!j.at_root) add("../", () => browse(j.parent), true);
  for (const d2 of j.dirs) add(d2 + "/", () => browse(dir ? dir+"/"+d2 : d2), true);
  for (const f of j.files) add(f.name + "  (" + fmt(f.size) + ")",
      ev => { picked = dir ? dir + "/" + f.name : f.name;
              [...box.children].forEach(c => c.classList.remove("sel"));
              ev.target.classList.add("sel"); }, false);
  if (!j.dirs.length && !j.files.length) {
    const el = document.createElement("div");
    el.className = "empty";
    el.textContent = "no folders or data files here";
    box.appendChild(el);
  }
}
$("godir").onclick = () => browse($("dirpath").value.trim());
$("dirpath").addEventListener("keydown", e => {
  if (e.key === "Enter") { e.preventDefault(); browse($("dirpath").value.trim()); }
});
function add(text, fn, isDir) {
  const el = document.createElement("div");
  el.textContent = text; if (isDir) el.className = "dir";
  el.onclick = fn; $("files").appendChild(el);
}
function fmt(n) { return n > 1e6 ? (n/1e6).toFixed(1)+" MB"
                 : n > 1e3 ? (n/1e3).toFixed(0)+" kB" : n+" B"; }
$("go").onclick = async () => {
  $("err").textContent = "";
  if (!picked) { $("err").textContent = "pick a file first"; return; }
  $("go").disabled = true; $("go").textContent = "working...";
  const body = { path: picked, data_kind: $("data_kind").value,
    rate: $("rate").value, estimator: $("estimator").value,
    taus: $("taus").value, max_af: $("max_af").value,
    column: $("column").value, time_column: $("time_column").value,
    unit: $("unit").value,
    confidence: $("confidence").checked, slopes: $("slopes").checked };
  try {
    const r = await fetch("/api/analyze", { method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) });
    const j = await r.json();
    if (j.error) { $("err").textContent = j.error; return; }
    render(j);
  } catch (e) { $("err").textContent = String(e); }
  finally { $("go").disabled = false; $("go").textContent = "Analyse"; }
};
function render(j) {
  const e = n => String(n).replace(/[&<>"]/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  let h = '<img src="data:image/png;base64,' + j.png + '">';
  h += '<p class="note">' + e(j.kind) + ' · ' + j.n + ' points · '
     + j.rate_hz + ' Hz · read as ' + e(j.read_as) + ' in ' + e(j.unit)
     + ', column ' + j.column + '</p>';
  for (const n of j.notes) h += '<p class="note">' + e(n) + '</p>';
  h += "<table><tr><th>AF</th><th>tau (s)</th><th>#</th><th>" + e(j.kind)
     + "</th>" + (j.has_ci ? "<th>lo</th><th>hi</th><th>EDOF</th><th>via</th>" : "")
     + "</tr>";
  for (const r of j.rows) {
    h += "<tr><td>" + r.af + "</td><td>" + r.tau.toPrecision(5) + "</td><td>"
       + r.n + "</td><td>" + r.dev.toExponential(5) + "</td>";
    if (j.has_ci) h += "<td>" + r.lo.toExponential(4) + "</td><td>"
       + r.hi.toExponential(4) + "</td><td>" + r.edof.toFixed(1)
       + '</td><td class="via">' + e(r.via) + "</td>";
    h += "</tr>";
  }
  document.getElementById("main").innerHTML = h + "</table>";
}
browse("");
</script></body></html>
"""

PAGE = PAGE.replace("%ESTIMATORS%", json.dumps(list(ESTIMATORS)))
PAGE = PAGE.replace("%UNITS%", json.dumps(
    {"phase": list(PHASE_UNITS), "frequency": list(FREQ_UNITS)}))
