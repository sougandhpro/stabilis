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

import numpy as np

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
    """Run one estimator over every requested column and plot them together.

    Several columns overlay happily *here*, unlike in the preview: a
    sigma-tau plot is already log-log and comparing channels on one axis is
    the entire reason to have more than one.

    The table is returned alongside the image on purpose: a picture is not
    a result, and anything worth pasting into a notebook is in the table.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import plot as plotting
    from .io import load_frequency, load_phase
    from .registry import get
    from .stats import confidence

    path = _safe(root, req["path"])
    rate = float(req["rate"]) if req.get("rate") else None
    is_phase = req.get("data_kind", "phase") == "phase"
    loader = load_phase if is_phase else load_frequency
    time_column = int(req["time_column"]) if req.get("time_column") else None

    name = req.get("estimator", "oadev")
    if name not in ESTIMATORS:
        raise ValueError(f"unknown estimator {name!r}")
    want_ci = bool(req.get("confidence", True))

    ax, out_series, notes = None, [], []
    for col, label, span in _columns_from(req):
        series = loader(path, rate_hz=rate, column=col, time_column=time_column)
        series = _rescale(series, req.get("unit"), is_phase)
        series, prep = _prepare(series, dict(req, start=span[0], stop=span[1]))

        kwargs = {"taus": req.get("taus", "octave")}
        if req.get("max_af"):
            kwargs["max_af"] = int(req["max_af"])
        # Coefficient arrays exist only for the confidence interval, and
        # they cost one array of order n per tau. Asking for them when the
        # bounds are off is how a large record becomes gigabytes for
        # nothing. MTIE and TIE rms take no such argument.
        if name not in ("mtie", "tierms"):
            kwargs["keep_coeffs"] = want_ci
        result = get("estimator", name)(series, **kwargs)

        ci = None
        if want_ci and result.is_variance:
            ci = confidence(result, level=float(req.get("level", 0.6827)))
        ax = plotting.sigma_tau(result, confidence=ci, label=label, ax=ax)

        rows = []
        for i, af in enumerate(result.af.tolist()):
            row = {"af": af, "tau": result.tau_s[i],
                   "n": int(result.n_terms[i]), "dev": result.dev[i]}
            if ci is not None:
                row["lo"], row["hi"] = ci.dev_lo[i], ci.dev_hi[i]
                row["edof"], row["via"] = ci.edof[i], ci.method[i]
            rows.append(row)
        out_series.append({
            "label": label, "column": col, "rows": rows,
            "prep": prep, "n": series.n, "has_ci": ci is not None,
        })
        if series.n_gaps:
            notes.append(f"{label}: {series.n_gaps} gaps; see "
                         "`stabilis docs gaps` for how they were handled.")
        if ci is None and not result.is_variance:
            notes.append(f"{name} is not a variance, so it has no chi-square "
                         "interval and none is shown. See "
                         "`stabilis docs mtie`.")

    if req.get("slopes"):
        plotting.slope_guides(ax)
    ax.set_title(f"{name} \u2014 {path.name}")
    buf = io.BytesIO()
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)

    first = out_series[0]
    return {
        "png": base64.b64encode(buf.getvalue()).decode("ascii"),
        "kind": name, "n": first["n"], "rate_hz": rate or 1.0,
        "rows": first["rows"], "notes": notes,
        "has_ci": first["has_ci"],
        "series": out_series,
        "prep": first["prep"],
        "read_as": "phase" if is_phase else "frequency",
        "unit": req.get("unit") or ("s" if is_phase else "fractional"),
        "column": first["column"],
    }


# A preview is a picture, and a picture has a few thousand pixels. Sending
# a million points to matplotlib to draw on top of each other is slow and
# looks identical.
PREVIEW_POINTS = 2000


def _decimate(values: np.ndarray, target: int = PREVIEW_POINTS):
    """Min/max decimation: one pair per bucket, so spikes survive.

    **Not stride sampling.** Taking every k-th point would hide exactly the
    outliers this view exists to find -- a one-sample glitch has a (k-1)/k
    chance of vanishing, 96% at the reduction used here. Keeping the
    minimum and maximum of each bucket preserves the visual envelope, so a
    spike is still a spike.

    Returns `(index, value)` pairs at the *original* sample indices, so a
    range dragged on the preview means the same thing to `select_range`.
    """
    n = values.size
    if n <= target:
        return np.arange(n), values
    buckets = max(1, target // 2)
    edges = np.linspace(0, n, buckets + 1).astype(int)
    idx, out = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        chunk = values[lo:hi]
        finite = np.isfinite(chunk)
        if not finite.any():
            idx.append(lo)
            out.append(np.nan)
            continue
        where = np.flatnonzero(finite)
        lo_i = where[np.argmin(chunk[where])]
        hi_i = where[np.argmax(chunk[where])]
        for j in sorted((lo_i, hi_i)):
            idx.append(lo + int(j))
            out.append(float(chunk[j]))
    return np.asarray(idx), np.asarray(out)


def _columns_from(req: dict):
    """The (column, label, range) triples the page asked for.

    One entry means the old single-column behaviour; the request shape is
    the same either way so nothing special-cases the common case.
    """
    cols = req.get("columns")
    if not cols:
        cols = [int(req.get("column") or 0)]
    cols = [int(c) for c in cols]
    labels = list(req.get("labels") or [])
    ranges = list(req.get("ranges") or [])
    out = []
    for i, col in enumerate(cols):
        label = (labels[i] if i < len(labels) and labels[i] else f"column {col}")
        span = ranges[i] if i < len(ranges) else None
        if span is None:
            span = [req.get("start"), req.get("stop")]
        out.append((col, str(label), span))
    return out


def _preview(root: Path, req: dict) -> dict:
    """The raw columns, decimated, plus what stabilis would suggest.

    Stacked subplots sharing one x-axis rather than curves overlaid on one:
    different columns of a real export are different quantities on wildly
    different scales, and overlaying them makes every one but the largest
    a flat line. Sharing x is what lets a range dragged on any panel mean
    the same sample indices in all of them.

    Deliberately cheap and deliberately *before* any estimator runs: the
    point is to see the data in order to choose a range, and a preview that
    made you wait would not get used.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .io.text import read_text
    from .tau import suggest

    path = _safe(root, req["path"])
    wanted = _columns_from(req)
    time_column = int(req["time_column"]) if req.get("time_column") else None
    is_phase = req.get("data_kind", "phase") == "phase"
    factor = (PHASE_UNITS if is_phase else FREQ_UNITS).get(req.get("unit"), 1.0)

    rate = float(req["rate"]) if req.get("rate") else None
    jitter, defects, n = None, None, 0
    series_info, columns = [], []
    for col, label, span in wanted:
        values, times, defects = read_text(path, column=col,
                                           time_column=time_column)
        values = values * factor
        n = int(values.size)
        if times is not None and rate is None:
            from .io.timebase import infer_rate, parse_times
            try:
                seconds, _epoch, _fmt = parse_times(times, req.get("time_format"))
                rate, jitter = infer_rate(seconds)
            except Exception:
                pass
        columns.append((label, values, span))
    rate = rate or 1.0

    height = max(2.4, 1.5 * len(columns) + 0.9)
    fig, axes = plt.subplots(len(columns), 1, figsize=(9.0, height),
                             sharex=True, squeeze=False)
    for ax, (label, values, span) in zip(axes[:, 0], columns):
        idx, shown = _decimate(values)
        ax.plot(idx / rate, shown, linewidth=0.8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        lo, hi = _span_indices(span, values.size)
        if lo != 0 or hi != values.size:
            ax.axvspan(lo / rate, hi / rate, alpha=0.18, color="tab:orange")
        finite = np.isfinite(values)
        series_info.append({
            "label": label, "n": int(values.size),
            "n_gaps": int((~finite).sum()), "shown": int(shown.size),
            "start": int(lo), "stop": int(hi),
        })
    axes[-1, 0].set_xlabel("time from start (s)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)

    first = series_info[0]
    return {
        "png": base64.b64encode(buf.getvalue()).decode("ascii"),
        "n": n,
        "rate_hz": float(rate),
        "jitter": None if jitter is None else float(jitter),
        "panels": len(columns),
        "series": series_info,
        # Mirrors of the first panel, so a single-column caller need not
        # know that the shape grew a list.
        "shown": first["shown"],
        "n_gaps": first["n_gaps"],
        "defects": None if defects is None or defects.clean else defects.describe(),
        "suggest": suggest(n, float(rate)),
    }


def _span_indices(span, size: int) -> tuple[int, int]:
    """A `[start, stop]` pair from the page into real indices."""
    lo, hi = (span or [None, None])[:2]
    lo = 0 if lo in (None, "") else max(0, int(lo))
    hi = size if hi in (None, "") else min(size, int(hi))
    return lo, hi


def _prepare(series, req: dict):
    """Apply the range and cleanup the page asked for, and say what it did.

    Returns `(series, lines)` where `lines` are the equivalent
    non-interactive calls. **A selection that exists only as a mouse
    gesture is not evidence**, so the page shows the code that reproduces
    it -- which doubles as how a user learns the API for their own scripts.
    """
    from .edit import drop_outliers, select_range, summary

    lines = []

    def _bound(value):
        # `str(None)` is "None", which is truthy -- so testing the string
        # accepted a None straight through to int() and raised. Check the
        # value itself.
        if value is None or str(value).strip() == "":
            return None
        return int(value)

    start = _bound(req.get("start"))
    stop = _bound(req.get("stop"))
    if start is not None or stop is not None:
        reason = req.get("reason") or "selected in the stabilis gui"
        series = select_range(series, start, stop, by="index", reason=reason)
        lines.append(f"series = edit.select_range(series, {start}, {stop}, "
                     f"by='index', reason={reason!r})")

    threshold = str(req.get("outlier_threshold", "")).strip()
    if threshold:
        reason = req.get("reason") or "outlier rejection in the stabilis gui"
        series = drop_outliers(series, threshold=float(threshold),
                               reason=reason)
        lines.append(f"series = edit.drop_outliers(series, "
                     f"threshold={float(threshold)}, reason={reason!r})")
    return series, {"calls": lines, "summary": summary(series)}


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
        handler = {"/api/analyze": _analyze, "/api/preview": _preview}.get(
            self.path)
        if handler is None:
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
            self._json(200, handler(self.root, req))
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
#prevwrap { position: relative; display: inline-block; cursor: crosshair;
            user-select: none; }
#band { position: absolute; top: 0; bottom: 0; background: rgba(255,150,0,.25);
        border-left: 1px solid #f90; border-right: 1px solid #f90;
        pointer-events: none; display: none; }
pre.calls { background: rgba(128,128,128,.14); padding: 8px 10px;
            border-radius: 4px; font-size: 12px; overflow-x: auto;
            white-space: pre-wrap; }
.pill { display: inline-block; padding: 1px 7px; border-radius: 9px;
        background: rgba(128,128,128,.25); font-size: 11px; margin-right: 6px; }
.warn { color: #d08000; }
#serieslist { width: 100%; border-collapse: collapse; font-size: 12px; }
#serieslist td { padding: 2px 3px; }
#serieslist input { padding: 3px; font-size: 12px; }
#serieslist tr { cursor: pointer; }
#serieslist tr.active { outline: 1px solid #f90; background: rgba(255,150,0,.10); }
#serieslist .num { width: 62px; }
.aim { font-size: 11px; opacity: .7; }
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
    <div><label>data columns</label><input id="columns" value="0"></div>
    <div><label>time column</label><input id="time_column" placeholder="none"></div>
  </div>
  <label>series &mdash; click a row to aim the drag at it</label>
  <table id="serieslist"></table>
  <div class="chk"><input type="checkbox" id="samerange">
    <label for="samerange" style="margin:0">one range for all series</label></div>
  <label>reason for the selection (goes in the record)</label>
  <input id="reason" placeholder="e.g. after warm-up">
  <label>outlier threshold (MAD sigmas, blank = keep all)</label>
  <input id="outlier_threshold" placeholder="off">
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
<div id="main">
  <div id="prep"></div>
  <div id="out"><p class="muted">Pick a file on the left. A preview appears,
    then drag on it to choose a range.</p></div>
</div>
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
              ev.target.classList.add("sel"); preview(); }, false);
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
// ---- series: one row per column, each with its own range --------------
let series = [{column: 0, label: "", start: "", stop: ""}];
let active = 0;
let previewInfo = null;

function parseColumns() {
  const raw = $("columns").value.split(/[,\s]+/).filter(s => s !== "");
  const cols = raw.map(Number).filter(n => Number.isInteger(n) && n >= 0);
  const kept = cols.length ? cols : [0];
  // Keep whatever the user already typed for a column that is still here,
  // so editing the list does not silently discard a label or a range.
  const before = {};
  for (const s of series) before[s.column] = s;
  series = kept.map(c => before[c] || {column: c, label: "", start: "", stop: ""});
  if (active >= series.length) active = 0;
  renderSeries();
}
function renderSeries() {
  const rows = series.map((s, i) =>
    '<tr class="' + (i === active ? "active" : "") + '" data-i="' + i + '">'
    + '<td class="aim">' + s.column + '</td>'
    + '<td><input data-f="label" data-i="' + i + '" value="' + esc(s.label)
    + '" placeholder="label"></td>'
    + '<td><input class="num" data-f="start" data-i="' + i + '" value="'
    + esc(s.start) + '" placeholder="from"></td>'
    + '<td><input class="num" data-f="stop" data-i="' + i + '" value="'
    + esc(s.stop) + '" placeholder="to"></td></tr>').join("");
  $("serieslist").innerHTML = rows;
  for (const tr of $("serieslist").querySelectorAll("tr"))
    tr.onclick = ev => {
      if (ev.target.tagName === "INPUT") return;
      active = Number(tr.dataset.i); renderSeries();
    };
  for (const el of $("serieslist").querySelectorAll("input"))
    el.onchange = () => {
      series[Number(el.dataset.i)][el.dataset.f] = el.value;
      preview();
    };
}
function esc(s) {
  return String(s === undefined || s === null ? "" : s)
    .replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}
function requestBody(extra) {
  return Object.assign({
    path: picked, data_kind: $("data_kind").value,
    columns: series.map(s => s.column),
    labels: series.map(s => s.label),
    ranges: series.map(s => [s.start === "" ? null : Number(s.start),
                             s.stop === "" ? null : Number(s.stop)]),
    time_column: $("time_column").value, unit: $("unit").value,
    rate: $("rate").value, reason: $("reason").value,
    outlier_threshold: $("outlier_threshold").value,
  }, extra || {});
}

// ---- preview: the point is to see the data before choosing a range ----
async function preview() {
  if (!picked) return;
  $("err").textContent = "";
  $("prep").innerHTML = '<p class="muted">reading...</p>';
  try {
    const r = await fetch("/api/preview", { method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(requestBody()) });
    const j = await r.json();
    if (j.error) { $("prep").innerHTML = ""; $("err").textContent = j.error; return; }
    previewInfo = j;
    if (!$("rate").value || $("time_column").value) $("rate").value = j.rate_hz;
    drawPreview(j);
  } catch (e) { $("err").textContent = String(e); }
}
function drawPreview(j) {
  let h = '<div id="prevwrap"><img id="prev" draggable="false" '
        + 'src="data:image/png;base64,' + j.png + '"><div id="band"></div></div>';
  h += '<p class="note"><span class="pill">' + j.n + ' points</span>'
     + '<span class="pill">' + j.rate_hz.toPrecision(6) + ' Hz</span>'
     + '<span class="pill">' + j.panels + ' series</span>'
     + '<span class="pill">' + j.n_gaps + ' gaps</span></p>';
  if (j.jitter !== null && j.jitter > 1e-3)
    h += '<p class="note warn">timestamps are ragged: interquartile spread '
       + (j.jitter*100).toFixed(2) + '% of the median interval. The rate '
       + 'above is the median; the grid is not uniform.</p>';
  if (j.defects)
    h += '<p class="note warn">file defects tolerated: ' + esc(j.defects) + '</p>';
  h += '<p class="note">drag on the plot to set the range for <b>'
     + esc(series[active].label || ("column " + series[active].column))
     + '</b>' + ($("samerange").checked ? " and every other series" : "")
     + '. Suggested tau grid: <b>' + esc(j.suggest.mode) + '</b> to AF '
     + j.suggest.max_af + ' (' + j.suggest.n_taus + ' points) '
     + '<button id="usesuggest" class="inline">use it</button></p>';
  $("prep").innerHTML = h;
  $("usesuggest").onclick = () => {
    $("taus").value = j.suggest.mode;
    $("max_af").value = j.suggest.max_af;
  };
  wireDrag();
}

// Dragging maps pixels back to sample indices, so what is dragged and what
// `edit.select_range` receives are the same thing.
function wireDrag() {
  const wrap = $("prevwrap"), img = $("prev"), band = $("band");
  if (!wrap || !previewInfo) return;
  let from = null;
  const toSample = ev => {
    const box = img.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (ev.clientX - box.left) / box.width));
    return Math.round(f * previewInfo.n);
  };
  const paint = (a, b) => {
    const box = img.getBoundingClientRect();
    const lo = Math.min(a, b) / previewInfo.n * box.width;
    const hi = Math.max(a, b) / previewInfo.n * box.width;
    band.style.display = "block";
    band.style.left = lo + "px";
    band.style.width = (hi - lo) + "px";
  };
  // preventDefault matters: an <img> is natively draggable, and without it
  // the browser starts its own image drag on mousedown and swallows every
  // mousemove and mouseup after -- which is why selection did nothing.
  img.ondragstart = () => false;
  wrap.onmousedown = ev => { ev.preventDefault(); from = toSample(ev);
                             paint(from, from); };
  wrap.onmousemove = ev => { if (from !== null) paint(from, toSample(ev)); };
  // Listen for the release on the window, so letting go outside the plot
  // still completes the selection instead of leaving it stuck.
  window.onmouseup = ev => {
    if (from === null) return;
    const a = from, b = toSample(ev);
    from = null;
    const lo = Math.min(a, b), hi = Math.max(a, b);
    if (hi - lo < 2) {
      band.style.display = "none";
      applyRange("", "");
      return;
    }
    applyRange(lo, hi);
  };
}
function applyRange(lo, hi) {
  const targets = $("samerange").checked
    ? series.map((_s, i) => i) : [active];
  for (const i of targets) { series[i].start = lo; series[i].stop = hi; }
  renderSeries();
  preview();
}

$("columns").addEventListener("change", parseColumns);
$("samerange").addEventListener("change", () => { if (previewInfo) preview(); });
for (const id of ["time_column", "unit", "data_kind"])
  $(id).addEventListener("change", preview);
parseColumns();

$("go").onclick = async () => {
  $("err").textContent = "";
  if (!picked) { $("err").textContent = "pick a file first"; return; }
  $("go").disabled = true; $("go").textContent = "working...";
  const body = requestBody({ estimator: $("estimator").value,
    taus: $("taus").value, max_af: $("max_af").value,
    confidence: $("confidence").checked, slopes: $("slopes").checked });
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
  h += '<p class="note">' + e(j.kind) + ' · ' + j.rate_hz + ' Hz · read as '
     + e(j.read_as) + ' in ' + e(j.unit) + '</p>';
  h += '<p class="note">the same thing without the mouse &mdash; an exclusion '
     + 'that exists only as a gesture is not evidence:</p>';
  for (const n of j.notes) h += '<p class="note">' + e(n) + '</p>';
  for (const s of j.series) {
    if (j.series.length > 1) h += "<h3>" + e(s.label) + "</h3>";
    h += "<table><tr><th>AF</th><th>tau (s)</th><th>#</th><th>" + e(j.kind)
       + "</th>" + (s.has_ci ? "<th>lo</th><th>hi</th><th>EDOF</th><th>via</th>" : "")
       + "</tr>";
    for (const r of s.rows) {
      h += "<tr><td>" + r.af + "</td><td>" + r.tau.toPrecision(5) + "</td><td>"
         + r.n + "</td><td>" + r.dev.toExponential(5) + "</td>";
      if (s.has_ci) h += "<td>" + r.lo.toExponential(4) + "</td><td>"
         + r.hi.toExponential(4) + "</td><td>" + r.edof.toFixed(1)
         + '</td><td class="via">' + e(r.via) + "</td>";
      h += "</tr>";
    }
    h += "</table>";
    if (s.prep && s.prep.calls.length)
      h += '<pre class="calls">' + e(s.prep.calls.join("\n")) + "</pre>";
  }
  document.getElementById("out").innerHTML = h;
}
browse("");
</script></body></html>
"""

PAGE = PAGE.replace("%ESTIMATORS%", json.dumps(list(ESTIMATORS)))
PAGE = PAGE.replace("%UNITS%", json.dumps(
    {"phase": list(PHASE_UNITS), "frequency": list(FREQ_UNITS)}))
