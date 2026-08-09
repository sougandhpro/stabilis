"""The page's endpoints. Everything numeric, nothing rendered.

The old page sent pictures: matplotlib drew a PNG on the server and the
browser displayed it, so every zoom, every drag and every relabelling was a
round trip through a figure. This one sends **numbers** -- a min/max
envelope for the raw view, a table of tau and sigma for the result -- and
the browser draws them on a canvas. That is what makes panning and zooming
free, and it takes matplotlib out of the GUI's dependencies entirely, so
`stabilis gui` is now as numpy-only as the core.

The split across the endpoints follows what actually costs time:

* `capabilities` asks the library what it can do, so the page holds no
  second copy of the estimator list to fall behind it.
* `open` reads the file, once, and answers everything about its shape. It
  is the only slow call.
* `traces` slices and decimates what is already in memory. This is the one
  the page fires on every interaction, and it must stay in milliseconds.
* `analyze` -- in `analysis.py` -- is the only one that runs an estimator.

**No endpoint computes a statistic itself.** `traces` reduces points for
drawing and says so; every number a user could quote comes from `stats/`.
"""

from __future__ import annotations

import numpy as np

from . import analysis, bench, expr, files, prepare
from .request import floats, labels_for, rate_for, specs, table_for

__all__ = ["estimators", "capabilities", "GET_ROUTES", "POST_ROUTES"]


def estimators() -> tuple[str, ...]:
    """Every estimator the library has registered, in name order.

    Read from the registry rather than listed here, so an estimator added to
    `stats/` -- or supplied by a third party through the `stabilis.estimator`
    entry point -- appears in the page without this file being touched. A
    hard-coded list is how a GUI ends up quietly older than its engine: the
    day this was changed the library had grown three estimators the page did
    not offer, and nothing anywhere said so.
    """
    from ..registry import available

    return tuple(available("estimator"))


# ---------------------------------------------------------------------------
# GET /api/capabilities
# ---------------------------------------------------------------------------

def capabilities(root, query: dict) -> dict:
    """What the page should offer, asked of the library rather than assumed.

    The estimator list, the tau grids and the unit tables all live in the
    engine. Fetching them means the page has no second copy to drift.
    """
    from ..bench.model import SCHEMA_VERSION
    from ..tau import MODES

    return {
        "estimators": list(estimators()),
        "tau_modes": list(MODES),
        "units": {"phase": list(prepare.PHASE_UNITS),
                  "frequency": list(prepare.FREQ_UNITS)},
        # So a bench the page writes carries the version the library would
        # have written, rather than a number typed into a script that then
        # has to be remembered when the layout changes.
        "bench_schema": SCHEMA_VERSION,
    }


# ---------------------------------------------------------------------------
# GET /api/files
# ---------------------------------------------------------------------------

def list_files(root, query: dict) -> dict:
    """Directories and data files, for the picker."""
    return files.listing(root, query.get("dir", [""])[0])


# ---------------------------------------------------------------------------
# POST /api/open
# ---------------------------------------------------------------------------

def open_file(root, req: dict) -> dict:
    """Read the file once and describe it. The only slow call in the page.

    Returns what the first stage has to decide from: the column names and
    whether the file supplied them or the reader invented them, how much of
    each column is real, which columns tick upwards like a timebase, and
    what had to be tolerated to read the thing at all.
    """
    from ..tau import suggest

    path = files.safe_path(root, req["path"])
    table = table_for(root, req)
    facts = files.column_facts(table)
    return {
        "path": req["path"],
        "name": path.name,
        "n": table.n,
        "n_columns": table.n_columns,
        "labels": list(table.labels),
        "has_header": table.has_header,
        "defects": table.defects,
        "columns": facts,
        "time_candidates": [f["index"] for f in facts if f["monotonic"]],
        "suggest": suggest(table.n, 1.0) if table.n >= 2 else None,
    }


# ---------------------------------------------------------------------------
# POST /api/traces
# ---------------------------------------------------------------------------

def traces(root, req: dict) -> dict:
    """Decimated traces for the raw view, over the window the page is showing.

    `view` is a `[first, last]` pair of sample indices, so zooming in asks
    for a narrower slice of the same cached array and gets back detail that
    was averaged away at the previous zoom -- level of detail, at the cost
    of one slice and two `reduceat` calls. There is no re-reading and no
    re-parsing, which is the whole reason this is a separate endpoint.
    """
    table = table_for(root, req)
    labels = labels_for(req, table)
    is_phase = req.get("data_kind", "phase") == "phase"
    factor = prepare.unit_factor(req.get("unit"), is_phase) \
        if req.get("scaled") else 1.0
    rate, jitter, note = rate_for(req, table)

    view = req.get("view") or [0, table.n]
    lo = max(0, int(view[0])) if view[0] not in (None, "") else 0
    hi = min(table.n, int(view[1])) if view[1] not in (None, "") else table.n
    target = int(req.get("target") or prepare.PREVIEW_POINTS)

    out = []
    for text, label, windows in specs(req, table):
        values, n_blown = expr.evaluate(text, labels, table.rows)
        if req.get("scaled"):
            # Stage 2 is captioned "value handed to the library", so it has
            # to be exactly that -- pedestal removed and in library units,
            # through the same call `analyze` makes. Stage 1 asks for the
            # numbers as they sit in the file and gets them untouched.
            values, _pedestal = prepare.to_library_units(values, factor)
        drawn = prepare.envelope(values, lo, hi, target)
        spans = prepare.clamp_windows(windows, values.size)
        out.append({
            "label": label, "expr": text,
            "kind": drawn["kind"],
            "i0": np.asarray(drawn["i0"]).astype(int).tolist(),
            "lo": floats(drawn["lo"]), "hi": floats(drawn["hi"]),
            "windows": [list(span) for span in spans],
            "divide_by_zero": n_blown,
            "stats": _stats(values, spans),
        })
    return {"n": table.n, "rate_hz": rate, "jitter": jitter, "rate_note": note,
            "view": [lo, hi], "series": out, "defects": table.defects}


def _stats(values: np.ndarray, spans) -> dict:
    """Count, gaps and span of what is actually selected.

    Over the selected windows rather than the whole column, because that is
    the record about to be analysed and the number a user checks before
    pressing the button.
    """
    chunk = (np.concatenate([values[lo:hi] for lo, hi in spans]) if spans
             else values)
    finite = np.isfinite(chunk)
    n_finite = int(finite.sum())
    return {
        "n": int(chunk.size),
        "n_gaps": int(chunk.size - n_finite),
        "min": float(np.min(chunk[finite])) if n_finite else None,
        "max": float(np.max(chunk[finite])) if n_finite else None,
        "mean": float(np.mean(chunk[finite])) if n_finite else None,
    }


GET_ROUTES = {"/api/files": list_files, "/api/capabilities": capabilities}
POST_ROUTES = {"/api/open": open_file, "/api/traces": traces,
               "/api/analyze": analysis.analyze,
               "/api/bench/check": bench.check_bench,
               "/api/bench/solve": bench.solve_bench,
               "/api/bench/load": bench.load_bench}
