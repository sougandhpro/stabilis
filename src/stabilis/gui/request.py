"""Reading a request from the page: what file, which columns, at what rate.

Every endpoint is handed the same JSON object and has to answer the same few
questions of it. They live here rather than in `api.py` so that `analysis.py`
can ask them too without the two importing each other.

Nothing here touches the numbers. It resolves a path, names some columns and
settles a sample rate; what those columns then mean is `prepare`'s job and
what they are worth is `stats/`'s.
"""

from __future__ import annotations

import numpy as np

from . import files

__all__ = ["floats", "table_for", "labels_for", "specs", "rate_for"]


def floats(a) -> list:
    """A float array as JSON, with every non-finite value as `null`.

    `json.dumps` writes a bare `NaN`, which `JSON.parse` refuses outright --
    so a gapped column would take the whole page down rather than showing a
    break in the line. `null` is both valid JSON and what the canvas plotter
    already treats as "lift the pen", which is what a gap is.
    """
    a = np.asarray(a, dtype=np.float64)
    out = a.tolist()
    for i in np.flatnonzero(~np.isfinite(a)).tolist():
        out[i] = None
    return out


def table_for(root, req) -> files.Table:
    """The requested file, from cache when it has not changed on disk."""
    path = files.safe_path(root, req["path"])
    return files.load_table(path, req.get("delimiter") or None)


def labels_for(req, table) -> tuple[str, ...]:
    """The column names to resolve expressions against.

    The file's header if it has one, the reader's positional names if not,
    and **whatever the user typed in preference to either**. A header is a
    hint from whoever wrote the logger, not an authority: `ch1` and `ch2`
    are legal names and useless ones, and the person reading the plot is the
    one who knows which is the maser. An override of the wrong length is
    ignored rather than partially applied, since a shifted name list would
    silently analyse the wrong column.
    """
    given = req.get("labels")
    if not given or len(given) != table.n_columns:
        return table.labels
    return tuple(str(name).strip() or table.labels[i]
                 for i, name in enumerate(given))


def specs(req, table):
    """The series the page asked for: `(expression, label, windows)`.

    Windows are a property of the *record*, not of one column: they are
    chosen by dragging on a shared sample axis, and every column of an export
    is on that same timebase. So the request-level `windows` applies to all
    of them, and a series may still carry its own list to override it --
    which is what a caller driving this endpoint from a script would want,
    even though the page has no control that does it.
    """
    given = req.get("series") or [
        {"expr": table.labels[0] if table.labels else "#0"}]
    shared = req.get("windows") or []
    out = []
    for spec in given:
        text = str(spec.get("expr", "")).strip()
        if not text:
            continue
        label = str(spec.get("label") or "").strip() or text
        out.append((text, label, spec.get("windows") or shared))
    if not out:
        raise ValueError("no series to work on: every expression is empty")
    return out


def rate_for(req, table):
    """`(rate_hz, jitter, note)` -- given by the user, or read off a timebase.

    An explicit rate always wins. A rate inferred from timetags is a
    measurement with its own scatter, and the page shows that scatter, but a
    user who knows their counter runs at exactly 1 Hz should not have to
    argue with a median.
    """
    if req.get("rate") not in (None, ""):
        return float(req["rate"]), None, "given"
    column = req.get("time_column")
    if column in (None, ""):
        return 1.0, None, "assumed 1 Hz -- no rate and no time column given"
    return files.timebase(table, int(column), req.get("time_format") or None)
