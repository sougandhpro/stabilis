"""`POST /api/analyze` -- the only endpoint that runs an estimator.

Split out of `api.py` because it is the half with a different job. The rest
of the page reads files and reduces arrays for drawing; this takes a series
the library will accept, hands it to `stats/`, and flattens what comes back
into parallel arrays the canvas can plot.

**Nothing here computes a statistic.** Every variance, degree of freedom and
confidence bound comes from the library, through the same calls `cli/`
makes, which is what stops the page and the command line from being able to
disagree about a number. There is a test that runs the recipe this module
prints, by hand, and demands identical output.
"""

from __future__ import annotations

import numpy as np

from . import expr, prepare
from .request import floats, labels_for, rate_for, specs, table_for

__all__ = ["analyze"]

DEFAULT_REASON = "selected in the stabilis gui"


def analyze(root, req: dict) -> dict:
    """Run one estimator over every selected stretch of every series.

    Each window becomes its own curve unless `join` is set, in which case the
    stretches between windows become gaps in one longer record. Both are
    honest; neither invents a sample. See `prepare.exclude_between`.
    """
    from .api import estimators

    table = table_for(root, req)
    labels = labels_for(req, table)
    is_phase = req.get("data_kind", "phase") == "phase"
    factor = prepare.unit_factor(req.get("unit"), is_phase)
    rate, jitter, rate_note = rate_for(req, table)

    name = req.get("estimator", "oadev")
    known = estimators()
    if name not in known:
        raise ValueError(f"unknown estimator {name!r}; "
                         f"expected one of {list(known)}")
    want_ci = bool(req.get("confidence", True))
    reason = str(req.get("reason") or "").strip() or DEFAULT_REASON
    join = bool(req.get("join"))

    out, notes = [], []
    for text, label, windows in specs(req, table):
        values, n_blown = expr.evaluate(text, labels, table.rows)
        if n_blown:
            notes.append(f"{label}: {n_blown} points became gaps because a "
                         "divisor was zero.")
        for entry in _one_series(values, windows, join, rate, is_phase, factor,
                                 name, req, want_ci, reason):
            entry["label"] = label + entry.pop("suffix")
            entry["expr"] = text
            note = entry.pop("ci_note", None)
            out.append(entry)
            if note and note not in notes:
                notes.append(note)
            if entry["summary"]["n_gaps"]:
                notes.append(f"{entry['label']}: {entry['summary']['n_gaps']} "
                             "gaps; see `stabilis docs gaps` for how they were "
                             "handled.")
    return {
        "kind": name, "rate_hz": rate, "jitter": jitter, "rate_note": rate_note,
        "read_as": "phase" if is_phase else "frequency",
        "unit": req.get("unit") or ("s" if is_phase else "fractional"),
        "series": out, "notes": notes,
    }


def _one_series(values, windows, join, rate, is_phase, factor, name, req,
                want_ci, reason):
    """Every stretch of one column expression, as finished result rows."""
    from ..registry import get

    spans = prepare.clamp_windows(windows, values.size)
    base, pedestal = prepare.build_series(values, rate, is_phase, factor)

    if join and len(spans) > 1:
        base = prepare.exclude_between(base, spans, reason)
        spans = [(spans[0][0], spans[-1][1])]
        suffixes = [f" [{len(windows)} windows joined]"]
    elif not spans:
        spans, suffixes = [(0, base.n)], [""]
    else:
        suffixes = [f" [{lo}:{hi}]" if len(spans) > 1 else ""
                    for lo, hi in spans]

    estimator = get("estimator", name)
    kwargs = {"taus": req.get("taus", "octave")}
    if req.get("max_af") not in (None, ""):
        kwargs["max_af"] = int(req["max_af"])
    # Coefficient arrays exist only for the confidence interval, and cost one
    # array of order n per tau. Asking for them with the bounds off is how a
    # large record becomes gigabytes for nothing.
    if _accepts_keep_coeffs(estimator):
        kwargs["keep_coeffs"] = want_ci

    out = []
    for (lo, hi), suffix in zip(spans, suffixes):
        lines = [f"series = stabilis.load_phase(path, rate_hz={rate!r})"
                 if is_phase else
                 f"series = stabilis.load_frequency(path, rate_hz={rate!r})"]
        series = prepare.crop(base, lo, hi, reason, lines)
        series = prepare.reject_outliers(series, req.get("outlier_threshold"),
                                         reason, lines)
        result = estimator(series, **kwargs)
        ci, note = _interval_for(result, want_ci, req)
        lines.append(f"result = stabilis.{name}(series, "
                     f"taus={kwargs['taus']!r})")
        if ci is not None:
            lines.append("ci = stabilis.confidence(result)")
        out.append({
            "suffix": suffix, "window": [lo, hi], "n": series.n,
            "pedestal": pedestal, "has_ci": ci is not None, "ci_note": note,
            "rows": _rows(result, ci), "calls": lines,
            "summary": _summary(series), "steps": _steps(result.provenance),
        })
    return out


def _accepts_keep_coeffs(fn) -> bool:
    """Whether this estimator can be asked to retain its coefficient arrays.

    Asked of the function rather than looked up in a list of names here, and
    the reason is empirical: a list of names was written, and within the same
    afternoon the library grew `theo1`, `totdev` and `wdev`, two of which
    take no `keep_coeffs` at all. The page called them with it and they
    raised `TypeError`. A registry the page reads dynamically cannot be
    paired with a blacklist the page maintains by hand.

    This is the one place the GUI inspects a signature. It is not the
    metaprogramming the style contract rules out -- nothing is generated,
    nothing is patched -- it is asking a callable what it accepts, which is
    the only honest question when the set of callables is open.
    """
    import inspect

    return "keep_coeffs" in inspect.signature(fn).parameters


def _interval_for(result, want_ci: bool, req: dict):
    """`(confidence, note)` -- and when there is none, say which reason.

    There are two of them and they are not the same fact. MTIE is not a
    variance, so no chi-square interval exists for it at all and none ever
    will. Thêo1 and the wavelet deviations *are* variances but do not carry
    the coefficient autocovariance the data-driven degrees of freedom are
    computed from, so they have no interval **yet**. Collapsing both into a
    blank column would tell a reader the same thing about a permanent
    property and a present limitation.

    The library raises with the distinction already in the message, so the
    message is what gets shown rather than a sentence reinvented here that
    could drift from it.
    """
    from ..stats import confidence

    if not want_ci or not result.is_variance:
        note = None if not want_ci else (
            f"{result.kind} is not a variance, so it has no chi-square "
            "interval and none is drawn. See `stabilis docs mtie`.")
        return None, note
    try:
        return confidence(result, level=float(req.get("level", 0.6827))), None
    except ValueError as exc:
        return None, f"no interval: {exc}"


def _steps(provenance) -> list[dict]:
    """The step log, flattened for the page to show.

    Shown rather than merely carried, because a decision taken on the user's
    behalf that nobody ever reads is not much better than one that was never
    recorded -- and the tau grid, the gap policy and the degrees-of-freedom
    estimator actually used are all in here.
    """
    return [{"module": step.module, "action": step.action,
             "outcome": step.outcome,
             "params": {k: str(v) for k, v in step.params.items()}}
            for step in provenance.steps]


def _rows(result, ci) -> dict:
    """The result as parallel arrays, which is what the plotter wants."""
    out = {
        "af": np.asarray(result.af).astype(int).tolist(),
        "tau": floats(result.tau_s),
        "n_terms": np.asarray(result.n_terms).astype(int).tolist(),
        "dev": floats(result.dev),
    }
    if ci is not None:
        out["lo"] = floats(ci.dev_lo)
        out["hi"] = floats(ci.dev_hi)
        out["edof"] = floats(ci.edof)
        out["via"] = list(ci.method)
    return out


def _summary(series) -> dict:
    from ..edit import summary

    return summary(series)
