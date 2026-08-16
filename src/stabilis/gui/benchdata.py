"""Turning one measurement's assignment into a result the solver can take.

This is the half of stage 4 that reaches a file. `bench.py` orchestrates --
check the graph, gather the rows, hand them over -- and everything between
"the user pointed at a column" and "here is a `StabilityResult`" happens
here, once per measurement.

Split out of `bench.py` for the same reason `analysis.py` is split out of
`api.py`: it is the half with a different job, and a different set of ways
to be wrong. The orchestration is about a graph; this is about somebody
else's counter output.

**It assumes nothing about the data.** Six measurements from six
instruments are six formats, six delimiters, six timebases and six
different kinds of mess, so every one of those is a per-measurement
setting rather than a property of the bench. What a file needed tolerating
to be read at all, how many points were gaps, how many an outlier
threshold removed, and how far the timetags wander from a clean grid are
all reported per row -- "something was tolerated somewhere" is not
actionable when the something belongs to one of six counters.
"""

from __future__ import annotations

from . import expr, prepare
from .request import floats, labels_for, rate_for, table_for

__all__ = ["measure", "rate_notes", "recipe", "preamble", "explain_failure"]

DEFAULT_REASON = "assigned in the stabilis gui"


def rate_notes(prepared: list) -> list[str]:
    """Say when the counters disagree, before the tau grids fail to meet.

    Different instruments run at different rates, which is normal and fine
    -- but `bench.solve` intersects tau grids and interpolates nothing, so
    rates that are not related by a power of two share very few taus and
    may share none. When that happens the failure arrives from
    `_common_taus` as "the measurements share no averaging time", which is
    accurate and says nothing about which row is the odd one out. This
    names them while the numbers are still on screen.

    A timebase read off timetags also has scatter, and a counter dropping
    the occasional point shows up here as jitter long before it shows up in
    a deviation.
    """
    out = []
    rates = {round(info["rate_hz"], 9) for info in prepared}
    if len(rates) > 1:
        detail = ", ".join(f"{i['id']} at {i['rate_hz']:g} Hz"
                           for i in prepared)
        out.append(
            f"the measurements run at {len(rates)} different rates ({detail})."
            " Nothing is interpolated onto a common grid, so only taus every"
            " record actually has are solved -- and if they share none, the"
            " solve refuses rather than inventing one.")
    for info in prepared:
        if info["jitter"] is not None and info["jitter"] > 1e-3:
            out.append(
                f"{info['id']}: the timetags in {info['path']} have an "
                f"interquartile spread of {info['jitter'] * 100:.3f}% of the "
                "median interval. That is a counter dropping or repeating "
                "points, not a clean grid -- give an explicit rate if you "
                "know it.")
    return out


def explain_failure(exc: ValueError, prepared: list) -> ValueError:
    """Add what the page knows to what the solver said.

    `bench.solve` refuses records that share no averaging time. That is the
    right refusal -- nothing here interpolates onto a common grid -- and it
    names the spans, but it cannot say why two counters that both ran at
    1 Hz ended up on different grids. The page can: it knows each row's rate
    and whether that rate was given or read off a timetag column.

    The usual culprit is float64, and an MJD column is the standard way to
    meet it. Day number 59000 has a float64 spacing of 7.3e-12 d = 6.3e-7 s,
    so a one-second step is resolved to about 2e-7 relative; `af/rate` then
    misses an exact 1.000 s by far more than the nanosecond tolerance the
    intersection allows. **The record is fine. The timebase is quantized.**
    An explicit rate fixes it, and saying so is worth more than repeating
    that the spans do not meet.

    Returns the exception to raise rather than raising, so the caller keeps
    the `raise ... from` chain and unrelated failures pass through
    untouched.
    """
    if "averaging time" not in str(exc) or not prepared:
        return exc
    rows = "\n".join(
        f"  {i['id']}: {i['rate_hz']!r} Hz ({i['rate_note']})"
        + (f", timetag spread {i['jitter'] * 100:.4f}%"
           if i["jitter"] is not None else "")
        for i in prepared)
    hint = ""
    if any(i["jitter"] is not None for i in prepared):
        hint = (
            "\n\nAt least one rate was read off a timetag column rather than "
            "given. A timebase in MJD days, or any absolute epoch, is "
            "quantized by float64 -- at MJD 59000 the spacing is 6.3e-7 s, so "
            "a 1 Hz grid infers as 0.99999976 Hz and its taus miss round "
            "numbers by more than the intersection tolerates. Nothing is "
            "wrong with the record. Give those rows an explicit rate, or a "
            "tau0 on the bench, and they will land on the same grid.")
    return ValueError(f"{exc}\n\nRates the page used:\n{rows}{hint}")


def measure(root, measurement, spec: dict, name: str, taus, max_af):
    """One measurement, from file to `StabilityResult`.

    **The bench is authoritative about what the quantity is.** Phase or
    frequency comes from `Measurement.kind` and the sample interval from
    `Measurement.tau0_s` when it is set, not from anything typed on this
    screen -- those two facts belong to the experiment, they are what
    `bench.json` carries to the CLI, and letting a control here contradict
    them is precisely how the page and `stabilis bench solve` would come to
    disagree about one file. The unit, the column expression and the crop
    describe the *file*, which the bench says nothing about, so those are
    the page's to supply.
    """
    from ..registry import get

    table = table_for(root, spec)
    labels = labels_for(spec, table)
    text = str(spec.get("expr") or "").strip()
    if not text:
        raise ValueError(
            f"{measurement.id}: a file is assigned but no column. Name the "
            f"column that holds this measurement -- one of "
            f"{[str(name) for name in labels]}, or #0, #1 by position.")

    is_phase = measurement.kind == "phase"
    factor = prepare.unit_factor(spec.get("unit"), is_phase)
    reason = str(spec.get("reason") or "").strip() or DEFAULT_REASON
    if measurement.tau0_s:
        rate, jitter, rate_note = 1.0 / measurement.tau0_s, None, \
            f"1/tau0_s = {1.0 / measurement.tau0_s:g} Hz, from the bench"
    else:
        rate, jitter, rate_note = rate_for(spec, table)

    values, n_blown = expr.evaluate(text, labels, table.rows)
    series, pedestal = prepare.build_series(values, rate, is_phase, factor)
    lines = recipe(measurement, spec, table, labels, text, is_phase, factor,
                    rate)

    spans = prepare.clamp_windows(spec.get("windows"), series.n)
    if len(spans) > 1:
        series = prepare.exclude_between(series, spans, reason)
        lines.append(f"# {len(spans)} windows joined; everything between them "
                     "is a gap, nothing interpolated")
        lo, hi = spans[0][0], spans[-1][1]
    else:
        lo, hi = spans[0] if spans else (0, series.n)
    series = prepare.crop(series, lo, hi, reason, lines)

    # Counted either side of the rejection rather than inferred from the
    # threshold, because how many points a MAD threshold removes is a fact
    # about the record and not about the number typed in. A rejection that
    # quietly took a third of the data is the thing worth seeing.
    from ..edit import summary
    kept = summary(series)
    series = prepare.reject_outliers(series, spec.get("outlier_threshold"),
                                     reason, lines)
    stats = summary(series)
    n_rejected = stats["n_gaps"] - kept["n_gaps"]

    result = get("estimator", name)(series, taus=taus, max_af=max_af)
    lines.append(f"results[{measurement.id!r}] = stabilis.{name}("
                 f"series, taus={taus!r}, max_af={max_af!r})")
    return result, {
        "id": measurement.id, "path": spec.get("path"), "expr": text,
        "kind": measurement.kind, "unit": spec.get("unit"),
        "rate_hz": rate, "rate_note": rate_note, "jitter": jitter,
        "pedestal": pedestal, "window": [lo, hi], "n": series.n,
        "n_gaps": stats["n_gaps"], "divide_by_zero": n_blown,
        "gap_fraction": stats["gap_fraction"], "runs": stats["runs"],
        "longest_run": stats["longest_run"], "n_windows": len(spans),
        "n_rejected": n_rejected,
        # What the reader had to tolerate to read this file at all: ragged
        # rows, text columns, unparseable fields. Per measurement rather
        # than once for the bench, because six counters produce six
        # different kinds of mess and "something was tolerated somewhere"
        # is not actionable.
        "defects": table.defects,
        "tau_s": floats(result.tau_s), "dev": floats(result.dev),
    }, lines


# ---------------------------------------------------------------------------
# the equivalent script
# ---------------------------------------------------------------------------

def preamble() -> list[str]:
    """Import lines and setup shared by every generated bench script."""
    return [
        "import numpy as np",
        "import stabilis",
        "from stabilis import edit",
        "from stabilis.io.text import read_table",
        "from stabilis.bench.solve import solve",
        "from stabilis.bench.store import load",
        "",
        "bench = load('bench.json')",
        "results = {}",
        "",
    ]


def recipe(measurement, spec, table, labels, text, is_phase, factor, rate):
    """The exact calls that rebuild this measurement's series.

    `read_table` rather than the friendlier `load_phase(column=...)` even
    when the expression is a single column, and the reason is the pedestal.
    `prepare.build_series` subtracts the first finite sample before scaling
    the units -- which is what buys back the float64 resolution a 0.35 s
    phase offset would otherwise spend -- and `load_phase` does not. The
    shorter recipe would therefore reproduce the answer without reproducing
    the arithmetic, and a recipe that is only nearly what happened is worth
    less than no recipe at all.
    """
    kind = "PhaseSeries" if is_phase else "FreqSeries"
    path = str(spec.get("path") or "")
    note = ("  (back-to-back: measures the channel)"
            if measurement.is_back_to_back else "")
    lines = [f"# {measurement.id}: {measurement.source_a} vs "
             f"{measurement.source_b} through {measurement.channel_id}{note}",
             f"labels, rows, _defects = read_table({path!r})"]
    if table.header_row_dropped:
        lines.append("rows = rows[1:]  # the header line parsed as a row of gaps")
    lines.append(f"values = {numpy_form(text, labels)}")
    scale = f" * {factor!r}" if factor != 1.0 else ""
    lines.append(f"values = (values - values[np.isfinite(values)][0]){scale}")
    lines.append(f"series = stabilis.{kind}(values, rate_hz={rate!r})")
    return lines


def numpy_form(text: str, labels) -> str:
    """The expression as runnable numpy, from the same tokens the page used.

    Reusing `expr.tokenize` rather than writing a second reader is the whole
    point: the closed grammar stays the only thing that has ever looked at
    the string, and this walk cannot admit a token the parser would have
    refused.
    """
    out = []
    for kind, value in expr.tokenize(text, labels):
        out.append(f"rows[:, {value}]" if kind == "col"
                   else repr(value) if kind == "num" else str(value))
    return " ".join(out)
