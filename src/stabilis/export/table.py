"""A stability curve as CSV or JSON.

Two audiences and two shapes. JSON is the complete record -- every field of
the result, the interval, and the provenance behind both -- and is what a
program should read. CSV is a table for a spreadsheet, and it is lossy on
purpose: a rectangular grid cannot carry the per-tau EDOF method or the
reason a point declined one, so those live in the JSON only and the CSV
does not pretend otherwise.

**The CSV never exceeds eight columns.** That is not a style choice:
`io/text.py` refuses a file wider than `MAX_COLUMNS = 8`, so a ninth column
would produce a file this library could not read back.

Computes nothing. Everything printed here was decided by whoever chose the
estimator and the interval, which is the same rule `report.build` follows --
an export must never be a second place where an analysis option gets picked.
"""

from __future__ import annotations

import json

import numpy as np

from ..provenance import to_dict as prov_to_dict

__all__ = ["COLUMNS", "to_dict", "to_json", "to_csv"]

# In the order they are written, and never more than `io.text.MAX_COLUMNS`
# of them at once. `alpha` and the interval columns appear only when there
# is an interval to report.
COLUMNS = ("af", "tau_s", "n_terms", "alpha", "dev_lo", "dev", "dev_hi", "edof")


def to_dict(result, ci=None) -> dict:
    """The complete result as plain Python, ready for `json.dumps`.

    Carries `var` as well as `dev`. The library stores variances and
    derives deviations, and an export that shipped only the deviation would
    force every consumer to square it back -- which is exactly the round
    trip `CLAUDE.md` forbids internally, exported.

    The provenance taken is `(ci or result)`'s, not the result's: computing
    an interval appends steps, and the shorter record would silently drop
    the EDOF method from the audit trail.
    """
    payload = {
        "kind": result.kind,
        "af": result.af.tolist(),
        "tau_s": result.tau_s.tolist(),
        "var": result.var.tolist(),
        "dev": result.dev.tolist(),
        "n_terms": result.n_terms.tolist(),
        "rate_hz": result.rate_hz,
        "is_variance": result.is_variance,
        "provenance": prov_to_dict((ci or result).provenance),
    }
    if ci is not None:
        payload["confidence"] = {
            "level": ci.level,
            "edof": ci.edof.tolist(),
            "edof_method": list(ci.method),
            "dev_lo": ci.dev_lo.tolist(),
            "dev_hi": ci.dev_hi.tolist(),
            "alpha": None if ci.alpha is None else ci.alpha.tolist(),
            "declined": list(ci.declined),
        }
    return payload


def to_json(result, ci=None, indent: int = 2) -> str:
    """Serialize a result and its interval. The 2-space convention."""
    return json.dumps(to_dict(result, ci), indent=indent, default=str)


def to_csv(result, ci=None, delimiter: str = ",", digits: int = 17) -> str:
    """One row per tau, with a column-name line the reader can recover.

    The names go on the first line because that is the only line
    `io.text.read_table` inspects for them. Seventeen digits by default,
    for the reason `theory/export.md` gives: it is the width at which a
    float64 survives the trip.
    """
    columns = _columns(result, ci)
    fmt = f"%.{digits}g"
    lines = ["# " + delimiter.join(name for name, _ in columns)]
    for row in zip(*(values for _, values in columns)):
        lines.append(delimiter.join(
            str(int(v)) if isinstance(v, (int, np.integer)) else fmt % v
            for v in row))
    return "\n".join(lines) + "\n"


def _columns(result, ci) -> list[tuple[str, np.ndarray]]:
    """The columns this result actually supports, in `COLUMNS` order.

    A column is present when there is something to put in it, rather than
    padded with a filler value. An empty cell in a numeric table is read as
    a gap by our own reader, and a zero would be read as a measurement.
    """
    columns = [("af", result.af), ("tau_s", result.tau_s),
               ("n_terms", result.n_terms)]
    if ci is None:
        columns.append(("dev", result.dev))
        return columns
    if ci.alpha is not None:
        columns.append(("alpha", ci.alpha))
    columns += [("dev_lo", ci.dev_lo), ("dev", result.dev),
                ("dev_hi", ci.dev_hi), ("edof", ci.edof)]
    return columns
