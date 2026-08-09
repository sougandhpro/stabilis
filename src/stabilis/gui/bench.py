"""The bench stage: which clock, against which, through which channel.

Stages 1-3 answer "what is this record worth". This one answers the
question a calibration lab actually starts from -- *I have four oscillators
and two counters; which of these numbers belongs to which clock?* -- and it
is the same three questions as before with the arithmetic done by
`bench/`: describe the setup, assign each measurement its data, solve.

**The library is handed results, never files.** `bench.solve` takes a
`{measurement id -> StabilityResult}` mapping and does one non-negative
least-squares solve per tau; it has no notion of columns, units or headers
and must not grow one. So the stripping is this module's job -- pick the
columns out of the file, subtract the pedestal, scale the units, crop to
the selected stretch -- and by the time anything the library owns is
called, a measurement is an array with a sample rate. That split is why
`Measurement` needs no column field: the page never asks the engine to go
looking for a column, so there is no second answer to disagree with.

The work is exactly the work `analysis.py` does for one series, done once
per measurement, through the same `expr` -> `prepare` -> registry path. The
recipe printed beside the answer is the whole thing as runnable code,
including the pedestal and the unit scaling, so the session reproduces
without the mouse.

Two names spelled `bench` meet in this file. `..bench` is the library's
measurement-setup model; this module is the page's client of it, and every
import of the former is written relatively so the two never blur.
"""

from __future__ import annotations

import numpy as np

from . import benchdata
from .request import floats

__all__ = ["check_bench", "solve_bench", "load_bench"]


def _bench_from(req: dict):
    """The draft in the request as a `Bench`, validated by its own model.

    A bench the page has just built carries no `schema`, so this supplies
    the current one -- that is a bench being *constructed* here, not a
    foreign file being guessed at. A bench that came off disk keeps whatever
    it was loaded with, so `store.from_dict` still refuses a version this
    reader does not know.
    """
    from ..bench.model import SCHEMA_VERSION
    from ..bench.store import from_dict

    draft = dict(req.get("bench") or {})
    draft.setdefault("schema", SCHEMA_VERSION)
    return from_dict(draft)


# ---------------------------------------------------------------------------
# POST /api/bench/check
# ---------------------------------------------------------------------------

def check_bench(root, req: dict) -> dict:
    """Can this setup separate its oscillators? Reads no data file.

    Fired on every edit, because the answer is worth most while the
    experiment is still being planned -- "go and measure B against D" is
    cheap advice on a Tuesday and expensive advice after a fortnight of
    logging. `root` is unused for exactly that reason and is kept only
    because every route is called the same way.
    """
    from ..bench.graph import check
    from ..bench.solve import common_mode_warnings
    from ..bench.store import to_dict

    bench = _bench_from(req)
    ident = check(bench)
    measured = set(ident.sources)
    return {
        # The draft back through the library's own serializer, so what the
        # page offers as a download is byte-for-byte what `store.save`
        # would have written rather than a second guess at the layout.
        "bench": to_dict(bench),
        "identifiable": ident.identifiable,
        "rank": ident.rank,
        "n_unknowns": ident.n_unknowns,
        "deficiency": ident.deficiency,
        "sources": list(ident.sources),
        "free_channels": list(ident.channels),
        "components": [list(c) for c in ident.components],
        "bipartite": list(ident.bipartite),
        "suggestions": [list(p) for p in ident.suggestions],
        "derivable": [list(e) for e in ident.derivable],
        "notes": list(ident.notes),
        "report": ident.report(),
        "warnings": list(common_mode_warnings(bench)),
        # Page-level facts, not verdicts: an empty bench is trivially
        # "identifiable" with nothing to determine, and a source nobody
        # measures silently has no column in the solve. Both look like
        # success from the rank alone, so they are reported separately
        # rather than left for a green tick to misrepresent.
        "n_measurements": len(bench.measurements),
        "unmeasured": [s.id for s in bench.sources if s.id not in measured],
        "fixed_channels": [c.id for d in bench.devices for c in d.channels
                           if c.floor is not None],
    }


# ---------------------------------------------------------------------------
# POST /api/bench/load
# ---------------------------------------------------------------------------

def load_bench(root, req: dict) -> dict:
    """Read a `bench.json` from under the root, for editing in the page.

    Read only. The page saves by handing the browser a download, so this
    server never writes to the disk it can read -- the bind is to localhost
    because a file reader is already something worth confining, and a file
    writer would be a different conversation.

    Hashes are checked and *reported*, never enforced: an unsolvable bench
    is still one worth opening and correcting, and refusing to display it
    would leave the user with a text editor and no diagnostic.
    """
    from .files import safe_path
    from ..bench.store import load, to_dict, verify

    path = safe_path(root, req["path"])
    bench = load(path, verify_hashes=False)
    return {"path": req["path"], "bench": to_dict(bench),
            "hash_problems": verify(bench)}


# ---------------------------------------------------------------------------
# POST /api/bench/solve
# ---------------------------------------------------------------------------

def solve_bench(root, req: dict) -> dict:
    """Strip every measurement to its column, estimate, then hand it over.

    Nothing here decides anything a user would quote. The estimator comes
    from the registry, the solve from `bench/`, and the only arithmetic in
    this function is picking the source block out of the answer.
    """
    from ..bench.solve import solve
    from ..registry import available

    bench = _bench_from(req)
    data = req.get("data") or {}
    name = req.get("estimator", "oadev")
    if name not in available("estimator"):
        raise ValueError(f"unknown estimator {name!r}; expected one of "
                         f"{list(available('estimator'))}")
    taus = req.get("taus", "octave")
    max_af = int(req["max_af"]) if req.get("max_af") not in (None, "") else None
    strict = not bool(req.get("allow_unidentifiable"))

    results, prepared, notes = {}, [], []
    calls = benchdata.preamble()
    for m in bench.measurements:
        spec = data.get(m.id) or {}
        if not str(spec.get("path") or "").strip():
            notes.append(f"{m.id}: no data assigned, so it takes no part in "
                         "the solve. The bench still counts it when deciding "
                         "what is identifiable.")
            continue
        result, info, lines = benchdata.measure(
            root, m, spec, name, taus, max_af)
        results[m.id] = result
        prepared.append(info)
        calls.extend(lines + [""])
        if info["divide_by_zero"]:
            notes.append(f"{m.id}: {info['divide_by_zero']} points became gaps "
                         "because a divisor was zero.")
        if info["n_gaps"]:
            notes.append(
                f"{m.id}: {info['n_gaps']} gaps ("
                f"{info['gap_fraction'] * 100:.2f}%), {info['runs']} runs, "
                f"longest {info['longest_run']}; see `stabilis docs gaps` "
                "for how they were handled.")
        if info["defects"]:
            notes.append(f"{m.id}: tolerated while reading {info['path']} -- "
                         f"{info['defects']}")
        if info["n_rejected"]:
            notes.append(
                f"{m.id}: the outlier threshold removed {info['n_rejected']} "
                f"points of {info['n']}. They became gaps, not deletions -- "
                "nothing was filled in, and the step log records the "
                "threshold and the reason.")
    notes.extend(benchdata.rate_notes(prepared))

    if not results:
        raise ValueError(
            "no measurement has a file assigned yet. Each row needs a file "
            "and a column expression before there is anything to solve; "
            "`bench check` is the part that runs without data.")

    calls.append(f"solution = solve(bench, results, "
                 f"require_identifiable={strict!r})")
    try:
        sol = solve(bench, results, require_identifiable=strict)
    except ValueError as exc:
        # The solver's refusals are correct and terse. The page knows things
        # it does not -- which file each row came from and how its rate was
        # arrived at -- so it adds them rather than restating the refusal.
        raise benchdata.explain_failure(exc, prepared) from exc

    n_src = len(sol.sources)
    return {
        "estimator": sol.estimator,
        "tau_s": floats(sol.tau_s),
        "sources": list(sol.sources),
        "channels": list(sol.channels),
        "source_dev": {s: floats(sol.dev(s)) for s in sol.sources},
        "source_var_signed": {s: floats(sol.signed_var(s)) for s in sol.sources},
        "channel_dev": {c: floats(np.sqrt(sol.channel_var[:, i]))
                        for i, c in enumerate(sol.channels)},
        "channel_var_signed": {c: floats(sol.unconstrained[:, n_src + i])
                               for i, c in enumerate(sol.channels)},
        "residual": floats(sol.residual),
        "kkt": floats(sol.kkt),
        "warnings": list(sol.warnings),
        "any_negative": sol.any_negative,
        "measurements": prepared,
        "notes": notes,
        "calls": calls,
        "steps": _steps(sol.provenance),
    }


def _steps(provenance) -> list[dict]:
    """The step log, flattened for the page -- shown, not merely carried."""
    return [{"module": step.module, "action": step.action,
             "outcome": step.outcome,
             "params": {k: str(v) for k, v in step.params.items()}}
            for step in provenance.steps]
