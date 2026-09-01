"""`stabilis bench` -- sources, devices and channels as a graph.

`check` reads no data files at all: identifiability is a property of the
setup, so it answers before anything is measured, and names the one extra
edge that would fix an underdetermined bench. `recommend` goes further and
says which technique the graph supports, since the hat and cross-correlation
are not alternatives.

`bench/` is the optional setup model `stats/` may not import, and the arrow
points the same way here: this module reaches into it, nothing reaches back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ..provenance import to_dict
from ..tau import MODES

__all__ = ["add_bench"]


def add_bench(sub) -> None:
    p_bench = sub.add_parser(
        "bench", help="the measurement setup: what it can and cannot determine"
    )
    bench_sub = p_bench.add_subparsers(dest="bench_command", required=True)

    p_check = bench_sub.add_parser(
        "check",
        help="can this setup separate its oscillators? reads no data files",
    )
    p_check.add_argument("path", help="bench.json")
    p_check.add_argument("--json", action="store_true", help="emit JSON")
    p_check.set_defaults(func=_cmd_bench_check)

    p_rec = bench_sub.add_parser(
        "recommend",
        help="which technique this setup supports -- hat, cross-correlation, "
             "or both -- and which assumption each answer rests on",
    )
    p_rec.add_argument("path", help="bench.json")
    p_rec.add_argument("--json", action="store_true", help="emit JSON")
    p_rec.set_defaults(func=_cmd_bench_recommend)

    p_solve = bench_sub.add_parser(
        "solve", help="N-cornered hat and channel floors, one solve per tau"
    )
    p_solve.add_argument("path", help="bench.json")
    p_solve.add_argument(
        "--estimator", default="oadev",
        choices=("adev", "oadev", "mdev", "tdev", "ohdev", "totdev"),
    )
    p_solve.add_argument("--taus", default="octave", choices=MODES)
    p_solve.add_argument("--max-af", type=int, help="largest averaging factor")
    p_solve.add_argument(
        "--allow-unidentifiable", action="store_true",
        help="solve a singular bench anyway; the answer is then one of "
             "infinitely many that fit equally well",
    )
    p_solve.add_argument(
        "--no-verify", action="store_true",
        help="skip the SHA-256 check on the referenced data files",
    )
    p_solve.add_argument("--json", action="store_true", help="emit JSON")
    p_solve.set_defaults(func=_cmd_bench_solve)


def _cmd_bench_check(args) -> int:
    """Can this setup determine its unknowns? Reads no data files.

    Worth running while the experiment is still being planned, which is the
    only time the answer -- "go and measure B against D" -- is cheap.
    """
    from ..bench import check
    from ..bench.store import load

    ident = check(load(args.path))
    if args.json:
        print(json.dumps({
            "identifiable": ident.identifiable,
            "rank": ident.rank,
            "n_unknowns": ident.n_unknowns,
            "sources": list(ident.sources),
            "free_channels": list(ident.channels),
            "components": [list(c) for c in ident.components],
            "bipartite": list(ident.bipartite),
            "suggestions": [list(p) for p in ident.suggestions],
            "derivable": [list(e) for e in ident.derivable],
            "notes": list(ident.notes),
        }, indent=2))
    else:
        print(ident.report())
    return 0 if ident.identifiable else 2


def _cmd_bench_recommend(args) -> int:
    """Hat, cross-correlation, or both. Reads no data files.

    Exit status is 0 when *something* determines the sources -- the hat on
    its own, or the hat once a repeated edge has removed the channel term.
    A bench that neither closes is still a 2, the same as `bench check`.
    """
    from ..bench import load
    from ..bench.recommend import recommend

    bench = load(args.path, verify_hashes=False)
    rec = recommend(bench)
    if args.json:
        print(json.dumps({
            "hat_identifiable": rec.hat_identifiable,
            "hat_deficiency": rec.hat_deficiency,
            "would_identify": rec.would_identify,
            "channels_determined": list(rec.channels_determined),
            "opportunities": [
                {"source_a": o.source_a, "source_b": o.source_b,
                 "measurements": list(o.measurements),
                 "channels": list(o.channels), "devices": list(o.devices),
                 "quality": o.quality}
                for o in rec.opportunities
            ],
            "notes": list(rec.notes),
        }, indent=2))
    else:
        print(rec.report())
    return 0 if (rec.hat_identifiable or rec.would_identify) else 2


def _cmd_bench_solve(args) -> int:
    """The N-cornered hat, channel floors and the common-mode warnings."""
    from ..bench.solve import solve
    from ..bench.store import load
    from ..io import load_frequency, load_phase
    from ..registry import get

    bench = load(args.path, verify_hashes=not args.no_verify)
    base = Path(args.path).parent
    estimator = get("estimator", args.estimator)

    results = {}
    for m in bench.measurements:
        if not m.path:
            continue
        target = Path(m.path)
        if not target.is_absolute():
            target = base / target
        loader = load_phase if m.kind == "phase" else load_frequency
        rate = 1.0 / m.tau0_s if m.tau0_s else None
        series = loader(target, rate_hz=rate)
        results[m.id] = estimator(series, taus=args.taus, max_af=args.max_af)
    if not results:
        print("stabilis: no measurement in this bench names a data file",
              file=sys.stderr)
        return 1

    sol = solve(bench, results,
                require_identifiable=not args.allow_unidentifiable)

    if args.json:
        print(json.dumps({
            "estimator": sol.estimator,
            "tau_s": sol.tau_s.tolist(),
            "sources": list(sol.sources),
            "channels": list(sol.channels),
            "source_dev": {s: sol.dev(s).tolist() for s in sol.sources},
            "source_var_signed": {
                s: sol.signed_var(s).tolist() for s in sol.sources},
            "channel_var": {c: sol.channel_var[:, i].tolist()
                            for i, c in enumerate(sol.channels)},
            "residual": sol.residual.tolist(),
            "kkt": sol.kkt.tolist(),
            "warnings": list(sol.warnings),
            "provenance": to_dict(sol.provenance),
        }, indent=2, default=str))
        return 0

    print(f"# bench solve of {args.path}, estimator {sol.estimator}")
    for text in sol.warnings:
        print(f"# WARNING: {text}")
    if sol.any_negative:
        print("# WARNING: the unconstrained solution went negative somewhere. "
              "An independence assumption is violated; see the signed columns.")
    header = f"{'Tau (s)':>12}" + "".join(f"{s:>14}" for s in sol.sources)
    header += "".join(f"{c:>14}" for c in sol.channels)
    print("\n# deviations, sqrt of the constrained (non-negative) solution")
    print(header)
    for k, tau in enumerate(sol.tau_s):
        row = f"{tau:>12.4e}"
        row += "".join(f"{np.sqrt(v):>14.6e}" for v in sol.source_var[k])
        row += "".join(f"{np.sqrt(v):>14.6e}" for v in sol.channel_var[k])
        print(row)
    if sol.any_negative:
        print("\n# unconstrained variances, sign intact -- the diagnostic")
        print(f"{'Tau (s)':>12}" + "".join(f"{s:>14}" for s in sol.sources))
        for k, tau in enumerate(sol.tau_s):
            row = f"{tau:>12.4e}"
            row += "".join(f"{v:>14.3e}" for v in sol.unconstrained[k, :len(sol.sources)])
            print(row)
    return 0
