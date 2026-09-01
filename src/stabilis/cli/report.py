"""`stabilis report` -- the customer-facing measurement report.

Conformity is decided on the interval, not the point, so a deviation under
the limit whose error bar crosses it does not pass. `_spec_points` and
`_report_grid` are shared with `export bundle`, which builds a report as
part of the directory it writes -- the one import edge inside this package,
and it points the right way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..stats import ONE_SIGMA, confidence
from ..tau import MODES
from .common import _add_input_args, _load

__all__ = ["add_report"]


def add_report(sub) -> None:
    p_rep = sub.add_parser(
        "report",
        help="an HTML measurement report, with a conformity verdict that "
             "respects the error bars",
    )
    _add_input_args(p_rep)
    p_rep.add_argument(
        "--estimator", default="oadev",
        choices=("adev", "oadev", "mdev", "tdev", "ohdev", "totdev"),
    )
    p_rep.add_argument("--taus", default="octave", choices=MODES)
    p_rep.add_argument("--max-af", type=int, help="largest averaging factor")
    p_rep.add_argument(
        "--confidence", type=float, default=ONE_SIGMA,
        help="confidence level for every interval and for the verdict",
    )
    p_rep.add_argument(
        "--spec", action="append", metavar="TAU:SIGMA", default=[],
        help="one specification point, repeatable. The tau grid is extended "
             "to hit each one exactly, so a limit is never interpolated",
    )
    p_rep.add_argument(
        "--mask", action="append", default=[], dest="masks", metavar="NAME",
        help="judge against a published limit too; repeatable, and "
             "`stabilis mask --list-masks` shows the catalogue")
    p_rep.add_argument("--bench", help="bench.json to read spec points from")
    p_rep.add_argument(
        "--source", help="which source in --bench carries the specification")
    p_rep.add_argument("--title", default="Frequency stability measurement")
    p_rep.add_argument("--subject", default="", help="what was measured")
    p_rep.add_argument("--operator", default="", help="who measured it")
    p_rep.add_argument("--notes", default="", help="conditions worth stating")
    p_rep.add_argument(
        "--decompose", action="store_true",
        help="include the power-law decomposition (adds an mdev pass)")
    p_rep.add_argument(
        "--no-figure", action="store_true",
        help="omit the plot; the report needs no matplotlib without it")
    p_rep.add_argument("--out", help="write HTML here (default: stdout)")
    p_rep.add_argument("--json", action="store_true", help="emit JSON instead")
    p_rep.set_defaults(func=_cmd_report)


def _spec_points(args):
    """Specification points and a label, from `--spec` pairs or a bench.

    Both sources may be given; the bench's points come second so a
    `--spec` override is visible in the table rather than replacing
    anything silently.
    """
    points = []
    for item in args.spec:
        tau, sep, sigma = item.partition(":")
        if not sep:
            raise ValueError(f"--spec wants TAU:SIGMA, got {item!r}")
        points.append((float(tau), float(sigma)))
    label = "supplied points" if points else ""
    if args.bench:
        if not args.source:
            raise ValueError("--bench needs --source to say which oscillator")
        from ..bench import load

        bench = load(args.bench)
        matches = [s for s in bench.sources if s.id == args.source]
        if not matches:
            raise KeyError(
                f"no source {args.source!r} in {args.bench}; it has "
                f"{[s.id for s in bench.sources]}"
            )
        source = matches[0]
        if not source.spec_adev:
            raise ValueError(f"source {source.id!r} carries no spec_adev points")
        points.extend((float(t), float(v)) for t, v in source.spec_adev)
        label = f"{source.id} datasheet" if not label else f"{label} + {source.id}"
    return tuple(sorted(points)), label


def _report_grid(args, series, spec):
    """The tau grid, extended to land exactly on every reachable spec tau.

    Spec points beyond the grid's own `max_af` policy are left out rather
    than pushing the analysis past it; `report.build` then reports them as
    not evaluated, which is the honest outcome for a limit stated at a tau
    the record cannot support.
    """
    from ..tau import averaging_factors
    from ..report import spec_afs

    base = averaging_factors(series.n, args.taus, args.max_af)
    if not spec or base.size == 0:
        return base
    extra = spec_afs(spec, series.tau0_s)
    return np.unique(np.concatenate([base, extra[extra <= base.max()]]))


def _masks_from(args):
    """Resolve `--mask NAME` to catalogue entries, refusing an unknown one."""
    from ..mask import CATALOG, available

    out = []
    for name in getattr(args, "masks", ()) or ():
        if name not in CATALOG:
            raise KeyError(f"no mask {name!r}; available: {available()}")
        out.append(CATALOG[name])
    return tuple(out)


def _cmd_report(args) -> int:
    """A measurement report: the document a laboratory hands to a customer.

    Distinct from `validate`, which is evidence about the software rather
    than about a measurement. The conformity verdict here is taken on the
    confidence interval, not on the point estimate, so a record too short
    to decide says so instead of guessing.
    """
    from ..registry import get
    from ..report import build, figure_uri, to_html, to_json

    series = _load(args)
    spec, spec_label = _spec_points(args)
    grid = _report_grid(args, series, spec)
    result = get("estimator", args.estimator)(series, taus=grid)
    ci = confidence(result, level=args.confidence)

    fit = None
    if args.decompose:
        from ..stats import mdev
        from ..stats.powerlaw import decompose

        fit = decompose([result, mdev(series, taus=grid)])

    report = build(
        series, result, ci,
        masks=_masks_from(args),
        title=args.title,
        subject=args.subject or f"{Path(args.path).name} ({args.estimator})",
        operator=args.operator,
        notes=args.notes,
        spec=spec,
        spec_label=spec_label,
        fit=fit,
    )
    if args.json:
        print(to_json(report))
        return 0

    uri = "" if args.no_figure else figure_uri(result, ci, spec, spec_label)
    html = to_html(report, uri)
    if not args.out:
        print(html)
        return 0
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out} -- verdict {report.verdict.upper()}, "
          f"digest {report.digest[:16]}")
    if report.worst is not None:
        w = report.worst
        print(f"worst margin {(w.margin - 1) * 100:+.1f}% at tau = "
              f"{w.tau_used_s:g} s (limit {w.limit:.3e}, upper bound "
              f"{w.dev_hi:.3e})")
    return 0
