"""Command-line interface.

Subcommands are thin: each parses arguments, calls one library function and
formats the result. Nothing here computes anything, so the CLI can never
disagree with the library.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .docs import list_topics, read_topic, topic_summary
from .log import configure
from .preprocess import basic_stats
from .provenance import to_dict
from .stats import ONE_SIGMA, confidence
from .stats.core import GAP_POLICIES
from .stats.edof import EDOF_METHODS
from .stats.greenhall import GREENHALL_RUN_POLICIES
from .gui import DEFAULT_PORT as GUI_DEFAULT_PORT
from .stats.powerlaw import WEIGHT_METHODS
from .tau import MODES

__all__ = ["main", "build_parser"]

_LOG_LEVELS = ("debug", "info", "warning", "error")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stabilis", description="Frequency stability analysis."
    )
    parser.add_argument("--version", action="version", version=f"stabilis {__version__}")
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=_LOG_LEVELS,
        help="how much of the step log to print to stderr (default: warning)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_docs = sub.add_parser("docs", help="read the theory pages")
    p_docs.add_argument("topic", nargs="?", help="topic name, e.g. noise-types")
    p_docs.add_argument(
        "--list", action="store_true", dest="list_topics", help="list all topics"
    )
    p_docs.set_defaults(func=_cmd_docs)

    p_val = sub.add_parser("validate", help="run the validation evidence report")
    p_val.add_argument("--out", help="write an HTML report to this path")
    p_val.add_argument("--json", action="store_true", help="emit JSON to stdout")
    p_val.add_argument(
        "--trials",
        type=int,
        default=300,
        help="Monte-Carlo depth; raised to 300 if lower (default: 300, "
        "published figures use 800)",
    )
    p_val.add_argument(
        "--self-test",
        action="store_true",
        help="inject a check designed to fail, proving the report can",
    )
    p_val.set_defaults(func=_cmd_validate)

    p_stats = sub.add_parser("stats", help="basic statistics for a data file")
    _add_input_args(p_stats)
    p_stats.add_argument("--json", action="store_true", help="emit JSON")
    p_stats.set_defaults(func=_cmd_stats)

    from .stats import adev, mdev, mtie, oadev, ohdev, tdev, tierms, totdev

    # MTIE and TIE rms take no confidence arguments: they are not variances,
    # so the chi-square interval does not apply and `confidence` refuses
    # them. Offering the flags and then ignoring them would be worse than
    # not offering them.
    for name, fn in (
        ("adev", adev),
        ("oadev", oadev),
        ("mdev", mdev),
        ("tdev", tdev),
        ("ohdev", ohdev),
        ("totdev", totdev),
        ("mtie", mtie),
        ("tierms", tierms),
    ):
        p = sub.add_parser(name, help=fn.__doc__.splitlines()[0])
        _add_input_args(p)
        p.add_argument("--taus", default="octave", choices=MODES, help="tau grid")
        p.add_argument("--max-af", type=int, help="largest averaging factor")
        p.add_argument(
            "--gap-policy",
            default="auto",
            choices=GAP_POLICIES,
            help="how a gap affects nearby terms; see `stabilis docs gaps`",
        )
        if name in ("mtie", "tierms"):
            p.add_argument("--json", action="store_true", help="emit JSON")
            p.set_defaults(func=_cmd_estimate, estimator=fn,
                           estimator_name=name, no_confidence=True,
                           edof_method="auto", confidence=ONE_SIGMA)
            continue
        p.add_argument(
            "--confidence",
            type=float,
            default=ONE_SIGMA,
            help="confidence level for the sigma bounds (default: 1 sigma, 0.6827)",
        )
        p.add_argument(
            "--edof-method",
            default="auto",
            choices=EDOF_METHODS,
            help="degrees-of-freedom estimator; see `stabilis docs edof`",
        )
        p.add_argument(
            "--alpha",
            type=int,
            choices=(2, 1, 0, -1, -2, -3, -4),
            help="power law to assume, with --edof-method greenhall. Omit to "
            "identify it per tau; pass Stable32's Alpha column to reproduce "
            "one of its runs exactly. See `stabilis docs greenhall`",
        )
        p.add_argument(
            "--run-policy",
            default="pool",
            choices=GREENHALL_RUN_POLICIES,
            help="on a gapped record, whether greenhall pools the runs or "
            "declines to them (default: pool)",
        )
        p.add_argument(
            "--no-confidence",
            action="store_true",
            help="skip the bounds; also frees the coefficient arrays, which "
            "matter on a record of millions of points",
        )
        p.add_argument("--json", action="store_true", help="emit JSON")
        p.set_defaults(func=_cmd_estimate, estimator=fn, estimator_name=name)

    p_cmp = sub.add_parser(
        "compare-edof",
        help="every EDOF method side by side, and what each does to the bounds",
    )
    _add_input_args(p_cmp)
    p_cmp.add_argument(
        "--estimator", default="oadev",
        choices=("adev", "oadev", "mdev", "tdev", "ohdev", "totdev")
    )
    p_cmp.add_argument("--taus", default="octave", choices=MODES, help="tau grid")
    p_cmp.add_argument("--max-af", type=int, help="largest averaging factor")
    p_cmp.add_argument(
        "--confidence", type=float, default=ONE_SIGMA, help="confidence level"
    )
    p_cmp.add_argument(
        "--alpha",
        type=int,
        choices=(2, 1, 0, -1, -2, -3, -4),
        help="power law for the greenhall column; omit to identify per tau",
    )
    p_cmp.add_argument("--json", action="store_true", help="emit JSON")
    p_cmp.set_defaults(func=_cmd_compare_edof)

    p_pl = sub.add_parser(
        "powerlaw",
        help="decompose sigma(tau) into IEEE 1139 h_alpha levels and drift",
    )
    _add_input_args(p_pl)
    p_pl.add_argument("--taus", default="octave", choices=MODES, help="tau grid")
    p_pl.add_argument("--max-af", type=int, help="largest averaging factor")
    p_pl.add_argument(
        "--fh",
        type=float,
        help="measurement system bandwidth in Hz. NOT identifiable from the "
        "data -- omit it and Nyquist is assumed and the result says so",
    )
    p_pl.add_argument(
        "--weights",
        default="edof",
        choices=WEIGHT_METHODS,
        help="row weighting; 'edof' uses each point's own sampling uncertainty",
    )
    p_pl.add_argument("--json", action="store_true", help="emit JSON")
    p_pl.set_defaults(func=_cmd_powerlaw)

    p_plot = sub.add_parser("plot", help="plot a data file (needs matplotlib)")
    _add_input_args(p_plot)
    p_plot.add_argument(
        "--draw", default="sigma-tau",
        choices=("sigma-tau", "phase", "frequency"),
        help="what to draw (default: sigma-tau). Distinct from --kind, "
             "which says how to read the input file",
    )
    p_plot.add_argument(
        "--estimator", default="oadev",
        choices=("adev", "oadev", "mdev", "tdev", "ohdev", "totdev", "mtie",
                 "tierms"),
    )
    p_plot.add_argument("--taus", default="octave", choices=MODES)
    p_plot.add_argument("--max-af", type=int, help="largest averaging factor")
    p_plot.add_argument(
        "--confidence", type=float, default=ONE_SIGMA,
        help="confidence level for the error bars (default: 1 sigma)",
    )
    p_plot.add_argument(
        "--no-confidence", action="store_true", help="draw no error bars")
    p_plot.add_argument(
        "--slopes", action="store_true",
        help="overlay power-law reference slopes, anchored to the first point",
    )
    p_plot.add_argument("--title", help="figure title")
    p_plot.add_argument(
        "--out", help="write to this path; omit to open a window")
    p_plot.add_argument("--dpi", type=int, default=150)
    p_plot.set_defaults(func=_cmd_plot)

    p_gui = sub.add_parser(
        "gui", help="a small local page in your browser, instead of flags")
    p_gui.add_argument("--port", type=int, default=GUI_DEFAULT_PORT)
    p_gui.add_argument(
        "--root", help="confine the file browser to this directory "
                       "(default: the current one)")
    p_gui.add_argument("--no-browser", action="store_true",
                       help="just print the URL")
    p_gui.set_defaults(func=_cmd_gui)

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

    return parser


def _add_input_args(p: argparse.ArgumentParser) -> None:
    """Input arguments shared by every data-consuming subcommand."""
    p.add_argument("path", help="input data file")
    p.add_argument(
        "--kind",
        choices=("phase", "frequency"),
        default="phase",
        help="whether the column holds phase in seconds or fractional frequency",
    )
    p.add_argument("--rate", type=float, help="sample rate in Hz")
    p.add_argument("--column", type=int, default=0, help="data column, 0-based")
    p.add_argument("--time-column", type=int, help="timetag column, to infer --rate")
    p.add_argument("--delimiter", help="override delimiter sniffing")
    p.add_argument(
        "--gap-value",
        type=float,
        help="value marking a gap (Stable32 writes 0; not assumed)",
    )


def _load(args):
    """Read the input file into a PhaseSeries or FreqSeries."""
    from .io import load_frequency, load_phase

    loader = load_phase if args.kind == "phase" else load_frequency
    return loader(
        args.path,
        rate_hz=args.rate,
        column=args.column,
        time_column=args.time_column,
        delimiter=args.delimiter,
        gap_value=args.gap_value,
    )


def _cmd_docs(args) -> int:
    if args.list_topics or args.topic is None:
        for topic in list_topics():
            print(f"  {topic:<16} {topic_summary(topic)}")
        return 0
    try:
        sys.stdout.write(read_topic(args.topic))
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2
    return 0


def _cmd_stats(args) -> int:
    series = _load(args)
    values = series.x if args.kind == "phase" else series.y
    stats = basic_stats(values)

    if args.json:
        print(
            json.dumps(
                {
                    "stats": stats,
                    "rate_hz": series.rate_hz,
                    "tau0_s": series.tau0_s,
                    "duration_s": series.duration_s,
                    "provenance": to_dict(series.provenance),
                },
                indent=2,
                default=str,
            )
        )
        return 0

    unit = "s" if args.kind == "phase" else ""
    print(f"file      {args.path}")
    print(f"kind      {args.kind}")
    print(f"n         {stats['n']}  ({stats['n_gaps']} gaps)")
    print(f"tau0      {series.tau0_s:.9g} s   rate {series.rate_hz:.9g} Hz")
    print(f"duration  {series.duration_s:.9g} s")
    print(f"min       {stats['min']:.9e} {unit}")
    print(f"max       {stats['max']:.9e} {unit}")
    print(f"mean      {stats['mean']:.9e} {unit}")
    print(f"median    {stats['median']:.9e} {unit}")
    print(f"std       {stats['std']:.9e} {unit}   (descriptive only, see docs gaps)")
    return 0


def _cmd_validate(args) -> int:
    """Run the validation hierarchy and emit the evidence report.

    Exit status is the verdict: 0 if every check passed, 1 otherwise. That
    makes the report usable as a gate in a lab's own qualification pipeline
    rather than only as something to read.
    """
    from .validate import run, to_html, to_json

    report = run(trials=args.trials, self_test=args.self_test)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(to_html(report))
    if args.json:
        print(to_json(report))
    else:
        for check in report.checks:
            mark = "pass" if check.passed else "FAIL"
            print(f"[{mark}] {check.tier:<10} {check.name}")
            print(f"         {check.measured}")
        passed = sum(c.passed for c in report.checks)
        print(
            f"\n{'PASS' if report.passed else 'FAIL'}: {passed}/"
            f"{len(report.checks)} checks   digest {report.digest[:16]}"
        )
        if args.out:
            print(f"report written to {args.out}")
    return 0 if report.passed else 1


def _step_params(result, action: str) -> dict:
    """Parameters of the last step with this action. The audit trail, reused."""
    return [s for s in result.provenance.steps if s.action == action][-1].params


def _confidence_for(args, result, series):
    """Bounds for one result, wiring up the noise-type arguments.

    `series` is handed through only for `greenhall`, which needs a power
    law and identifies one per tau when `--alpha` was not given.
    """
    return confidence(
        result,
        level=args.confidence,
        method=args.edof_method,
        alpha=getattr(args, "alpha", None),
        series=series,
        run_policy=getattr(args, "run_policy", "pool"),
    )


def _cmd_estimate(args) -> int:
    """Run one estimator and print the tau table.

    Column layout deliberately mirrors Stable32's -- AF, Tau, #, Min Sigma,
    Sigma, Max Sigma -- so a user migrating can diff the two outputs
    directly. The extra EDOF column is not in Stable32's output and is the
    point of difference: it says how much data each bound rests on, and
    which estimator produced it.
    """
    series = _load(args)
    want_ci = not args.no_confidence
    extra = {} if args.estimator_name in ("mtie", "tierms") else {
        "keep_coeffs": want_ci}
    result = args.estimator(
        series,
        taus=args.taus,
        max_af=args.max_af,
        gap_policy=args.gap_policy,
        **extra,
    )
    want_ci = want_ci and result.is_variance
    policy = _step_params(result, "evaluate")
    ci = _confidence_for(args, result, series) if want_ci else None

    if args.json:
        payload = {
            "kind": result.kind,
            "af": result.af.tolist(),
            "tau_s": result.tau_s.tolist(),
            "var": result.var.tolist(),
            "dev": result.dev.tolist(),
            "n_terms": result.n_terms.tolist(),
            "provenance": to_dict((ci or result).provenance),
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
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(f"# {result.kind.upper()} for {args.path}")
    print(f"# tau grid: {args.taus}   n = {series.n}   rate = {series.rate_hz:g} Hz")
    if series.n_gaps:
        print(
            f"# {series.n_gaps} gaps, {policy['runs']} gap-free runs, "
            f"policy '{policy['gap_policy']}', longest run {policy['longest_run']}"
        )
    name = result.kind.upper()
    if ci is None:
        print(f"{'AF':>7}  {'Tau (s)':>13}  {'#':>8}  {name:>14}")
        for af, tau, n, dev in zip(result.af, result.tau_s, result.n_terms, result.dev):
            print(f"{af:>7}  {tau:>13.4e}  {n:>8}  {dev:>14.6e}")
        return 0

    basis = "an assumed power law" if ci.alpha is not None else "data-driven EDOF"
    print(f"# bounds at {ci.level:.4%} confidence, chi-square on {basis}")
    alpha_col = f"  {'Alpha':>5}" if ci.alpha is not None else ""
    print(
        f"{'AF':>7}  {'Tau (s)':>13}  {'#':>8}{alpha_col}  {'Min ' + name:>14}  "
        f"{name:>14}  {'Max ' + name:>14}  {'EDOF':>9}  {'via':>9}"
    )
    alphas = ci.alpha if ci.alpha is not None else [None] * result.af.size
    rows = zip(
        result.af, result.tau_s, result.n_terms, alphas, ci.dev_lo, result.dev,
        ci.dev_hi, ci.edof, ci.method,
    )
    for af, tau, n, a, lo, dev, hi, eta, how in rows:
        # A declined point has no assumed power law, so the column is blank
        # rather than carrying the sentinel out into the user's table.
        shown = "" if a is None else ("" if how != "greenhall" else f"{a:d}")
        cell = f"  {shown:>5}" if ci.alpha is not None else ""
        print(
            f"{af:>7}  {tau:>13.4e}  {n:>8}{cell}  {lo:>14.6e}  {dev:>14.6e}  "
            f"{hi:>14.6e}  {eta:>9.1f}  {how:>9}"
        )
    reasons = sorted({d for d in ci.declined if d})
    for why in reasons:
        print(f"# declined greenhall where: {why}")
    return 0


def _cmd_compare_edof(args) -> int:
    """Every EDOF method on one dataset, and what each costs in bounds.

    The diagnostic that makes the design argument concrete. `eta1` reads
    the degrees of freedom off the data; `eta3` counts non-overlapping
    windows and is a deliberate lower bound; `greenhall` assumes a power
    law and is what Stable32 prints. Where they disagree is where the
    assumption is doing the work, and a metrologist deciding which number
    to put on a certificate should be able to see that in one table.
    """
    from .registry import get

    series = _load(args)
    result = get("estimator", args.estimator)(
        series, taus=args.taus, max_af=args.max_af
    )
    cis = {}
    for method in ("eta1", "eta3", "greenhall"):
        cis[method] = confidence(
            result, level=args.confidence, method=method,
            alpha=args.alpha, series=series,
        )
    greenhall = cis["greenhall"]

    if args.json:
        print(json.dumps({
            "kind": result.kind,
            "af": result.af.tolist(),
            "tau_s": result.tau_s.tolist(),
            "dev": result.dev.tolist(),
            "alpha": greenhall.alpha.tolist(),
            "edof": {k: v.edof.tolist() for k, v in cis.items()},
            "dev_lo": {k: v.dev_lo.tolist() for k, v in cis.items()},
            "dev_hi": {k: v.dev_hi.tolist() for k, v in cis.items()},
            "method_used": {k: list(v.method) for k, v in cis.items()},
            "declined": list(greenhall.declined),
            "provenance": to_dict(greenhall.provenance),
        }, indent=2, default=str))
        return 0

    print(f"# EDOF comparison for {args.path}, {result.kind}")
    print(f"# bounds at {args.confidence:.4%}; ratio is the width of the "
          "greenhall interval over eta1's")
    print(
        f"{'AF':>7}  {'Tau (s)':>13}  {'Alpha':>5}  {'eta1':>9}  {'eta3':>9}  "
        f"{'greenhall':>9}  {'via':>9}  {'width g/1':>9}"
    )
    for i, af in enumerate(result.af.tolist()):
        used = greenhall.method[i]
        a = "" if used != "greenhall" else f"{greenhall.alpha[i]:d}"
        width_1 = cis["eta1"].dev_hi[i] - cis["eta1"].dev_lo[i]
        width_g = greenhall.dev_hi[i] - greenhall.dev_lo[i]
        print(
            f"{af:>7}  {result.tau_s[i]:>13.4e}  {a:>5}  "
            f"{cis['eta1'].edof[i]:>9.1f}  {cis['eta3'].edof[i]:>9.1f}  "
            f"{greenhall.edof[i]:>9.1f}  {used:>9}  {width_g / width_1:>9.2f}"
        )
    for why in sorted({d for d in greenhall.declined if d}):
        print(f"# declined greenhall where: {why}")
    return 0


def _cmd_powerlaw(args) -> int:
    """Decompose a measured curve into IEEE 1139 noise levels and drift.

    Stacks the overlapping Allan and modified Allan curves of the same
    record, because the two PM noises do not separate from AVAR alone. The
    fit is a non-negative least squares, so no level ever comes back
    negative and the solution carries an optimality certificate.
    """
    from .powerlaw_coeffs import ALPHAS
    from .stats import mdev, oadev
    from .stats.powerlaw import decompose
    from .synth import ALPHA_NAMES

    series = _load(args)
    curves = [oadev(series, taus=args.taus, max_af=args.max_af),
              mdev(series, taus=args.taus, max_af=args.max_af)]
    fit = decompose(curves, f_h=args.fh, weights=args.weights)

    if args.json:
        print(json.dumps({
            "h": {str(a): fit.h[a] for a in ALPHAS},
            "drift_per_s": fit.drift,
            "dominant": fit.dominant,
            "f_h": fit.f_h,
            "assumed_fh": fit.assumed_fh,
            "weights": fit.weights,
            "n_rows": fit.n_rows,
            "dropped_rows": fit.dropped_rows,
            "residual": fit.residual,
            "kkt": fit.kkt,
            "provenance": to_dict(fit.provenance),
        }, indent=2, default=str))
        return 0

    print(f"# power-law decomposition of {args.path}")
    print(f"# {fit.n_rows} rows from oadev + mdev, {fit.dropped_rows} dropped "
          f"outside 2*pi*f_h*tau >= 10; weights={fit.weights}")
    if fit.assumed_fh:
        print(f"# WARNING: f_h assumed = {fit.f_h:.6g} Hz (Nyquist). It is not "
              "identifiable from the data. The white-PM and flicker-PM levels")
        print("#          below scale with it and must not be quoted without it.")
    else:
        print(f"# f_h = {fit.f_h:.6g} Hz, as given")
    print(f"\n{'alpha':>6}  {'noise type':<16}  {'h_alpha':>14}")
    for a in ALPHAS:
        print(f"{a:>+6}  {ALPHA_NAMES[a]:<16}  {fit.h[a]:>14.6e}")
    print(f"{'':>6}  {'linear drift':<16}  {fit.drift:>14.6e}  per second")
    dominant = ("none resolved" if fit.dominant is None
                else f"{fit.dominant:+d} ({ALPHA_NAMES[fit.dominant]})")
    print(f"\n# dominant at the shortest fitted tau: {dominant}")
    print(f"# residual {fit.residual:.4e}, KKT optimality {fit.kkt:.2e}")
    return 0


def _cmd_plot(args) -> int:
    """Draw one file. The only subcommand that needs matplotlib."""
    from . import plot as plotting
    from .registry import get

    series = _load(args)
    if args.draw == "phase":
        from .stats.core import as_phase
        ax = plotting.phase(as_phase(series), label=Path(args.path).name)
    elif args.draw == "frequency":
        from .types import to_frequency
        fs = series if hasattr(series, "y") else to_frequency(series)
        ax = plotting.frequency(fs, label=Path(args.path).name)
    else:
        result = get("estimator", args.estimator)(
            series, taus=args.taus, max_af=args.max_af)
        # MTIE and TIE rms have no chi-square interval, and `confidence`
        # says so rather than inventing one -- so do not ask.
        want_ci = not args.no_confidence and result.is_variance
        ci = _confidence_for(
            argparse.Namespace(confidence=args.confidence,
                               edof_method="auto"), result, series
        ) if want_ci else None
        ax = plotting.sigma_tau(result, confidence=ci,
                                label=Path(args.path).name)
        if args.slopes:
            plotting.slope_guides(ax)
    if args.title:
        ax.set_title(args.title)

    if args.out:
        plotting.save(ax, args.out, dpi=args.dpi)
        print(f"wrote {args.out}")
    else:
        import matplotlib.pyplot as plt
        plt.show()
    return 0


def _cmd_gui(args) -> int:
    """Serve the local page. Blocks until interrupted."""
    from .gui import serve

    serve(port=args.port, root=args.root, open_browser=not args.no_browser)
    return 0


def _cmd_bench_check(args) -> int:
    """Can this setup determine its unknowns? Reads no data files.

    Worth running while the experiment is still being planned, which is the
    only time the answer -- "go and measure B against D" -- is cheap.
    """
    from .bench import check
    from .bench.store import load

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
            "notes": list(ident.notes),
        }, indent=2))
    else:
        print(ident.report())
    return 0 if ident.identifiable else 2


def _cmd_bench_solve(args) -> int:
    """The N-cornered hat, channel floors and the common-mode warnings."""
    from .bench.solve import solve
    from .bench.store import load
    from .io import load_frequency, load_phase
    from .registry import get

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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure(args.log_level)
    try:
        return args.func(args)
    except (ValueError, KeyError, OSError) as exc:
        print(f"stabilis: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
