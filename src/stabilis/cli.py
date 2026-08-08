"""Command-line interface.

Subcommands are thin: each parses arguments, calls one library function and
formats the result. Nothing here computes anything, so the CLI can never
disagree with the library.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .docs import list_topics, read_topic, topic_summary
from .log import configure
from .preprocess import basic_stats
from .provenance import to_dict
from .stats import ONE_SIGMA, confidence
from .stats.core import GAP_POLICIES
from .stats.edof import EDOF_METHODS
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

    from .stats import adev, mdev, oadev, ohdev, tdev

    for name, fn in (
        ("adev", adev),
        ("oadev", oadev),
        ("mdev", mdev),
        ("tdev", tdev),
        ("ohdev", ohdev),
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
            "--no-confidence",
            action="store_true",
            help="skip the bounds; also frees the coefficient arrays, which "
            "matter on a record of millions of points",
        )
        p.add_argument("--json", action="store_true", help="emit JSON")
        p.set_defaults(func=_cmd_estimate, estimator=fn, estimator_name=name)

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
    result = args.estimator(
        series,
        taus=args.taus,
        max_af=args.max_af,
        keep_coeffs=want_ci,
        gap_policy=args.gap_policy,
    )
    policy = _step_params(result, "evaluate")
    ci = (
        confidence(result, level=args.confidence, method=args.edof_method)
        if want_ci
        else None
    )

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

    print(f"# bounds at {ci.level:.4%} confidence, chi-square on data-driven EDOF")
    print(
        f"{'AF':>7}  {'Tau (s)':>13}  {'#':>8}  {'Min ' + name:>14}  "
        f"{name:>14}  {'Max ' + name:>14}  {'EDOF':>9}  {'via':>5}"
    )
    rows = zip(
        result.af, result.tau_s, result.n_terms, ci.dev_lo, result.dev,
        ci.dev_hi, ci.edof, ci.method,
    )
    for af, tau, n, lo, dev, hi, eta, how in rows:
        print(
            f"{af:>7}  {tau:>13.4e}  {n:>8}  {lo:>14.6e}  {dev:>14.6e}  "
            f"{hi:>14.6e}  {eta:>9.1f}  {how:>5}"
        )
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
