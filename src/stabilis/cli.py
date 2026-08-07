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


def _cmd_estimate(args) -> int:
    """Run one estimator and print the tau table.

    Column layout deliberately mirrors Stable32's, so a user migrating can
    diff the two outputs directly. Confidence bounds arrive in M2; until
    then the header says so rather than leaving a suggestive blank.
    """
    series = _load(args)
    result = args.estimator(
        series, taus=args.taus, max_af=args.max_af, keep_coeffs=False
    )

    if args.json:
        print(
            json.dumps(
                {
                    "kind": result.kind,
                    "af": result.af.tolist(),
                    "tau_s": result.tau_s.tolist(),
                    "var": result.var.tolist(),
                    "dev": result.dev.tolist(),
                    "n_terms": result.n_terms.tolist(),
                    "provenance": to_dict(result.provenance),
                },
                indent=2,
                default=str,
            )
        )
        return 0

    print(f"# {result.kind.upper()} for {args.path}")
    print(f"# tau grid: {args.taus}   n = {series.n}   rate = {series.rate_hz:g} Hz")
    print("# confidence bounds not yet implemented (M2)")
    print(f"{'AF':>7}  {'Tau (s)':>13}  {'#':>8}  {result.kind.upper():>14}")
    for af, tau, n, dev in zip(result.af, result.tau_s, result.n_terms, result.dev):
        print(f"{af:>7}  {tau:>13.4e}  {n:>8}  {dev:>14.6e}")
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
