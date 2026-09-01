"""Command-line interface.

Subcommands are thin: each parses arguments, calls one library function and
formats the result. Nothing here computes anything, so the CLI can never
disagree with the library.
"""

from __future__ import annotations

import argparse
import sys

from .. import __version__
from ..log import configure
from .bench import add_bench
from .docs import add_docs
from .edof import add_compare_edof
from .estimators import add_estimators, add_wdev
from .export import add_export
from .fitting import add_fit
from .gui import add_gui
from .mask import add_mask
from .multi import add_cross, add_reject
from .plot import add_plot
from .powerlaw import add_powerlaw
from .preprocess import add_stats
from .psd import add_psd
from .report import add_report
from .spur import add_spur
from .stream import add_davar, add_stream
from .validate import add_validate

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
    add_docs(sub)
    add_validate(sub)
    add_stats(sub)
    add_estimators(sub)
    add_wdev(sub)
    add_compare_edof(sub)
    add_powerlaw(sub)
    add_fit(sub)
    add_cross(sub)
    add_reject(sub)
    add_plot(sub)
    add_report(sub)
    add_gui(sub)
    add_spur(sub)
    add_mask(sub)
    add_psd(sub)
    add_davar(sub)
    add_stream(sub)
    add_export(sub)
    add_bench(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure(args.log_level)
    try:
        return args.func(args)
    except (ValueError, KeyError, OSError) as exc:
        print(f"stabilis: {exc}", file=sys.stderr)
        return 1

