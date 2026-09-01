"""`stabilis plot` -- the only subcommand that needs matplotlib.

`plotting` is imported inside the handler, not at module scope, so
`stabilis --help` still works in an environment that has only numpy.
`tests/test_core_deps.py` is the test that keeps that true for the library;
nothing yet asserts it for the command line.

Note the two dots in `from .. import plot as plotting`. One dot would be a
legal `from package import submodule` naming *this module*, which imports
cleanly, binds a module object, and then fails at the first attribute
access -- and `tests/test_plot.py` opens with `importorskip`, so on a
numpy-only machine the whole file skips and that ships.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..stats import ONE_SIGMA
from ..stats.edof import EDOF_METHODS
from ..tau import MODES
from .common import _add_input_args, _confidence_for, _load

__all__ = ["add_plot"]


def add_plot(sub) -> None:
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


def _cmd_plot(args) -> int:
    """Draw one file. The only subcommand that needs matplotlib."""
    from .. import plot as plotting
    from ..registry import get

    series = _load(args)
    if args.draw == "phase":
        from ..stats.core import as_phase
        ax = plotting.phase(as_phase(series), label=Path(args.path).name)
    elif args.draw == "frequency":
        from ..types import to_frequency
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
