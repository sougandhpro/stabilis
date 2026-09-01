"""Argument groups, loading and formatting shared across subcommands.

Only what more than one family needs lives here. A helper with a single
caller stays in the module that calls it -- otherwise this file becomes the
place things are put rather than the place they belong.

`_summary` is the one exception to that rule, and deliberately so: it
encodes a shipped bug rather than a convenience, so a future family taking
help text from a docstring has to be able to find it.
"""

from __future__ import annotations

import argparse

import numpy as np

from ..stats import confidence

__all__ = [
    "NO_INTERVAL", "_summary", "_add_input_args", "_load",
    "_confidence_for", "_nan_to_none",
]

# Estimators with no confidence interval today, for two different reasons,
# and offering the flags anyway would be worse than not having them. MTIE
# and TIE rms are not variances at all, so the chi-square interval does not
# apply and `confidence` refuses them. The Thêo family *is* a variance
# family, but it reports no coefficient arrays for the data-driven path and
# its own EDF (SP 1065 Table 9) is not implemented -- see theory/theoh.md.
# None of them takes `keep_coeffs` either, which is why this one list drives
# both decisions.
NO_INTERVAL = ("mtie", "tierms", "theo1", "theobr", "theoh")


def _summary(fn) -> str:
    """First docstring line, safe to hand to argparse as help text.

    argparse runs every help string through `%`-formatting, so a bare
    percent sign in a docstring -- "75 % of the record length" -- is read
    as a conversion spec and `--help` dies with a TypeError instead of
    printing. Doubling it is the documented escape.
    """
    return fn.__doc__.splitlines()[0].replace("%", "%%")


def _add_input_args(p: argparse.ArgumentParser,
                    required_path: bool = True) -> None:
    """Input arguments shared by every data-consuming subcommand.

    `required_path=False` for a subcommand that also has a listing mode and
    so can be run with no file at all.
    """
    if required_path:
        p.add_argument("path", help="input data file")
    else:
        p.add_argument("path", nargs="?", help="input data file")
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
    from ..io import load_frequency, load_phase

    loader = load_phase if args.kind == "phase" else load_frequency
    return loader(
        args.path,
        rate_hz=args.rate,
        column=args.column,
        time_column=args.time_column,
        delimiter=args.delimiter,
        gap_value=args.gap_value,
    )


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


def _nan_to_none(a):
    """NaN is not JSON. A withheld value is `null`, which parsers understand."""
    return [None if not np.isfinite(v) else float(v) for v in a]
