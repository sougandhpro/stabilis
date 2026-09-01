"""Command-line interface.

Subcommands are thin: each parses arguments, calls one library function and
formats the result. Nothing here computes anything, so the CLI can never
disagree with the library.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import chain
from pathlib import Path

import numpy as np

from .. import __version__
from ..log import configure
from ..provenance import to_dict
from ..stats import ONE_SIGMA, confidence
from ..stats.core import GAP_POLICIES
from ..stats.edof import EDOF_METHODS
from ..export import TABLE_FORMATS
from ..io.write import FORMATS, TIME_COLUMNS
from ..stream.davar import STEP_DIVISOR
from ..tau import MODES
from .common import _add_input_args, _confidence_for, _load, _nan_to_none
from .docs import add_docs
from .edof import add_compare_edof
from .estimators import add_estimators, add_wdev
from .fitting import add_fit
from .gui import add_gui
from .mask import add_mask
from .plot import add_plot
from .preprocess import add_stats
from .powerlaw import add_powerlaw
from .psd import add_psd
from .spur import add_spur
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

    p_x = sub.add_parser(
        "cross",
        help="cross-variance of two channels watching one clock, which "
             "rejects the instruments' own noise",
    )
    _add_input_args(p_x)
    p_x.add_argument("path_b", help="the second channel's data file")
    p_x.add_argument(
        "--column-b", type=int,
        help="data column in the second file, if it differs from --column",
    )
    p_x.add_argument(
        "--estimator", default="oadev",
        choices=("adev", "oadev", "mdev", "tdev", "ohdev"),
        help="any variance estimator; MTIE and TIE rms have no cross form",
    )
    p_x.add_argument("--taus", default="octave", choices=MODES, help="tau grid")
    p_x.add_argument("--max-af", type=int, help="largest averaging factor")
    p_x.add_argument("--json", action="store_true", help="emit JSON")
    p_x.set_defaults(func=_cmd_cross)

    p_rej = sub.add_parser(
        "reject",
        help="average every pair of M channels watching one clock, which "
             "rejects instrument noise deeper than any single pair",
    )
    _add_input_args(p_rej)
    p_rej.add_argument(
        "extra", nargs="+",
        help="the remaining channels' data files, at least one",
    )
    p_rej.add_argument(
        "--estimator", default="oadev",
        choices=("adev", "oadev", "mdev", "tdev", "ohdev"),
    )
    p_rej.add_argument("--taus", default="octave", choices=MODES, help="tau grid")
    p_rej.add_argument("--max-af", type=int, help="largest averaging factor")
    p_rej.add_argument(
        "--matrix", action="store_true",
        help="also print the full pairwise correlation matrix at each tau",
    )
    p_rej.add_argument("--json", action="store_true", help="emit JSON")
    p_rej.set_defaults(func=_cmd_reject)

    add_plot(sub)

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

    add_gui(sub)

    add_spur(sub)

    add_mask(sub)

    add_psd(sub)

    p_dav = sub.add_parser(
        "davar", help="dynamic stability: sigma(tau) as a function of when you looked"
    )
    _add_input_args(p_dav)
    p_dav.add_argument("--window", type=int, required=True,
                       help="window length in SAMPLES; rounding a duration "
                            "silently is how two runs land on different grids")
    p_dav.add_argument("--step", type=int,
                       help=f"window stride in samples (default: window/"
                            f"{STEP_DIVISOR})")
    p_dav.add_argument("--estimator", default="oadev",
                       help="any registered estimator that honours an "
                            "explicit tau list (default: oadev)")
    p_dav.add_argument("--taus", default="octave", choices=MODES)
    p_dav.add_argument("--max-af", type=int, dest="max_af")
    p_dav.add_argument("--level", type=float,
                       help="confidence level per window, e.g. 0.6827; "
                            "must be asked for now, as the intervals cannot "
                            "be added after the coefficients are released")
    p_dav.add_argument("--at-af", type=int, dest="at_af",
                       help="print the time series at this averaging factor "
                            "(default: the smallest)")
    p_dav.add_argument("--json", action="store_true",
                       help="the whole surface, not one cut through it")
    p_dav.set_defaults(func=_cmd_davar)

    p_stream = sub.add_parser(
        "stream", help="run the Allan deviation over a file as it is written"
    )
    p_stream.add_argument("path", help="file to read, or to follow with --follow")
    p_stream.add_argument("--rate", type=float, required=True,
                          help="sample rate in Hz; a stream has no header to "
                               "read it from")
    p_stream.add_argument("--af", help="comma-separated averaging factors")
    p_stream.add_argument("--taus", default="octave", choices=MODES,
                          help="with --expect, the grid to build (default: octave)")
    p_stream.add_argument("--expect", type=int,
                          help="expected record length, to size the tau grid; "
                               "a stream has no length, so one of --af or "
                               "--expect is required")
    p_stream.add_argument("--column", type=int, default=0)
    p_stream.add_argument("--time-column", type=int, dest="time_column",
                          help="timetag column; without one a dropped sample "
                               "leaves no trace and gaps are undetectable")
    p_stream.add_argument("--delimiter")
    p_stream.add_argument("--follow", action="store_true",
                          help="keep reading as the file grows")
    p_stream.add_argument("--timeout", type=float, default=5.0,
                          help="give up after this many seconds with no new "
                               "data (default: 5)")
    p_stream.add_argument("--every", type=int,
                          help="print the running result every N samples")
    p_stream.add_argument("--json", action="store_true")
    p_stream.set_defaults(func=_cmd_stream)

    p_export = sub.add_parser(
        "export", help="write data, curves and report bundles back out"
    )
    export_sub = p_export.add_subparsers(dest="export_command", required=True)

    p_xdata = export_sub.add_parser(
        "data",
        help="the record itself, in a format another tool can read",
    )
    _add_input_args(p_xdata)
    p_xdata.add_argument(
        "--format", default="stabilis", choices=FORMATS,
        help="stabilis: a comment header carrying the rate and epoch, so "
             "this library reads the series back whole. csv: the same, "
             "comma-separated. stable32: bare numbers, no header, for "
             "Stable32 and TimeLab",
    )
    p_xdata.add_argument(
        "--time-column-out", default="", choices=TIME_COLUMNS,
        metavar="{elapsed,unix,mjd}",
        help="prepend a timetag column (default: none; the rate goes in "
             "the header instead, which is exact)",
    )
    p_xdata.add_argument(
        "--gap-out", type=float, metavar="VALUE",
        help="write gaps as this value, for a format with no NaN",
    )
    p_xdata.add_argument(
        "--digits", type=int, default=17,
        help="significant digits (default 17, the width at which float64 "
             "reads back exactly)",
    )
    p_xdata.add_argument("--out", required=True, help="write here")
    p_xdata.set_defaults(func=_cmd_export_data)

    p_xtab = export_sub.add_parser(
        "table",
        help="the sigma-tau curve, for a spreadsheet, a program, or Stable32",
    )
    _add_input_args(p_xtab)
    p_xtab.add_argument(
        "--estimator", default="oadev",
        choices=("adev", "oadev", "mdev", "tdev", "ohdev", "totdev"),
    )
    p_xtab.add_argument("--taus", default="octave", choices=MODES)
    p_xtab.add_argument("--max-af", type=int, help="largest averaging factor")
    p_xtab.add_argument("--gap-policy", default="segment", choices=GAP_POLICIES)
    p_xtab.add_argument("--confidence", type=float, default=ONE_SIGMA)
    p_xtab.add_argument("--edof-method", default="auto", choices=EDOF_METHODS)
    p_xtab.add_argument(
        "--alpha", type=int,
        help="assumed power law; required by --format stable32 unless the "
             "interval already carries one",
    )
    p_xtab.add_argument("--no-confidence", action="store_true",
                        help="curve only, no interval")
    p_xtab.add_argument(
        "--format", default="csv", choices=TABLE_FORMATS,
        help="csv: eight columns for a spreadsheet. json: every field, "
             "including what the csv cannot carry. stable32: the block "
             "Stable32 prints, to its own bytes -- and to its four decimals",
    )
    p_xtab.add_argument("--out", help="write here (default: stdout)")
    p_xtab.set_defaults(func=_cmd_export_table)

    p_xbun = export_sub.add_parser(
        "bundle",
        help="a directory carrying the data, the curve, the report and a "
             "manifest of what everything hashes to",
    )
    _add_input_args(p_xbun)
    p_xbun.add_argument(
        "--estimator", default="oadev",
        choices=("adev", "oadev", "mdev", "tdev", "ohdev", "totdev"),
    )
    p_xbun.add_argument("--taus", default="octave", choices=MODES)
    p_xbun.add_argument("--max-af", type=int, help="largest averaging factor")
    p_xbun.add_argument("--gap-policy", default="segment", choices=GAP_POLICIES)
    p_xbun.add_argument("--confidence", type=float, default=ONE_SIGMA)
    p_xbun.add_argument("--edof-method", default="auto", choices=EDOF_METHODS)
    p_xbun.add_argument("--alpha", type=int, help="assumed power law")
    p_xbun.add_argument("--spec", action="append", default=[],
                        metavar="TAU:SIGMA",
                        help="a datasheet limit, repeatable")
    p_xbun.add_argument("--bench", help="bench.json, to take the limits from")
    p_xbun.add_argument("--source", help="which oscillator in --bench")
    p_xbun.add_argument("--title", default="")
    p_xbun.add_argument("--subject", default="")
    p_xbun.add_argument("--operator", default="")
    p_xbun.add_argument("--notes", default="")
    p_xbun.add_argument(
        "--no-data", action="store_true",
        help="reference the source record by hash instead of copying it in")
    p_xbun.add_argument("--no-figure", action="store_true")
    p_xbun.add_argument(
        "--no-stamp", action="store_true",
        help="omit the creation time, so two bundles of the same analysis "
             "are byte-identical")
    p_xbun.add_argument("--overwrite", action="store_true",
                        help="write over a directory that already holds one")
    p_xbun.add_argument("--out", required=True, metavar="DIR")
    p_xbun.set_defaults(func=_cmd_export_bundle)

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

    return parser








def _cmd_davar(args) -> int:
    """Was it doing the same thing all along? The question one curve cannot ask.

    Prints one cut through the surface rather than the whole grid, because
    a surface is `windows x taus` numbers and a terminal is not where you
    read one. The cut is a time series at a single averaging factor, which
    is the form an event shows up in; `--json` gives the rest.
    """
    from ..stream import davar

    series = _load(args)
    surface = davar(series, window=args.window, step=args.step,
                    taus=args.taus, kind=args.estimator, max_af=args.max_af,
                    level=args.level)

    if args.json:
        print(json.dumps(
            {"kind": surface.kind, "af": surface.af.tolist(),
             "tau_s": surface.tau_s.tolist(), "t_s": surface.t_s.tolist(),
             "start_index": surface.start_index.tolist(),
             "var": [_nan_to_none(row) for row in surface.var],
             "n_terms": surface.n_terms.tolist(),
             "var_lo": None if surface.var_lo is None
                       else [_nan_to_none(r) for r in surface.var_lo],
             "var_hi": None if surface.var_hi is None
                       else [_nan_to_none(r) for r in surface.var_hi],
             "level": surface.level, "window": surface.window,
             "step": surface.step, "rate_hz": surface.rate_hz,
             "provenance": to_dict(surface.provenance)},
            indent=2, default=str))
        return 0

    af = int(surface.af[0]) if args.at_af is None else args.at_af
    col = np.flatnonzero(surface.af == af)
    if col.size == 0:
        raise KeyError(
            f"averaging factor {af} is not in this surface: "
            f"{surface.af.tolist()}"
        )
    dev = np.sqrt(surface.at_af(af))
    tau = float(surface.tau_s[col[0]])
    print(f"# {surface.kind} at tau = {tau:g} s over {surface.n_windows} "
          f"windows of {surface.window} samples, step {surface.step}")
    live = dev[np.isfinite(dev)]
    if live.size:
        worst = int(np.nanargmax(dev))
        print(f"# quiet {np.nanmin(dev):.4e}, worst {np.nanmax(dev):.4e} at "
              f"t = {surface.t_s[worst]:g} s -- a factor "
              f"{np.nanmax(dev) / np.nanmin(dev):.2f}")
    print(f"{'t (s)':>14}  {surface.kind:>14}  {'terms':>7}")
    for t, d, n in zip(surface.t_s, dev, surface.n_terms[:, col[0]]):
        cell = f"{d:>14.6e}" if np.isfinite(d) else f"{'-':>14}"
        print(f"{t:>14.6g}  {cell}  {n:>7d}")
    return 0


def _cmd_stream(args) -> int:
    """The deviation of a measurement that has not finished yet.

    Memory stays at the accumulator's `2*max(m)+1` samples however long
    the file grows, which is the point -- and the answer is the batch
    answer to a couple of ulp, which is the other point. See
    theory/stream.md.
    """
    from ..stream import (feed, online_avar, online_push, online_result,
                          parse_lines, tail_file)
    from ..stream.collect import on_grid
    from ..tau import averaging_factors

    if args.af:
        afs = [int(v) for v in args.af.split(",") if v.strip()]
    elif args.expect:
        afs = averaging_factors(args.expect, args.taus).tolist()
    else:
        raise ValueError(
            "a stream has no length, so the tau grid cannot be derived from "
            "it: give --af 1,2,4,... or --expect N to size a grid from an "
            "expected record length"
        )

    lines = (tail_file(args.path, timeout_s=args.timeout) if args.follow
             else open(args.path, encoding="utf-8"))
    pairs = parse_lines(lines, column=args.column,
                        time_column=args.time_column, rate_hz=args.rate,
                        delimiter=args.delimiter)
    try:
        try:
            first = next(pairs)
        except StopIteration:
            raise ValueError(f"{args.path}: no samples to read") from None
        # The epoch is the first timestamp, as it is for `collect`: a file
        # whose timetags are MJD seconds would otherwise be placed on a
        # grid starting at zero and refused as one enormous outage.
        acc = online_avar(args.rate, afs, epoch=first[0], source=args.path)
        pairs = chain([first], pairs)
        if args.every:
            for _, value, _ in on_grid(pairs, acc.rate_hz, acc.epoch):
                online_push(acc, value)
                if acc.n_seen % args.every == 0:
                    _print_stream(acc, online_result(acc), args, final=False)
        else:
            feed(acc, pairs)
    finally:
        if not args.follow:
            lines.close()

    _print_stream(acc, online_result(acc), args, final=True)
    return 0


def _print_stream(acc, result, args, final: bool) -> None:
    """One report from a running accumulator, JSON or table."""
    if args.json:
        print(json.dumps(
            {"n_seen": acc.n_seen, "gaps": acc.n_gaps, "final": final,
             "af": result.af.tolist(), "tau_s": result.tau_s.tolist(),
             "dev": result.dev.tolist(), "n_terms": result.n_terms.tolist(),
             "provenance": to_dict(result.provenance)},
            default=str))
        return
    if args.time_column is None:
        note = " (no timetag column: a dropped sample leaves no trace here)"
    else:
        note = ""
    print(f"# {acc.n_seen} samples, {acc.n_gaps} gaps{note}")
    print(f"{'tau (s)':>14}  {'oadev':>14}  {'terms':>9}")
    for tau, dev, n in zip(result.tau_s, result.dev, result.n_terms):
        print(f"{tau:>14.6g}  {dev:>14.6e}  {n:>9d}")


def _cmd_export_data(args) -> int:
    """Write the record back out, in a format something else can read.

    The point of the default `stabilis` format is that the file says what
    it is: the rate and the epoch ride in a comment block, so reading it
    back needs no arguments and cannot be given the wrong ones. The
    `stable32` format drops all of that, because the tools it is aimed at
    were not asked to tolerate our comments.
    """
    from ..io import save_frequency, save_phase

    series = _load(args)
    save = save_phase if args.kind == "phase" else save_frequency
    written = save(
        series,
        args.out,
        format=args.format,
        time_column=args.time_column_out,
        gap_value=args.gap_out,
        digits=args.digits,
    )
    print(f"wrote {written.path} -- {series.n} samples, {series.n_gaps} gaps, "
          f"{written.n_bytes} bytes, sha256 {written.sha256[:16]}")
    return 0


def _cmd_export_table(args) -> int:
    """The curve itself, in whichever of the three shapes is wanted.

    Same library calls the estimator subcommands make, so an exported table
    and a printed one can never disagree.
    """
    from ..export import to_csv, to_json, to_stable32
    from ..registry import get

    series = _load(args)
    estimator = get("estimator", args.estimator)
    want_ci = not args.no_confidence
    result = estimator(series, taus=args.taus, max_af=args.max_af,
                       gap_policy=args.gap_policy, keep_coeffs=want_ci)
    ci = _confidence_for(args, result, series) if want_ci else None

    if args.format == "csv":
        text, newline = to_csv(result, ci), "\n"
    elif args.format == "json":
        text, newline = to_json(result, ci) + "\n", "\n"
    else:
        text, newline = to_stable32(series, result, ci, alpha=args.alpha,
                                    source=Path(args.path).name), ""

    if not args.out:
        print(text, end="")
        return 0
    # newline="" leaves the CRLF the Stable32 writer put in; the others
    # asked for LF and must not be translated to the platform's ending.
    with open(args.out, "w", encoding="utf-8", newline=newline) as fh:
        fh.write(text)
    print(f"wrote {args.out} -- {result.af.size} taus, {args.format} format")
    return 0


def _cmd_export_bundle(args) -> int:
    """The whole measurement as a directory, with a manifest of its hashes.

    Everything a recipient needs to check the number: the record itself,
    the curve in two forms, the report, the provenance, and a SHA-256 for
    each. The analysis is done here and handed to `export.bundle.build`,
    which computes nothing itself -- so a bundle can never carry options
    the caller did not choose.
    """
    from ..export.bundle import build as build_bundle
    from ..registry import get
    from ..report import build as build_report, figure_uri, to_html

    series = _load(args)
    spec, spec_label = _spec_points(args)
    grid = _report_grid(args, series, spec)
    result = get("estimator", args.estimator)(series, taus=grid,
                                              gap_policy=args.gap_policy,
                                              keep_coeffs=True)
    ci = _confidence_for(args, result, series)
    report = build_report(
        series, result, ci,
        title=args.title,
        subject=args.subject or f"{Path(args.path).name} ({args.estimator})",
        operator=args.operator, notes=args.notes,
        spec=spec, spec_label=spec_label,
    )
    html = to_html(report, "" if args.no_figure
                   else figure_uri(result, ci, spec, spec_label))

    bundle = build_bundle(
        args.out, series, result, ci,
        report=report, report_html=html, source_path=args.path,
        include_data=not args.no_data, include_figure=not args.no_figure,
        stamp=not args.no_stamp, overwrite=args.overwrite,
    )
    print(f"wrote {bundle.path} -- {len(bundle.files) + 2} files, "
          f"verdict {report.verdict.upper()}, digest {bundle.digest[:16]}")
    return 0
























def _cmd_cross(args) -> int:
    """Cross-variance of two records of one clock, seen through two channels.

    Prints the two auto-deviations beside the cross so the rejection is
    visible rather than asserted: where the cross sits well below both, the
    instruments' noise has been removed from the answer. Negative points
    print as `--` in the deviation column, because there is no deviation
    there and a number would invite one to be read.
    """
    from ..multi import cross_variance

    a = _load(args)
    column_b = args.column if args.column_b is None else args.column_b
    b = _load(argparse.Namespace(**{**vars(args), "path": args.path_b,
                                    "column": column_b}))
    cross = cross_variance(a, b, args.estimator, taus=args.taus,
                           max_af=args.max_af)

    if args.json:
        print(json.dumps({
            "kind": cross.kind,
            "tau_s": cross.tau_s.tolist(),
            "af": cross.af.tolist(),
            "var": cross.var.tolist(),
            "sem": cross.sem.tolist(),
            "auto_a": cross.auto_a.tolist(),
            "auto_b": cross.auto_b.tolist(),
            "eta": cross.eta.tolist(),
            "edof_method": list(cross.edof_method),
            "correlation": cross.correlation.tolist(),
            "rejection_db": cross.rejection_db.tolist(),
            "n_terms": cross.n_terms.tolist(),
            "provenance": to_dict(cross.provenance),
        }, indent=2, default=str))
        return 0

    print(f"# {cross.kind} of {args.path} against {args.path_b}")
    print("# WARNING: this assumes the two channels' noise is independent of "
          "each other and of")
    print("#          the clock. A shared reference, supply or thermal "
          "environment breaks that")
    print("#          silently -- no arrangement of these two records can "
          "test it.")
    print(f"\n{'tau (s)':>12} {'sigma_a':>12} {'sigma_b':>12} {'cross':>12} "
          f"{'+/- (var)':>12} {'rho':>7} {'rej dB':>7} {'eta':>10}")
    for i, tau in enumerate(cross.tau_s):
        dev = "--" if cross.var[i] <= 0.0 else f"{cross.dev[i]:.4e}"
        print(f"{tau:12.6g} {np.sqrt(cross.auto_a[i]):12.4e} "
              f"{np.sqrt(cross.auto_b[i]):12.4e} {dev:>12} "
              f"{cross.sem[i]:12.4e} {cross.correlation[i]:7.3f} "
              f"{cross.rejection_db[i]:7.2f} {cross.eta[i]:10.1f}")
    n_neg = int(cross.negative.sum())
    if n_neg:
        print(f"\n# {n_neg} of {cross.tau_s.size} points came out negative and "
              "have no deviation. That is")
        print("# ordinary at long tau: the common part is below the scatter "
              "of the estimate.")
    print("\n# see `stabilis docs cross` for the error bar and what it assumes")
    return 0


def _cmd_reject(args) -> int:
    """M channels of one clock, averaged pairwise to reject their own noise.

    Every file is read the same way, which is the honest default for M
    channels of one capture; per-file delimiters and units are the bench
    stage's job, not this one's.
    """
    from ..multi import cross_matrix, dataset, reject

    paths = [args.path, *args.extra]
    series = {}
    for path in paths:
        name = Path(path).stem
        while name in series:
            name += "'"
        series[name] = _load(argparse.Namespace(**{**vars(args), "path": path}))
    ds = dataset(series)
    matrix = cross_matrix(ds, args.estimator, taus=args.taus,
                          max_af=args.max_af)
    r = reject(matrix)

    if args.json:
        print(json.dumps({
            "kind": r.kind,
            "channels": list(r.names),
            "n_pairs": r.n_pairs,
            "n": ds.n,
            "tau_s": r.tau_s.tolist(),
            "var": r.var.tolist(),
            "sem": r.sem.tolist(),
            "auto": r.auto.tolist(),
            "floor": r.floor.tolist(),
            "eta": r.eta.tolist(),
            "edof_method": list(r.edof_method),
            "correlation": r.correlation.tolist(),
            "rejection_db": r.rejection_db.tolist(),
            "matrix": matrix.var.tolist(),
            "notes": list(r.notes),
            "provenance": to_dict(r.provenance),
        }, indent=2, default=str))
        return 0

    print(f"# {r.kind} over {len(r.names)} channels, {r.n_pairs} pairs: "
          f"{', '.join(r.names)}")
    print(f"# {ds.n} common samples at {ds.rate_hz:g} Hz")
    print("# WARNING: every pair is assumed to have independent channel "
          "noise. M channels of")
    print("#          one clock are often M channels of one instrument, "
          "which share a timebase,")
    print("#          an LO, a supply and a thermal environment -- and that "
          "part does not cancel.")
    print(f"\n{'tau (s)':>12} {'1-channel':>12} {'rejected':>12} "
          f"{'+/- (var)':>12} {'rho':>7} {'rej dB':>7} {'eta':>10}")
    for i, tau in enumerate(r.tau_s):
        dev = "--" if r.var[i] <= 0.0 else f"{r.dev[i]:.4e}"
        print(f"{tau:12.6g} {np.sqrt(r.floor[i]):12.4e} {dev:>12} "
              f"{r.sem[i]:12.4e} {r.correlation[i]:7.3f} "
              f"{r.rejection_db[i]:7.2f} {r.eta[i]:10.1f}")
    if args.matrix:
        for i, tau in enumerate(r.tau_s):
            print(f"\n# correlation at tau = {tau:g} s")
            print("           " + "".join(f"{n:>12}" for n in r.names))
            for j, name in enumerate(r.names):
                row = "".join(f"{v:12.3f}" for v in matrix.correlation[i, j])
                print(f"{name:>10} {row}")
    for note in r.notes:
        print(f"\n# {note}")
    print("\n# see `stabilis docs cross` for the error bar and what it assumes")
    return 0


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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure(args.log_level)
    try:
        return args.func(args)
    except (ValueError, KeyError, OSError) as exc:
        print(f"stabilis: {exc}", file=sys.stderr)
        return 1

