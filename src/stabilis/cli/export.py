"""`stabilis export` -- the record, the curve, or everything with a manifest.

`export data` writes a self-describing file: it carries its own rate and
epoch, so a round trip needs no arguments and cannot be given the wrong
ones. `export bundle` builds a report on the way, which is why it borrows
`_spec_points` and `_report_grid` from `cli/report.py`.
"""

from __future__ import annotations

from pathlib import Path

from ..export import TABLE_FORMATS
from ..io.write import FORMATS, TIME_COLUMNS
from ..stats import ONE_SIGMA
from ..stats.core import GAP_POLICIES
from ..stats.edof import EDOF_METHODS
from ..tau import MODES
from .common import _add_input_args, _confidence_for, _load
from .report import _report_grid, _spec_points

__all__ = ["add_export"]


def add_export(sub) -> None:
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
