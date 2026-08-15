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
from .fitting import DEFAULT_BOOTSTRAP, OBJECTIVES
from .log import configure
from .preprocess import basic_stats
from .provenance import to_dict
from .stats import ONE_SIGMA, confidence
from .stats.core import GAP_POLICIES
from .stats.edof import EDOF_METHODS
from .stats.greenhall import GREENHALL_RUN_POLICIES
from .gui import DEFAULT_PORT as GUI_DEFAULT_PORT
from .export import TABLE_FORMATS
from .io.write import FORMATS, TIME_COLUMNS
from .stats.powerlaw import WEIGHT_METHODS
from .stats.psd import DETRENDS, KINDS as PSD_KINDS
from .stats.window import WINDOWS
from .tau import MODES

__all__ = ["main", "build_parser"]

_LOG_LEVELS = ("debug", "info", "warning", "error")

# Estimators with no confidence interval today, for two different reasons,
# and offering the flags anyway would be worse than not having them. MTIE
# and TIE rms are not variances at all, so the chi-square interval does not
# apply and `confidence` refuses them. The Thêo family *is* a variance
# family, but it reports no coefficient arrays for the data-driven path and
# its own EDF (SP 1065 Table 9) is not implemented -- see theory/theoh.md.
# None of them takes `keep_coeffs` either, which is why this one list drives
# both decisions.
NO_INTERVAL = ("mtie", "tierms", "theo1", "theobr", "theoh")


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

    from .stats import (adev, mdev, mtie, oadev, ohdev, tdev, theo1, theobr,
                    theoh, tierms, totdev)

    for name, fn in (
        ("adev", adev),
        ("oadev", oadev),
        ("mdev", mdev),
        ("tdev", tdev),
        ("ohdev", ohdev),
        ("totdev", totdev),
        ("theo1", theo1),
        ("theobr", theobr),
        ("theoh", theoh),
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
        if name in NO_INTERVAL:
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

    # wdev gets its own parser rather than a place in the loop above: its
    # scales are dyadic by construction, so --taus and --max-af have nothing
    # to act on, and the MODWT masks gaps rather than choosing a policy.
    from .stats.wavelet import FILTERS, wdev

    p_w = sub.add_parser(
        "wdev", help=wdev.__doc__.splitlines()[0])
    _add_input_args(p_w)
    p_w.add_argument(
        "--filter", default="haar", choices=sorted(FILTERS),
        help="wavelet family. haar (K=1) reproduces the overlapping ADEV "
             "exactly; d4 (K=2) additionally annihilates a linear frequency "
             "drift, d6 a quadratic, d8 a cubic",
    )
    p_w.add_argument(
        "--levels", type=int,
        help="stop after this many octaves (default: as many as the record "
             "supports)",
    )
    p_w.add_argument(
        "--biased", action="store_true",
        help="keep the coefficients that wrapped the circular boundary. "
             "Shift-invariant and wrong: it treats the end of the record as "
             "adjacent to its start",
    )
    p_w.add_argument(
        "--confidence", type=float, default=ONE_SIGMA,
        help="confidence level for the sigma bounds (default: 1 sigma)",
    )
    p_w.add_argument(
        "--edof-method", default="auto", choices=EDOF_METHODS,
        help="degrees-of-freedom estimator; see `stabilis docs edof`",
    )
    p_w.add_argument(
        "--no-confidence", action="store_true", help="skip the bounds")
    p_w.add_argument("--json", action="store_true", help="emit JSON")
    p_w.set_defaults(func=_cmd_wdev, estimator_name="wdev",
                     alpha=None, run_policy="pool")

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

    p_fit = sub.add_parser(
        "fit",
        help="fit a model to a measured sigma-tau curve, with uncertainties "
             "that count the correlation between taus",
    )
    _add_input_args(p_fit)
    # Deliberately not a `choices` list: a third-party model registered
    # through the plugin entry points is as valid as a built-in, and
    # `registry.get` already refuses an unknown name with the full list --
    # including those plugins, which a `choices` fixed at parser-build time
    # would not.
    p_fit.add_argument(
        "--model", default="powerlaw",
        help="model name -- powerlaw, drift, gauss_markov, quantization, "
             "bump, or a registered plugin. Use `stabilis powerlaw` instead "
             "for the IEEE 1139 decomposition, which is an exact solve "
             "rather than an optimization",
    )
    p_fit.add_argument(
        "--estimator", default="oadev",
        choices=("adev", "oadev", "mdev", "tdev", "ohdev", "totdev"),
        help="which curve to fit. A decimating estimator such as adev has no "
             "measurable cross-tau correlation, so its error bars come back "
             "as lower bounds",
    )
    p_fit.add_argument("--taus", default="octave", choices=MODES,
                       help="tau grid")
    p_fit.add_argument("--max-af", type=int, help="largest averaging factor")
    p_fit.add_argument(
        "--objective", default="ml", choices=OBJECTIVES,
        help="'ml' is the exact chi-square likelihood; the others are for "
             "comparability with tools that fit differently",
    )
    p_fit.add_argument(
        "--freeze", action="append", metavar="NAME=VALUE", default=[],
        help="hold a parameter at a value; repeatable",
    )
    p_fit.add_argument(
        "--bootstrap", type=int, default=0, metavar="N",
        help=f"resampled refits for a percentile interval "
             f"(a typical run is {DEFAULT_BOOTSTRAP}); 0 skips it",
    )
    p_fit.add_argument("--seed", type=int,
                       help="seed for the bootstrap, so a report is reproducible")
    p_fit.add_argument("--level", type=float, default=ONE_SIGMA,
                       help="confidence level (default: one sigma)")
    p_fit.add_argument("--json", action="store_true", help="emit JSON")
    p_fit.set_defaults(func=_cmd_fit)

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

    p_gui = sub.add_parser(
        "gui", help="a small local page in your browser, instead of flags")
    p_gui.add_argument("--port", type=int, default=GUI_DEFAULT_PORT)
    p_gui.add_argument(
        "--root", help="confine the file browser to this directory "
                       "(default: the current one)")
    p_gui.add_argument("--no-browser", action="store_true",
                       help="just print the URL")
    p_gui.set_defaults(func=_cmd_gui)

    p_mask = sub.add_parser(
        "mask", help="judge a record against a limit mask, with the margin "
                     "at the worst point"
    )
    _add_input_args(p_mask, required_path=False)
    p_mask.add_argument("--mask", help="mask name; --list-masks to see them")
    p_mask.add_argument("--list-masks", action="store_true",
                        help="print the catalogue and exit")
    p_mask.add_argument("--taus", default="octave", choices=MODES)
    p_mask.add_argument("--max-af", type=int)
    p_mask.add_argument("--confidence", type=float, default=ONE_SIGMA)
    p_mask.add_argument("--edof-method", default="auto", choices=EDOF_METHODS)
    p_mask.add_argument("--alpha", type=int, help="assumed power law")
    p_mask.add_argument("--no-confidence", action="store_true",
                        help="judge the point estimate; for a variance this "
                             "is simple acceptance and is not the default")
    p_mask.add_argument("--json", action="store_true")
    p_mask.set_defaults(func=_cmd_mask)

    p_psd = sub.add_parser(
        "psd", help="power spectral density, and L(f) where it is meaningful"
    )
    _add_input_args(p_psd)
    p_psd.add_argument("--nperseg", type=int,
                       help="segment length (default: ~8 segments of the "
                            "longest gap-free run, rounded down to a power of 2)")
    p_psd.add_argument("--overlap", type=float, default=0.5)
    p_psd.add_argument("--window", default="hann", choices=WINDOWS,
                       dest="window_name")
    p_psd.add_argument("--detrend", choices=DETRENDS,
                       help="default: linear for phase, mean for frequency")
    p_psd.add_argument("--spectrum", default=None, choices=PSD_KINDS,
                       help="convert to this spectrum (default: the one the "
                            "input yields natively -- S_x from phase, S_y "
                            "from frequency)")
    p_psd.add_argument("--nu0", type=float,
                       help="carrier frequency in Hz; required for S_phi "
                            "and for L(f), and never assumed")
    p_psd.add_argument("--script-l", action="store_true",
                       help="report L(f) in dBc/Hz instead of a density")
    p_psd.add_argument("--json", action="store_true")
    p_psd.set_defaults(func=_cmd_psd)

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


def _cmd_mask(args) -> int:
    """Does this unit meet spec -- the question a calibration report answers.

    The mask names the statistic it limits, so the estimator is chosen from
    the mask rather than from a flag: a limit on TDEV says nothing about
    MTIE, and letting the two be selected independently would let them be
    mismatched.
    """
    from .mask import CATALOG, available, evaluate
    from .registry import get

    if args.list_masks:
        for name in available():
            m = CATALOG[name]
            print(f"  {name:<24} {m.quantity:<6} "
                  f"tau {m.tau_min:g}..{m.tau_max:g} s   {m.source}")
        return 0
    if not args.mask:
        raise ValueError("give --mask NAME, or --list-masks to see them")
    if not args.path:
        raise ValueError("give a data file to judge against the mask")
    if args.mask not in CATALOG:
        raise KeyError(f"no mask {args.mask!r}; available: {available()}")

    mask = CATALOG[args.mask]
    series = _load(args)
    estimator = get("estimator", mask.quantity)
    want_ci = not args.no_confidence and mask.quantity not in NO_INTERVAL
    result = estimator(series, taus=args.taus, max_af=args.max_af,
                       **({"keep_coeffs": True} if want_ci else {}))
    ci = _confidence_for(args, result, series) if want_ci else None
    verdict = evaluate(result, mask, ci, series)

    if args.json:
        from dataclasses import asdict

        print(json.dumps(
            {"mask": verdict.mask, "quantity": verdict.quantity,
             "verdict": verdict.verdict,
             "source": mask.source, "notes": list(mask.notes),
             "checks": [asdict(c) for c in verdict.checks],
             "worst": asdict(verdict.worst) if verdict.worst else None,
             "provenance": to_dict(verdict.provenance)},
            indent=2, default=str))
        return 0

    print(f"# {mask.name} against {args.path}")
    print(f"# {mask.source}")
    for note in mask.notes:
        print(f"# note: {note}")
    print(f"{'Tau (s)':>12}  {mask.quantity.upper():>13}  {'Limit':>13}  "
          f"{'Margin':>8}  {'Verdict':<14}")
    for c in verdict.checks:
        limit = "--" if c.limit is None else f"{c.limit:.4e}"
        margin = "--" if c.margin is None else f"{c.margin:>7.2f}x"
        print(f"{c.tau_s:>12.4g}  {c.measured:>13.4e}  {limit:>13}  "
              f"{margin:>8}  {c.verdict:<14}"
              + (f"  # {c.reason}" if c.reason else ""))
    print(f"\nVERDICT: {verdict.verdict.upper()}")
    if verdict.worst is not None:
        w = verdict.worst
        print(f"worst margin {w.margin:.2f}x at tau = {w.tau_s:g} s "
              f"({w.measured:.4e} against {w.limit:.4e})")
    else:
        print("no tau could be compared with this mask")
    return 0 if verdict.verdict in ("pass", "not_evaluated") else 1


def _cmd_psd(args) -> int:
    """The spectrum, and L(f) only where L = S_phi/2 actually holds.

    The frequency-domain view of the same noise sigma-tau describes. It is
    not a second opinion: the two are related by a published integral, and
    `theory/psd.md` gives the identity that ties this module's output to
    `powerlaw.py`'s coefficients.
    """
    from .stats.psd import convert, script_l, welch

    series = _load(args)
    spec = welch(series, nperseg=args.nperseg, overlap=args.overlap,
                 window_name=args.window_name, detrend=args.detrend,
                 nu0_hz=args.nu0)
    if args.spectrum:
        spec = convert(spec, args.spectrum, args.nu0)

    if args.script_l:
        f, dbc, valid = script_l(spec, args.nu0)
        if args.json:
            print(json.dumps(
                {"f_hz": f.tolist(), "script_l_dbc": _nan_to_none(dbc),
                 "small_angle_valid": valid.tolist(),
                 "nu0_hz": args.nu0, "enbw_hz": spec.enbw_hz,
                 "n_segments": spec.n_segments,
                 "provenance": to_dict(spec.provenance)},
                indent=2, default=str))
            return 0
        print(f"# L(f) at nu0 = {args.nu0:g} Hz, from {spec.n_segments} "
              f"segments of {spec.nperseg}")
        if not valid.all():
            print(f"# {int((~valid).sum())} bins withheld: the phase "
                  "excursion above them exceeds the small-angle limit, "
                  "where L is not S_phi/2")
        print(f"{'f (Hz)':>14}  {'L (dBc/Hz)':>12}")
        for fi, di in zip(f, dbc):
            if fi > 0.0:
                print(f"{fi:>14.6e}  " +
                      (f"{di:>12.2f}" if np.isfinite(di) else f"{'withheld':>12}"))
        return 0

    if args.json:
        print(json.dumps(
            {"kind": spec.kind, "units": spec.units, "f_hz": spec.f.tolist(),
             "psd": spec.psd.tolist(), "rate_hz": spec.rate_hz,
             "nperseg": spec.nperseg, "n_segments": spec.n_segments,
             "window": spec.window, "enbw_hz": spec.enbw_hz,
             "detrend": spec.detrend, "nu0_hz": spec.nu0_hz,
             "provenance": to_dict(spec.provenance)},
            indent=2, default=str))
        return 0

    print(f"# {spec.kind} ({spec.units}) for {args.path}")
    print(f"# {spec.n_segments} segments of {spec.nperseg}, {spec.window} "
          f"window, detrend '{spec.detrend}'")
    print(f"# bin spacing {spec.df_hz:.6g} Hz, resolution (ENBW) "
          f"{spec.enbw_hz:.6g} Hz")
    print(f"{'f (Hz)':>14}  {spec.kind:>14}")
    for fi, pi in zip(spec.f, spec.psd):
        if fi > 0.0:
            print(f"{fi:>14.6e}  {pi:>14.6e}")
    return 0


def _nan_to_none(a):
    """NaN is not JSON. A withheld value is `null`, which parsers understand."""
    return [None if not np.isfinite(v) else float(v) for v in a]


def _cmd_export_data(args) -> int:
    """Write the record back out, in a format something else can read.

    The point of the default `stabilis` format is that the file says what
    it is: the rate and the epoch ride in a comment block, so reading it
    back needs no arguments and cannot be given the wrong ones. The
    `stable32` format drops all of that, because the tools it is aimed at
    were not asked to tolerate our comments.
    """
    from .io import save_frequency, save_phase

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
    from .export import to_csv, to_json, to_stable32
    from .registry import get

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
    from .export.bundle import build as build_bundle
    from .registry import get
    from .report import build as build_report, figure_uri, to_html

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
    extra = {} if args.estimator_name in NO_INTERVAL else {
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
    header = (f"# tau grid: {args.taus}   n = {series.n}   "
              f"rate = {series.rate_hz:g} Hz")
    if series.n_gaps:
        header += (
            f"\n# {series.n_gaps} gaps, {policy['runs']} gap-free runs, "
            f"policy '{policy['gap_policy']}', longest run {policy['longest_run']}"
        )
    return _print_result(args, result, ci, header)


def _print_result(args, result, ci, header: str) -> int:
    """The tau table, shared by every estimator that prints one."""
    if args.json:
        # One implementation of the result schema, in `export.table`, so the
        # command line and the library file cannot drift apart.
        from .export import to_json as result_json

        print(result_json(result, ci))
        return 0

    print(f"# {result.kind.upper()} for {args.path}")
    print(header)
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


def _cmd_wdev(args) -> int:
    """The wavelet deviation, on the Allan scale and one row per octave.

    Same table as every other estimator, because with the Haar filter it is
    the same number: `wdev --filter haar` and `oadev` agree at every shared
    tau, bounds included. The `--filter` column is where they part company.
    """
    from .stats.wavelet import FILTERS, wdev

    series = _load(args)
    want_ci = not args.no_confidence
    result = wdev(series, name=args.filter, levels=args.levels,
                  unbiased=not args.biased, keep_coeffs=want_ci)
    ci = _confidence_for(args, result, series) if want_ci else None
    step = _step_params(result, "wavelet_variance")
    header = (f"# filter: {args.filter}, K = {FILTERS[args.filter]} vanishing "
              f"moments   n = {series.n}   rate = {series.rate_hz:g} Hz")
    if series.n_gaps:
        header += (f"\n# {series.n_gaps} gaps masked "
                   f"{step['coefficients_masked']} coefficients")
    return _print_result(args, result, ci, header)


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


def _parse_freeze(items) -> dict:
    """`NAME=VALUE` pairs from the command line into a dict."""
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--freeze wants NAME=VALUE, got {item!r}")
        try:
            out[name.strip()] = float(value)
        except ValueError:
            raise ValueError(
                f"--freeze {item!r}: {value!r} is not a number") from None
    return out


def _cmd_fit(args) -> int:
    """Fit a model to a measured curve, honouring the chi-square sampling law.

    **Prints the parameters only when they are fit to quote.** If the
    optimizer did not converge, or the deviance is outside what its
    effective degrees of freedom admit, the deviance and the reason are
    printed instead. A parameter from a model that does not describe the
    data is worse than no parameter, because it looks like a measurement.

    Two standard errors are shown for each parameter: the one that counts
    the correlation between taus, and the one that ignores it. Their ratio
    is what the correlation was costing.
    """
    from .fitting import fit as run_fit
    from .registry import get as registry_get

    series = _load(args)
    curve = registry_get("estimator", args.estimator)(
        series, taus=args.taus, max_af=args.max_af)
    result = run_fit(
        curve, args.model, freeze=_parse_freeze(args.freeze),
        objective=args.objective, level=args.level,
        bootstrap=args.bootstrap, seed=args.seed,
    )

    if args.json:
        print(json.dumps({
            "model": result.model,
            "ok": result.ok,
            "params": result.params,
            "free": list(result.free),
            "frozen": result.frozen,
            "stderr": result.stderr,
            "stderr_uncorrelated": result.stderr_naive,
            "understatement": result.understatement,
            "interval": {k: list(v) for k, v in result.interval.items()},
            "bootstrap": {k: list(v) for k, v in result.bootstrap.items()},
            "n_bootstrap": result.n_bootstrap,
            "n_bootstrap_failed": result.n_bootstrap_failed,
            "level": result.level,
            "objective": result.objective,
            "correlation": result.correlation,
            "correlation_note": result.correlation_note,
            "deviance": result.gof.deviance,
            "dof_effective": result.gof.dof,
            "deviance_scale": result.gof.scale,
            "p_value": result.gof.p_value,
            "gof_ok": result.gof.ok,
            "gof_note": result.gof.note,
            "converged": result.converged,
            "reason": result.reason,
            "tau_s": result.tau_s.tolist(),
            "measured": result.measured.tolist(),
            "fitted": result.fitted.tolist(),
            "edof": result.edof.tolist(),
            "provenance": to_dict(result.provenance),
        }, indent=2, default=str))
        return 0 if result.ok else 1

    print(f"# fit of {result.model!r} to {args.estimator} of {args.path}")
    print(f"# objective={result.objective}, {result.tau_s.size} taus, "
          f"deviance {result.gof.deviance:.4g} on {result.gof.dof:.1f} "
          f"effective dof, p = {result.gof.p_value:.3g}")
    if result.correlation == "isserlis":
        print("# cross-tau correlation measured from the coefficients")
    else:
        print(f"# WARNING: the taus were treated as independent "
              f"({result.correlation_note})")
        print("#          so every uncertainty below is a lower bound.")

    if not result.ok:
        print("\n# REFUSING TO PRINT PARAMETERS")
        if not result.converged:
            print(f"#   the optimizer did not converge: {result.reason}")
        if not result.gof.ok:
            print(f"#   {result.gof.note}")
        print("# The fitted values exist on the result object; they are "
              "withheld here")
        print("# because a number from a model that does not describe the "
              "data reads")
        print("# like a measurement and is not one.")
        return 1

    pct = 100.0 * result.level
    print(f"\n{'parameter':<22}{'value':>14}{'std err':>13}"
          f"{'uncorrelated':>14}{'ratio':>8}")
    for name in result.free:
        print(f"{name:<22}{result.params[name]:>14.6e}"
              f"{result.stderr[name]:>13.3e}{result.stderr_naive[name]:>14.3e}"
              f"{result.understatement[name]:>8.2f}")
    for name, value in result.frozen.items():
        print(f"{name:<22}{value:>14.6e}{'frozen':>13}")

    print(f"\n# {pct:.1f}% intervals")
    for name in result.free:
        lo, hi = result.interval[name]
        line = f"#   {name:<20}[{lo:.6e}, {hi:.6e}]"
        if name in result.bootstrap:
            b_lo, b_hi = result.bootstrap[name]
            line += f"   bootstrap [{b_lo:.6e}, {b_hi:.6e}]"
        print(line)
    if result.n_bootstrap:
        print(f"# {result.n_bootstrap} bootstrap refits"
              + (f", {result.n_bootstrap_failed} dropped as unconverged"
                 if result.n_bootstrap_failed else ""))
    if result.gof.note:
        print(f"# {result.gof.note}")
    return 0


def _cmd_cross(args) -> int:
    """Cross-variance of two records of one clock, seen through two channels.

    Prints the two auto-deviations beside the cross so the rejection is
    visible rather than asserted: where the cross sits well below both, the
    instruments' noise has been removed from the answer. Negative points
    print as `--` in the deviation column, because there is no deviation
    there and a number would invite one to be read.
    """
    from .multi import cross_variance

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
    from .multi import cross_matrix, dataset, reject

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
        from .bench import load

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
    from .tau import averaging_factors
    from .report import spec_afs

    base = averaging_factors(series.n, args.taus, args.max_af)
    if not spec or base.size == 0:
        return base
    extra = spec_afs(spec, series.tau0_s)
    return np.unique(np.concatenate([base, extra[extra <= base.max()]]))


def _cmd_report(args) -> int:
    """A measurement report: the document a laboratory hands to a customer.

    Distinct from `validate`, which is evidence about the software rather
    than about a measurement. The conformity verdict here is taken on the
    confidence interval, not on the point estimate, so a record too short
    to decide says so instead of guessing.
    """
    from .registry import get
    from .report import build, figure_uri, to_html, to_json

    series = _load(args)
    spec, spec_label = _spec_points(args)
    grid = _report_grid(args, series, spec)
    result = get("estimator", args.estimator)(series, taus=grid)
    ci = confidence(result, level=args.confidence)

    fit = None
    if args.decompose:
        from .stats import mdev
        from .stats.powerlaw import decompose

        fit = decompose([result, mdev(series, taus=grid)])

    report = build(
        series, result, ci,
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
    from .bench import load
    from .bench.recommend import recommend

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
