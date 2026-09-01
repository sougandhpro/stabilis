"""`stabilis davar` and `stabilis stream` -- measurements still running.

`davar` asks whether the oscillator was doing the same thing all along,
which one curve cannot answer; `stream` runs the estimator over a file as
it is written.

A sample that never arrived becomes a NaN in its own slot, never the
disappearance of a slot. Appending arriving values to a list passes every
other test and fails only that one -- a counter that drops fifty readings
would hand back a record fifty samples shorter and perfectly smooth.
"""

from __future__ import annotations

import json
from itertools import chain

import numpy as np

from ..provenance import to_dict
from ..stream.davar import STEP_DIVISOR
from ..tau import MODES
from .common import _add_input_args, _load, _nan_to_none

__all__ = ["add_davar", "add_stream"]


def add_davar(sub) -> None:
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


def add_stream(sub) -> None:
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
