"""`stabilis stats`, `drift` and `decimate` -- conditioning a record."""

from __future__ import annotations

import json

from ..io import save_frequency, save_phase
from ..preprocess import (DRIFT_MODELS, basic_stats, decimate, fit_drift,
                          remove_drift)
from ..provenance import to_dict
from .common import _add_input_args, _load, _summary

__all__ = ["add_stats", "add_drift", "add_decimate"]


def add_stats(sub) -> None:
    p_stats = sub.add_parser("stats", help="basic statistics for a data file")
    _add_input_args(p_stats)
    p_stats.add_argument("--json", action="store_true", help="emit JSON")
    p_stats.set_defaults(func=_cmd_stats)


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


def _add_out_args(p) -> None:
    p.add_argument("--out", help="write the conditioned record here")
    p.add_argument("--digits", type=int, default=17,
                   help="significant digits in --out (default: 17, round-trip)")
    p.add_argument("--json", action="store_true", help="emit JSON")


def _save(series, args):
    if not args.out:
        return None
    save = save_phase if args.kind == "phase" else save_frequency
    return save(series, args.out, digits=args.digits)


def add_drift(sub) -> None:
    p = sub.add_parser("drift", help=_summary(fit_drift))
    _add_input_args(p)
    models = sorted(set(DRIFT_MODELS["phase"]) | set(DRIFT_MODELS["frequency"]))
    p.add_argument("--model", choices=models,
                   help="default: quadratic for phase, linear for frequency")
    _add_out_args(p)
    p.set_defaults(func=_cmd_drift)


def _cmd_drift(args) -> int:
    series = _load(args)
    fit = fit_drift(series, args.model)
    written = _save(remove_drift(series, fit.model), args) if args.out else None
    if args.json:
        print(json.dumps({
            "model": fit.model, "kind": fit.kind,
            "drift_per_s": fit.drift_per_s, "drift_per_day": fit.drift_per_day,
            "out": args.out,
            "sha256": None if written is None else written.sha256,
            "provenance": to_dict(fit.provenance),
        }, indent=2, default=str))
        return 0
    print(f"file      {args.path}")
    print(f"model     {fit.model} ({fit.kind})")
    print(f"drift     {fit.drift_per_s:.6e} /s   {fit.drift_per_day:.6e} /day")
    if written is not None:
        print(f"written   {args.out}  sha256 {written.sha256[:16]}...")
    return 0


def add_decimate(sub) -> None:
    p = sub.add_parser("decimate", help=_summary(decimate))
    _add_input_args(p)
    p.add_argument("--factor", "-m", type=int, required=True,
                   help="averaging factor m; the new tau0 is m * tau0")
    _add_out_args(p)
    p.set_defaults(func=_cmd_decimate)


def _cmd_decimate(args) -> int:
    series = _load(args)
    out = decimate(series, args.factor)
    written = _save(out, args)
    if args.json:
        print(json.dumps({
            "factor": args.factor, "n_in": series.n, "n_out": out.n,
            "tau0_s": out.tau0_s, "out": args.out,
            "sha256": None if written is None else written.sha256,
            "provenance": to_dict(out.provenance),
        }, indent=2, default=str))
        return 0
    print(f"file      {args.path}")
    print(f"n         {series.n} -> {out.n}")
    print(f"tau0      {series.tau0_s:.9g} s -> {out.tau0_s:.9g} s")
    if written is not None:
        print(f"written   {args.out}  sha256 {written.sha256[:16]}...")
    return 0
