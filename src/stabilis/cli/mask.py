"""`stabilis mask` -- judge a record against a limit, with the margin.

The verdict is decided on the interval rather than the point: a deviation
under the limit whose error bar crosses it has not passed. A record that
cannot be evaluated against the standard's own measurement conditions
yields `not evaluated` rather than a green tick.
"""

from __future__ import annotations

import json

from ..provenance import to_dict
from ..stats import ONE_SIGMA
from ..stats.edof import EDOF_METHODS
from ..tau import MODES
from .common import (NO_INTERVAL, _add_input_args, _confidence_for, _load)

__all__ = ["add_mask"]


def add_mask(sub) -> None:
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


def _cmd_mask(args) -> int:
    """Does this unit meet spec -- the question a calibration report answers.

    The mask names the statistic it limits, so the estimator is chosen from
    the mask rather than from a flag: a limit on TDEV says nothing about
    MTIE, and letting the two be selected independently would let them be
    mismatched.
    """
    from ..mask import CATALOG, available, evaluate
    from ..registry import get

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
