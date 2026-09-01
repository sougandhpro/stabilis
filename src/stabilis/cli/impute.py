"""`stabilis impute` -- fill gaps with the Total Imputer, and say so."""

from __future__ import annotations

import json

import numpy as np

from ..impute import fill_gaps
from ..io import save_frequency
from ..provenance import to_dict
from .common import _add_input_args, _load, _summary

__all__ = ["add_impute"]


def add_impute(sub) -> None:
    p = sub.add_parser("impute", help=_summary(fill_gaps))
    _add_input_args(p)
    p.add_argument("--reason", required=True,
                   help="why the gaps exist; required, and recorded")
    p.add_argument("--out", help="write the filled frequency record here")
    p.add_argument("--mask-out",
                   help="write the imputed mask here, one 0 or 1 per line")
    p.add_argument("--digits", type=int, default=17,
                   help="significant digits in --out (default: 17, round-trip)")
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.set_defaults(func=_cmd_impute)


def _cmd_impute(args) -> int:
    series = _load(args)
    if args.kind == "phase":
        # Not a TypeError from the library with a stack behind it: the fix
        # is a flag on this command line, so say that here.
        print("stabilis: impute fills frequency samples. Re-run with "
              "--kind frequency, or convert the file first -- the phase "
              "across a gap is not in the data, so it is not recoverable "
              "even once the frequency samples are.")
        return 1

    filled, mask = fill_gaps(series, reason=args.reason)
    written = None
    if args.out:
        written = save_frequency(filled, args.out, digits=args.digits)
    if args.mask_out:
        np.savetxt(args.mask_out, mask.astype(np.int8), fmt="%d")

    step = next(s for s in filled.provenance.steps if s.action == "fill_gaps")
    if args.json:
        print(json.dumps({
            "gaps": step.params["gaps"],
            "imputed": step.params["imputed"],
            "imputed_fraction": step.params.get("imputed_fraction", 0.0),
            "reason": args.reason,
            "out": args.out,
            "sha256": None if written is None else written.sha256,
            "provenance": to_dict(filled.provenance),
        }, indent=2, default=str))
        return 0

    print(f"file      {args.path}")
    print(f"n         {series.n}")
    print(f"gaps      {step.params['gaps']}")
    print(f"imputed   {step.params['imputed']} "
          f"({step.params.get('imputed_fraction', 0.0):.2%})")
    print(f"reason    {args.reason}")
    if written is not None:
        print(f"written   {args.out}  sha256 {written.sha256[:16]}...")
    if args.mask_out:
        print(f"mask      {args.mask_out}")
    if step.params["imputed"]:
        print("note      imputed samples are not measurements. State the "
              "gaps and their cause with any result derived from this "
              "record, and pass the mask to stabilis.live_terms so the "
              "confidence interval does not count them.")
    return 0
