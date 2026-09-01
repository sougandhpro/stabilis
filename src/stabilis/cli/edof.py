"""`stabilis compare-edof` -- every EDOF method side by side.

A diagnostic rather than a measurement: it exists so the cost of assuming a
power law can be read off directly, next to the data-driven answer, instead
of being argued about.
"""

from __future__ import annotations

import json

from ..provenance import to_dict
from ..stats import ONE_SIGMA, confidence
from ..tau import MODES
from .common import _add_input_args, _load

__all__ = ["add_compare_edof"]


def add_compare_edof(sub) -> None:
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


def _cmd_compare_edof(args) -> int:
    """Every EDOF method on one dataset, and what each costs in bounds.

    The diagnostic that makes the design argument concrete. `eta1` reads
    the degrees of freedom off the data; `eta3` counts non-overlapping
    windows and is a deliberate lower bound; `greenhall` assumes a power
    law and is what Stable32 prints. Where they disagree is where the
    assumption is doing the work, and a metrologist deciding which number
    to put on a certificate should be able to see that in one table.
    """
    from ..registry import get

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
