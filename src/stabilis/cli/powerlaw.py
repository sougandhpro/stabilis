"""`stabilis powerlaw` -- sigma(tau) back into the IEEE 1139 levels.

Stacks Allan and modified rows, because the two PM noises do not separate
from the Allan variance alone: its high-frequency cutoff is fixed, while
the modified variance's scales with tau.
"""

from __future__ import annotations

import json

from ..provenance import to_dict
from ..stats.powerlaw import WEIGHT_METHODS
from ..tau import MODES
from .common import _add_input_args, _load

__all__ = ["add_powerlaw"]


def add_powerlaw(sub) -> None:
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


def _cmd_powerlaw(args) -> int:
    """Decompose a measured curve into IEEE 1139 noise levels and drift.

    Stacks the overlapping Allan and modified Allan curves of the same
    record, because the two PM noises do not separate from AVAR alone. The
    fit is a non-negative least squares, so no level ever comes back
    negative and the solution carries an optimality certificate.
    """
    from ..powerlaw_coeffs import ALPHAS
    from ..stats import mdev, oadev
    from ..stats.powerlaw import decompose
    from ..synth import ALPHA_NAMES

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
