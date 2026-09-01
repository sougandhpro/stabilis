"""`stabilis psd` -- power spectra, and L(f) where it is meaningful.

The carrier frequency is required rather than assumed: a spectrum converted
with a guessed one is wrong by a constant nothing downstream can detect.
L(f) is withheld where the phase excursion is too large for `L = S_phi/2`.
"""

from __future__ import annotations

import json

import numpy as np

from ..provenance import to_dict
from ..stats.psd import DETRENDS, KINDS as PSD_KINDS
from ..stats.window import WINDOWS
from .common import _add_input_args, _load, _nan_to_none

__all__ = ["add_psd"]


def add_psd(sub) -> None:
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


def _cmd_psd(args) -> int:
    """The spectrum, and L(f) only where L = S_phi/2 actually holds.

    The frequency-domain view of the same noise sigma-tau describes. It is
    not a second opinion: the two are related by a published integral, and
    `theory/psd.md` gives the identity that ties this module's output to
    `powerlaw.py`'s coefficients.
    """
    from ..stats.psd import convert, script_l, welch

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
