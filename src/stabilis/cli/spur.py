"""`stabilis spur` -- discrete tones: find them, measure them, say so.

Every removal is recorded, because a record with a tone taken out describes
an oscillator that does not exist.
"""

from __future__ import annotations

import json

from ..provenance import to_dict
from .common import _add_input_args, _load

__all__ = ["add_spur"]


def add_spur(sub) -> None:
    p_spur = sub.add_parser(
        "spur", help="discrete tones: find them, measure them, optionally "
                     "remove them and say so"
    )
    _add_input_args(p_spur)
    p_spur.add_argument("--nperseg", type=int)
    p_spur.add_argument("--threshold-db", type=float, default=10.0,
                        help="how far above the local floor a bin must "
                             "stand (default 10)")
    p_spur.add_argument("--max-spurs", type=int, default=16)
    p_spur.add_argument("--floor-bins", type=int, default=33,
                        help="width of the running-median floor, in bins; "
                             "also sets the band that can be searched")
    p_spur.add_argument("--notch-out", metavar="FILE",
                        help="write the record with the tones subtracted; "
                             "every removal is recorded in its provenance")
    p_spur.add_argument("--json", action="store_true")
    p_spur.set_defaults(func=_cmd_spur)


def _cmd_spur(args) -> int:
    """Tones, measured against the record rather than read off the spectrum.

    The spectrum says where to look; the amplitude and frequency come from
    a least-squares fit to the samples, which is exact for a tone that fell
    between bins and needs no flat-top window to be so.
    """
    from ..stats.psd import welch
    from ..stats.spur import detect, notch

    series = _load(args)
    spec = welch(series, nperseg=args.nperseg)
    table = detect(spec, series, threshold_db=args.threshold_db,
                   max_spurs=args.max_spurs, floor_bins=args.floor_bins)

    if args.json:
        from dataclasses import asdict

        print(json.dumps(
            {"spurs": [asdict(s) for s in table.spurs],
             "threshold_db": table.threshold_db,
             "expected_false_alarms": table.expected_false_alarms,
             "searched_hz": list(table.searched_hz),
             "n_segments": spec.n_segments,
             "provenance": to_dict(table.provenance)},
            indent=2, default=str))
    else:
        unit = "s" if args.kind == "phase" else ""
        print(f"# tones in {args.path}, {args.threshold_db:g} dB above the "
              f"local floor")
        print(f"# searched {table.searched_hz[0]:.5g} .. "
              f"{table.searched_hz[1]:.5g} Hz over {spec.n_segments} segments; "
              f"{table.expected_false_alarms:.2g} expected by chance")
        if not table.spurs:
            print("# none found")
        else:
            print(f"{'f (Hz)':>16}  {'amplitude':>13}  {'rms':>13}  "
                  f"{'excess (dB)':>11}")
            for sp in table.spurs:
                print(f"{sp.f_hz:>16.9g}  {sp.amplitude:>13.4e}  "
                      f"{sp.rms:>13.4e}  {sp.excess_db:>11.1f}  {unit}")

    if args.notch_out:
        from ..io import save_frequency, save_phase

        if not table.spurs:
            raise ValueError("no tones were found, so there is nothing to "
                             "notch and no file was written")
        cleaned = notch(series, table)
        save = save_phase if args.kind == "phase" else save_frequency
        written = save(cleaned, args.notch_out)
        print(f"wrote {written.path} -- {len(table)} tones subtracted. "
              f"This record no longer describes the device as measured; the "
              f"removals are in its provenance.")
    return 0
