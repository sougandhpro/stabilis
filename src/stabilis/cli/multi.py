"""`stabilis cross` and `stabilis reject` -- several channels at once.

The estimate is signed and goes negative at long tau. That is reported
rather than clipped: a negative cross-variance is the estimator saying the
common part is below its own scatter, and hiding it would turn a fact about
the measurement into a fact about the display.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..provenance import to_dict
from ..tau import MODES
from .common import _add_input_args, _load

__all__ = ["add_cross", "add_reject"]


def add_cross(sub) -> None:
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


def add_reject(sub) -> None:
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


def _cmd_cross(args) -> int:
    """Cross-variance of two records of one clock, seen through two channels.

    Prints the two auto-deviations beside the cross so the rejection is
    visible rather than asserted: where the cross sits well below both, the
    instruments' noise has been removed from the answer. Negative points
    print as `--` in the deviation column, because there is no deviation
    there and a number would invite one to be read.
    """
    from ..multi import cross_variance

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
    from ..multi import cross_matrix, dataset, reject

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
