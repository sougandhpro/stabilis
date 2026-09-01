"""The variance family on the command line, and the table it prints.

One loop registers the eleven estimators that take the same arguments, and
`wdev` sits outside it because its scales are dyadic: `--taus` and
`--max-af` would have nothing to act on. The tuple the loop walks *is* the
order `adev ... tierms` appears in `--help`, so it is not sorted and must
not be.

`_print_result` and `_step_params` live here rather than in `common`
because they have exactly two callers between them, both in this file.
"""

from __future__ import annotations

from ..stats import ONE_SIGMA
from ..stats.core import GAP_POLICIES
from ..stats.edof import EDOF_METHODS
from ..stats.greenhall import GREENHALL_RUN_POLICIES
from ..tau import MODES
from .common import (NO_INTERVAL, _add_input_args, _confidence_for, _load,
                     _summary)

__all__ = ["add_estimators", "add_wdev"]


def add_estimators(sub) -> None:
    from ..stats import (adev, mdev, mtie, oadev, ohdev, tdev, theo1, theobr,
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
        p = sub.add_parser(name, help=_summary(fn))
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


def add_wdev(sub) -> None:
    # wdev gets its own parser rather than a place in the loop above: its
    # scales are dyadic by construction, so --taus and --max-af have nothing
    # to act on, and the MODWT masks gaps rather than choosing a policy.
    from ..stats.wavelet import FILTERS, wdev

    p_w = sub.add_parser(
        "wdev", help=_summary(wdev))
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


def _step_params(result, action: str) -> dict:
    """Parameters of the last step with this action. The audit trail, reused."""
    return [s for s in result.provenance.steps if s.action == action][-1].params


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
        from ..export import to_json as result_json

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
    from ..stats.wavelet import FILTERS, wdev

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
