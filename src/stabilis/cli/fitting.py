"""`stabilis fit` -- a model fitted under the chi-square likelihood.

`--model` deliberately carries no `choices`: models arrive through the
registry, including from third-party plugins, so a fixed list here would be
a list that goes stale the first time someone registers one.
"""

from __future__ import annotations

import json

from ..fitting import DEFAULT_BOOTSTRAP, OBJECTIVES
from ..provenance import to_dict
from ..stats import ONE_SIGMA
from ..tau import MODES
from .common import _add_input_args, _load

__all__ = ["add_fit"]


def add_fit(sub) -> None:
    p_fit = sub.add_parser(
        "fit",
        help="fit a model to a measured sigma-tau curve, with uncertainties "
             "that count the correlation between taus",
    )
    _add_input_args(p_fit)
    # Deliberately not a `choices` list: a third-party model registered
    # through the plugin entry points is as valid as a built-in, and
    # `registry.get` already refuses an unknown name with the full list --
    # including those plugins, which a `choices` fixed at parser-build time
    # would not.
    p_fit.add_argument(
        "--model", default="powerlaw",
        help="model name -- powerlaw, drift, gauss_markov, quantization, "
             "bump, or a registered plugin. Use `stabilis powerlaw` instead "
             "for the IEEE 1139 decomposition, which is an exact solve "
             "rather than an optimization",
    )
    p_fit.add_argument(
        "--estimator", default="oadev",
        choices=("adev", "oadev", "mdev", "tdev", "ohdev", "totdev"),
        help="which curve to fit. A decimating estimator such as adev has no "
             "measurable cross-tau correlation, so its error bars come back "
             "as lower bounds",
    )
    p_fit.add_argument("--taus", default="octave", choices=MODES,
                       help="tau grid")
    p_fit.add_argument("--max-af", type=int, help="largest averaging factor")
    p_fit.add_argument(
        "--objective", default="ml", choices=OBJECTIVES,
        help="'ml' is the exact chi-square likelihood; the others are for "
             "comparability with tools that fit differently",
    )
    p_fit.add_argument(
        "--freeze", action="append", metavar="NAME=VALUE", default=[],
        help="hold a parameter at a value; repeatable",
    )
    p_fit.add_argument(
        "--bootstrap", type=int, default=0, metavar="N",
        help=f"resampled refits for a percentile interval "
             f"(a typical run is {DEFAULT_BOOTSTRAP}); 0 skips it",
    )
    p_fit.add_argument("--seed", type=int,
                       help="seed for the bootstrap, so a report is reproducible")
    p_fit.add_argument("--level", type=float, default=ONE_SIGMA,
                       help="confidence level (default: one sigma)")
    p_fit.add_argument("--json", action="store_true", help="emit JSON")
    p_fit.set_defaults(func=_cmd_fit)


def _parse_freeze(items) -> dict:
    """`NAME=VALUE` pairs from the command line into a dict."""
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--freeze wants NAME=VALUE, got {item!r}")
        try:
            out[name.strip()] = float(value)
        except ValueError:
            raise ValueError(
                f"--freeze {item!r}: {value!r} is not a number") from None
    return out


def _cmd_fit(args) -> int:
    """Fit a model to a measured curve, honouring the chi-square sampling law.

    **Prints the parameters only when they are fit to quote.** If the
    optimizer did not converge, or the deviance is outside what its
    effective degrees of freedom admit, the deviance and the reason are
    printed instead. A parameter from a model that does not describe the
    data is worse than no parameter, because it looks like a measurement.

    Two standard errors are shown for each parameter: the one that counts
    the correlation between taus, and the one that ignores it. Their ratio
    is what the correlation was costing.
    """
    from ..fitting import fit as run_fit
    from ..registry import get as registry_get

    series = _load(args)
    curve = registry_get("estimator", args.estimator)(
        series, taus=args.taus, max_af=args.max_af)
    result = run_fit(
        curve, args.model, freeze=_parse_freeze(args.freeze),
        objective=args.objective, level=args.level,
        bootstrap=args.bootstrap, seed=args.seed,
    )

    if args.json:
        print(json.dumps({
            "model": result.model,
            "ok": result.ok,
            "params": result.params,
            "free": list(result.free),
            "frozen": result.frozen,
            "stderr": result.stderr,
            "stderr_uncorrelated": result.stderr_naive,
            "understatement": result.understatement,
            "interval": {k: list(v) for k, v in result.interval.items()},
            "bootstrap": {k: list(v) for k, v in result.bootstrap.items()},
            "n_bootstrap": result.n_bootstrap,
            "n_bootstrap_failed": result.n_bootstrap_failed,
            "level": result.level,
            "objective": result.objective,
            "correlation": result.correlation,
            "correlation_note": result.correlation_note,
            "deviance": result.gof.deviance,
            "dof_effective": result.gof.dof,
            "deviance_scale": result.gof.scale,
            "p_value": result.gof.p_value,
            "gof_ok": result.gof.ok,
            "gof_note": result.gof.note,
            "converged": result.converged,
            "reason": result.reason,
            "tau_s": result.tau_s.tolist(),
            "measured": result.measured.tolist(),
            "fitted": result.fitted.tolist(),
            "edof": result.edof.tolist(),
            "provenance": to_dict(result.provenance),
        }, indent=2, default=str))
        return 0 if result.ok else 1

    print(f"# fit of {result.model!r} to {args.estimator} of {args.path}")
    print(f"# objective={result.objective}, {result.tau_s.size} taus, "
          f"deviance {result.gof.deviance:.4g} on {result.gof.dof:.1f} "
          f"effective dof, p = {result.gof.p_value:.3g}")
    if result.correlation == "isserlis":
        print("# cross-tau correlation measured from the coefficients")
    else:
        print(f"# WARNING: the taus were treated as independent "
              f"({result.correlation_note})")
        print("#          so every uncertainty below is a lower bound.")

    if not result.ok:
        print("\n# REFUSING TO PRINT PARAMETERS")
        if not result.converged:
            print(f"#   the optimizer did not converge: {result.reason}")
        if not result.gof.ok:
            print(f"#   {result.gof.note}")
        print("# The fitted values exist on the result object; they are "
              "withheld here")
        print("# because a number from a model that does not describe the "
              "data reads")
        print("# like a measurement and is not one.")
        return 1

    pct = 100.0 * result.level
    print(f"\n{'parameter':<22}{'value':>14}{'std err':>13}"
          f"{'uncorrelated':>14}{'ratio':>8}")
    for name in result.free:
        print(f"{name:<22}{result.params[name]:>14.6e}"
              f"{result.stderr[name]:>13.3e}{result.stderr_naive[name]:>14.3e}"
              f"{result.understatement[name]:>8.2f}")
    for name, value in result.frozen.items():
        print(f"{name:<22}{value:>14.6e}{'frozen':>13}")

    print(f"\n# {pct:.1f}% intervals")
    for name in result.free:
        lo, hi = result.interval[name]
        line = f"#   {name:<20}[{lo:.6e}, {hi:.6e}]"
        if name in result.bootstrap:
            b_lo, b_hi = result.bootstrap[name]
            line += f"   bootstrap [{b_lo:.6e}, {b_hi:.6e}]"
        print(line)
    if result.n_bootstrap:
        print(f"# {result.n_bootstrap} bootstrap refits"
              + (f", {result.n_bootstrap_failed} dropped as unconverged"
                 if result.n_bootstrap_failed else ""))
    if result.gof.note:
        print(f"# {result.gof.note}")
    return 0
