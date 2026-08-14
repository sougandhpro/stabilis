"""Fitting a model to a measured stability curve, end to end.

What this assembles, and why each piece is not optional:

* the **objective** is the exact chi-square likelihood, because a stability
  estimate is not Gaussian about its truth and is least Gaussian at long
  tau, where the specification is;
* the **degrees of freedom** are the same ones `confidence` puts on the
  plotted error bars, so a fit cannot silently weight a point differently
  from how it was drawn;
* the **covariance between taus** is measured rather than assumed away, so
  the parameter uncertainties are not the too-narrow ones that come from
  pretending the points are independent;
* the **goodness of fit** is reported against the distribution the deviance
  actually has, and the parameters are withheld when it is out of range,
  because a number quoted from a model that does not describe the data is
  worse than no number.

**Use `stats.powerlaw.decompose` instead whenever it applies.** The five
IEEE 1139 noise powers and drift are linear in their parameters, so that
fit is an exact non-negative least squares with no starting guess, no
convergence tolerance and no local minimum. This module is for the shapes
that are not: a correlation time, a resonance, a quantization floor, and
whatever the customer brings.

See theory/fit.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .._specfun import norm_ppf
from ..log import record, warn
from ..provenance import Provenance
from ..stats.ci import ONE_SIGMA
from ..stats.core import StabilityResult
from ..stats.edof import DEFAULT_LAG_LIMIT
from ..stats.taucov import TauCovariance, tau_covariance
from .lm import (
    DEFAULT_MAX_ITER,
    levenberg_marquardt,
    to_bounded,
    to_free,
)
from .models import Model, model as lookup_model
from .objective import goodness_of_fit, residuals
from .uncertainty import (
    bootstrap_parameters,
    naive_covariance,
    residual_hat,
    sandwich_covariance,
)

__all__ = ["FitResult", "fit", "DEFAULT_BOOTSTRAP"]

# Draws when a bootstrap is asked for without a count. 200 gives a
# percentile interval whose own edges are stable to a few percent, which is
# finer than the interval is meaningful to.
DEFAULT_BOOTSTRAP = 200


@dataclass(frozen=True)
class FitResult:
    """Fitted parameters, two accounts of how well they are known, and why."""

    model: str
    params: dict[str, float]
    free: tuple[str, ...]
    frozen: dict[str, float]
    # Standard errors from the sandwich, which uses the measured cross-tau
    # covariance, and from the same expression with that covariance
    # replaced by its diagonal. The second is what every other tool
    # reports; carrying both makes the difference a measurement.
    stderr: dict[str, float]
    stderr_naive: dict[str, float]
    cov: np.ndarray
    interval: dict[str, tuple[float, float]]
    # Percentile interval over the resampled fits, empty when no bootstrap
    # was run. It is not an average with the sandwich and must not be
    # presented as one -- where the two disagree, the model is misspecified
    # or a parameter is badly non-linear, and that is the finding.
    bootstrap: dict[str, tuple[float, float]] = field(default_factory=dict)
    n_bootstrap: int = 0
    n_bootstrap_failed: int = 0
    level: float = ONE_SIGMA
    objective: str = "ml"
    correlation: str = "independent"
    correlation_note: str = ""
    tau_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    measured: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fitted: np.ndarray = field(default_factory=lambda: np.zeros(0))
    edof: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gof: object = None
    converged: bool = False
    reason: str = ""
    n_iter: int = 0
    provenance: Provenance | None = None

    @property
    def ok(self) -> bool:
        """Whether these parameters are fit to quote at all.

        Both conditions, not either: an unconverged optimizer has not found
        the minimum, and a converged one on a model that does not describe
        the curve has found the wrong question's answer.
        """
        return bool(self.converged and self.gof is not None and self.gof.ok)

    @property
    def understatement(self) -> dict[str, float]:
        """Sandwich standard error over naive, per free parameter.

        The number this whole subsystem exists to produce. Above one means
        the correlation between taus was costing confidence that was never
        there; one exactly means the taus really were independent, which
        happens only for a decimating estimator or a record too gappy to
        measure the correlation on.
        """
        return {
            name: (self.stderr[name] / self.stderr_naive[name]
                   if self.stderr_naive[name] > 0.0 else float("nan"))
            for name in self.free
        }

    @property
    def residual_ratio(self) -> np.ndarray:
        """Measured over fitted, the thing to plot under the curve."""
        return self.measured / self.fitted


def _resolve_model(spec: Model | str) -> Model:
    return spec if isinstance(spec, Model) else lookup_model(spec)


def _resolve_bounds(m: Model, bounds) -> dict:
    """Default bounds from the model's physics, overridden per name.

    A parameter the model declares positive gets `(0, None)` unless the
    caller says otherwise. That is not a convenience: a variance or a
    correlation time that goes negative mid-iteration makes the likelihood
    undefined, and the optimizer would then report a failure whose cause is
    two steps upstream of where it appears.
    """
    out = {name: ((0.0, None) if name in m.positive else (None, None))
           for name in m.params}
    for name, pair in (bounds or {}).items():
        if name not in out:
            raise ValueError(
                f"fit: bounds name {name!r} is not a parameter of "
                f"{m.name!r} ({list(m.params)})")
        lo, hi = pair
        if lo is not None and hi is not None and not lo < hi:
            raise ValueError(f"fit: bounds for {name!r} are not increasing: {pair}")
        out[name] = (lo, hi)
    return out


def _check_curve(result: StabilityResult, n_free: int) -> None:
    if not result.is_variance:
        raise ValueError(
            f"fit: {result.kind!r} is not a variance, so it has no "
            "chi-square sampling law and this likelihood does not apply to "
            "it. An extreme value like MTIE needs a different objective")
    if np.any(result.var <= 0.0):
        raise ValueError(
            "fit: at least one measured variance is not positive, and the "
            "log-likelihood has no value there. A negative cross-variance is "
            "a real result -- see multi/cross.py -- but it cannot be fitted "
            "with this objective")
    if n_free == 0:
        raise ValueError("fit: every parameter is frozen, so there is nothing "
                         "to fit")
    if result.af.size <= n_free:
        raise ValueError(
            f"fit: {result.af.size} taus for {n_free} free parameters leaves "
            "no degrees of freedom. Widen the tau grid, freeze a parameter, "
            "or use a simpler model")


def fit(
    result: StabilityResult,
    model: Model | str,
    guess: dict | None = None,
    bounds: dict | None = None,
    freeze: dict | None = None,
    objective: str = "ml",
    level: float = ONE_SIGMA,
    bootstrap: int = 0,
    seed: int | None = None,
    edof_method: str = "auto",
    lag_limit: int = DEFAULT_LAG_LIMIT,
    max_iter: int = DEFAULT_MAX_ITER,
    taucov: TauCovariance | None = None,
) -> FitResult:
    """Fit `model` to the sigma-tau curve in `result`.

    `model` is a `Model` record or the name of a registered one. `guess`
    overrides the model's own starting values per parameter; `bounds` and
    `freeze` are per-parameter, `{name: (lo, hi)}` and `{name: value}`.

    `bootstrap` is a number of resampled refits. Zero, the default, skips
    it: the sandwich standard errors are always computed and are the cheap
    answer, while the bootstrap costs that many optimizations and is worth
    it when the parameters are strongly non-linear or the model is in
    doubt. Pass `seed` to make it reproducible, which a report requires.

    The returned `FitResult.ok` is False when the optimizer did not
    converge *or* the deviance is outside what its degrees of freedom
    admit. Callers should honour it rather than printing the parameters
    anyway; `stabilis fit` does.
    """
    m = _resolve_model(model)
    frozen = dict(freeze or {})
    unknown = set(frozen) - set(m.params)
    if unknown:
        raise ValueError(
            f"fit: cannot freeze {sorted(unknown)}; not parameters of "
            f"{m.name!r} ({list(m.params)})")
    free = tuple(p for p in m.params if p not in frozen)
    _check_curve(result, len(free))

    limits = _resolve_bounds(m, bounds)
    tc = taucov or tau_covariance(result, lag_limit, edof_method)
    eta, tau = tc.edof, result.tau_s
    start = dict(m.guess(tau, result.var))
    start.update(guess or {})
    start.update(frozen)

    def variance_of(theta_free: np.ndarray) -> np.ndarray:
        p = dict(frozen)
        p.update(dict(zip(free, np.asarray(theta_free, dtype=np.float64))))
        return m.variance(tau, p)

    def unpack(z: np.ndarray) -> np.ndarray:
        return np.array([to_bounded(float(z[i]), *limits[n])
                         for i, n in enumerate(free)])

    def variance_of_z(z: np.ndarray) -> np.ndarray:
        return variance_of(unpack(z))

    def residual_of(z: np.ndarray, measured: np.ndarray) -> np.ndarray:
        return residuals(objective, measured, variance_of_z(z), eta)

    z0 = np.array([to_free(float(start[n]), *limits[n]) for n in free])
    out = levenberg_marquardt(lambda z: residual_of(z, result.var), z0,
                              max_iter=max_iter)
    theta = unpack(out.z)
    fitted = variance_of(theta)
    params = dict(frozen)
    params.update(dict(zip(free, (float(v) for v in theta))))

    # The covariances are computed in the parameters themselves, not in the
    # optimizer's transformed coordinates. The transformation was there to
    # keep the search inside the model's domain; the optimum is interior,
    # and a covariance is easier to read where it is quoted. Both routes
    # agree -- the sandwich is equivariant under a smooth change of
    # variables -- and this one does not need the delta method to say so.
    cov = sandwich_covariance(objective, result.var, variance_of, eta, theta,
                              tc.cov)
    cov_naive = naive_covariance(objective, result.var, variance_of, eta, theta,
                                 tc.cov)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    se_naive = np.sqrt(np.clip(np.diag(cov_naive), 0.0, None))
    z_level = float(norm_ppf(0.5 * (1.0 + level)))
    hat = residual_hat(objective, result.var, variance_of, eta, theta)
    gof = goodness_of_fit(float(out.cost), int(tau.size), len(free), tc.corr,
                          hat)

    boot, n_boot, n_failed = {}, 0, 0
    worst_eig = 0.0
    if bootstrap:
        boot, n_boot, n_failed, worst_eig = _run_bootstrap(
            fitted, eta, tc.corr, residual_of, out.z, limits, free, max_iter,
            int(bootstrap), level, seed,
        )

    prov = record(
        tc.provenance, "fit", "fit",
        {
            "model": m.name,
            "objective": objective,
            "free": list(free),
            "frozen": {k: float(v) for k, v in frozen.items()},
            "bounds": {k: list(v) for k, v in limits.items()},
            "correlation": tc.method,
            "edof_method": edof_method,
            "level": level,
            "bootstrap": n_boot,
            "seed": seed,
            "deviance": float(out.cost),
            "dof_effective": float(gof.dof),
            "iterations": out.n_iter,
        },
        f"{out.reason}; deviance {out.cost:.4g} on {gof.dof:.1f} dof",
    )
    if not out.converged:
        prov = warn(prov, "fit", "not_converged", {"reason": out.reason},
                    "the parameters are where the optimizer stopped, not "
                    "where the minimum is; do not quote them")
    if not gof.ok:
        prov = warn(prov, "fit", "goodness_of_fit",
                    {"deviance": float(out.cost), "dof": float(gof.dof),
                     "p_value": float(gof.p_value)}, gof.note)
    if tc.method != "isserlis":
        prov = warn(prov, "fit", "uncorrelated",
                    {"reason": tc.reason},
                    "the taus were treated as independent, so these "
                    "parameter uncertainties are lower bounds")
    if worst_eig < 0.0:
        prov = warn(prov, "fit", "correlation_repaired",
                    {"worst_eigenvalue": worst_eig},
                    "the measured correlation matrix was not positive "
                    "definite and was projected before resampling")
    if n_failed:
        prov = warn(prov, "fit", "bootstrap_failures",
                    {"failed": n_failed, "drawn": n_boot + n_failed},
                    "resampled fits that did not converge were dropped, not "
                    "retried; the interval rests on the rest")

    return FitResult(
        model=m.name,
        params=params,
        free=free,
        frozen={k: float(v) for k, v in frozen.items()},
        stderr=dict(zip(free, (float(v) for v in se))),
        stderr_naive=dict(zip(free, (float(v) for v in se_naive))),
        cov=cov,
        interval={n: (params[n] - z_level * float(se[i]),
                      params[n] + z_level * float(se[i]))
                  for i, n in enumerate(free)},
        bootstrap=boot,
        n_bootstrap=n_boot,
        n_bootstrap_failed=n_failed,
        level=level,
        objective=objective,
        correlation=tc.method,
        correlation_note=tc.reason,
        tau_s=tau,
        measured=result.var,
        fitted=fitted,
        edof=eta,
        gof=gof,
        converged=out.converged,
        reason=out.reason,
        n_iter=out.n_iter,
        provenance=prov,
    )


def _run_bootstrap(fitted, eta, corr, residual_of, z_hat, limits, free,
                   max_iter, draws, level, seed):
    """Resample from the fitted law, refit each draw, take percentiles.

    Each refit starts from the original solution rather than from the
    model's guess. That is not a shortcut for speed: starting every draw
    from the same point makes the spread of the answers a property of the
    data, not of how far each happened to start.
    """
    rng = np.random.default_rng(seed)

    def refit(x):
        out = levenberg_marquardt(lambda z: residual_of(z, x), z_hat,
                                  max_iter=max_iter)
        if not out.converged:
            return None
        return np.array([to_bounded(out.z[i], *limits[n])
                         for i, n in enumerate(free)])

    samples, failed, worst = bootstrap_parameters(fitted, eta, corr, refit,
                                                  draws, rng)
    if samples.size == 0:
        return {}, 0, failed, worst
    alpha = 0.5 * (1.0 - level)
    lo = np.quantile(samples, alpha, axis=0)
    hi = np.quantile(samples, 1.0 - alpha, axis=0)
    interval = {n: (float(lo[i]), float(hi[i])) for i, n in enumerate(free)}
    return interval, int(samples.shape[0]), failed, worst
