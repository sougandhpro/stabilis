"""Levenberg-Marquardt, in numpy, with bounds by reparameterization.

**There is no scipy fast path here, and that is deliberate.** scipy would
be faster and is already an optional dependency elsewhere in this library.
But a fit ends up in a signed report, and a report whose numbers depend on
what happens to be installed on the machine that produced it is not a
report. scipy appears in `tests/test_fit.py` instead, as a differential
check that `scipy.optimize.least_squares` reaches the same minimum -- which
is better evidence than a speedup and costs the caller nothing.

**Bounds are transformations, not projections.** A parameter constrained to
be positive is fitted as `exp(z)`; one constrained to an interval is fitted
as a logistic. The optimizer then runs unconstrained, cannot take a step
that lands outside the model's domain, and never reports a parameter
sitting exactly on a bound as though that were a measurement. The price is
that a parameter whose true value *is* the bound cannot be reached, only
approached -- which is the right trade here, because a variance that is
genuinely zero is a job for the exact non-negative solve in
`stats/powerlaw.py`, not for an optimizer.

The covariance comes back in the transformed space and is carried to the
original by the delta method, `Cov_theta = G Cov_z G.T` with `G` the
diagonal of `d theta / d z`. Exact for the transformation, first-order in
the usual way for everything else.

See theory/fit.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LMResult", "levenberg_marquardt", "numeric_jacobian",
           "to_free", "to_bounded", "dtheta_dz", "DEFAULT_MAX_ITER"]

DEFAULT_MAX_ITER = 200

# Relative step for the central-difference Jacobian. eps**(1/3) balances
# truncation error, which falls as h**2, against rounding, which grows as
# 1/h -- the classic optimum for a central difference, and about 6e-6.
_JAC_STEP = float(np.finfo(np.float64).eps) ** (1.0 / 3.0)

# Damping schedule. Marquardt's own: start small, multiply by 10 on a
# rejected step and divide by 10 on an accepted one.
_LAMBDA0 = 1e-3
_LAMBDA_UP = 10.0
_LAMBDA_DOWN = 0.1
_LAMBDA_MAX = 1e12


def to_free(theta: float, lo: float | None, hi: float | None) -> float:
    """Bounded parameter to the unconstrained coordinate the optimizer sees.

        (-inf, inf):  z = theta
        [lo,   inf):  z = ln(theta - lo)
        (-inf,  hi]:  z = ln(hi - theta)
        [lo,    hi]:  z = ln((theta-lo) / (hi-theta))
    """
    if lo is not None and hi is not None:
        if not lo < theta < hi:
            raise ValueError(
                f"starting value {theta!r} is not strictly inside its bounds "
                f"({lo}, {hi}); a transformed fit cannot start on a bound")
        return float(np.log((theta - lo) / (hi - theta)))
    if lo is not None:
        if theta <= lo:
            raise ValueError(
                f"starting value {theta!r} is not above its lower bound {lo}")
        return float(np.log(theta - lo))
    if hi is not None:
        if theta >= hi:
            raise ValueError(
                f"starting value {theta!r} is not below its upper bound {hi}")
        return float(np.log(hi - theta))
    return float(theta)


def to_bounded(z: float, lo: float | None, hi: float | None) -> float:
    """The inverse of `to_free`.

    The two-sided branch is written as a logistic on the *negative*
    magnitude, `1/(1 + exp(-|z|))` folded by sign, so neither `exp(z)` nor
    `exp(-z)` can overflow when the optimizer pushes a parameter hard
    against a bound. Writing it the obvious way gives inf/inf and a NaN
    parameter, several iterations before anything else notices.
    """
    if lo is not None and hi is not None:
        t = 1.0 / (1.0 + np.exp(-abs(z)))
        frac = t if z >= 0.0 else 1.0 - t
        return float(lo + (hi - lo) * frac)
    if lo is not None:
        return float(lo + np.exp(z))
    if hi is not None:
        return float(hi - np.exp(z))
    return float(z)


def dtheta_dz(z: float, lo: float | None, hi: float | None) -> float:
    """Derivative of `to_bounded`, for the delta method."""
    if lo is not None and hi is not None:
        theta = to_bounded(z, lo, hi)
        return float((theta - lo) * (hi - theta) / (hi - lo))
    if lo is not None:
        return float(np.exp(z))
    if hi is not None:
        return float(-np.exp(z))
    return 1.0


def numeric_jacobian(fn, z: np.ndarray, step: float = _JAC_STEP,
                     scale=None) -> np.ndarray:
    """Central-difference Jacobian of a vector function, `J[i, j] = df_i/dz_j`.

    The step is `h_j = step * scale_j`, and getting `scale` right is the
    whole difficulty: the parameters of a stability model span decades -- a
    correlation time of 1e3 s beside a variance of 1e-24 -- so no absolute
    step suits both.

    The default, `max(|z_j|, 1)`, is for the **optimizer's** coordinates,
    where a positive parameter is a logarithm of order ten and the floor
    keeps the step alive as a coordinate passes through zero. It is wrong
    for raw parameters: differencing a variance of 1e-24 by 6e-6 does not
    measure a derivative, it evaluates the model somewhere else entirely
    and, for a positive quantity, somewhere it does not exist. Callers
    working in the parameters themselves pass a relative `scale`.
    """
    z = np.asarray(z, dtype=np.float64)
    if scale is None:
        scale = np.maximum(np.abs(z), 1.0)
    scale = np.broadcast_to(np.asarray(scale, dtype=np.float64), z.shape)
    out = []
    for j in range(z.size):
        h = step * float(scale[j])
        up, dn = z.copy(), z.copy()
        up[j] += h
        dn[j] -= h
        out.append((fn(up) - fn(dn)) / (2.0 * h))
    return np.column_stack(out) if out else np.zeros((0, 0))


@dataclass(frozen=True)
class LMResult:
    """Where the optimizer stopped, and why."""

    z: np.ndarray
    residual: np.ndarray
    cost: float
    n_iter: int
    converged: bool
    reason: str


def levenberg_marquardt(
    residual_fn,
    z0,
    max_iter: int = DEFAULT_MAX_ITER,
    ftol: float = 1e-12,
    xtol: float = 1e-12,
    gtol: float = 1e-12,
) -> LMResult:
    """Minimize `sum(residual_fn(z)**2)` over the unconstrained vector `z`.

        (J.T J + lam * diag(J.T J)) delta = -J.T r
        z <- z + delta   if the cost fell, and lam <- lam/10
        lam <- lam*10    otherwise, and try again from the same z

    Damping is scaled by `diag(J.T J)` -- Marquardt's form -- rather than by
    the identity. That makes the step invariant to rescaling any one
    parameter, which matters here because the parameters are not
    commensurable: a bare `lam * I` would damp a variance of 1e-24 and a
    correlation time of 1e3 by wildly different relative amounts and the
    fit would crawl along one coordinate.

    Stops on any of three conditions, and says which: the cost stopped
    falling (`ftol`), the step became negligible (`xtol`), or the gradient
    vanished (`gtol`). Running out of iterations is reported as
    **not converged** rather than as an answer, because a fit that stopped
    early and a fit that finished look identical from the outside and only
    one of them should reach a report.
    """
    z = np.asarray(z0, dtype=np.float64).copy()
    r = np.asarray(residual_fn(z), dtype=np.float64)
    cost = float(r @ r)
    lam = _LAMBDA0

    for it in range(1, max_iter + 1):
        j = numeric_jacobian(residual_fn, z)
        gradient = j.T @ r
        if float(np.max(np.abs(gradient), initial=0.0)) < gtol * max(cost, 1.0):
            return LMResult(z, r, cost, it, True, "gradient vanished")

        jtj = j.T @ j
        scale = np.diag(jtj).copy()
        # A parameter the residual does not depend on gives a zero column.
        # Damping it by zero makes the system singular; damping it by the
        # largest diagonal freezes it, which is the honest outcome.
        scale[scale <= 0.0] = float(np.max(scale, initial=1.0)) or 1.0

        stepped = False
        while lam < _LAMBDA_MAX:
            try:
                delta = np.linalg.solve(jtj + lam * np.diag(scale), -gradient)
            except np.linalg.LinAlgError:
                lam *= _LAMBDA_UP
                continue
            trial = z + delta
            r_trial = np.asarray(residual_fn(trial), dtype=np.float64)
            cost_trial = float(r_trial @ r_trial)
            if np.isfinite(cost_trial) and cost_trial < cost:
                dropped = cost - cost_trial
                step = float(np.linalg.norm(delta))
                z, r, cost = trial, r_trial, cost_trial
                lam = max(lam * _LAMBDA_DOWN, 1e-14)
                stepped = True
                if dropped <= ftol * max(cost, 1.0):
                    return LMResult(z, r, cost, it, True, "cost stopped falling")
                if step <= xtol * (float(np.linalg.norm(z)) + xtol):
                    return LMResult(z, r, cost, it, True, "step became negligible")
                break
            lam *= _LAMBDA_UP
        if not stepped:
            return LMResult(z, r, cost, it, True, "no downhill step exists")

    return LMResult(
        z, r, cost, max_iter, False,
        f"stopped at the {max_iter}-iteration limit without converging",
    )
