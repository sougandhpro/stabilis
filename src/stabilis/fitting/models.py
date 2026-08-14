"""Models a sigma-tau curve can be fitted to.

**A model returns a variance**, `f(tau, params) -> sigma2`, not a
deviation. That is the library-wide convention -- nothing here takes a
square root that a caller will square -- and it is also what the likelihood
in `objective.py` is written in, so no square root enters the fit at all.

A `Model` is a record, not an object with behaviour: a name, the function,
its parameter names, an initial-guess function and a one-line description.
There is no base class to subclass and no method to override. A customer's
own model is the same five fields, registered under the same registry the
built-ins use -- which is the plugin design rule of `registry.py` applied
here rather than asserted.

**Variances add**, so `sum_of` composes models with no new algebra: a
power-law floor plus a servo bump plus a drift is one call, and the
composite's parameters are the components' parameters with their model name
prefixed so two components can both have an `A`.

The built-ins are the shapes a tau-domain curve actually takes that
`stats/powerlaw.py` cannot fit. That module already handles anything linear
in its parameters -- the five IEEE 1139 noise powers and drift -- exactly,
by non-negative least squares, with no starting guess and no convergence
tolerance. **Use it in preference to this one whenever it applies.** What
is here is the rest: a correlation time, a resonance, a quantization floor,
and whatever the customer brings.

See theory/fit.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..registry import register

__all__ = ["Model", "sum_of", "model", "available_models",
           "POWERLAW", "DRIFT", "GAUSS_MARKOV", "QUANTIZATION", "BUMP"]


@dataclass(frozen=True)
class Model:
    """One fittable shape. Data only -- the behaviour is `variance`."""

    name: str
    # variance(tau, params) -> ndarray of the same shape as tau.
    variance: Callable[[np.ndarray, dict], np.ndarray]
    params: tuple[str, ...]
    # guess(tau, var) -> dict of starting values. A fit is only as good as
    # where it starts, and asking the user for a starting point they have
    # no way to know is how a fitting tool becomes unusable.
    guess: Callable[[np.ndarray, np.ndarray], dict]
    doc: str
    # Parameters that cannot be negative on physical grounds -- a noise
    # power, a correlation time, a width. `fit` applies these as default
    # bounds so the optimizer cannot wander somewhere the model does not
    # mean anything, and the caller can still override them.
    positive: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = set(self.positive) - set(self.params)
        if unknown:
            raise ValueError(
                f"model {self.name!r}: positive names {sorted(unknown)} are "
                f"not parameters of it ({list(self.params)})"
            )


def _as_tau(tau) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64)
    if np.any(tau <= 0.0):
        raise ValueError("model: tau must be positive")
    return tau


# --------------------------------------------------------------------------
# a single power law
# --------------------------------------------------------------------------


def _powerlaw(tau, p) -> np.ndarray:
    """sigma2(tau) = A * tau**b."""
    return p["A"] * _as_tau(tau) ** p["b"]


def _powerlaw_guess(tau, var) -> dict:
    """Least-squares line through log(var) against log(tau).

    Exact for a pure power law and harmless for anything else, which is all
    a starting point has to be. Note it is unweighted: at this stage the
    degrees of freedom are not yet in play and a weighted regression would
    only make a guess look more authoritative than it is.
    """
    ok = var > 0.0
    if np.count_nonzero(ok) < 2:
        return {"A": float(np.max(var, initial=1.0)), "b": -1.0}
    b, a = np.polyfit(np.log(tau[ok]), np.log(var[ok]), 1)
    return {"A": float(np.exp(a)), "b": float(b)}


POWERLAW = Model(
    name="powerlaw",
    variance=_powerlaw,
    params=("A", "b"),
    guess=_powerlaw_guess,
    doc="a single power law A*tau^b, the two-parameter baseline",
    positive=("A",),
)


# --------------------------------------------------------------------------
# deterministic linear frequency drift
# --------------------------------------------------------------------------


def _drift(tau, p) -> np.ndarray:
    """sigma2(tau) = D**2 * tau**2 / 2, for a linear frequency drift D.

    The exact Allan variance of `y(t) = D*t`, not an approximation, and the
    same expression as `powerlaw_coeffs.drift_coeff` times `D**2`. Fitted
    as `D` rather than `D**2` because the sign of a drift is physical even
    though the variance cannot see it; the fitted `D` is its magnitude.
    """
    return 0.5 * p["D"] ** 2 * _as_tau(tau) ** 2


def _drift_guess(tau, var) -> dict:
    """Attribute the longest tau entirely to drift -- an upper bound.

    Deliberately an over-estimate. Starting a drift term above the truth
    and letting it come down is stable; starting it at zero is not, because
    the gradient in `D` vanishes there.
    """
    return {"D": float(np.sqrt(2.0 * np.max(var)) / tau[-1])}


DRIFT = Model(
    name="drift",
    variance=_drift,
    params=("D",),
    guess=_drift_guess,
    doc="deterministic linear frequency drift, D**2*tau**2/2",
    positive=("D",),
)


# --------------------------------------------------------------------------
# first-order Gauss-Markov (Ornstein-Uhlenbeck) frequency noise
# --------------------------------------------------------------------------


def _gauss_markov(tau, p) -> np.ndarray:
    """Allan variance of an OU frequency process, exactly.

        rho = tau / T
        sigma2(tau) = (sigma_y**2 / rho**2)
                      * (2*rho - 3 + 4*exp(-rho) - exp(-2*rho))

    Derived in theory/fit.md from `R(t) = sigma_y**2 * exp(-|t|/T)` by
    integrating the averaged frequency twice; no approximation is made
    anywhere in it. The two limits are worth knowing because they are what
    a stability curve shows:

        tau >> T:  -> 2*sigma_y**2*T/tau     white FM, h0 = 4*sigma_y**2*T
        tau << T:  -> (2/3)*sigma_y**2*tau/T random-walk-like, slope +1

    so a Gauss-Markov process is a **knee**, and `T` is where it sits. That
    is the one number a customer wants from a disciplined oscillator or a
    temperature-driven ageing loop, and no power-law fit can produce it.

    **`y` here is the average over each tau0, not a point sample.** That is
    what a counter with a gate time produces, so it is right for real data
    -- but a *simulated* AR(1) sequence of point samples is a different
    process at short tau, and its exact Allan variance sits 50% above this
    at m=1, 3% at m=4, and within 1e-4 by m=64. Fitting simulated point
    samples across the whole grid is correctly rejected by the goodness of
    fit. See tests/test_fit_models.py, which pins both the closed form and
    the departure.

    The small-rho branch uses the series `2*rho**3/3 - rho**4/2 + ...`: the
    bracket is a difference of terms each of order 1 whose leading four
    orders cancel exactly, so evaluating it directly at rho = 1e-4 leaves
    nothing but rounding -- 1e-16 of cancellation error against a true
    value of 7e-13.
    """
    rho = _as_tau(tau) / p["T"]
    small = rho < 1e-2
    out = np.empty_like(rho)
    r = rho[~small]
    out[~small] = (2.0 * r - 3.0 + 4.0 * np.exp(-r) - np.exp(-2.0 * r)) / r**2
    r = rho[small]
    out[small] = r * (2.0 / 3.0 - 0.5 * r + 0.2 * r * r)
    return p["sigma_y"] ** 2 * out


def _gauss_markov_guess(tau, var) -> dict:
    """Solve the two asymptotes for the two parameters.

        var[0]  ~ (2/3) * sigma_y**2 * tau[0] / T      (short tau)
        var[-1] ~ 2 * sigma_y**2 * T / tau[-1]         (long tau)

    Their ratio gives `T**2` and their product gives `sigma_y**4`, so the
    guess is closed-form rather than a search. It is only right when the
    grid actually straddles the knee; when it does not, it still lands on
    the correct side of it, which is what the optimizer needs.
    """
    long_side = 0.5 * var[-1] * tau[-1]          # sigma_y**2 * T
    short_side = 1.5 * var[0] / tau[0]           # sigma_y**2 / T
    t = float(np.sqrt(long_side / short_side))
    return {"sigma_y": float(np.sqrt(long_side / t)), "T": t}


GAUSS_MARKOV = Model(
    name="gauss_markov",
    variance=_gauss_markov,
    params=("sigma_y", "T"),
    guess=_gauss_markov_guess,
    doc="first-order Gauss-Markov frequency noise: a knee at correlation time T",
    positive=("sigma_y", "T"),
)


# --------------------------------------------------------------------------
# a time-quantization floor
# --------------------------------------------------------------------------


def _quantization(tau, p) -> np.ndarray:
    """sigma2(tau) = 3 * sigma_x**2 / tau**2.

    The white-PM law, written in the time error that causes it rather than
    in `h_2`. A counter with a least-significant bit `q` and no
    interpolation contributes `sigma_x = q/sqrt(12)`, the standard deviation
    of a uniform distribution over one bit, so a fitted `sigma_x` is
    directly comparable with the instrument's datasheet resolution -- which
    is the point of parameterizing it this way.

    **The 3 belongs to the unmodified Allan deviation.** Fitting this to an
    `mdev` curve will return a `sigma_x` too large by a known factor and no
    warning, because the modified estimator's white-PM law falls as
    `tau**-3`. That is what `sum_of` and a `powerlaw` term are for.
    """
    return 3.0 * p["sigma_x"] ** 2 / _as_tau(tau) ** 2


def _quantization_guess(tau, var) -> dict:
    """Attribute the shortest tau entirely to quantization."""
    return {"sigma_x": float(np.sqrt(var[0] / 3.0) * tau[0])}


QUANTIZATION = Model(
    name="quantization",
    variance=_quantization,
    params=("sigma_x",),
    guess=_quantization_guess,
    doc="a white-PM floor set by a time quantization, 3*sigma_x**2/tau**2",
    positive=("sigma_x",),
)


# --------------------------------------------------------------------------
# a bump
# --------------------------------------------------------------------------


def _bump(tau, p) -> np.ndarray:
    """sigma2(tau) = A * exp(-ln(tau/tau_c)**2 / (2*w**2)).

    A Gaussian in `ln tau`, which is the right shape on a log-log plot: a
    servo, a thermal cycle or a mains-related line shows up as a localized
    hump of roughly constant fractional width, and this has exactly that
    property -- `w` is a width in decades, independent of where `tau_c`
    sits.

    It is a **descriptive** term, not a physical one. It says "there is a
    feature here, this big, this wide" and does not claim to know what
    causes it. Reported as such.
    """
    z = np.log(_as_tau(tau) / p["tau_c"]) / p["w"]
    return p["A"] * np.exp(-0.5 * z * z)


def _bump_guess(tau, var) -> dict:
    """Centre on the largest point; half a decade wide."""
    i = int(np.argmax(var))
    return {"A": float(var[i]), "tau_c": float(tau[i]), "w": 0.5 * np.log(10.0)}


BUMP = Model(
    name="bump",
    variance=_bump,
    params=("A", "tau_c", "w"),
    guess=_bump_guess,
    doc="a localized feature, Gaussian in ln(tau) -- descriptive, not physical",
    positive=("A", "tau_c", "w"),
)


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------


def sum_of(*models: Model, name: str | None = None) -> Model:
    """One model whose variance is the sum of the components' variances.

    Variances add under independence, which is the same assumption the
    power-law decomposition and the bench solver already rest on, so this
    needs no new algebra -- only bookkeeping. Each component's parameters
    are prefixed with its model name, so `sum_of(POWERLAW, BUMP)` has
    parameters `powerlaw_A`, `powerlaw_b`, `bump_A`, `bump_tau_c`,
    `bump_w`, and the two `A`s do not collide.

    The composite's guess calls each component's guess on the whole curve.
    That over-counts -- every component is started as if it explained
    everything -- and it is the right failure: starting each term too high
    and letting the fit share the curve out converges, while starting them
    at their fair share requires knowing the answer.
    """
    if len(models) < 2:
        raise ValueError("sum_of: give at least two models")
    seen = [m.name for m in models]
    if len(set(seen)) != len(seen):
        raise ValueError(
            f"sum_of: model names must be distinct, got {seen}; they are the "
            "parameter prefixes, so a repeat would collide"
        )

    prefixes = {m.name: tuple(f"{m.name}_{p}" for p in m.params) for m in models}
    params = tuple(p for m in models for p in prefixes[m.name])
    positive = tuple(
        f"{m.name}_{p}" for m in models for p in m.positive
    )

    def variance(tau, p):
        total = np.zeros_like(_as_tau(tau))
        for m in models:
            total = total + m.variance(
                tau, {q: p[f"{m.name}_{q}"] for q in m.params}
            )
        return total

    def guess(tau, var):
        out = {}
        for m in models:
            for q, value in m.guess(tau, var).items():
                out[f"{m.name}_{q}"] = value
        return out

    return Model(
        name=name or " + ".join(seen),
        variance=variance,
        params=params,
        guess=guess,
        doc="sum of " + ", ".join(seen),
        positive=positive,
    )


# --------------------------------------------------------------------------
# lookup
# --------------------------------------------------------------------------

for _m in (POWERLAW, DRIFT, GAUSS_MARKOV, QUANTIZATION, BUMP):
    register("model", _m.name)(_m)


def model(name: str) -> Model:
    """The `Model` registered under `name`, third-party plugins included."""
    from ..registry import get

    return get("model", name)


def available_models() -> list[str]:
    """Names of every registered model, built-in and third-party."""
    from ..registry import available

    return available("model")
