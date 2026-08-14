"""What "a good fit" means when the sampling law is chi-square.

A stability estimate is not Gaussian about its truth. With `eta` equivalent
degrees of freedom,

    eta * x / s  ~  chi2(eta),      x = measured variance, s = model variance

which is skewed, and increasingly so as `eta` falls at long tau -- exactly
where the interesting physics is and exactly where the customer's spec
lives. Least squares on `x` is the wrong objective there, not marginally
but structurally: it treats an interval that runs from 0.55*sigma to
2.2*sigma as though it were symmetric.

**The likelihood, and why Levenberg-Marquardt still applies.** Dropping
terms free of the parameters,

    -2 ln L = sum_i eta_i * ( ln s_i + x_i / s_i )

Subtract its own minimum, at `s_i = x_i`, and every term becomes
non-negative:

    D = sum_i eta_i * ( u_i - ln u_i - 1 ),        u_i = x_i / s_i

This is the Gamma deviance, equivalently the Itakura-Saito divergence.
Because each term is a non-negative number that vanishes only at a perfect
fit, it has a square root, and

    r_i = sign(1 - u_i) * sqrt( eta_i * (u_i - ln u_i - 1) )

is a genuine residual vector with `D = sum r_i**2`. So the exact
chi-square likelihood is minimized by an ordinary least-squares optimizer
with no Gaussian approximation anywhere. Near the optimum
`u - ln u - 1 -> (u-1)**2 / 2`, so `r_i -> -sqrt(eta_i/2) * (u_i - 1)` and
the residual passes smoothly through zero rather than turning a corner.

Two alternatives are offered, and neither is ever the default. They exist
so that a result stays comparable with a tool that computed it differently,
and the one actually used is recorded in provenance.

See theory/fit.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._specfun import chi2_cdf, digamma, trigamma

__all__ = ["OBJECTIVES", "GoodnessOfFit", "residuals", "deviance",
           "goodness_of_fit", "GOF_BAND"]

# "ml"       -- the exact chi-square likelihood above. The default.
# "wlsq"     -- least squares weighted by Var(x) = 2 s**2 / eta. Agrees with
#               "ml" to first order and differs where it matters.
# "logspace" -- least squares on ln x, with the exact chi-square moments of
#               the logarithm rather than their leading terms.
OBJECTIVES = ("ml", "wlsq", "logspace")

# Probability band on the deviance outside which parameters are not quoted.
# Two-sided, because a deviance far *below* expectation is also a finding:
# it means the model has more freedom than the data can constrain, or the
# degrees of freedom are over-estimated. Set wide deliberately -- this is a
# refusal to print, not a hypothesis test, and a report that cries wolf is
# no more evidence than one that cannot fail.
GOF_BAND = (0.001, 0.999)

# Below this |u - 1| the deviance's bracket is evaluated by series. The
# three terms of `u - ln u - 1 = e**2/2 - e**3/3 + e**4/4 - ...` are each
# of order `e` while their combination is of order `e**2`, so the direct
# form loses half its digits near a good fit -- which is where a fit spends
# every iteration after the first few.
_SERIES_EDGE = 1e-4


def _divergence(u: np.ndarray) -> np.ndarray:
    """`u - ln u - 1`, accurately, including where it nearly vanishes.

    Non-negative for every `u > 0`, zero only at `u = 1`, and convex. The
    series branch is `e**2 * (1/2 - e/3 + e**2/4)` with `e = u - 1`.
    """
    if np.any(u <= 0.0):
        raise ValueError(
            "objective: the model returned a non-positive variance, which "
            "has no chi-square likelihood. Constrain the parameters, or use "
            "a model that cannot go negative"
        )
    e = u - 1.0
    near = np.abs(e) < _SERIES_EDGE
    out = np.empty_like(u)
    out[~near] = u[~near] - np.log(u[~near]) - 1.0
    en = e[near]
    out[near] = en * en * (0.5 - en / 3.0 + 0.25 * en * en)
    return out


def deviance(measured, model_var, eta) -> float:
    """`D = sum eta_i (u_i - ln u_i - 1)`, twice the log-likelihood ratio.

    The goodness-of-fit statistic, and the objective `residuals("ml", ...)`
    minimizes. Asymptotically chi-square on `n - p` degrees of freedom when
    the points are independent; `goodness_of_fit` handles the case where
    they are not, which for an overlapping estimator is always.
    """
    x = np.asarray(measured, dtype=np.float64)
    s = np.asarray(model_var, dtype=np.float64)
    w = np.asarray(eta, dtype=np.float64)
    return float(np.sum(w * _divergence(x / s)))


def residuals(objective: str, measured, model_var, eta) -> np.ndarray:
    """Residual vector whose sum of squares is the objective.

        ml:       sign(1-u) sqrt( eta (u - ln u - 1) )
        wlsq:     sqrt(eta/2) (u - 1)
        logspace: (ln x - psi(eta/2) + ln(eta/2) - ln s) / sqrt(psi'(eta/2))

    with `u = x / s`. The signs differ between the first two -- one is
    signed by which side of the model the measurement fell on, the other by
    the residual itself -- and it does not matter, because only the squares
    enter. What does matter is that all three have unit variance at the
    truth, so the sum of squares of any of them is comparable with `n - p`.

    **`wlsq` weights by the model variance, not the measured one.** Using
    `x**2` in the denominator lets a point that happened to come out low
    claim a small uncertainty and pull the fit down onto itself, which
    biases the whole curve towards the low realizations. It is the classic
    Pearson-versus-Neyman chi-square trap and it is worth a line of code to
    avoid.

    **`logspace` uses the exact moments of `ln chi2`, not `2/eta`.** The
    logarithm of a chi-square variable is biased: `E[ln(chi2_eta/eta)]` is
    `psi(eta/2) - ln(eta/2)`, which is -0.58 at eta = 2 rather than 0, and
    its variance is `psi'(eta/2)`, which is 1.64 rather than 1. Ignoring
    either -- as a naive log-log fit does -- pulls every long-tau point
    down and weights it as though it were three times better known than it
    is. This is the same Gamma-function correction that makes the last
    point of an ADEV curve 11.4% low in expectation.
    """
    if objective not in OBJECTIVES:
        raise ValueError(
            f"objective must be one of {OBJECTIVES}, got {objective!r}")
    x = np.asarray(measured, dtype=np.float64)
    s = np.asarray(model_var, dtype=np.float64)
    w = np.asarray(eta, dtype=np.float64)
    u = x / s

    if objective == "ml":
        return np.sign(1.0 - u) * np.sqrt(w * _divergence(u))
    if objective == "wlsq":
        if np.any(s <= 0.0):
            raise ValueError("objective: the model returned a non-positive "
                             "variance, which cannot weight anything")
        return np.sqrt(0.5 * w) * (u - 1.0)
    if np.any(x <= 0.0):
        raise ValueError(
            "objective='logspace' needs the logarithm of every measured "
            "variance, and at least one is not positive. A negative estimate "
            "is a real result -- see multi/cross.py -- but it has no log; "
            "use objective='ml'"
        )
    bias = digamma(0.5 * w) - np.log(0.5 * w)
    return (np.log(x) - bias - np.log(s)) / np.sqrt(trigamma(0.5 * w))


@dataclass(frozen=True)
class GoodnessOfFit:
    """Whether the model describes the data, and on what dof."""

    deviance: float
    # Effective degrees of freedom and scale after accounting for the
    # correlation between taus. With independent points these are `n - p`
    # and 1.0 exactly.
    dof: float
    scale: float
    p_value: float
    ok: bool
    note: str

    @property
    def reduced(self) -> float:
        """Deviance per effective degree of freedom. Near 1 for a good fit."""
        return self.deviance / (self.scale * self.dof)


def goodness_of_fit(
    dev: float, n_points: int, n_params: int, corr=None, hat=None
) -> GoodnessOfFit:
    """The deviance read against the distribution it actually has.

    With independent points `D ~ chi2(n - p)`. The points of an overlapping
    estimator are not independent, and the correction is Satterthwaite's.
    Near the optimum the fitted residual is `(I - H) r` with `H` the hat
    matrix of the residual Jacobian, and `r` has covariance `P`, the
    measured correlation matrix -- exactly, because `taucov` imposes a
    diagonal that leaves each residual unit-variance. So `D` is a quadratic
    form in correlated normals with

        M = (I - H) P,    E[D] = tr(M),    Var[D] = 2 tr(M @ M)

    and matching those two moments to `scale * chi2(dof)` gives

        scale = tr(M @ M) / tr(M),    dof = tr(M)**2 / tr(M @ M)

    At `P = I` that is `scale = 1`, `dof = n - p`, exactly.

    **The projector cannot be summarized as "p dimensions".** Fitting
    removes `tr(H P)`, not `p`, and when the points are correlated the
    model lies along the smooth directions where `P` carries most of its
    mass, so `tr(H P)` runs to several times `p`. Assuming otherwise
    predicted a mean deviance of 8.0 for a two-parameter fit to ten
    correlated octave taus where the measured mean was 5.8 +/- 0.4 -- five
    standard errors out, and in the direction that makes a correct fit look
    suspiciously good. `corr` and `hat` are therefore required together:
    there is no defensible way to use one without the other, and offering
    a `corr`-only path would only offer that mistake back.

    `ok` is False on both tails. Too large is the familiar case -- the
    model does not describe the data. Too small means the model has more
    freedom than the curve can constrain, or the degrees of freedom feeding
    it are over-estimated; either way the parameters are not defensible and
    the caller should say so instead of printing them.
    """
    if (corr is None) != (hat is None):
        raise ValueError(
            "goodness_of_fit: `corr` and `hat` go together. The correlation "
            "alone cannot say how much of the deviance the fit absorbed, and "
            "assuming it absorbed `p` is the error this argument exists to "
            "avoid")
    n = int(n_points)
    dof = float(n - int(n_params))
    if dof <= 0.0:
        return GoodnessOfFit(
            deviance=float(dev), dof=0.0, scale=1.0, p_value=float("nan"),
            ok=False,
            note=f"{n} points for {n_params} parameters leaves no degrees of "
                 "freedom, so the fit cannot be tested at all",
        )

    scale, note = 1.0, ""
    if corr is not None:
        m = (np.eye(n) - np.asarray(hat, dtype=np.float64)) @ np.asarray(
            corr, dtype=np.float64)
        tr1, tr2 = float(np.trace(m)), float(np.trace(m @ m))
        if tr1 <= 0.0 or tr2 <= 0.0:
            note = ("the correlation left no positive effective degrees of "
                    "freedom, so the deviance is read naively")
        else:
            scale, dof = tr2 / tr1, tr1 * tr1 / tr2
            if abs(dof - (n - int(n_params))) > 0.05 * n:
                note = (f"the taus are correlated, so the deviance is read "
                        f"against {dof:.1f} effective degrees of freedom "
                        f"rather than {n - int(n_params)}")

    cdf = float(chi2_cdf(float(dev) / scale, dof))
    lo, hi = GOF_BAND
    ok = lo <= cdf <= hi
    if not ok:
        note = (
            f"the deviance is {'far above' if cdf > hi else 'far below'} what "
            f"{dof:.1f} effective degrees of freedom admit"
            + (": the model does not describe this curve"
               if cdf > hi else
               ": the model has more freedom than the curve constrains, or "
               "the degrees of freedom are over-estimated")
        )
    return GoodnessOfFit(
        deviance=float(dev), dof=dof, scale=scale, p_value=1.0 - cdf,
        ok=ok, note=note,
    )
