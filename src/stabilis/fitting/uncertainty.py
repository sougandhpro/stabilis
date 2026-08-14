"""How well the parameters are known -- twice, by two different routes.

**The sandwich.** The objective in `objective.py` treats the taus as
independent. They are not: the points of an overlapping estimator come
from the same phase samples, and `stats/taucov.py` measures how much they
move together. When the working model for the errors is wrong but the
estimating equation is still unbiased -- which is exactly this situation --
the covariance of the estimator is not the inverse Hessian but

    Cov(theta) = (Jr.T Jr)^-1 (Jr.T Cov(r) Jr) (Jr.T Jr)^-1

with `Jr` the Jacobian of the residual vector and `Cov(r) = G C G` carrying
the measured covariance `C` through the residual's own sensitivity `G` to
each measurement. Substitute the independence assumption, `C = diag(2
s^2/eta)`, and it collapses to the familiar `2 (Jr.T Jr)^-1` -- exactly,
which is a bit-checkable test rather than a plausibility argument.

Written generically in the residual rather than derived per objective, so
`ml`, `wlsq` and `logspace` all get it from one implementation and none of
them can drift away from the others.

**The bootstrap.** Resample the curve from the fitted model's own sampling
law and refit, many times. To be worth having it must honour both the
marginal law -- chi-square with `eta` degrees of freedom, not Gaussian --
and the correlation, so the draw is a Gaussian copula: correlated normals
through their own CDF into uniform, then through the chi-square quantile
into a variance. The marginals are then exact and the rank correlation
matches what was measured.

**Both are reported and neither is reduced to the other.** They agree when
the model is right and the linearization holds. They disagree when it is
not -- a badly non-linear parameter, a model that cannot describe the
curve -- and that disagreement is information the user wants rather than a
discrepancy to hide behind an average.

See theory/fit.md.
"""

from __future__ import annotations

import numpy as np

from .._specfun import chi2_ppf, norm_cdf
from .lm import numeric_jacobian
from .objective import residuals

__all__ = ["sandwich_covariance", "naive_covariance", "residual_hat",
           "copula_draw", "bootstrap_parameters", "nearest_correlation"]

# Relative step for differentiating the residual with respect to a
# measurement. Larger than the parameter step because the residual is
# nearly linear in `x` near the optimum, so truncation error is negligible
# and rounding is what matters.
_X_STEP = 1e-6

# Eigenvalue floor, as a fraction of the largest eigenvalue, when a
# measured correlation matrix has to be made positive definite for a
# Cholesky factor. Any repair is reported rather than done quietly.
_PSD_FLOOR = 1e-10


def _residual_jacobians(objective, measured, model_variance, eta, theta):
    """`(Jr, G)`: how the residual moves with the parameters, and with the data.

    `Jr[i, p] = d r_i / d theta_p` by central differences, stepped in
    proportion to each parameter's own magnitude. That relative step is why
    these functions can be handed the parameters directly rather than the
    optimizer's transformed coordinates: a variance of 1e-24 and a
    correlation time of 1e3 both get a fractional step of 6e-6, and neither
    is displaced somewhere the model does not exist. A parameter that is
    exactly zero has no scale of its own and falls back to one.

    `G[i] = d r_i / d x_i`, and it is a vector rather than a matrix because
    residual `i` depends on measurement `i` and no other. That also makes
    it one pair of evaluations instead of `n`: perturbing every measurement
    at once perturbs every residual independently.
    """
    x = np.asarray(measured, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    scale = np.where(np.abs(theta) > 0.0, np.abs(theta), 1.0)
    jr = numeric_jacobian(
        lambda t: residuals(objective, x, model_variance(t), eta), theta,
        scale=scale,
    )
    h = _X_STEP * np.abs(x)
    s = model_variance(theta)
    g = (residuals(objective, x + h, s, eta)
         - residuals(objective, x - h, s, eta)) / (2.0 * h)
    return jr, g


def _equilibrate(jr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(Jr / d, d)` with `d` the column norms. Scaling, not regularization.

    **Without this the covariance silently comes back rank-deficient.** The
    parameters are not commensurable -- a fitted variance of 1e-22 against a
    fitted exponent of -1 -- so `Jr` has one column of order 1e23 beside one
    of order 10, and `Jr.T Jr` spans 43 orders of magnitude. A pseudo-inverse
    then discards the smaller singular value as numerical noise, and every
    standard error comes back as exactly zero: a plausible-looking answer
    that is entirely wrong. Normalizing the columns first makes the inverse
    well conditioned and is undone exactly afterwards, so nothing is
    approximated.

    A column that is identically zero -- a parameter the residual does not
    depend on -- is left alone rather than divided by, and its standard
    error comes back zero, which is then the true answer rather than an
    artefact.
    """
    d = np.linalg.norm(jr, axis=0)
    d = np.where(d > 0.0, d, 1.0)
    return jr / d, d


def _sandwich(jr: np.ndarray, g: np.ndarray, cov_x: np.ndarray) -> np.ndarray:
    """`(Jr.T Jr)^-1 (Jr.T G C G Jr) (Jr.T Jr)^-1`, the meat in the bread."""
    scaled, d = _equilibrate(jr)
    bread = np.linalg.pinv(scaled.T @ scaled)
    weighted = scaled * g[:, None]
    meat = weighted.T @ cov_x @ weighted
    return (bread @ meat @ bread) / np.outer(d, d)


def naive_covariance(objective, measured, model_variance, eta, theta_free,
                     cov_x):
    """Parameter covariance with the taus' correlations set to zero.

    The same sandwich as `sandwich_covariance`, over the same measured
    per-point variances, with only the off-diagonal of `C` deleted. That is
    what makes the ratio of the two a clean statement about the
    *correlation* and nothing else -- had this used a differently-derived
    diagonal, say `2 s^2 / eta` from the fitted model rather than from the
    measurement, the ratio would also carry the fit's own residual and
    would not come back to exactly one when there is no correlation to
    find.
    """
    jr, g = _residual_jacobians(objective, measured, model_variance, eta,
                                theta_free)
    return _sandwich(jr, g, np.diag(np.diag(np.asarray(cov_x, float))))


def sandwich_covariance(objective, measured, model_variance, eta, theta_free,
                        cov_x):
    """Parameter covariance using the measured cross-tau covariance `cov_x`.

    Reduces to `naive_covariance` exactly when `cov_x` is diagonal -- not
    approximately, and there is a test.
    """
    jr, g = _residual_jacobians(objective, measured, model_variance, eta,
                                theta_free)
    return _sandwich(jr, g, np.asarray(cov_x, dtype=np.float64))


def residual_hat(objective, measured, model_variance, eta, theta_free):
    """`H = Jr (Jr.T Jr)^-1 Jr.T`, the projector onto the fitted directions.

    What the goodness-of-fit test needs and cannot get from a parameter
    count: with correlated points the fit absorbs `tr(H P)` of the
    deviance rather than `p` of it, and the two differ by a factor of
    several. Idempotent and symmetric by construction, with trace equal to
    the number of free parameters.
    """
    jr, _ = _residual_jacobians(objective, measured, model_variance, eta,
                                theta_free)
    scaled, _ = _equilibrate(jr)
    return scaled @ np.linalg.pinv(scaled.T @ scaled) @ scaled.T


def nearest_correlation(corr) -> tuple[np.ndarray, float]:
    """A positive-definite correlation matrix, and how far it had to move.

    A measured correlation matrix need not be positive definite: the lag
    sum behind it is truncated, and truncation is not a positive operation.
    The Cholesky factor the copula needs does not exist in that case, so
    the negative eigenvalues are floored and the diagonal renormalized to
    one. Returns the repaired matrix and the most negative eigenvalue
    encountered, which is zero when nothing was done -- the caller reports
    it rather than repairing quietly.
    """
    p = np.asarray(corr, dtype=np.float64)
    p = 0.5 * (p + p.T)
    values, vectors = np.linalg.eigh(p)
    worst = float(values.min())
    if worst > 0.0:
        return p, 0.0
    floor = _PSD_FLOOR * float(values.max())
    repaired = vectors @ np.diag(np.maximum(values, floor)) @ vectors.T
    d = np.sqrt(np.diag(repaired))
    return repaired / np.outer(d, d), worst


def copula_draw(model_variance_at_fit, eta, chol, rng) -> np.ndarray:
    """One resampled curve, chi-square in the margins and correlated across tau.

        z ~ N(0, P)      via the Cholesky factor of P
        u = Phi(z)                   uniform, but not independent
        x = s * chi2_ppf(u, eta) / eta

    The middle step is the copula: `u` has exactly uniform margins whatever
    the correlation is, so the last step gives each point exactly the
    sampling law it should have while the joint structure is inherited from
    `P`. Resampling from a multivariate normal instead would get the
    correlation right and the tails wrong, which at eta = 2 is most of the
    distribution.
    """
    s = np.asarray(model_variance_at_fit, dtype=np.float64)
    z = chol @ rng.standard_normal(s.size)
    u = np.clip(norm_cdf(z), 1e-12, 1.0 - 1e-12)
    return s * np.asarray(chi2_ppf(u, np.asarray(eta, float))) / np.asarray(eta,
                                                                            float)


def bootstrap_parameters(model_variance_at_fit, eta, corr, refit, n_draws,
                         rng) -> tuple[np.ndarray, int, float]:
    """Refit `n_draws` resampled curves; return the samples that converged.

    `refit(x)` returns the free parameter vector for one resampled curve,
    or `None` where the fit failed to converge. **Failures are counted and
    dropped, not retried with a different seed** -- a bootstrap that quietly
    resamples until it succeeds reports the spread of the fits that happened
    to work, which is narrower than the truth and biased towards wherever
    the optimizer is well behaved.

    Returns `(samples, n_failed, worst_eigenvalue)`.
    """
    repaired, worst = nearest_correlation(corr)
    chol = np.linalg.cholesky(repaired)
    samples, failed = [], 0
    for _ in range(int(n_draws)):
        x = copula_draw(model_variance_at_fit, eta, chol, rng)
        got = refit(x)
        if got is None:
            failed += 1
        else:
            samples.append(got)
    return (np.array(samples) if samples else np.zeros((0, 0)), failed, worst)
