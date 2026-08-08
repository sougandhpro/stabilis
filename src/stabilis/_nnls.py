"""Non-negative least squares, exactly, without scipy.

    minimise  ||A x - b||_2   subject to   x >= 0

This is the one numerical routine two very different features both need,
which is why it lives here rather than inside either of them.

**The power-law decomposition** fits `S_y(f) = sum_alpha h_alpha f^alpha`
to a measured sigma-tau curve. Each `h_alpha` is a noise power and cannot
be negative; parameterize the drift term as `D**2` and the model is
*linear* in its parameters, so the constrained fit is this problem and
nothing more. No optimizer, no starting guess, no convergence tolerance to
tune.

**The bench solver** (M5) does the same thing for a different physics.
Under independence a measurement of sources `i` and `j` through channel
`c` has `sigma2_meas = sigma2_i + sigma2_j + sigma2_c`, so a whole bench
of measurements is `A x = b` with `x >= 0` -- one solve per tau. The
three-cornered hat is the 3x3 case, the N-cornered hat is any larger
graph, and instrument noise-floor subtraction is what a back-to-back row
does. All three collapse into this function.

**Why exact and not an optimizer.** Lawson & Hanson's algorithm terminates
in finitely many steps at the true constrained minimum -- it is a
combinatorial search over which constraints are active, not an iterative
descent. A negative variance coming back from a three-cornered hat is then
unambiguously a statement about the measurement (the sources are not
independent, or one is misattributed) rather than possibly a statement
about the solver. That distinction is the whole value of the feature.

The solution carries its own proof: see `kkt_residual`. See
theory/nnls.md.
"""

from __future__ import annotations

import numpy as np

__all__ = ["nnls", "kkt_residual", "DEFAULT_MAX_ITER_FACTOR"]

# Outer iterations allowed, as a multiple of the number of unknowns. The
# algorithm is proven to terminate, so hitting this means a numerical
# breakdown rather than slow progress -- it raises instead of returning
# whatever it had reached.
DEFAULT_MAX_ITER_FACTOR = 3


def _gradient_tolerance(a: np.ndarray, b: np.ndarray) -> float:
    """When a component of `w = A.T @ (b - A x)` counts as zero.

        tol = 10 * eps * n * ||A||_inf * ||b||_inf

    **Both factors are needed, and this is where a plausible-looking
    version goes wrong.** `w` has the units of `A` times the units of `b`,
    so a threshold built from `A` alone -- or worse, floored at 1.0 to
    "be safe" -- is not a tolerance but an absolute cutoff in disguise.
    This library's variances are of order 1e-24, where `w` is of order
    1e-48; a threshold floored at 1e-15 declares every gradient zero and
    the solver returns x = 0 for every problem, silently and with a
    perfect-looking residual of ||b||.

    No floor, therefore. If `A` or `b` is exactly zero then `w` is exactly
    zero, `w > 0` is false, and the algorithm correctly stops at once.
    """
    row_sum = float(np.abs(a).sum(axis=1).max(initial=0.0))
    return 10.0 * np.finfo(np.float64).eps * a.shape[1] * row_sum * float(
        np.abs(b).max(initial=0.0)
    )


def nnls(a, b, tol: float | None = None, max_iter: int | None = None):
    """Lawson & Hanson's active-set algorithm, NNLS (1974, chapter 23).

        minimise ||A x - b||_2  subject to  x >= 0

    The variables are partitioned into a *passive* set, currently allowed
    to be non-zero, and an *active* set held at zero. Each outer step moves
    into the passive set the active variable whose gradient
    `w = A.T @ (b - A x)` most wants to grow; the inner loop then solves
    the unconstrained least-squares problem on the passive columns and, if
    that solution has gone negative anywhere, steps only as far as the
    first constraint it hits and returns those variables to the active
    set.

    Returns `(x, residual_norm)`. The partition is what terminates the
    algorithm: there are finitely many subsets and each step strictly
    decreases the residual, so no subset can repeat.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"nnls: A must be 2-D, got shape {a.shape}")
    if b.shape != (a.shape[0],):
        raise ValueError(f"nnls: b must have shape ({a.shape[0]},), got {b.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("nnls: A and b must be finite; no NaN or inf")

    n = a.shape[1]
    tol = _gradient_tolerance(a, b) if tol is None else float(tol)
    limit = DEFAULT_MAX_ITER_FACTOR * n if max_iter is None else int(max_iter)

    x = np.zeros(n, dtype=np.float64)
    passive = np.zeros(n, dtype=bool)

    for _ in range(limit):
        w = a.T @ (b - a @ x)
        w[passive] = -np.inf
        if not np.any(w > tol):
            return x, float(np.linalg.norm(b - a @ x))
        passive[int(np.argmax(w))] = True
        x = _inner(a, b, x, passive, limit)

    raise ValueError(
        f"nnls: no termination in {limit} iterations for a {a.shape[0]}x{n} "
        "system. The algorithm is proven to terminate, so this is numerical "
        "breakdown -- check the conditioning of A"
    )


def _inner(a, b, x, passive, limit: int) -> np.ndarray:
    """Solve on the passive columns, retreating to the first blocking constraint.

        s_P = argmin ||A_P s - b||,   s_Z = 0
        if all s_P > 0:  return s
        alpha = min over j in P with s_j <= 0 of x_j / (x_j - s_j)
        x <- x + alpha*(s - x),  drop the newly zeroed variables from P

    `alpha` is the largest step along `s - x` that keeps every variable
    non-negative, so the update lands exactly on a constraint boundary
    rather than near it. `passive` is modified in place, which is what the
    caller's loop iterates on.

    The drop threshold is computed here rather than passed in because it
    is in the units of `x`, not of the gradient the outer loop tests. In
    exact arithmetic the blocking variable lands on zero exactly; in
    floating point it lands within rounding of it, and if that rounding
    leaves it a hair positive the variable never leaves the passive set
    and the inner loop stalls.
    """
    eps_n = 10.0 * np.finfo(np.float64).eps * x.size
    for _ in range(limit):
        s = np.zeros_like(x)
        if np.any(passive):
            s[passive] = np.linalg.lstsq(a[:, passive], b, rcond=None)[0]
        if np.all(s[passive] > 0.0):
            return s
        blocked = passive & (s <= 0.0)
        denom = x[blocked] - s[blocked]
        usable = denom > 0.0
        if not np.any(usable):
            return s
        alpha = float(np.min(x[blocked][usable] / denom[usable]))
        x = x + alpha * (s - x)
        dropped = passive & (x <= eps_n * float(np.abs(x).max(initial=0.0)))
        x[dropped] = 0.0
        passive &= ~dropped

    raise ValueError(f"nnls: inner loop did not terminate in {limit} steps")


def kkt_residual(a, b, x) -> float:
    """How far `x` is from satisfying the optimality conditions. Zero is optimal.

    For this problem the Karush-Kuhn-Tucker conditions on
    `w = A.T @ (b - A x)` are

        x_j >  0  =>  w_j == 0     (free variable, gradient must vanish)
        x_j == 0  =>  w_j <= 0     (held at the boundary, and pushing into it)
        x_j >= 0                   (feasibility)

    and this returns the largest violation of any of them, scaled by
    `||A.T b||` so the number is dimensionless.

    It exists because it is an **independent certificate**: it is computed
    from `A`, `b` and `x` alone and knows nothing about how `x` was found,
    so it checks the answer rather than re-running the algorithm. The
    bench solver reports it beside every solution, because "the fit is
    optimal" is a claim a report should be able to substantiate rather
    than assert.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    w = a.T @ (b - a @ x)
    scale = max(float(np.abs(a.T @ b).max(initial=0.0)), 1.0)
    free = x > 0.0
    violation = max(
        float(np.abs(w[free]).max(initial=0.0)),
        float(np.max(w[~free], initial=0.0)),
        float(np.max(-x, initial=0.0)) * scale,
    )
    return violation / scale
