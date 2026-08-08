"""Non-negative least squares, against an oracle that uses no algorithm.

The NNLS optimum is always the unconstrained least-squares fit on *some*
subset of the variables, with every component non-negative. For a small
problem that subset can be enumerated -- all 2**n of them -- and the best
feasible one is the answer by construction. That brute-force search is the
oracle here: it shares no code and no idea with the active-set algorithm
it checks, in the same spirit as the exact-rational oracle the estimators
are checked against.
"""

import itertools

import numpy as np
import pytest

from stabilis._nnls import kkt_residual, nnls


def _exhaustive(a, b):
    """The optimum by enumeration. No algorithm, just every possibility.

    Deliberately the stupidest correct thing: fit each subset of columns
    without constraints, discard the fits that went negative, keep the
    smallest residual. Exponential, and irrelevant at these sizes.
    """
    n = a.shape[1]
    best_r, best_x = np.inf, None
    for size in range(n + 1):
        for subset in itertools.combinations(range(n), size):
            x = np.zeros(n)
            if subset:
                x[list(subset)] = np.linalg.lstsq(a[:, subset], b, rcond=None)[0]
            if np.all(x >= -1e-12):
                r = float(np.linalg.norm(a @ x - b))
                if r < best_r:
                    best_r, best_x = r, x
    return best_r, best_x


# --- cases with an answer known on paper ---------------------------------


def test_the_three_cornered_hat_solves_exactly():
    """The M5 bench solver's smallest case, and the reason for this routine.

        s1 + s2 = 3,  s2 + s3 = 5,  s1 + s3 = 4

    Summing gives 2(s1+s2+s3) = 12, so s3 = 3, s1 = 1, s2 = 2 -- all
    positive, so the constraint is inactive and NNLS must return the exact
    solve. The matrix is the unsigned incidence matrix of a triangle, which
    is exactly why three sources are enough and four in a ring are not.
    """
    a = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    b = np.array([3.0, 5.0, 4.0])
    x, rnorm = nnls(a, b)
    assert x == pytest.approx([1.0, 2.0, 3.0], abs=1e-12)
    assert rnorm == pytest.approx(0.0, abs=1e-12)


def test_a_negative_unconstrained_answer_clamps_to_zero():
    """min ||[1;1] x - [-1;-1]|| over x >= 0 has its optimum at the boundary.

    Unconstrained the answer is -1. Constrained it is 0, and the residual
    is then ||b|| = sqrt(2). Both are exact, so this is asserted tightly.
    """
    a = np.array([[1.0], [1.0]])
    b = np.array([-1.0, -1.0])
    x, rnorm = nnls(a, b)
    assert x == pytest.approx([0.0])
    assert rnorm == pytest.approx(np.sqrt(2.0), rel=1e-15)


def test_an_already_positive_answer_is_just_least_squares():
    """Where no constraint binds, the constrained fit must equal the plain one.

    If it did not, the constraint machinery would be perturbing answers it
    has no business touching -- which on a bench with no negative
    variances is every answer.
    """
    rng = np.random.default_rng(1)
    a = np.abs(rng.standard_normal((12, 3))) + 0.5
    x_true = np.array([1.0, 2.0, 3.0])
    b = a @ x_true
    x, _ = nnls(a, b)
    assert x == pytest.approx(np.linalg.lstsq(a, b, rcond=None)[0], rel=1e-9)
    assert x == pytest.approx(x_true, rel=1e-9)


def test_a_hat_with_a_dependent_source_returns_a_boundary_solution():
    """The famous negative-variance case, and what the user should see.

    If the measured pair variances cannot be explained by three
    independent sources -- here the A-B measurement is smaller than
    physically possible given the other two -- the unconstrained solve
    wants a negative variance. The constrained answer pins it at zero and
    leaves a residual, and *that residual is the diagnostic*: it says the
    independence assumption failed, which is a statement about the bench
    and not about the solver.
    """
    a = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    b = np.array([0.1, 5.0, 4.0])
    x, rnorm = nnls(a, b)
    assert np.all(x >= 0.0)
    assert rnorm > 0.0
    assert kkt_residual(a, b, x) < 1e-12


# --- against the exhaustive oracle ---------------------------------------


def test_matches_the_exhaustive_optimum_on_random_problems():
    """400 random systems, every one solved by enumeration as well.

    This is the whole correctness claim for the module. The oracle knows
    nothing of active sets, gradients or tolerances -- it fits every subset
    and picks the best feasible one -- so agreement is evidence rather
    than a restatement.
    """
    rng = np.random.default_rng(0)
    worst_kkt = 0.0
    for _ in range(400):
        m, n = int(rng.integers(2, 10)), int(rng.integers(1, 7))
        a = rng.standard_normal((m, n))
        b = rng.standard_normal(m)
        x, rnorm = nnls(a, b)
        best, _ = _exhaustive(a, b)
        assert np.all(x >= 0.0)
        assert rnorm <= best + 1e-9, f"suboptimal by {rnorm - best:.3e}"
        worst_kkt = max(worst_kkt, kkt_residual(a, b, x))
    assert worst_kkt < 1e-13, f"worst KKT residual {worst_kkt:.3e}"


def test_badly_scaled_problems_still_reach_the_optimum():
    """The tolerance is relative to ||A||, so the scale must not matter.

    A power-law design matrix spans many decades between its columns --
    tau**-2 against tau**+1 -- so this is the realistic case, not a corner
    one. An absolute zero threshold fails here.
    """
    rng = np.random.default_rng(7)
    for exponent in (-12, -6, 6, 12):
        a = rng.standard_normal((8, 4)) * 10.0**exponent
        b = rng.standard_normal(8) * 10.0**exponent
        x, rnorm = nnls(a, b)
        best, _ = _exhaustive(a, b)
        assert rnorm <= best * (1.0 + 1e-9) + 1e-9 * abs(best)
        assert np.all(x >= 0.0)


def test_rank_deficient_and_duplicate_columns_are_handled():
    """Two identical columns make the solution non-unique but the fit unique.

    The bench matrix is singular whenever the measurement graph has no odd
    cycle, which is a setup a user can and does build. It must return a
    feasible optimum rather than raising -- the identifiability warning is
    the graph layer's job, not this one's.
    """
    a = np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    b = np.array([2.0, 1.0, 1.0])
    x, rnorm = nnls(a, b)
    best, _ = _exhaustive(a, b)
    assert np.all(x >= 0.0)
    assert rnorm <= best + 1e-9


def test_a_zero_right_hand_side_gives_a_zero_solution():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((6, 4))
    x, rnorm = nnls(a, np.zeros(6))
    assert x == pytest.approx(np.zeros(4))
    assert rnorm == pytest.approx(0.0)


# --- the optimality certificate ------------------------------------------


def test_the_kkt_residual_is_zero_at_the_solution_and_not_elsewhere():
    """It has to be able to fail, or it certifies nothing.

    Perturbing a free variable breaks the vanishing-gradient condition;
    making a variable negative breaks feasibility. Both must register.
    """
    rng = np.random.default_rng(5)
    a = np.abs(rng.standard_normal((10, 3))) + 0.5
    b = a @ np.array([1.0, 2.0, 3.0])
    x, _ = nnls(a, b)
    assert kkt_residual(a, b, x) < 1e-12

    perturbed = x.copy()
    perturbed[0] += 0.5
    assert kkt_residual(a, b, perturbed) > 1e-3

    infeasible = x.copy()
    infeasible[1] = -1.0
    assert kkt_residual(a, b, infeasible) > 1e-3


# --- input handling ------------------------------------------------------


def test_shapes_and_finiteness_are_checked():
    a = np.ones((3, 2))
    with pytest.raises(ValueError, match="must be 2-D"):
        nnls(np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match=r"b must have shape"):
        nnls(a, np.ones(5))
    with pytest.raises(ValueError, match="must be finite"):
        nnls(a, np.array([1.0, np.nan, 1.0]))
    with pytest.raises(ValueError, match="must be finite"):
        nnls(np.array([[1.0, np.inf], [1.0, 1.0], [1.0, 1.0]]), np.ones(3))


# --- interoperability ----------------------------------------------------


def test_never_worse_than_scipy():
    """Differential, where scipy is installed. Note what is asserted.

    Not equality: on 2 of the 400 random problems in the oracle test above,
    `scipy.optimize.nnls` returned a point whose residual was strictly
    larger than the exhaustive optimum, so equality is not a true claim to
    make. What must hold is that this implementation is never the worse of
    the two, and the oracle test is what establishes that it is right.

    scipy is an optional dependency and is never required for a correct
    answer -- see tests/test_core_deps.py.
    """
    optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(11)
    for _ in range(200):
        m, n = int(rng.integers(2, 12)), int(rng.integers(1, 9))
        a = rng.standard_normal((m, n))
        b = rng.standard_normal(m)
        ours, r_ours = nnls(a, b)
        theirs = optimize.nnls(a, b)[0]
        r_theirs = float(np.linalg.norm(a @ theirs - b))
        assert np.all(ours >= 0.0)
        assert r_ours <= r_theirs + 1e-9, f"worse by {r_ours - r_theirs:.3e}"


def test_the_answer_is_exactly_scale_invariant():
    """nnls(cA, cb) == nnls(A, b) and nnls(A, cb) == c * nnls(A, b).

    Both are exact properties of the problem, and together they are the
    sharpest possible test that no threshold in the implementation is
    secretly absolute. This caught a real one: the gradient tolerance was
    built from ||A|| alone and floored at 1.0, so on a problem scaled to
    1e-12 -- which is a *typical* variance in this field, not a corner
    case -- every gradient tested as zero and the solver returned x = 0
    with a residual of ||b|| that looked entirely reasonable.
    """
    rng = np.random.default_rng(13)
    a = rng.standard_normal((9, 4))
    b = rng.standard_normal(9)
    base, _ = nnls(a, b)
    assert np.any(base > 0.0), "degenerate fixture: nothing to compare"
    for c in (1e-18, 1e-6, 1e6, 1e18):
        both, _ = nnls(a * c, b * c)
        assert both == pytest.approx(base, rel=1e-9)
        rhs, _ = nnls(a, b * c)
        assert rhs == pytest.approx(base * c, rel=1e-9)
