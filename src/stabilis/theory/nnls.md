# Non-negative least squares

## The problem, and why it is the same problem twice

    minimise  ||A x - b||₂   subject to   x ≥ 0

Two features in this library reduce to exactly this, which is why the
routine lives on its own in `_nnls.py` rather than inside either of them.

**Power-law decomposition.** Fit `S_y(f) = Σ_α h_α f^α` to a measured
σ(τ) curve. Each `h_α` is a noise power and cannot be negative. Every
variance in the family is *linear* in the `h_α`, and if drift enters as
`D²` rather than `D` the drift term is linear too — so the whole fit is
this problem, with no optimizer, no starting guess and no convergence
tolerance to tune.

**The bench solver (M5).** Under independence a measurement of sources `i`
and `j` through channel `c` satisfies

    σ²_meas = σ²_i + σ²_j + σ²_c

so a whole bench of measurements is `A x = b`, `x ≥ 0`, one solve per τ.
The three-cornered hat is the 3×3 case, the N-cornered hat is any larger
graph, and instrument noise-floor subtraction is what a back-to-back row
(both endpoints on the same source) does. Three features, one solve.

## Why exact, not an optimizer

Lawson & Hanson's algorithm is a **finite combinatorial search** over which
constraints are active, not an iterative descent. It reaches the true
constrained minimum in finitely many steps and then stops.

That matters more here than speed. A negative variance falling out of a
three-cornered hat is a famous and *informative* result — it says the
sources are not independent, or one has been misattributed, or the
measurement system is contributing more than assumed. For that reading to
be available, the clamping at zero has to be provably the constrained
optimum. With a descent method, "this source came back at zero" could
always instead be "the solver stopped early", and the physics conclusion
evaporates.

## The algorithm

Variables are partitioned into a **passive** set, currently allowed to be
non-zero, and an **active** set held at zero.

    w = Aᵀ(b − A x)                       gradient of ½||Ax − b||²,  negated

Outer step: move into the passive set the active variable with the largest
positive `w` — the one most wanting to grow. Stop when no active variable
does.

Inner step: solve the *unconstrained* least-squares problem on the passive
columns alone,

    s_P = argmin ||A_P s − b||,   s_Z = 0

If every passive component of `s` is positive, take it. Otherwise step only
as far as the first constraint in the way,

    α = min over j ∈ P with s_j ≤ 0 of  x_j / (x_j − s_j)
    x ← x + α(s − x)

which lands exactly *on* a boundary rather than near it, and return the
newly-zeroed variables to the active set.

Termination: there are finitely many subsets, and the residual strictly
decreases at each outer step, so no subset can recur.

### Two thresholds, and they have different units

    w_tol = 10 · eps · n · ‖A‖_∞ · ‖b‖_∞      when a gradient counts as zero
    x_tol = 10 · eps · n · ‖x‖_∞              when a variable counts as zero

Both factors in `w_tol` are needed, and this is where the obvious version
goes wrong. `w = Aᵀ(b − Ax)` carries the units of `A` times those of `b`,
so a threshold built from `A` alone — or floored at 1.0 to "be safe" —
is an absolute cutoff wearing a relative disguise. The variances this
library handles are of order 1e-24, where `w` is of order 1e-48; a
threshold floored at 1e-15 declares every gradient zero, and the solver
returns `x = 0` for every problem with a residual of `‖b‖` that looks
entirely reasonable on a plot. That bug was written here and caught by the
scale-invariance test below.

There is no floor. If `A` or `b` is exactly zero then `w` is exactly zero,
`w > 0` is false, and the algorithm correctly stops at once.

`x_tol` is separate because it lives in the units of `x`, not of the
gradient. In exact arithmetic the blocking variable lands on zero exactly;
in floating point it lands within rounding, and if that rounding leaves it
a hair positive the variable never leaves the passive set and the inner
loop stalls.

## The solution carries its own proof

`kkt_residual(A, b, x)` returns the largest violation of the
Karush–Kuhn–Tucker conditions,

    x_j >  0  ⟹  w_j = 0      free variable, gradient must vanish
    x_j =  0  ⟹  w_j ≤ 0      held at the boundary, and pushing into it
    x_j ≥  0                  feasibility

scaled by `||Aᵀb||` so the number is dimensionless. Zero means optimal.

It is computed from `A`, `b` and `x` alone and knows nothing about how `x`
was found, so it **checks the answer instead of re-running the algorithm**.
The bench solver reports it beside every solution, because "this fit is
optimal" is a claim an ISO-17025 report should substantiate rather than
assert.

## Validation status

**Validated against an exhaustive oracle**, in `tests/test_nnls.py`.

The NNLS optimum is always the unconstrained least-squares fit on *some*
subset of the variables, with every component non-negative. For small
problems that subset can simply be enumerated — 2ⁿ of them — and the best
feasible one is the answer, by construction and with no algorithm
involved. Over 400 random problems of up to 9×6 this implementation
attains the exhaustive optimum every time, and its KKT residual stays
below 1e-14.

**Scale invariance is asserted as an exact identity**, and it is the
sharpest test here: `nnls(cA, cb) = nnls(A, b)` and
`nnls(A, cb) = c·nnls(A, b)`, over eighteen decades of `c`. Nothing
absolute can survive it, which is how the tolerance bug above was found.

Also asserted: the three-cornered-hat system solves exactly; a problem
whose unconstrained answer is negative clamps to the analytic constrained
one; a problem whose unconstrained answer is already non-negative agrees
with `numpy.linalg.lstsq`; rank-deficient and duplicate-column systems
return a feasible optimum rather than raising; and the KKT certificate is
shown to *fail* on a perturbed and on an infeasible point, since a
certificate that cannot fail certifies nothing.

A differential test against `scipy.optimize.nnls` runs where scipy is
installed. It asserts this implementation's residual is **never worse**,
rather than asserting equality: on 2 of those 400 random problems scipy's
result had a strictly larger residual than the exhaustive optimum, so
equality is not the right claim to make. scipy is an optional dependency
here and is never required for a correct answer.

## References

* Lawson & Hanson, *Solving Least Squares Problems*, Prentice-Hall 1974,
  chapter 23 — the algorithm, and the finite-termination proof.
* Bro & de Jong, "A fast non-negativity-constrained least squares
  algorithm", *J. Chemometrics* 11, 1997 — the caching speedup, not used
  here: these systems have a handful of unknowns and clarity wins.
