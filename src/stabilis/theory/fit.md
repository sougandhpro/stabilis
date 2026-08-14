# Fitting a model to a σ(τ) curve

## Use the exact solve when you can

`stats/powerlaw.py` fits the five IEEE 1139 noise powers and a drift term
by **non-negative least squares**. That model is linear in its parameters,
so the fit is exact: no starting guess, no convergence tolerance, no local
minimum, and a KKT certificate attached to every answer. Anything that
decomposition covers should go through it.

This page is about the rest — a Gauss–Markov correlation time, a servo
bump, a quantization floor, a customer's own model — where the parameters
enter non-linearly and an optimizer is unavoidable.

## The objective

A stability estimate is not Gaussian about its truth. With `η` equivalent
degrees of freedom,

    η·x / s ~ χ²(η)

where `x` is the measured variance and `s = σ²_model(τ, θ)`. That law is
skewed, and increasingly so as `η` falls at long τ — which is where the
specification usually is. Least squares on `x` is not a slightly worse
choice there; it is the wrong objective for a distribution that at η = 2
runs from 0.55σ to 2.2σ.

Dropping terms free of the parameters,

    −2 ln L = Σᵢ ηᵢ ( ln sᵢ + xᵢ / sᵢ )

Its minimum over `s` is at `sᵢ = xᵢ`. Subtracting that minimum turns the
objective into a sum of **non-negative** terms,

    D = Σᵢ ηᵢ ( uᵢ − ln uᵢ − 1 ),      uᵢ = xᵢ / sᵢ

the Gamma deviance, equivalently the Itakura–Saito divergence.

## Why Levenberg–Marquardt applies to it

A non-negative term has a square root, so

    rᵢ = sign(1 − uᵢ)·√( ηᵢ (uᵢ − ln uᵢ − 1) )

is a genuine residual vector with `D = Σ rᵢ²`. **The exact χ² likelihood is
therefore minimized by an ordinary least-squares optimizer**, with no
Gaussian approximation introduced anywhere. Near the optimum
`u − ln u − 1 → (u−1)²/2`, so `rᵢ → −√(ηᵢ/2)·(uᵢ − 1)` and the residual
passes smoothly through zero rather than turning a corner there.

`u − ln u − 1` is evaluated by series below `|u − 1| = 1e-4`. Written
directly it is a difference of two terms of order `e` whose leading orders
cancel: at `e = 1e-8` float64 returns exactly zero where the true value is
5e-17, and that is precisely where an optimizer spends every iteration
after the first few.

## Two alternatives, neither ever the default

They exist so a result stays comparable with a tool that computed it
differently. Which one was used is recorded in provenance.

**`wlsq`** — least squares weighted by `Var(x) = 2s²/η`. It weights by the
**model** variance, not the measured one. Using `x²` in the denominator
lets a point that happened to come out low claim a small uncertainty and
drag the fit onto itself; that is the Pearson-versus-Neyman χ² trap, and
it biases the whole curve towards the low realizations.

**`logspace`** — least squares on `ln x`, with the **exact** χ² moments of
the logarithm:

    E[ln x] = ln s + ψ(η/2) − ln(η/2)
    Var[ln x] = ψ′(η/2)

not their leading terms 0 and 2/η. At η = 2 the true values are −0.577 and
1.645, so a naive log-log fit pulls every long-τ point down by 42 % and
weights it as though it were 1.6× better known than it is. This is the same
Γ-function correction that makes the last point of an ADEV curve 11.4 % low
in expectation — see `theory/bias.md`.

## Parameter uncertainty, twice

### The sandwich

The objective treats the τ as independent. They are not — see
`theory/taucov.md` — so the covariance of the estimator is not the inverse
Hessian but

    Cov(θ) = (Jr′Jr)⁻¹ (Jr′ · G C G · Jr) (Jr′Jr)⁻¹

with `Jr` the Jacobian of the residual vector, `G` its sensitivity to each
measurement, and `C` the measured cross-τ covariance. Substituting the
independence assumption `C = diag(2s²/η)` collapses this to `2(Jr′Jr)⁻¹`
**exactly**, which is asserted as a bit-level test rather than argued.

Both are computed and both are reported. Their ratio is the number that
matters: it says how much confidence the correlation was costing, measured
rather than assumed. On white FM over ten octave τ it runs about 1.25–1.35,
so error bars quoted the naive way are a quarter to a third too narrow.

The sandwich is computed in the optimizer's own transformed coordinates and
carried back by the delta method. That is not an approximation — the
sandwich is equivariant under a smooth change of variables — and it removes
the need for a per-parameter differencing step, since in the transformed
space every positive parameter is a logarithm and one relative step suits
all of them.

### The bootstrap

Resample the curve from the fitted model's own sampling law and refit. To
be worth having it must honour both the marginal law and the correlation,
so the draw is a **Gaussian copula**:

    z ~ N(0, P)  →  u = Φ(z)  →  x* = s · χ²⁻¹(u, η) / η

The margins are then exactly χ² whatever the correlation is, and the rank
correlation matches what was measured. Drawing from a multivariate normal
instead would get the correlation right and the tails wrong, which at η = 2
is most of the distribution.

Resampled fits that do not converge are **counted and dropped, never
retried**. A bootstrap that resamples until it succeeds reports the spread
of the fits that happened to work, which is narrower than the truth.

Sandwich and bootstrap agree when the model is right and the linearization
holds, and disagree when a parameter is badly non-linear or the model is
misspecified. Neither is reduced to the other, because that disagreement is
the finding.

## Goodness of fit, and the refusal

`D` is asymptotically χ² — on how many degrees of freedom is the subtle
part. Near the optimum the fitted residual is `(I − H)r` with `H` the hat
matrix of `Jr`, and `r` has covariance `P`, the measured correlation matrix
exactly (because `taucov` imposes a diagonal that leaves each residual
unit-variance). So `D` is a quadratic form in correlated normals:

    M = (I − H)P,   E[D] = tr M,   Var[D] = 2 tr M²
    scale = tr M² / tr M,   dof = (tr M)² / tr M²

At `P = I` that is `scale = 1`, `dof = n − p`, exactly.

**The projector cannot be summarized as "p dimensions."** Fitting removes
`tr(HP)`, not `p`, and with correlated points the model lies along the
smooth directions where `P` carries most of its mass, so `tr(HP)` runs to
several times `p`. Assuming otherwise predicted a mean deviance of 8.0 for
a two-parameter fit to ten correlated octave τ where the measured mean was
5.8 ± 0.4 — five standard errors out, in the direction that makes a correct
fit look suspiciously good. The correlation matrix and the projector are
required together for that reason.

`FitResult.ok` is False when the optimizer did not converge **or** the
deviance is outside the band its degrees of freedom admit, and `stabilis
fit` prints the deviance instead of the parameters in that case. Both tails
count: a deviance far below expectation means the model has more freedom
than the curve constrains, or the degrees of freedom are over-estimated.

## The models

All return a **variance**, `f(τ, params) → σ²`. That is the library-wide
convention, and the likelihood above is written in the variance, so no
square root enters the fit at all.

**Gauss–Markov** is the one worth deriving here, because it is the shape no
power-law fit can produce. For an Ornstein–Uhlenbeck frequency process with
`R(t) = σ_y² e^{−|t|/T}`, the average over one τ has

    Var(ȳ) = 2σ_y²(ρ − 1 + e^{−ρ}) / ρ²
    Cov(ȳ₀, ȳ₁) = σ_y²(1 − e^{−ρ})² / ρ²,      ρ = τ/T

and σ²_A = Var(ȳ) − Cov(ȳ₀, ȳ₁) gives

    σ²(τ) = (σ_y²/ρ²)·(2ρ − 3 + 4e^{−ρ} − e^{−2ρ})

with no approximation. Its limits are what a stability plot shows:

* `τ ≫ T`: → `2σ_y²T/τ`, white FM with `h₀ = 4σ_y²T`;
* `τ ≪ T`: → `(2/3)σ_y²τ/T`, rising with slope +1 like random-walk FM.

So a Gauss–Markov process is a **knee**, and `T` is where it sits — the one
number a customer wants out of a disciplined oscillator or a
temperature-driven ageing loop. The small-ρ branch is evaluated by series
because the bracket's leading four orders cancel exactly.

The others: `powerlaw` (`A·τ^b`), `drift` (`D²τ²/2`, bit-exact against
`powerlaw_coeffs.drift_coeff`), `quantization` (`3σ_x²/τ²`, parameterized
in the time error so a fitted `σ_x` compares directly with a datasheet's
resolution — note the 3 belongs to the *unmodified* ADEV), and `bump`, a
Gaussian in `ln τ` that is descriptive rather than physical and is reported
as such. Variances add, so `sum_of` composes them.

## Bounds, and what they cost

Bounds are **transformations, not projections** — `exp` for one-sided, a
logistic for two-sided. The optimizer runs unconstrained, cannot step
outside the model's domain, and never reports a parameter sitting exactly
on a bound as though that were a measurement. The price is that a parameter
whose true value *is* the bound can only be approached. That is the right
trade: a variance that is genuinely zero is a job for the exact
non-negative solve in `stats/powerlaw.py`.

## There is no scipy fast path

scipy would be faster and is already an optional dependency elsewhere here.
But a fit ends up in a signed report, and a report whose numbers depend on
what happens to be installed on the machine that produced it is not a
report. scipy appears only in `tests/test_fit_models.py`, as a differential
check that `scipy.optimize.least_squares` reaches the same minimum — better
evidence than a speedup.

## Validation status

**Validated on exact identities and against Monte-Carlo truth**, in
`tests/test_fit.py` and `tests/test_fit_models.py`.

* The optimizer recovers parameters from a curve the model generated to
  1e-8, including a three-parameter composite started 200× wrong in the
  correlation time.
* The sandwich reduces to `2(Jr′Jr)⁻¹` to machine precision when the
  covariance is the independent diagonal.
* The deviance is calibrated: for white FM over ten octave τ the mean
  deviance is 5.96 ± 0.36 against a predicted 5.99, and the p-values are
  uniform (quartiles 0.29 / 0.48 / 0.68 against 0.25 / 0.50 / 0.75).
* Parameter recovery: `b = 1.006 ± 0.038` against a truth of 1 for
  random-walk FM over eight octave τ.
* The Gamma-deviance bracket is checked against a 50-digit `Decimal`
  oracle, where the direct float64 form returns exactly zero.

**Known and deliberate**: fitting a single power law across the *smallest*
τ of a synthesized random-walk-FM record is correctly rejected, because the
discrete synthesizer's ADEV at m = 1 and m = 2 sits 50 % and 13 % above the
continuous-time τ¹ line. That is a real departure of the data from the
model, not a defect of the fit — see the synthesizer's own calibration
notes.

Not yet validated: fitted parameter uncertainties against an independent
implementation. No other tool computes the cross-τ correlation, so there is
nothing to differ from; `VALIDATION-DEBT.md` records it.

## References

* Itakura & Saito 1968, and Cressie & Read, "Multinomial goodness-of-fit
  tests", *JRSS B* 46, 1984 — the divergence and its deviance reading.
* Huber, "The behavior of maximum likelihood estimates under nonstandard
  conditions", 1967, and White, *Econometrica* 48, 1980 — the sandwich.
* Satterthwaite, "An approximate distribution of estimates of variance
  components", *Biometrics Bulletin* 2, 1946 — the moment-matched χ².
* Marquardt, "An algorithm for least-squares estimation of nonlinear
  parameters", *SIAM J. Appl. Math.* 11, 1963 — the `diag(J′J)` damping.
* Nelsen, *An Introduction to Copulas*, 2nd ed., 2006 — the Gaussian
  copula construction used for the bootstrap draw.
