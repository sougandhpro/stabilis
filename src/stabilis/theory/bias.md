# Bias in the reported deviation

## The variance is unbiased. The deviation is not.

The square root is concave, so by Jensen's inequality `E[√V] < √(E[V])`.
For a chi-square sampling law with `η` degrees of freedom the factor is
exact:

    E[σ̂] / σ  =  c₄(η)  =  √(2/η) · Γ((η+1)/2) / Γ(η/2)

| η | c₄ | shortfall |
|---|---|---|
| 1 | 0.7979 | 20.2 % |
| 2 | 0.8862 | **11.4 %** |
| 10 | 0.9754 | 2.5 % |
| 50 | 0.9950 | 0.50 % |
| 100 | 0.99750 | 0.25 % |

Asymptotically `c₄ ≈ 1 − 1/(4η)`.

**So the bias is negligible at short τ and severe at long τ** — precisely
where an oscillator specification is written and where a customer reads the
number off the plot. The last ADEV point of a typical run is 11 % low on
average, and every tool in the field reports it that way.

## Two consequences run through the library

**Fit variances, never deviations.** Fitting a model to biased points
biases the parameters, and the bias varies point to point because `η`
does. `fit.py` (M4) therefore works in the variance domain throughout.

**The correction is offered, not applied.** `unbias_deviation(dev, eta)`
raises the value by `1/c₄`. It is opt-in because every other tool reports
the uncorrected deviation, so applying it silently would make `stabilis`
disagree with Stable32 by an amount that looks like a bug. When it *is*
applied it must be stated — IEEE Std 1139 clause 4.2 requires the
pretreatment of the data to be reported, and this is pretreatment of the
result.

## Implementation note

Evaluated through `lgamma` and a difference of logarithms, not as a ratio
of gamma functions: `Γ(η/2)` overflows float64 above `η ≈ 340`, which is
well inside the range of an ordinary record.

## Scope

B1, B2/B2_3, B3 and the TOTDEV and Thêo1 bias corrections are deferred to
M3/M7. Each exists to correct an estimator this library does not yet have,
and B1 is an additional noise-identification route that would sit beside
two methods already agreeing to within a tenth of a power law. What is
implemented here is the correction that changes numbers being reported
today.

## Validation status

**Validated.** `tests/test_noiseid.py` asserts the two closed forms
`c₄(2) = √π/2` and `c₄(1) = √(2/π)`; agreement with an exact two-step
recurrence

    c₄(η+2) = c₄(η)·(η+1)/√(η(η+2))

derived from `Γ(z+1) = z·Γ(z)`, which reaches any even `η` from `c₄(2)`
using multiplication and square roots alone and shares no code with the
`lgamma` path it checks; monotonicity and the bound `c₄ ∈ (0, 1]`;
agreement with the asymptotic form; survival at `η = 10⁵`; and a
Monte-Carlo measurement of `E[σ̂]/√(E[σ̂²])` against `c₄(η_true)` over 600
realizations, which agrees to within 0.02.

That recurrence check earned its place immediately: it caught a
transcribed constant for `c₄(100)` taken from a quality-control table
indexed by sample size `n` rather than by degrees of freedom `η = n−1`.

**Not yet validated:** B1/B2/B3 and the TOTDEV/Thêo1 corrections, none of
which are implemented.

## References

* NIST SP 1065 (Riley), §5.2 and §5.4 — bias functions and corrections.
* IEEE Std 1139-2022, clause 4.2 — pretreatment shall be stated.
