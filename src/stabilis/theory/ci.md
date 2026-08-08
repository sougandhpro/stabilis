# Confidence intervals

## The sampling law

A variance estimate with `η` equivalent degrees of freedom satisfies

    η·σ̂² / σ²  ~  χ²(η)

so a two-sided interval at confidence `level`, with `α = 1 − level`, is

    lower = η·σ̂² / χ²⁻¹(1 − α/2, η)
    upper = η·σ̂² / χ²⁻¹(α/2, η)

Note the crossed indices: the **upper** bound on the variance comes from
the **lower** quantile. Getting that backwards gives an inverted interval
that still looks plausible drawn on a log axis, so there is a test for it.

## The interval is not symmetric

It is strongly asymmetric, and increasingly so as `η` falls — which is
exactly what happens at long τ, which is where the interesting physics
lives. At `η = 2` the 68.27 % interval on the **deviation** runs

    0.737·σ̂   to   2.406·σ̂

The upper arm is 5.3 times the lower one. This is why nothing in this
library ever offers `σ ± δ`. A symmetric error bar at the last few τ —
exactly where a customer's specification lives — misrepresents the
distribution by more than a factor of two.

Deviation bounds are the square roots of the variance bounds. Square root
is monotone, so coverage is preserved exactly. They are *not* centred on
`σ̂`, and that is correct rather than a rounding artefact.

## Bias, which is a separate matter

The interval above is on the variance and is unbiased in the sense that
matters. The *deviation* is not: since √ is concave,

    E[σ̂] = σ·√(2/η)·Γ((η+1)/2)/Γ(η/2)

At `η = 2` that factor is 0.8862, so the last ADEV point is **11.4 % low on
average** — again, exactly where the specification lives. The correction
belongs in `bias.py` (M2) and is why anything fitted in this library is
fitted to variances, never to deviations.

## Which quantile function

`_specfun.chi2_ppf`, pure Python, no scipy. Wilson–Hilferty start then
Newton with a maintained bracket; the normal quantile is Wichura AS241 at
~1e-16 rather than Acklam at ~1e-9, because the claim being made is
metrology-grade. Bisection alone would be wrong here: `η` reaches 10⁴ on a
long record, where the distribution is narrow and far from the origin.

Measured accuracy of the underlying incomplete gamma is ≥ 12.4 correct
digits at the worst point on a wide grid, and ≥ 14 everywhere an interval
actually evaluates it.

## Measured coverage

The claim an assessor cares about is not that the formula is right but that
the intervals contain the truth as often as they say. Measured over 800
realizations of N = 512 at each power law:

| nominal | measured range |
|---|---|
| 68.27 % (1σ) | 0.61 – 0.77 |
| 95.45 % (2σ) | 0.90 – 0.99 |

The spread is worst at the last octave, where `η` is small and η₁ is
working from an autocorrelation it can barely resolve. **No competing tool
publishes this measurement**, which is the point of making it.

## Validation status

**Validated.** `tests/test_ci.py` asserts: the interval brackets the
estimate; the closed form at `η = 2` where `χ²⁻¹(p,2) = −2·ln(1−p)`;
monotonicity in both `level` and `η`; the asymmetry figures quoted above;
scale invariance; that `η₃` really does give wider intervals than `η₁`;
that the same code path serves the Hadamard family; and measured coverage,
quickly in the default run and over the full sweep under `-m slow`.

**Not yet validated:** agreement with Stable32's `Min Sigma` / `Max Sigma`
columns, which needs Greenhall EDOF (M3), and the deviation bias correction
(`bias.py`).

## References

* NIST SP 1065 (Riley), §5 — confidence intervals and the χ² construction.
* Percival & Walden, *Wavelet Methods for Time Series Analysis*, CUP 2000.
* Wichura, "Algorithm AS 241: The percentage points of the normal
  distribution", *Applied Statistics* 37, 1988.
