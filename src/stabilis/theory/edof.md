# Equivalent degrees of freedom

## What it is

A stability estimate is a mean of `M` squared coefficients. If those
coefficients were independent, the estimate would have `M` degrees of
freedom. They are not — an overlapping estimator reuses each phase sample
in many terms — so it has fewer, and the *equivalent* degrees of freedom
is the number that makes the chi-square sampling law come out right:

    Var(σ̂²) = 2·σ⁴ / η

That is the definition. Everything else on this page is a way of estimating
`η` from data, and the test suite validates them by measuring both sides.

## Why this is the differentiator

Stable32 identifies the noise type, looks `η` up in a table of closed forms
fitted to that power law, and applies it. The approach works when the noise
really is one of five pure power laws. Real oscillators are a mixture,
often on a drift, seen through a measurement system with noise of its own —
and when the noise is not one of the five, Stable32 has no answer at all.

`stabilis` estimates `η` from the estimator's own coefficient
autocovariance, so it never needs to know the noise type. That is the
wedge: **confidence intervals that do not require guessing the noise type
first.**

Greenhall's algorithm — the one that reproduces Stable32's `Min Sigma` and
`Max Sigma` columns — is implemented, in `greenhall`. It is a parity
feature, selected explicitly and never by `auto`, not the primary path.
The two are directly comparable, and the comparison is instructive: at
m = 1 the model-based answer is 17 % high for white FM and 24 % low for
random-walk FM against the exact values, while η₁ reads both correctly off
the data. See `greenhall` for the measurements.

## η₁ — from the coefficient autocorrelation

For a mean over `M` correlated terms of a stationary Gaussian process,

    Var(σ̂²) = (2·σ⁴/M) · Σ_k (1 − |k|/M) · ρ(k)²

Equating with the definition,

    η₁ = M / (1 + 2·Σ_{k=1}^{kmax} (1 − k/M)·ρ̂(k)²)

where `ρ̂` is the sample autocorrelation of the coefficients and
`kmax = min(M−1, 4m)`.

**The weight `1 − k/M` is not a tuning parameter** — it comes out of the
variance of a mean. The truncation is the choice, and it is essential.

### Why truncate

Summing `ρ̂(k)²` over all `M−1` lags adds a spurious contribution even when
the true autocorrelation is zero. Each `ρ̂(k)` has variance ≈ 1/M, so its
square has expectation ≈ 1/M, and `M−1` of them sum to ≈ 1. That fake 1
sits beside the real 1 in the denominator and roughly halves the estimate.
It is the same bias that motivates lag windows in spectral estimation
(Newey–West).

Overlapping coefficients are correlated only out to about the window
width — `2m` samples for the Allan family, `3m` for Hadamard and modified —
so lags past `4m` carry noise and nothing else. Discarding them removes the
bias and loses no signal.

### Why not taper to `kmax`

The natural-looking `1 − k/(kmax+1)`, which sends the last retained lag to
zero, was implemented first. It shrinks real correlation along with noise:
measured against Monte-Carlo truth it ran **6–25 % high** across every
power law, meaning error bars uniformly too narrow. With the theoretical
`1 − k/M` the same measurement centres near 1.0. There is no
positive-definiteness argument for the taper either, since the sum is over
*squares* and cannot go negative.

### Coefficients are not demeaned

A second or third difference of phase has zero mean by construction for any
process without a deterministic polynomial term, so subtracting a sample
mean would remove signal rather than a nuisance parameter.

## η₃ — the conservative fallback

    η₃ = max(M / (span·m), 1)

`span·m` is the width of one term in phase samples. Terms further apart
share no data, so this counts the non-overlapping terms and ignores the
information the overlapping ones do carry. It is therefore a **lower
bound**, not an estimate — for white FM it is about three times too small.

That is deliberate. It is the fallback when there are fewer than
`MIN_TERMS_FOR_ETA1 = 8` terms and the autocorrelation cannot be resolved,
and a fallback must err towards intervals that are too wide.

Which method was actually used is returned alongside the value and recorded
in provenance. "Silently fell back" is precisely the kind of decision an
ISO-17025 report has to be able to answer for.

## Measured behaviour

Against `2σ⁴/Var(σ̂²)` over 400–500 realizations of N = 512, for
α = 2, 1, 0, −1, −2:

| regime | η₁ / η_true |
|---|---|
| m ≤ N/32 (hundreds of terms) | within ±10 % |
| last one or two octaves | 0.7 – 1.4 |

At the last octave the estimate has a handful of degrees of freedom and the
Monte-Carlo truth is itself uncertain, so that spread is partly the
measurement's own.

## Validation status

**Validated.** `tests/test_edof.py` asserts: η₁ ≈ M for uncorrelated terms;
η₁ bounded in [1, M]; η₁ collapses for perfectly correlated terms; the
untruncated sum is demonstrably biased low; η₁ tracks Monte-Carlo truth
within ±10 % up to m = N/32 and within 0.6–1.5 everywhere; η₃ never exceeds
the truth; the FFT autocorrelation matches a direct sum; and the method
used is reported. The end-to-end check is coverage, in `tests/test_ci.py`.

Greenhall EDOF has its own page and its own suite,
`tests/test_greenhall.py`. What remains unvalidated there is agreement with
actual Stable32 output, which needs a reference session.

## References

* Percival, "On estimation of the wavelet variance", *Biometrika* 82, 1995.
* Percival & Walden, *Wavelet Methods for Time Series Analysis*, CUP 2000,
  §8.4 — η₁, η₂, η₃.
* Greenhall & Riley, "Uncertainty of Stability Variances Based on Finite
  Differences", PTTI 2003 — the algorithm Stable32 uses.
* Newey & West, *Econometrica* 55, 1987 — why the lag window is needed.
