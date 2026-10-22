# The MODWT and the wavelet variance

## The identity

**The Haar wavelet variance is the Allan variance.** For a fractional
frequency series,

    avar(tau_j) = 2 * nu^2_Haar(tau_j)

This is exact, not approximate and not statistical. The retained Haar
coefficients at scale `2^{j-1}` *are* the overlapping Allan second
differences, so the two estimators are one sum written two ways. Measured
across eleven octaves on a 4096-point record, `wdev(..., "haar")` and
`oadev` agree to **1e-15** — the project plan asked for 1e-12.

So the classical Allan variance is the `K = 1` member of one framework
rather than a separate thing that happens to resemble it. `wdev` returns
the deviation already scaled by `sqrt(2)` so the two plot on top of each
other.

## What the higher orders buy

**Vanishing moments.** A filter with `K` vanishing moments annihilates any
polynomial trend of degree `< K`. Measured as the inflation of the coarsest
scale when a linear frequency drift is added to fixed noise:

| filter | K | inflation |
|---|---|---|
| Haar | 1 | **1 636 161x** |
| D4 | 2 | **1.00x** |
| D6 | 3 | 1.00x |

D4 removes a linear frequency drift *with no pre-processing at all*. Drift
removal is otherwise a separate and contestable operation on the data —
here it is a property of the filter.

**Convergence for steeper noise.** The wavelet variance converges for
`alpha > -2K-1` rather than AVAR's `alpha > -3`, so flicker-walk and
random-run FM stop diverging.

## The filters are computed, not transcribed

Daubechies coefficients are exactly the sort of number that gets copied
wrong, and a filter with one bad low-order digit still looks like a filter.
So they are derived here by spectral factorization of the half-band
polynomial

    P(y) = sum_{j<k} C(k-1+j, j) y^j

keeping the roots inside the unit circle and multiplying by `(1 + z^-1)^k`.
The tests then **certify** them against the defining algebra:

* `||g|| = 1` and `sum(g) = sqrt(2)` (Percival & Walden's normalization);
* `sum(h) = 0` and `||h|| = 1`;
* orthonormality at every even shift;
* `K` vanishing moments, `sum_l h_l l^p = 0` for `p < K`.

A coefficient set satisfying all of those *is* the Daubechies filter,
however it was obtained. That is a stronger guarantee than a citation.

The overall sign of `h` is conventional — `h_l = (-1)^l g_{L-1-l}` gives
`[+, -]` for Haar — and the variance squares it away.

## MODWT, not DWT

The maximal-overlap transform does not decimate, so it is defined at every
sample:

    W_j[t] = sum_l h~_l V_{j-1}[(t - 2^{j-1} l) mod N]

with `h~ = h/sqrt(2)`. The consequence that matters:

| quantity | under a shift of the record |
|---|---|
| the coefficients | **exactly equivariant** — `roll(W(x)) == W(roll(x))`, bit for bit |
| the biased variance | **invariant to 2e-16** |
| the unbiased variance | *not* invariant, and should not be |

The decimating DWT fails the first outright, which would make a variance
depend on where the record happened to start.

The third row is a distinction worth keeping straight rather than filing as
a bug. The unbiased estimator drops the coefficients that wrapped around
the circular boundary, and that set is **positional** — rotating the record
discards different data, so the estimate moves by ordinary sampling
variation (a few percent at the coarsest scale, where a couple of hundred
coefficients survive). Keeping the wrapped coefficients would make it
perfectly invariant *and* wrong: it treats the end of the record as
adjacent to its start.

    L_j = (2^j - 1)(L - 1) + 1        coefficients dropped at level j

## Confidence intervals, and gaps

The retained coefficients are carried on the result, so `confidence` serves
the wavelet variance through the same single path it serves every other
estimator with. That is not an approximation borrowed from the Allan
family: `eta1` is an EDOF read off the autocorrelation of the terms being
averaged, and at scale `j` those terms *are* these coefficients. With the
Haar filter the whole interval — EDOF and both bounds — comes back equal to
the overlapping Allan interval to 1e-12, the same identity showing up in
the error bar.

`span_factor` is `2(L-1)`, the equivalent filter's width per unit of `m`,
since `L_j / m -> 2(L - 1)`. Haar gives 2, which is the Allan family's own
value.

One thing here was checked rather than reasoned. `eta1` truncates its lag
sum at `4m`, a width calibrated against the Allan window; the equivalent
filter at level `j` is `2(L-1)m` wide, so the concern was that the flat
truncation discards real correlation and returns bounds too narrow.

Median `eta1` over 400 realizations of N = 4096 white FM, against a
Monte-Carlo `2 sigma^4 / Var(sigma_hat^2)` from the same 400, as a ratio to
that truth. Scales to `m = N/32`, past which the truth is itself uncertain:

| filter | `kmax = 4m`, as shipped | `kmax = 4(L-1)m`, the filter's own width |
|---|---|---|
| haar | 0.86 – 1.15 | identical, `L - 1 = 1` |
| d4 | 0.84 – 1.13 | 0.67 – 1.12 |
| d6 | 0.83 – 1.15 | 0.65 – 1.09 |
| d8 | 0.86 – 1.15 | 0.68 – 1.07 |

Widening the sum to the filter's own width is **worse**, and the two
columns are not symmetric about 1: the wide one drifts steadily low towards
the coarse scales (d8 runs 0.95, 1.07, 1.04, 0.93, 0.84, 0.76, 0.83, 0.68),
which is bounds too *wide* rather than too narrow, but wrong in a way that
grows with tau. The extra lags carry more `rho_hat` noise than signal —
exactly the bias `eta1`'s truncation exists to remove. The flat `4m` stays,
and does about as well on the longer filters as on Haar.

**Gaps are masked, and only masked.** A gap is a non-finite sample; the
filter spreads it over every coefficient whose support covers it and those
are dropped, which is `gap_policy='mask'` arriving by another route. The
count reaches provenance. `'segment'` is refused rather than accepted and
ignored, because each run would land on its own dyadic grid and pooling
across grids that do not line up is not a decision the estimator can make
for the caller.

The price is steep at the coarse end and worth stating plainly: one bad
sample in 2048 removes the coarsest `d4` scale outright, because the
equivalent filter there is 1534 taps and every coefficient the boundary
left covers the gap. That is the argument for the gap-filling milestone,
not an argument for improvising an interpolation here -- and for the
imputation-free alternative, `gappy_wavelet_variance`, which keeps every
observed pair instead of every clean coefficient (see `gappy`).

## Validation status

**Validated, level included, for every filter.** `tests/test_wavelet.py`:
the filter algebra for every
family; the Allan identity to 1e-14 over eleven octaves, from both phase and
frequency input; shift equivariance of the coefficients as bit equality;
shift invariance of the biased variance to 1e-14; the deliberate
*non*-invariance of the unbiased one; the drift annihilation table above;
the EDOF against Monte-Carlo truth for `d4` and `d8`; the Haar confidence
interval against the overlapping Allan one; and the gap accounting.
`tests/test_docs_and_cli.py` runs the same identity through the CLI, since
that is where a user meets it.

### How the level is pinned

Until 2026-09-01 that list contained no check on the absolute level of any
filter but `haar`. The drift ratio, the EDOF ratio and the shift
invariances are every one of them scale-free, so a constant multiplicative
error in the `d4`/`d6`/`d8` variance passed all of them — measured, not
supposed: inflating the non-Haar variance by 3% leaves the whole of the
rest of `tests/test_wavelet.py` green.

Three checks close it, and none is a Monte-Carlo estimate of the truth.

1. **The equivalent filter, exactly.** `modwt` applied to a unit impulse
   returns the level-`j` equivalent wavelet filter. `validate/expected.py`
   builds that filter independently by the upsampling cascade,

       a_j = up(h~, 2^{j-1}) * c_{j-1},   c_j = up(g~, 2^{j-1}) * c_{j-1}

   with `c_0 = [1]`, and the two agree to 1e-17 over four families and six
   levels. That pins the shape: the `1/sqrt(2)` per level, the `2^{j-1}`
   stride, and where the wavelet filter enters rather than the scaling one.
   Length `L_j = (2^j - 1)(L - 1) + 1` is asserted with it, so a cascade
   run one level too deep — which agrees on its leading taps — still fails.

2. **The norm, exactly.** `||a_j||² = 2^-j`, the orthonormality of the
   transform read as an energy statement. It is what fixes the *level*
   rather than the shape: for white FM `E[ν_j²] = var(y)·||a_j||²`, so
   `2ν_j²` is the Allan variance at `τ_j = 2^{j-1} τ₀` for every filter,
   not only for `haar`. The published closed form `h₀/(2τ)` is asserted
   against it at 1e-12 for all four families.

3. **The estimator, end to end.** The two above pin the filters; the
   aggregation stacked on them — the boundary drop, the mean of squares,
   the τ assignment — is checked against

       E[ν_j²] = ‖a_j ⊛ h‖²

   where `h` is the Kasdin–Walter impulse response. `y = h ⊛ w` makes each
   coefficient one fixed linear functional of the driving white noise, so
   the expectation is *exact* — only the estimator side is an ensemble
   average, and the band is that ensemble's own standard error. Scored at
   `d4`, `d6`, `d8` × white PM, white FM, random-walk FM, and in the
   validation report over all four filters at every level.

This is the case the analytic tier exists for: Stable32 has no wavelet
variance, so no interop fixture could exist and none should be manufactured.
The same machinery does *not* reach TOTVAR or the Thêo family — both extend
the record by reflection, so they are linear functionals but not
convolutions, and still have no absolute `E[σ²(τ)]` anywhere in the
repository. MTIE is out permanently.

## Not yet

* **LA8**, the least-asymmetric variant. It differs from D8 only in which
  spectral roots are kept, and choosing them needs the least-asymmetry
  criterion rather than the minimum-phase one. D8 is here and covers the
  same `K = 4`; LA8 would buy a more symmetric impulse response, not more
  vanishing moments.
* **Percival & Walden's spectral EDOF.** The coefficient-based path above
  is measured and honest, but their mode-2 and mode-3 estimators derive the
  degrees of freedom from an assumed or fitted spectrum instead, and are
  the better answer where the record is short and `eta1` has too few terms
  to read an autocorrelation from.

## References

* D. B. Percival and A. T. Walden, *Wavelet Methods for Time Series
  Analysis*, CUP 2000 — the transform, the unbiased estimator, `L_j`.
* D. B. Percival and P. Guttorp, "Long-memory processes, the Allan variance
  and wavelets", 1994 — the Allan equivalence.
