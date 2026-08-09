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

## Validation status

**Validated.** `tests/test_wavelet.py`: the filter algebra for every family;
the Allan identity to 1e-14 over eleven octaves, from both phase and
frequency input; shift equivariance of the coefficients as bit equality;
shift invariance of the biased variance to 1e-14; the deliberate
*non*-invariance of the unbiased one; and the drift annihilation table
above.

## Not yet

* **LA8**, the least-asymmetric variant. It differs from D8 only in which
  spectral roots are kept, and choosing them needs the least-asymmetry
  criterion rather than the minimum-phase one. D8 is here and covers the
  same `K = 4`; LA8 would buy a more symmetric impulse response, not more
  vanishing moments.
* **The wavelet EDOF path.** `confidence` currently treats a wavelet result
  through the coefficient-based estimators; Percival & Walden's own
  spectral EDOF is a separate piece.
* **Mondal & Percival's gap-tolerant estimator**, the imputation-free route
  to a wavelet variance on gappy records.

## References

* D. B. Percival and A. T. Walden, *Wavelet Methods for Time Series
  Analysis*, CUP 2000 — the transform, the unbiased estimator, `L_j`.
* D. B. Percival and P. Guttorp, "Long-memory processes, the Allan variance
  and wavelets", 1994 — the Allan equivalence.
