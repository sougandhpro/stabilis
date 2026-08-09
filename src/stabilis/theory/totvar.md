# Total variance (TOTVAR / TOTDEV)

## Definition

    Tot var(tau) = 1/(2 tau^2 (N-2)) * sum_{i=2}^{N-1}
                       [x*_{i-m} - 2 x*_i + x*_{i+m}]^2

NIST SP 1065 eq. (25), §5.2.11. The `N` phase values are extended by
reflection about **both** endpoints into a virtual sequence `x*` running
from `i = 3-N` to `i = 2N-2`, of length `3N-4`, with the original data in
the centre:

    x*_{1-j} = 2 x_1 - x_{1+j}          j = 1 .. N-2
    x*_{N+j} = 2 x_N - x_{N-j}          j = 1 .. N-2

## Why it exists

The Allan variance runs out of data at long tau. At `tau = T/2` it has one
term left and the estimate collapses — which is precisely where a stability
measurement is usually trying to say something.

**TOTVAR keeps `N-2` terms at every tau**, because the sum always runs over
the original interior points however large `m` becomes. On a 2049-point
record the overlapping Allan variance falls from 2047 terms to 1025 at the
longest octave tau; TOTVAR reports 2047 throughout.

## The reflection is a point reflection, not a mirror

`2*x_1 - x_{1+j}` reflects *through* the endpoint. A plain mirror
(`x_{1+j}`) would put a corner at each end and manufacture a large second
difference exactly where the estimator is most sensitive.

The consequence: a straight ramp extends to a longer straight ramp, and
every second difference stays zero. A parabola does **not** extend to a
parabola — the reflection continues the slope but inverts the curvature.
That is why TOTVAR does not reproduce the exact drift identity:

| tau | TOTDEV / (D·tau/sqrt2) |
|---|---|
| tau0 | 1.000 (exact) |
| 32 tau0 | 0.986 |

Biased low, never high, growing with tau. Small, known, and tested in both
directions rather than left to be discovered.

## Bias, and why this estimator needs a noise type

The reflected data is not new information. SP 1065 §5.2.11: TOTVAR "has the
same expected value" as AVAR for **white PM, flicker PM and white FM**, but
for the steeper FM noises it is biased low and needs

    corrected = totvar / (1 - a*tau/T)

| noise | alpha | a |
|---|---|---|
| white PM, flicker PM, white FM | +2, +1, 0 | 0 |
| flicker FM | −1 | **0.481** |
| random-walk FM | −2 | **0.750** |

`stats.bias.totvar_bias` returns the factor rather than applying it, so a
caller can see and record what was done. It **refuses** past the point
where the correction would exceed a factor of two: beyond there the
correction is larger than the quantity being corrected, and a number
produced that way is not a measurement.

This is the coupling that makes TOTVAR different in kind from the classical
family — its answer depends on identifying the noise first, which is the
assumption this library's confidence intervals otherwise avoid.

## Degrees of freedom — do not count the terms

TOTVAR reports `N-2` terms at every tau *by reusing reflected data*. A
data-driven estimator handed those coefficients sees a long array and
reports a confidence the record cannot support. SP 1065 gives the real
count:

    edf = b*(T/tau) - c

| noise | b | c |
|---|---|---|
| white FM | 1.500 | 0 |
| flicker FM | 1.168 | 0.222 |
| random-walk FM | 0.927 | 0.358 |

For white and flicker PM it is the overlapping ADEV's edf "increased by 2",
so `totvar_edof` requires that value to be passed rather than guessing it.

At the longest tau of a 4096-point record this is under a hundredth of the
term count — the gap between what TOTVAR *reports* and what it *knows*.

## Validation status

**Validated.** `tests/test_total.py`:

* the reflection against SP 1065's own index formulae, by hand, on a
  five-point record where every entry is written out in the test;
* a ramp extends to a ramp (the point-reflection property);
* **TOTDEV equals OADEV at `m = 1` to 1e-15** — at the first averaging
  factor no reflected point enters any term, so they are the same sum;
* the term count is constant at `N-2` across every tau, while the Allan
  family's halves;
* **time reversal to 1e-14 relative** — both ends use the same rule, so
  reversing the record must give identical variances. This is the test the
  project plan flagged as non-trivial for reflection code, and a one-sided
  or off-by-one reflection breaks it and almost nothing else;
* the drift bias, asserted in both directions;
* **the published `a` coefficients against measured bias**: synthetic
  flicker FM and random-walk FM, 120 realizations at N=4096, TOTVAR/AVAR
  against `1 - a*tau/T` to 0.02 out to `tau = T/16`, and the correction
  shown to remove the bias it describes.

## References

* NIST SP 1065 (Riley), §5.2.11, eqs. (25)–(26), read from the digital
  original at `tf.nist.gov/general/pdf/2220.pdf`.
* D. A. Howe, "An extension of the Allan variance with increased confidence
  at long term", Proc. 1995 IEEE Int. Freq. Cont. Symp., 321–329.
* D. A. Howe and C. A. Greenhall, on the bias and its correction.
