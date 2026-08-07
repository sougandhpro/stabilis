# The Allan variance

## Definition

The two-sample or Allan variance of fractional frequency `y` averaged over
intervals of length `tau`:

    sigma_y**2(tau) = (1/2) * < ( ybar[k+1] - ybar[k] )**2 >

The half is what makes it a *two-sample* variance: the expected squared
difference of two adjacent averages is twice the variance of one. It
converges for power-law noise with `alpha > -3`, where the ordinary sample
variance does not converge at all for flicker or random-walk FM. That
non-convergence is the entire reason this statistic exists.

## Estimator as implemented

In phase, an adjacent difference of frequency averages is a second
difference of phase:

    d[i] = x[i+2m] - 2*x[i+m] + x[i],        i = 0 .. N-2m-1
    c[i] = d[i] / (m * tau0)
    avar = mean(c**2) / 2
         = sum(d**2) / (2 * (N-2m) * (m*tau0)**2)

That is the **overlapping** estimator (`oadev`): every phase sample starts
a term, so `N-2m` terms are formed from `N` points. The **non-overlapping**
estimator (`adev`) decimates first, `xd = x[::m]`, and forms only
`len(xd)-2` terms. Overlapping is preferred -- it uses the same data to
produce a lower-variance estimate -- but the terms are then correlated,
which is why the confidence interval cannot be computed from the term count
alone.

The scale factor 1/2 is applied outside the coefficients rather than as a
`1/sqrt(2)` divisor inside them. Folding it in would make every coefficient
irrational and destroy exactness in cases where the mathematics is exact.

## Expected tau-slopes

| alpha | noise | avar ~ | adev ~ |
|---|---|---|---|
| +2 | white PM | tau^-2 | tau^-1 |
| +1 | flicker PM | tau^-2 | tau^-1 |
| 0 | white FM | tau^-1 | tau^-1/2 |
| -1 | flicker FM | tau^0 | tau^0 |
| -2 | random-walk FM | tau^+1 | tau^+1/2 |

White PM and flicker PM share a slope: **the Allan variance cannot separate
them.** Use the modified Allan variance, whose effective bandwidth scales
with tau. See `mvar`.

A linear frequency drift `D` contributes `D**2 * tau**2 / 2` to the
variance, so a drifting oscillator shows a `tau^+1` deviation slope at long
tau regardless of its noise. Use the Hadamard variance, which annihilates
drift. See `hvar`.

## Validation status

**Validated to 15.7-16.4 correct digits** against exact rational
arithmetic. `tests/test_classical.py` compares against
`validate.certified.exact_oavar`, an independent implementation in
`fractions.Fraction`, on both NBS14 reference sets, requiring LRE >= 13.
Interop against the published Stable32 tables is asserted separately at
rtol 1e-6 and labelled as such.

Analytic identities in `tests/test_identities.py`:

* linear frequency drift gives `avar = D**2 tau**2 / 2` **bit-exactly**,
* a constant frequency offset changes nothing, bit-exactly,
* `adev == oadev == mdev` at AF 1,
* scale equivariance at powers of two, bit-exactly,
* time-reversal invariance, phase/frequency input agreement,
* constant phase gives exactly zero,
* white FM recovers a `tau^-1` variance slope.

## References

* Allan, "Statistics of atomic frequency standards", 1966.
* IEEE Std 1139.
* NIST SP 1065 (Riley), §5.
