# The Hadamard variance

## Why it exists

Every real oscillator drifts. A linear frequency drift `D` adds exactly
`D**2 * tau**2 / 2` to the Allan variance, so at long tau a drifting
oscillator's ADEV rises with slope `+1` no matter what its noise is doing.
The drift, being deterministic, then hides the stochastic behaviour that
was the object of the measurement.

The usual answer is to fit and subtract the drift first. That works, but it
makes the result depend on which of several drift models was chosen, and
the choice has to be recorded and defended. The Hadamard variance instead
uses a third difference, which annihilates any quadratic in phase --
equivalently, any linear drift in frequency -- exactly and with no fitting.

It also converges for steeper noise: `alpha > -5`, against AVAR's
`alpha > -3`, so flicker walk and random run FM are estimable.

## Definition and estimator

    d[i] = x[i+3m] - 3*x[i+2m] + 3*x[i+m] - x[i],   i = 0 .. N-3m-1
    c[i] = d[i] / (m * tau0)
    hvar = mean(c**2) / 6
         = sum(d**2) / (6 * (N-3m) * (m*tau0)**2)

The coefficients `1, -3, 3, -1` are the third row of Pascal's triangle with
alternating signs, and the 1/6 normalizes to agree with AVAR for white FM.

At `m = 1` the overlapping and non-overlapping Hadamard estimators coincide,
since there is no overlap to exploit. That identity is not merely decorative:
it is what revealed a transcription error in the published NBS14 table -- a
relayed value of `hdev(1) = 70.80608` contradicted `ohdev(1) = 70.80607`,
and the exact oracle certified `70.806073185850378`, settling it.

## Relationship to the wavelet view

The Allan variance is the Haar wavelet variance, which has one vanishing
moment and so kills a constant frequency offset. The Hadamard variance is
the next order up: two vanishing moments, killing a linear drift. The
Daubechies D4 wavelet variance has the same two vanishing moments and
therefore the same drift immunity by a completely different route, which
makes the two a strong mutual cross-check. That test arrives with the
wavelet module.

## Expected tau-slopes

| alpha | noise | hvar ~ |
|---|---|---|
| 0 | white FM | tau^-1 |
| -1 | flicker FM | tau^0 |
| -2 | random-walk FM | tau^+1 |
| -3 | flicker walk FM | tau^+2 |
| -4 | random run FM | tau^+3 |

Linear frequency drift contributes **zero**, which is the whole point.

## Validation status

**Validated to 15.8-16.7 correct digits** against
`validate.certified.exact_ohvar` on both NBS14 sets, LRE >= 13, in
`tests/test_classical.py`.

The decisive analytic test is in `tests/test_identities.py`: for a pure
quadratic phase, `hvar` is **exactly zero** -- bit-exact, not approximately
zero -- while the same test asserts `avar = D**2 tau**2 / 2` bit-exactly on
the same data. One test thereby validates both normalizations and the drift
convention at once. A companion test confirms AVAR is *not* blind to drift,
so the immunity test cannot pass vacuously.

## References

* Baugh, "Frequency modulation analysis with the Hadamard variance", 1971.
* Hutsell, "Relating the Hadamard variance to MCS Kalman filter clock
  estimation", 1995.
* NIST SP 1065 (Riley), §5.2.
