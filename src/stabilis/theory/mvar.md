# The modified Allan variance

## Why it exists

The Allan variance gives `tau^-2` for both white PM and flicker PM, so it
cannot tell them apart. The cause is structural rather than statistical:
AVAR's effective high-frequency cutoff sits at a fixed measurement
bandwidth `f_h`, so changing tau does not change which part of the spectrum
is being sampled at the top end.

The modified variance inserts an average over `m` phase positions, which
makes the effective cutoff scale as `1/(m*tau0)`. The two noise types then
separate: white PM gives `tau^-3` and flicker PM `tau^-2`.

This is also why higher-order wavelet filters do *not* solve the same
problem. More vanishing moments buy drift immunity and convergence for
steeper noise, but every order gives the same tau-slopes. Only the
modified family changes the bandwidth scaling.

## Definition and estimator

    z[i] = x[i+2m] - 2*x[i+m] + x[i]
    S[j] = sum_{i=j}^{j+m-1} z[i],           j = 0 .. N-3m
    c[j] = S[j] / (m**2 * tau0)
    mvar = mean(c**2) / 2
         = sum(S**2) / (2 * m**4 * tau0**2 * (N-3m+1))

At `m = 1` the inner sum is a single term and the estimator reduces exactly
to the Allan variance, which is a useful check.

The sliding sum is computed from a cumulative sum over a gap-zeroed copy,
together with a cumulative count of gaps, so a NaN spoils exactly the `m`
windows that contain it instead of poisoning everything after it.

## Expected tau-slopes

| alpha | noise | mvar ~ | mdev ~ |
|---|---|---|---|
| +2 | white PM | tau^-3 | tau^-3/2 |
| +1 | flicker PM | tau^-2 | tau^-1 |
| 0 | white FM | tau^-1 | tau^-1/2 |
| -1 | flicker FM | tau^0 | tau^0 |
| -2 | random-walk FM | tau^+1 | tau^+1/2 |

The slope-to-alpha map is a bijection over the whole range, unlike AVAR's.
That is why the modified deviation, not the Allan deviation, is the right
curve to read a noise type off, and why power-law decomposition must be fit
to AVAR and MVAR jointly rather than AVAR alone.

## Validation status

**Validated to ~16 correct digits** against `validate.certified.exact_mvar`
on both NBS14 sets, LRE >= 13, in `tests/test_classical.py`. Term count
`N-3m+1` asserted there too. Identities in `tests/test_identities.py`
cover `mdev == adev` at AF 1, scale equivariance, time reversal, phase and
frequency input agreement, and exact zero on a constant record.

## References

* Allan & Barnes, "A modified Allan variance with increased oscillator
  characterization ability", 1981.
* NIST SP 1065 (Riley), §5.2.
