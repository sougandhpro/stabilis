# The time deviation

## Definition

The time variance is a rescaling of the modified Allan variance into units
of time:

    tvar(tau) = (tau**2 / 3) * mvar(tau)
    tdev(tau) = (tau / sqrt(3)) * mdev(tau)

`tdev` is measured in seconds and answers a different question from `adev`.
Where `adev` asks "how stable is the frequency", `tdev` asks "how far will
the clock have wandered in time". Telecom specifications are written in
`tdev` for that reason, and the masks in ITU-T G.811, G.812, G.813 and
G.8272 are all `tdev` or MTIE limits.

The `1/sqrt(3)` is chosen so that for white PM the time deviation equals the
standard deviation of the time error itself.

## Estimator as implemented

Written in terms of the modified estimator's inner sum `S[j]`, the
rescaling collapses into the coefficients:

    c[j] = S[j] / m
    tvar = mean(c**2) / 6 = sum(S**2) / (6 * m**2 * (N-3m+1))

so no separate multiplication by `tau**2/3` is performed, and no
intermediate `mdev` is formed and re-squared. This matters because taking a
square root that is then squared again loses precision for nothing.

## Expected tau-slopes

| alpha | noise | tdev ~ |
|---|---|---|
| +2 | white PM | tau^0 |
| +1 | flicker PM | tau^+1/2 |
| 0 | white FM | tau^+1/2 |
| -1 | flicker FM | tau^+1 |
| -2 | random-walk FM | tau^+3/2 |

White PM is flat in `tdev`, which is the direct consequence of the
normalization above.

## Validation status

**Validated to 16.5-16.6 correct digits** against
`validate.certified.exact_tvar` on both NBS14 sets, LRE >= 13, in
`tests/test_classical.py`.

`tests/test_identities.py::test_tdev_is_mdev_scaled_by_tau_over_root_three`
asserts the defining relation `tdev = tau/sqrt(3) * mdev` directly at
rtol 1e-15, which is the check that the collapsed coefficient form above
is algebraically equivalent to the definition rather than merely close.

## References

* ITU-T Recommendation G.810, definitions and terminology.
* NIST SP 1065 (Riley), §5.2.
