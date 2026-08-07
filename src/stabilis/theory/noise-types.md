# Power-law noise types and their synthesis

## Definition

The fractional frequency `y(t) = (nu(t) - nu0) / nu0` of a real oscillator
is modelled as a sum of five power-law processes, each with one-sided
power spectral density

    S_y(f) = h_alpha * f**alpha

| alpha | name | S_y(f) | S_x(f) |
|---|---|---|---|
| +2 | white PM | f^2 | f^0 |
| +1 | flicker PM | f^1 | f^-1 |
| 0 | white FM | f^0 | f^-2 |
| -1 | flicker FM | f^-1 | f^-3 |
| -2 | random-walk FM | f^-2 | f^-4 |

Phase and frequency spectra are related by `S_x(f) = S_y(f) / (2*pi*f)**2`,
which is the `alpha - 2` shift in the right-hand column.

## Expected tau-slopes

For `sigma_y**2(tau) ~ tau**mu`, the Allan variance gives `mu = -alpha - 1`
over `-2 <= alpha <= 0`. Two things break that simple rule and both matter:

* **AVAR saturates at `mu = -2` for both alpha = +1 and alpha = +2.** White
  PM and flicker PM are *not separable by the Allan variance*. This is not a
  numerical problem, it is a property of the estimator: its effective
  high-frequency cutoff sits at a fixed `f_h`. Separating them requires the
  *modified* Allan variance, whose cutoff scales as `1/(m*tau0)`.
* **AVAR does not converge for `alpha <= -3`.** Flicker walk and random run
  need a higher-order estimator: the Hadamard variance (third difference),
  or a Daubechies wavelet variance with `K >= 2` vanishing moments, which
  converges for `alpha > -2K-1`.

## Estimator as implemented

`synth.power_law_noise` convolves white noise with the Kasdin-Walter
impulse response

    h[0] = 1;   h[k] = h[k-1] * (k - 1 + beta/2) / k,    beta = -alpha

which is the discrete fractional integrator of order `beta/2`, applied by
FFT. Two special cases make the recursion self-checking, because it
terminates:

* `beta = 2` (alpha = -2) gives `h = [1, 1, 1, ...]`, a cumulative sum.
  Integrating white noise gives a random walk.
* `beta = -2` (alpha = +2) gives `h = [1, -1, 0, 0, ...]`, plain
  differencing. Differencing white noise gives f^2 noise.

The generator takes an explicit `numpy.random.Generator`. It never touches
the global random state, because validation evidence that cannot be
reproduced bit for bit is not evidence.

## Validation status

**Validated.** `tests/test_synth.py`:

* `test_impulse_random_walk_is_cumulative_sum` and
  `test_impulse_white_pm_is_a_difference` assert the two terminating cases
  exactly. These are worth more than the statistical test: if the impulse
  response is right where it terminates, the recursion is right.
* `test_impulse_flicker_coefficients` pins the half-integrator series
  `1, 1/2, 3/8, 5/16`.
* `test_psd_slope_recovers_alpha` fits the log-log periodogram slope over
  `1e-3 < f < 0.1` for each alpha, tolerance 0.1.
* `test_no_global_rng_is_touched` guards reproducibility.

Amplitude is **not** calibrated: the output has no certified `h_alpha`.
That arrives with the Monte-Carlo unbiasedness tests, which check the
ensemble mean of `sigma_hat**2` against the closed forms `h_0/(2*tau)`,
`2*ln(2)*h_-1` (flat in tau) and `2*pi**2*h_-2*tau/3`.

## References

* Kasdin & Walter, "Discrete simulation of power law noise", 1992.
* NIST SP 1065 (Riley), *Handbook of Frequency Stability Analysis*, §5.
* IEEE Std 1139.
