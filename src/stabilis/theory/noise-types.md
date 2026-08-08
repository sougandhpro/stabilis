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

## Amplitude calibration

`h_alpha` is a settable parameter: `power_law_freq(..., h_alpha=1e-26)`
returns a series whose PSD really is `1e-26 * f**alpha`. At unit driving
variance the generator's own coefficient is

    h_unit(alpha, tau0) = 2 * tau0 * (2*pi*tau0)**alpha

so the requested amplitude costs one division. The derivation: a
unit-variance white sequence sampled at `tau0` has flat one-sided PSD
`2*tau0`, and the fractional integrator has `|H|**2 = (2*sin(pi*f*tau0))
**alpha` on the unit circle, giving the **exact** discrete PSD

    S_y(f) = 2 * tau0 * (2*sin(pi*f*tau0))**alpha,    0 < f <= 1/(2*tau0)

which linearizes to `h_unit * f**alpha` for `f << 1/tau0`.

### The sine matters, and by how much

The estimators see the sine, not its linearization, so the discrete
generator's `sigma_y**2(tau)` is not the textbook continuum form at small
`m`. The departures are exact and known, not error terms:

| alpha | exact `sigma_y**2(m*tau0)` | departure from the continuum form |
|---|---|---|
| +2 | `3*h_2 / (8*pi**2*tau0*tau**2)` | none — equals the brick-wall form at exactly `f_h = 1/(2*tau0)` |
| 0 | `h_0 / (2*tau)` | none, at any `m` |
| −1 | `2*ln(2)*h_-1 * r(m)`, `r(1) = 1/ln(2)` | +44% at `m=1`, +15% at 2, +5.0% at 4, +1.5% at 8, +0.46% at 16, +0.13% at 32 |
| −2 | `(2*pi**2*h_-2*tau/3) * (1 + 1/(2*m**2))` | +50% at `m=1`, falling as `1/(2*m**2)` |

Two of those are worth stating separately because they are useful rather
than merely true:

* **White PM is a brick-wall process at exactly Nyquist.** The `f_h`-
  dependent white-PM coefficient is not identifiable from tau-domain data,
  so a synthetic series whose `f_h` is known *by construction* is the only
  end-to-end check that coefficient can get. See `theory/powerlaw.md`.
* **Flicker FM at `m = 1` is exactly `2*h_-1`**, because the Allan kernel's
  `sin**4/sin**2` cancels the generator's `1/sin` and the integral
  collapses to `(2/pi)*int_0^(pi/2) sin(u) du`.

## Validation status

**Validated, including amplitude.** `tests/test_synth.py`:

* the four closed forms in the table above, asserted at `rel=1e-12` to
  `1e-14` against `exact_avar` — an independent path to `E[avar]` through
  the impulse response (`(g @ g)/2` for `g = correlate(a, h)`) rather than
  through a realization;
* `test_the_continuum_closed_forms_are_recovered_at_large_m` — the three
  published `f_h`-free forms `h_0/(2*tau)`, `2*ln(2)*h_-1` and
  `2*pi**2*h_-2*tau/3` at `m = 256`, to 2e-4. These contain `ln(2)` and
  `pi**2` factors that a scale error cannot reproduce by accident, which is
  what makes them evidence that the calibration is *the IEEE 1139 one*
  rather than merely self-consistent;
* `test_the_generator_actually_delivers_the_requested_h_alpha` — 200
  realizations through the real `oadev`, at two sampling rates, recovering
  the requested `h_alpha` to 6% for `m <= n/16`;
* linear scaling in `h_alpha`, and `h_alpha` recorded in provenance.

The recursion itself is checked separately, and more strongly:

* `test_impulse_random_walk_is_cumulative_sum` and
  `test_impulse_white_pm_is_a_difference` assert the two terminating cases
  exactly. These are worth more than the statistical test: if the impulse
  response is right where it terminates, the recursion is right.
* `test_impulse_flicker_coefficients` pins the half-integrator series
  `1, 1/2, 3/8, 5/16`.
* `test_psd_slope_recovers_alpha` fits the log-log periodogram slope over
  `1e-3 < f < 0.1` for each alpha, tolerance 0.1.
* `test_no_global_rng_is_touched` guards reproducibility.

## References

* Kasdin & Walter, "Discrete simulation of power law noise", 1992.
* NIST SP 1065 (Riley), *Handbook of Frequency Stability Analysis*, §5.
* IEEE Std 1139.
