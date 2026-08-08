# Power-law domain conversions

How `sigma_y(tau)` and `S_y(f) = h_alpha * f**alpha` convert into each
other, where the coefficients came from, and the two places the published
source cannot be taken at face value.

## The integrals everything descends from

NIST SP 1065 (Riley, *Handbook of Frequency Stability Analysis*), p. 73:

    sigma2_y(tau)     = 2 * int_0^fh S_y(f) * sin^4(pi tau f)/(pi tau f)^2 df      (65)

    Mod sigma2_y(tau) = 2/(m^4 pi^2 tau0^2)
                        * int_0^fh S_y(f) sin^6(pi tau f)/(f^2 sin^2(pi tau0 f)) df  (66)

`f_h` is the upper cutoff frequency of the measuring system. Everything
below is these two integrals evaluated for `S_y(f) = h_alpha f^alpha`.

## The coefficients

    sigma2_y(tau) = sum_alpha c_alpha(tau) * h_alpha + (tau^2/2) * D^2

| alpha | name | Allan `c_alpha` | Modified `c_alpha` |
|---|---|---|---|
| +2 | white PM | `3*f_h / (4*pi^2*tau^2)` | `3 / (8*pi^2*tau^3)` |
| +1 | flicker PM | `[1.038 + 3*ln(2*pi*f_h*tau)] / (4*pi^2*tau^2)` | `(24*ln2 - 9*ln3) / (8*pi^2*tau^2)` |
| 0 | white FM | `1/(2*tau)` | `1/(4*tau)` |
| −1 | flicker FM | `2*ln2` | `0.9352277` |
| −2 | random-walk FM | `(2*pi^2/3)*tau` | `(11*pi^2/20)*tau` |
| — | linear drift | `tau^2/2` | `tau^2/2` |

**The Allan column is transcribed** from SP 1065 section 7.1, eq. (67) and
the table on p. 74. **The modified column is derived here**, by quadrature
of eq. (66): the handbook tabulates no modified conversions — section 7.1
says its formulae extend to ThêoH, and stops.

The drift row is exact for both, not asymptotic, which is what lets drift
enter the design matrix as a `D**2` column and keep the model linear. It is
the same identity `tests/test_identities.py` asserts at 1e-13.

## Why the modified rows exist at all

White PM and flicker PM have **the same Allan tau-slope**. SP 1065 says so
in its own introduction to section 7: "The general conversion from time to
frequency domain is not unique because white and flicker phase noise have
the same Allan variance dependence on tau." No amount of AVAR data
separates them, and no choice of wavelet vanishing moments does either.

The modified variance breaks it, because its bandwidth scales as `1/tau`
instead of sitting at a fixed `f_h`:

* modified white PM falls as **`tau^-3`**,
* modified flicker PM falls as **`tau^-2`**.

One octave of tau then resolves the pair. That is the entire reason the
decomposition stacks AVAR and MVAR rows rather than fitting AVAR alone.

Note also that **no modified row carries `f_h`**. The `1/tau` bandwidth
removes it. The two unmodified PM rows are the only place in this library
where an unmeasurable instrument parameter enters an answer.

## `f_h` is an input, never an inference

`f_h` cannot be recovered from tau-domain data. It is a recorded input; when
defaulted, the result is flagged `assumed_fh` and that flag propagates into
every report and legend. Rows are used only where

    2*pi*f_h*tau >= 10

and rows violating it are dropped rather than fitted. Below that the
formulae are not merely imprecise, they are wrong in a way no error bar
reports — the parameter they are asymptotic in is exactly that product. The
white-FM row shows the scale honestly: at `2*pi*f_h*tau = 201` the closed
form `h_0/(2*tau)` is still 0.47% high, because the sinc tail beyond `f_h`
is real signal the closed form has integrated in.

## Two places the source cannot be taken at face value

### 1. The flicker-PM row is printed two different ways, and they disagree

SP 1065 gives the flicker-PM conversion twice, and the two do not match:

* **eq. (67), p. 73:** `[1.038 + 3*log_e(2*pi*f_h*tau)] / (2*pi^2*tau^2)`
* **the table, p. 74:** `D = 1.038 + 3*ln(2*pi*f_h*tau0)/4*pi^2`

The denominators differ by a factor of two, and the logarithm's argument
carries `tau` in one and `tau0` in the other. Both cannot be right.

Integrating the handbook's **own eq. (65)** settles it. Measured against the
quadrature at `tau0 = 1`, `f_h = 0.5`:

| candidate | quad / candidate at m=64 | at m=256 |
|---|---|---|
| eq. (67) as printed | 0.500017 | 0.500012 |
| table, `tau0` in the log | 3.789963 | 4.719887 |
| **table's `4*pi^2`, eq. (67)'s `tau`** | **1.000035** | **1.000024** |

So the shipped row is `[1.038 + 3*ln(2*pi*f_h*tau)] / (4*pi^2*tau^2)`.
Each printed form carries exactly one error, and they are different errors:
eq. (67)'s denominator is two-fold, and the table's log argument should be
`tau`. The `tau0` reading is not a close call — its error *drifts with m*,
which is the signature of the wrong variable inside a logarithm.

An independent second route agrees: the synthesizer's flicker-PM output has
a measured `sigma2*tau^2` rising by 0.6620 per octave of tau, against
`3*ln2/pi = 0.66187` for the `4*pi^2` form and twice that for eq. (67)'s.

This is the one place in this library where a published formula is not
reproduced verbatim. It is recorded here, in the module docstring, and in
`test_the_flicker_pm_row_settles_a_contradiction_in_the_source`.

### 2. The modified flicker-FM constant has no closed form

`Mod sigma2_y = K * h_-1` with **K = 0.9352277**, derived by quadrature of
eq. (66), converging as `1/m^2`:

| m | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| K(m) | 0.9352449 | 0.9352320 | 0.9352288 | 0.9352280 |

`27*ln(2)/20 = 0.9357487` is within 0.06% and is **not** it — it misses in
the fourth figure, far outside the quadrature's own convergence. A search
over `a*ln2 + b*ln3` with small rational `a, b` found nothing within 2e-9.
So the number ships as a number with its derivation attached, rather than
dressed up as a closed form it does not have. A test guards it against
being "tidied".

## Validation status

**Validated.** `tests/test_powerlaw_coeffs.py`:

* every Allan row against eq. (65) and every modified row against eq. (66),
  at two `tau0` — the second `tau0` because a missing `tau0` factor is
  invisible at `tau0 = 1`, and the modified rows have none;
* the tolerance is `3/(2*pi*f_h*tau)`, the approximation's own asymptotic
  parameter, rather than a round number chosen to pass;
* **white PM against a generator whose bandwidth is known a priori.** `f_h`
  is not identifiable from data, so this row could not otherwise be checked
  end to end. The synthesizer's alpha = +2 output is white phase of
  variance `tau0^2`, whose Allan variance is `3*tau0^2/tau^2` exactly —
  which is this row at exactly `f_h = 1/(2*tau0)`. Agreement to 1e-14;
* all ten coefficients against the shipped `oadev` and `mdev` over 300
  realizations of calibrated synthetic noise, to 3%;
* the `tau^-3` against `tau^-2` slope difference, at 1e-12.

**One exclusion, stated rather than hidden.** The *unmodified flicker-PM*
row is not checked against the synthesizer, because that coefficient
assumes a brick-wall `f_h` and the Kasdin–Walter flicker-PM process is not
brick-wall limited — its spectrum rises to Nyquist and stops. Its measured
constant is **1.764** where the brick-wall model says 1.038, equivalent to
an effective `f_h` about 27% above Nyquist. The white-PM row has no such
gap because that process *is* brick-wall at Nyquist exactly, and the
modified flicker-PM row has none because it carries no `f_h` at all. So the
gap appears exactly where theory says it must, and nowhere else.

## References

* NIST SP 1065 (Riley), *Handbook of Frequency Stability Analysis*, section
  7 "Domain Conversions", eqs. (63)–(67) and the table on p. 74. Read from
  the digital original at `tf.nist.gov/general/pdf/2220.pdf`.
* IEEE Std 1139, for the `h_alpha` definitions.
* `theory/noise-types.md` for the synthesizer's exact discrete behaviour,
  including the proof that its white PM is brick-wall at Nyquist.
