# Power-law noise identification

## Why it is here, and why it is not load-bearing

Nothing in this library's estimators or confidence intervals needs a noise
type. That is the design: the EDOF is estimated from the data (see
`edof`), so an interval can be quoted for noise that is a mixture, sits on
a drift, or is nothing on the list.

Identification is here for three other reasons. Users want to know what
they are looking at. Bias corrections for TOTVAR and Thêo1 genuinely
require a noise type, and arrive with those estimators. And Stable32 prints
an `Alpha` column, which parity demands.

## Two independent methods

They share no arithmetic, so agreement is evidence and disagreement is
information.

### Lag-1 autocorrelation (Riley & Greenhall)

Operating on **frequency** data:

    r₁    = lag-1 autocorrelation
    δ     = r₁ / (1 + r₁)
    if δ < 1/4:   α = −2·(δ + d)
    else:         difference the series, d ← d+1, repeat

`d` counts the differences needed before the series decorrelates, and each
difference steepens the underlying power law by 2 — which is why it enters
with the same weight as `δ`.

`δ` can be negative, which is how the formula reaches positive `α`. White
PM in the frequency domain is a differenced white sequence with `r₁ ≈ −1/2`,
so `δ ≈ −1` and `α ≈ +2`.

Phase input is converted to frequency first. Mixing the domains is a
two-unit error in `α` — two whole noise types.

### Modified-Allan log-log slope

    mdev(τ) ∝ τ^μ,      α = −2μ − 1

| α | μ |
|---|---|
| 2 (white PM) | −3/2 |
| 1 (flicker PM) | −1 |
| 0 (white FM) | −1/2 |
| −1 (flicker FM) | 0 |
| −2 (random-walk FM) | +1/2 |

**This is the method that separates white PM from flicker PM.** The plain
Allan deviation cannot: both give a τ⁻¹ slope, because AVAR's effective
bandwidth is set by the hardware cutoff `f_h` rather than by τ. MDEV
averages over `m` phase positions, so its bandwidth scales as `1/(mτ₀)` and
the degeneracy breaks. Any identification built on ADEV slopes alone is
guessing between those two.

## Noise type is a function of τ

A white-FM oscillator with a random-walk term is white FM at one second and
random-walk FM at a thousand. Reporting one `α` for a record hides exactly
the thing a customer is buying. `identify_curve` runs the identifier at
every averaging factor, averaging the frequency data to `mτ₀` first.

## Confidence

The two estimates are compared. Agreement within half a power law — the
point at which they would round to different types — gives `high`.
Otherwise `low`, and the reported `α` is the average of the two rather than
one of them, so a disagreement widens rather than picks a side. With no
`mdev` result supplied the answer comes back flagged `single-method`,
because one method is not evidence.

## Measured behaviour

Against synthesized noise of known power law:

| record | hit rate, α = 2, 1, 0, −1, −2 |
|---|---|
| N = 256 | 98 %, 98 %, 100 %, 95 %, 98 % |
| N = 512 | 99 %, 100 %, 100 %, 100 %, 100 % |
| N = 1024 | 100 % across the board |

Individual estimates at N = 4096 have a standard deviation of 0.03 for the
FM noises and 0.09 at white PM (lag-1), 0.07–0.15 (MDEV slope).

## Validation status

**Validated.** `tests/test_noiseid.py` asserts: both methods recover the
true α from synthesized noise of each type; MDEV slopes separate white PM
from flicker PM where ADEV slopes do not; the combined identifier hits the
right type at the rates above; phase and frequency input agree;
disagreement is flagged rather than hidden; and the identifier tracks a
record whose noise changes with τ.

Synthesized noise of known α is the only honest reference here — every
published noise-ID table is Stable32's classification of its own test data.

**Not yet validated:** agreement with Stable32's `Alpha` column (M3), and
the B1/B2/B3 bias functions, which are an additional identification route
and are deferred with the estimators that need them.

## References

* Riley & Greenhall, "Power Law Noise Identification Using the Lag 1
  Autocorrelation", EFTF 2004.
* NIST SP 1065 (Riley), §5.5 — noise identification.
* IEEE Std 1139-2022, clause 4.1 — what a stability report must state.
