# Power spectra, and the four places they can be quietly wrong

The Allan variance and the power spectral density describe the same noise.
They are related by a published integral, so a spectrum is not a second
opinion about a record — it is the same information in the coordinates
where a discrete tone is a spike instead of a bump.

Most of the work in `stats/psd.py` is not the transform. It is the four
constants that can be wrong without anything downstream noticing.

## 1. A periodogram is not a density until the window is divided out

Welch's method, one-sided:

    S(f_k) = 2 * |FFT(w * x)|**2 / (rate * sum(w**2))

with the DC and Nyquist bins not doubled, since neither has a
negative-frequency twin.

The `sum(w**2)` denominator is the whole point. Without it the level
depends on which window was chosen, which is a property of the analyst
rather than of the oscillator. With it, Hann and rectangular give the same
answer for the same noise — asserted by test.

The check that this is right is **Parseval**: a single rectangular-windowed
segment with no detrending must integrate to the record's own mean square,
and it does, to twelve digits. That identity is what makes every absolute
level in the module trustworthy; get the normalization wrong and every
number is off by a constant no later step can find.

**Bin spacing is not resolution.** `df = rate/nperseg` is how often the
spectrum is sampled; the *noise-equivalent bandwidth*

    enbw = rate * sum(w**2) / sum(w)**2

is how wide each of those samples actually is — exactly 1 bin for
rectangular and exactly 3/2 for Hann. Both are reported, because the number
a spur amplitude must be divided by to become a density is the second one,
and using the first turns a discrete tone into a noise level.

## 2. Each bin is chi-square, so use the mean

A Welch bin averaged over `N` segments is chi-square distributed with about
`2N` degrees of freedom. Its **mean** is the unbiased estimate of the true
density; its **median** is not, and at the default 15 segments the median
sits 3.4 % low.

This is recorded because the median looks like the robust choice, and
picking it would show up as a 3.4 % calibration error in the estimator —
which is the kind of small, self-consistent wrongness that survives review.
Measured across twelve seeds, the band mean recovers a synthesized
`h_0` to 0.998 ± 0.006.

## 3. The carrier is not in the data

    S_y(f)   = (2*pi*f)**2   * S_x(f)      1/Hz
    S_phi(f) = (2*pi*nu0)**2 * S_x(f)      rad**2/Hz

`nu0` is the carrier frequency, and no amount of looking at the samples
will reveal it — a 10 MHz OCXO and a 10 GHz source produce identical time
error records if their fractional stability is the same. A spectrum
converted with an assumed carrier is wrong by a constant factor, and
nothing downstream can detect it.

So `convert` **refuses** to produce `S_phi` without `nu0_hz`, and the value
used is recorded. This is the same rule already in force for `f_h`.

Converting *to* `S_x` drops the DC bin rather than returning infinity:
`S_x = S_y/(2*pi*f)**2` diverges at zero, and a plotted infinity is worse
than an absent point.

## 4. `L(f) = S_phi(f)/2` is only true near enough to zero

The relation is the small-angle limit of the phase-modulation expansion. It
holds while the mean-square phase deviation above `f` is small compared
with 1 rad²; close to the carrier of a noisy source it is not, and the
formula's output there is a plausible-looking number that means nothing.

`script_l` integrates `S_phi` from each `f` up to Nyquist and **withholds**
every bin where that exceeds `SMALL_ANGLE_RAD2 = 0.1`. Withheld bins are
NaN, not a value with a warning attached, and `plot.script_l` leaves the
gap in the line rather than joining across it.

The refusal is progressive, which is what a physical criterion should look
like. Measured on synthesized white FM:

| record | carrier | outcome |
|---|---|---|
| `h_0 = 1e-24` | 10 MHz | every bin valid |
| `h_0 = 1e-20` | 10 GHz | 1005 bins withheld; valid from 0.49 Hz up |
| `h_0 = 1e-16` | 10 GHz | nothing valid; the whole curve is withheld |

That near-carrier region is precisely the part a datasheet gets written
from, which is why it is refused rather than merely flagged.

## 5. A cumulative sum is not an ideal integrator

This one was found by measurement rather than reasoned out in advance, and
it is the reason a spectrum taken from a phase record and one taken from
the same record's *frequency* do not agree near Nyquist.

`convert` uses `S_y = (2*pi*f)**2 * S_x`, which is the continuous-time
identity and what IEEE 1139 defines. But `to_phase` integrates by
cumulative sum, whose transfer function is `tau0/(1 - exp(-j*w*tau0))`
rather than `1/(j*w)`. The ratio of their squared magnitudes is

    gain(f) = (pi*f*tau0)**2 / sin(pi*f*tau0)**2

which is 1 at low frequency and `(pi/2)**2 = 2.467` at Nyquist. Measured
against synthesized white FM it holds closely: 2.272 against 2.246
predicted in the top half-decade, 1.075 against 1.081 near `f/10`. Dividing
it out recovers the generated level to 0.15 %, against 70 % high without.

`discrete_integration_gain` returns the factor and **does not apply it**. A
caller comparing two spectra of one record wants it; a caller reporting
`S_y` to a datasheet does not, and a switch inside `convert` would let the
wrong one happen quietly. It is a property of sampled data, not an error —
the same standing as the discrete departures already recorded in
`synth.py`.

## 6. Closing the loop with the time domain

For white FM, `S_y(f) = h_0` flat and

    sigma_y(tau) = sqrt(h_0 / (2*tau))

An FFT and a second difference over the same samples, reaching the same
physical quantity through the published conversion. It agrees to 10 % at
the taus where the record supports it — and neither path can be adjusted to
please the other without breaking something checked elsewhere, which is
what makes it worth more than either estimator's own tests.

## Gaps, and detrending

**Gaps segment the record; they are never masked.** One NaN makes an entire
FFT NaN, and filling the hole would invent data. Each gap-free run is
transformed separately, runs shorter than `nperseg` are dropped *and
counted* in provenance, and a record with no run long enough is refused
with the length of its longest — not silently given a shorter segment.
`nperseg` is chosen from the longest gap-free run rather than the record
length, for the same reason.

**Detrending is a decision, so it is recorded.** A phase record's linear
term is its frequency offset: physically real, spectrally enormous, and it
puts an `f**-2` skirt over every low bin. The default is `linear` for phase
and `mean` for frequency, per segment. Per-segment detrending removes some
genuine low-frequency content from each segment, which biases the lowest
few bins downward — the usual Welch trade, stated here rather than left for
someone to rediscover.

## Validation status

**Validated.** `tests/test_psd.py`:

* Parseval to twelve digits on an untouched segment, and the level
  independent of the window;
* white FM recovered at three amplitudes spanning six decades, to 5 %, from
  a generator calibrated in SI units — a genuine closure, since the
  generator works in the time domain and the estimator in the frequency
  domain;
* the noise-equivalent bandwidth exact at 1 and 3/2, and Hann's coherent
  gain exact at 1/2;
* power-law slopes recovered for `alpha` = −2, 0 and +2;
* **the time-domain identity** `sigma_y(tau) = sqrt(h_0/(2*tau))` across six
  octaves, tying this module to `powerlaw.py` through two separate paths;
* the discrete-integration gain, predicted and measured, including its
  exact Nyquist value;
* the conversions exact and reversible to 1e-14, DC dropped rather than
  infinite, and `S_phi` refusing an assumed carrier;
* `L(f)` equal to `S_phi/2` where the angle is small, withheld where it is
  not, in all three regimes of the table above;
* gaps segmented, short runs counted, an impossible request refused.

**Not validated.** No confidence interval is attached to a spectral
estimate. The chi-square law above gives one directly and the degrees of
freedom follow from the segment count and the overlap correlation, but that
is unwritten — a spectrum here carries a level and no error bar, which is
less than every sigma-tau curve in this library carries. Recorded in
`VALIDATION-DEBT.md`.

No comparison against another implementation. Nothing in Stable32 produces
a PSD in these units, so there is no interop tier to sit in.
