# Spurs: a tone is not noise, and should not be averaged like one

A mains harmonic, a cooling fan, a PLL reference feedthrough. A spur is a
sinusoid sitting on top of the noise, and every statistic in this library
assumes it is not there. Left in, it inflates `sigma_y(tau)` at some taus
and — this is the surprising part — at others it is completely invisible.

## The identity everything else is checked against

For a phase record carrying one tone,

    x(t) = A sin(2*pi*nu*t + phi)

the second difference gives, averaged over `phi`,

    sigma_y(tau) = 2 * A * sin(pi*nu*tau)**2 / tau

Derivation, since it is three lines: the second difference of `exp(i*w*t)`
over an interval `tau` carries a factor `(exp(i*w*tau) - 1)**2`, whose
modulus is `4*sin(w*tau/2)**2`; the phase average of the squared imaginary
part halves it; and `avar` divides by `2*tau**2`. Verified against this
library's own `adev` to five significant figures over seven octaves.

**The zeros are real.** At `nu*tau = 1, 2, 3, ...` the deviation is exactly
zero: the second difference straddles a whole number of periods and cannot
see the tone at all. A stability plot on an octave grid can therefore miss
a large spur entirely, which is the reason to look for one in the spectrum
rather than infer it from a curve.

## Amplitude comes from the record, not from the spectrum

A tone that falls between DFT bins loses amplitude to scalloping. The
standard answer is a flat-top window — HFT248D and its relatives — whose
coefficients are transcribed from a paper and whose flatness is still only
approximate.

This module does something simpler and exact instead: it uses the spectrum
only to say *where to look*, then fits

    a*cos(2*pi*f*t) + b*sin(2*pi*f*t) + c + d*t

to the samples by least squares, and reports `hypot(a, b)`. This is exact
for a tone at any frequency, needs no coefficient table, and produces the
notch as a by-product. The constant and ramp are in the design so that a
phase record's frequency offset cannot be absorbed into the tone.

Two things had to be got right, and both were found by measurement:

**The frequency search needs more than one pass.** A single 65-point grid
across one bin, parabola-interpolated, locates the tone well enough to
explain nearly all its variance — and still leaves **0.3 %** of a 3 × 10⁻¹¹
tone behind, which stands 25 dB above the floor and is then detected as a
tone in its own right. Three passes, each narrowing the bracket by the
previous grid step, bring the frequency to about 1 nHz and the amplitude to
0.001 %.

**The notch must use the same design the measurement used.** Fitting a bare
`cos`/`sin` pair at notch time, without the constant and ramp, lets the
record's frequency offset leak into the sinusoid's coefficients and leaves
a comparable residue. Only the `cos`/`sin` part is subtracted; the trend
belongs to the oscillator and stays.

## The floor, and the band that can be searched

The floor is a **running median** of the spectrum, not a fit. A median
follows a sloping power-law background without assuming which power law it
is, and — being a median — cannot be pulled up by the very tone it exists
to reveal. A mean floor would be, and a strong spur would then hide a weak
one beside it.

**The outermost bins are not searched**, and the excluded band is reported
rather than assumed away. Their floor is a median over a *truncated*
window, estimated from fewer and systematically different bins; and at the
low end they also carry the residue of per-segment detrending that
`theory/psd.md` records. Together those produced a confident 14.8 dB
detection of a tone that did not exist, which is how the exclusion came to
be there.

## How often it cries wolf

Each Welch bin is chi-square distributed with about `2*n_segments` degrees
of freedom, so the probability that noise alone clears the threshold is
exact and is printed with every table. A detector that cannot say its own
false-alarm rate is not a measurement.

Two approximations in that number are stated rather than hidden, and both
push the same way, so it is a **lower bound**: overlapping segments are
correlated, which puts the true degrees of freedom below `2*n`; and the
floor is itself estimated, which adds scatter the calculation does not
model.

## Notching is an edit

`notch` returns a new series and records every tone it removed — frequency
and amplitude — in that series' provenance, with the outcome line stating
that the record no longer describes the device as measured.

That wording is deliberate. A spur is often physically real: it is part of
what the oscillator does in its actual environment. Removing it is
legitimate when the tone is known to be pickup in the *measurement* chain
and misleading when it is not, and the library is in no position to tell
the difference. So the decision stays with the caller, the evidence stays
with the data, and notching nothing at all is refused rather than recorded
— an audit trail claiming an edit that did not happen is worse than an
error.

## Validation status

**Validated.** `tests/test_spur.py`:

* the closed form against this library's own `adev`, phase-averaged over 40
  realizations at six taus, to 0.2 %;
* the zeros of that identity, so the invisibility is asserted and not just
  described;
* a tone deliberately off a bin centre recovered to 0.2 % in amplitude and
  1e-4 relative in frequency, with the bin shown to have been only a hint;
* two tones found and measured separately to 0.5 %;
* **no tones in pure noise, across six seeds** — the direction that matters,
  since a detector that finds spurs in noise is worthless;
* the reported excess and the threshold agreeing: raising the threshold
  past a tone's stated excess hides it, lowering it does not;
* the searched band excluding the edges, and saying so in provenance;
* the expected false-alarm count rising with a looser threshold;
* a truncated candidate list flagged rather than silently short;
* notching removing the tone to better than a part in a thousand, leaving a
  record where nothing is detectable, with the removed part's RMS equal to
  `A/sqrt(2)` to 2 %;
* the removal recorded with frequency and amplitude; notching nothing
  refused; the series type and epoch preserved;
* **the end-to-end closure** — the stability curve before notching matches
  the closed form to 5 %, and after notching is smaller by a factor of 100.

**Not validated.** No comparison against another implementation, and no
real record with a known injected tone from hardware. The detector's
behaviour on tones close enough to interfere with one another — within a
resolution bandwidth of each other — is untested, and the cluster rule
would report them as one.
