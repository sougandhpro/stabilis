# The Total Imputer: filling gaps with real data

Implemented in `impute.py`. Reference: D. A. Howe and N. Schlossberger,
"Characterizing Frequency Stability Measurements Having Multiple Data
Gaps," *IEEE Trans. UFFC* **69**(2), 468–472, Feb. 2022,
doi:10.1109/TUFFC.2021.3137425. Open copy at
`tf.nist.gov/general/pdf/3150.pdf`.

## What the authors ask, quoted

> The imputation algorithm in the link above is not meant to be used as a
> substitute for a measurement deficiency. Analysts should take seriously
> their measurement methodologies to eliminate gaps if at all feasible. If
> that is not possible or practical, the presence of gaps and the reason
> for them should be clearly stated along with the analysis results.

That is why `fill_gaps` requires a `reason`, records the imputed fraction
in provenance as a *warning* rather than a note, and returns the mask.

## Why not interpolate

Interpolation, zero-padding and adjoining live segments all fail the same
way: they insert an interval containing **no stochastic noise**. ADEV then
improves precisely because data was invented, and the methods destabilise
above roughly 10 % dead time. The Total Imputer instead fills a gap with
real measured data — the preceding run, reversed, with a linear slope added
— so the filled samples are drawn from the true noise process and carry its
correlation structure. It is TOTVAR's data extension applied to an interior
gap rather than to the record ends.

## The algorithm

For a gap of `n` missing points:

1. take the `n` samples immediately preceding the gap;
2. reverse their temporal order and insert them into the gap;
3. add a linear slope, so that the block's endpoint difference equals the
   difference of the **filtered** endpoints either side of the gap.

For multiple gaps, the order is not arbitrary — it always imputes outward
from the most trustworthy data. Find the longest contiguous run; fill the
gaps to its right, left to right, to the end of the record; then reverse
the record and repeat, which reaches the beginning. Every source block is
therefore live measurement, never itself imputed at one remove.

### Endpoint matching

Matching the raw endpoints produces a visible artificial sawtooth under
white PM (the paper's Fig. 5): a single endpoint that happens to sit far
from the local mean sets the slope for the whole inserted block, and under
white PM most endpoints do. Instead the match points are read from a
low-pass-filtered copy, eq. 3:

    x_filtered = ifft( fft(x) · exp(-2³ |f| τ₀² / T) )

The block is padded with a flipped copy of itself on each side before
transforming, because the FFT treats the data as periodic and an unpadded
transform manufactures a step between the first and last point; the match
points are then read at the reflection points, which are the block's own
ends.

## Two ambiguities resolved deliberately

Both are recorded here because resolving them is evidence that the
implementation was derived from the paper rather than transcribed from the
authors' code — which, see below, we may not read.

### 1. `f` in eq. 3 is a bin index, not a frequency

As printed, `f` is "the Fourier-frequency variable" and `T` "the total
length of the imputed data".

* Read with `f` a physical frequency `k/(Nτ₀)` and `T` a duration, the
  exponent collapses to `-8k/N²`, which at the Nyquist bin is `-4/N` —
  **no filtering whatsoever**, and flatter the longer the record.
* Read with `f` the FFT **bin index** and `T` the **number of points**
  (equivalently τ₀ = 1 sample), the exponent is `-8k/T`: a real low-pass
  with an e-folding at one eighth of the sample rate, which is what the
  paper's Fig. 6 plots.

The index reading is also the only dimensionally sane one — an
endpoint-matching filter whose behaviour changed when the user relabelled
τ₀ from seconds to milliseconds would be absurd. `T` is taken as the length
of the array actually transformed, so the cutoff stays at one eighth of the
sample rate whatever the padding factor. `tests/test_impute.py` asserts the
τ₀-invariance directly, as bit equality.

### 2. It fills the frequency samples, not the phase

The paper is loose about which series it fills. Section II motivates the
extension through TOTVAR, which is a phase construction, and its Fig. 2(c)
shows a phase-like extension; but eq. 1 defines AVAR over `y` data and the
text calls the record "the number of y data".

This one was settled by measurement, on the paper's own acceptance
criterion. Reversing a *phase* block flips the sign of the frequency at the
seam. For random-walk FM, where `y` wanders slowly and a sign flip is a
large excursion, that is catastrophic:

| filled series | AVAR(τ₀) bias, RWFM | trials in 100 with 90 % CI overlap |
|---|---|---|
| phase, reverse + slope | **74× high** | 3 |
| phase, reflect + invert about the last live point | **83× high** | 2 |
| **frequency** | 1.00 | **100** |

Reversing `y` preserves a random walk's continuity where reversing `x`
destroys it. `fill_gaps` therefore takes a `FreqSeries`.

A gapped `PhaseSeries` is **refused** rather than converted. The phase
*across* a gap is not in the data even once the frequency samples are
imputed: a counter that stopped and resumed carries an unknown phase step,
and integrating imputed frequencies through the gap would assert that step
was zero. That is exactly the assumption `preprocess.segments` exists to
avoid making silently.

## What it refuses

* **A gap reaching either end of the record.** There is live data on one
  side only, so the endpoint match does not apply — and a gap at an end
  bridges nothing and carries no information about what happened there.
  Trim it with `edit.select_range`, which loses no live data and puts the
  edit on the record.
* **A gap wider than the live run preceding it.** The method extends live
  data into a gap, so it needs at least as much of it as the gap is wide;
  the paper's own caveat is that the largest gap must be small compared
  with the initial run. Their remedy — impute from both sides and match
  averages in the middle — is in a conference paper we do not have, so it
  is refused rather than improvised.
* **A `PhaseSeries`**, as above.

## Confidence: imputed samples buy no degrees of freedom

The paper measures EDF rising under extension by **up to 6× for white
noise**, less for flicker and random walk, and treats that as a benefit.
For a confidence interval it is the opposite: an interval computed from the
filled term count is too narrow, and wrongly so in the optimistic
direction.

`impute.live_terms(result, mask)` counts, per τ, the terms whose support
`[t, t + span_factor·m)` touches no imputed sample, and
`confidence(result, live_terms=…)` scales each η by that fraction. An
imputed sample *is* real measured data — but it is real measured data
already in the record, a reflection of live samples the estimator has
counted once, so it carries no independent information and a term touching
one contributes nothing.

Where the live fraction is zero — every term at that τ rests on invented
data — **no interval is quoted**: NaN bounds, the averaging factors named
in provenance, and a warning. The point estimate is still there; it just
has no error bar that is not made of invented data. A χ² quantile at zero
degrees of freedom is not a wide interval, it is no interval, and printing
a number there would be the one thing this whole module exists to avoid.

## Validation status

**Validated against the paper's own published figure of merit.**
`tests/test_impute.py` and a row in `stabilis validate`.

The paper's Fig. 4 reports, for 513-point records with points 250–400
removed (29 %) and 100 trials per noise type, the number of trials in which
the 90 % confidence interval of the filled AVAR overlapped that of the
original. Reproduced here with the *overlapping* estimator, which is
tighter than the paper's powers-of-two eq. 1 and therefore a harder test:

| τ/τ₀ | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| white PM, published | 96 | 98 | 96 | 94 | 84 | 34 | 11 | 20 |
| white PM, here | 100 | 100 | 100 | 68 | 27 | 10 | 3 | 1 |
| white FM, published | 95 | 93 | 96 | 97 | 94 | 84 | 59 | 63 |
| white FM, here | 100 | 100 | 100 | 99 | 89 | 79 | 65 | 64 |
| flicker FM, published | 94 | 97 | 97 | 97 | 98 | 95 | 84 | 87 |
| flicker FM, here | 100 | 100 | 100 | 100 | 100 | 100 | 99 | 98 |
| random-walk FM, published | 92 | 99 | 97 | 96 | 94 | 98 | 100 | 98 |
| random-walk FM, here | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

**The exact counts are not comparable and are not asserted as such.** The
criterion's strictness is set entirely by the confidence interval, and the
paper does not state its EDOF convention; with the non-overlapping AVAR of
its eq. 1 the intervals at long τ are so wide that every column here rises
above the published one. What *is* asserted is the structure, which is
reproduced exactly: **random-walk FM best, then flicker FM, then white FM,
with white PM the failing case and the failure concentrated at long τ**,
where the terms are mostly imputed. That ordering is the paper's result,
and it is not what a broken implementation produces — the phase-domain
reading above inverts it.

Also asserted, and cheap: a record with no gaps comes back **bit-identical**
and with an empty mask; the fill is invariant to τ₀ as bit equality; live
samples are never modified; filling never narrows a confidence interval.

## Licence position: clean-room from the paper

The authors' reference implementation
(`github.com/nkschlos/time-series-imputation`, also Zenodo 5594587) carries
**no licence file at all**, so it is all-rights-reserved and must not be
vendored, ported or copied. Nothing here was read from it. The published
description is sufficient, and where it is ambiguous the reading taken is
argued above rather than guessed. Cite the paper, not the repository.

## Not implemented

* **Both-sided imputation with averages matched in the middle**, for a gap
  too large relative to the initial run. The paper's ref. [14] (Howe,
  Champagne and Schlossberger, IFCS-EFTF 2022) is the source and we do not
  have it. Refused rather than improvised.
* **Moving the reflection point to match the phase of an embedded
  periodic**, which the paper mentions for diurnals. It needs the period,
  which is an input this library does not currently take.
* **Mondal & Percival's gap-tolerant wavelet variance**, the
  imputation-free alternative — an unbiased estimator for gapped data that
  invents no samples at all. Since the Haar wavelet variance *is* the Allan
  variance, it drops into `stats/wavelet.py` and gives a gap-tolerant ADEV
  with no EDOF inflation. Shipping both, and reporting where they disagree,
  is a differentiator: Stable32 fills, `allantools` ignores, and nobody
  cross-checks the two.

## References

* D. A. Howe and N. Schlossberger, "Characterizing Frequency Stability
  Measurements Having Multiple Data Gaps," *IEEE Trans. UFFC* **69**(2),
  468–472, 2022 — the algorithm, eq. 3, and the Fig. 4 counts above.
* C. A. Greenhall, D. A. Howe and D. B. Percival, "Total variance, an
  estimator of long-term frequency stability," *IEEE Trans. UFFC* **46**(5),
  1183–1191, 1999 — the reflect-and-invert extension this generalises.
* D. Mondal and D. Percival, "Wavelet variance analysis for gappy time
  series," *Ann. Inst. Stat. Math.* **62**(5), 943–966, 2010 — the
  imputation-free alternative.
* `theory/gaps.md` for the segment-versus-mask decision this sits beside,
  and `theory/ci.md` for the interval the live-term count feeds.
