# MTOT, TTOT and HTOT

Implemented in `stats/mtotal.py`. NIST SP 1065 sections 5.2.12–5.2.14,
equations (27), (28) and (29), read from the digital original at
`tf.nist.gov/general/pdf/2220.pdf`.

## What they are

TOTVAR (`theory/totvar.md`) extends the **whole record** once by reflection
and computes the Allan variance on it. This family works the other way
round: it takes every 3m-point subsequence, removes *that subsequence's*
linear trend, extends *it* by reflection to 9m points, computes the
variance there, and averages over all N − 3m + 1 subsequences.

    subsequence   s  = a[n : n+3m]                       3m points
    detrend       s0 = s − c·i,   c from the half-means
    extend        s# = [s0 reversed, s0, s0 reversed]     9m points
    average       ā[j] = mean(s#[j : j+m])
    difference    z[i] = ā[i] − 2ā[i+m] + ā[i+2m],  i = 0 … 6m−1

with `a` the phase for MTOT and TTOT and the fractional frequency for HTOT.

| | on | scale | coefficient |
|---|---|---|---|
| MTOT | phase | 1/2 | `z/(mτ₀)` |
| TTOT | phase | 1/6 | `z` |
| HTOT | frequency | 1/6 | `z` |

TTOT is MTOT rescaled, `σ²_x,total = (τ²/3)·Mod Tot var`, eq. (28) — the
same relation that turns MVAR into TVAR, and written the same way here, as
a change of coefficient rather than a multiplication afterwards.

**Why it is worth the cost.** At the largest averaging factor `m = N/3` the
outer sum has one term left, but the inner one still has 6m. The estimate
does not collapse at long τ the way the Allan family's does — which is
where a stability measurement is usually trying to say something.

**The reflection here is uninverted**, unlike TOTVAR's. SP 1065 writes it
`0x#_{n−l} = 0x_{n+l−1}`, a plain half-point mirror that duplicates the
edge sample. TOTVAR's `2x₁ − x` continues the trend across the boundary
instead — it can, because it has not removed one. Here the trend has
already been taken out of the subsequence, so a mirror manufactures no
corner and inverting as well would be doing the job twice.

## Two things the text does not pin down

**The detrend divisor.** "Averaging the first and last halves of the
subsequence and dividing by half the interval" is exact when `3m` is even.
When `m` is odd the two halves cannot both be `1.5m` long, and this
implementation divides by the distance between the two half-means'
centroids, `3m − ⌊3m/2⌋`, which reduces to `1.5m` in the case the text
does describe.

**HTOT at `m = 1` is the overlapping Hadamard variance.** The subsequence
is three points, the detrend takes one degree of freedom out of them, and
the six second differences of one-point averages are all drawn from what
is left: the construction says nothing. Stable32 and `allantools` both
return OHVAR there and so does this. It is an identity rather than a
substitution — at `m = 1` the second difference of frequency *is* the third
difference of phase — and `tests/test_mtotal.py` asserts the two agree with
`allantools` bit for bit.

## No confidence interval, and why not a fallback

These three are the only estimators in the library whose coefficient array
is **not one time-ordered sequence**. It is `N − 3m + 1` blocks of 6m terms,
each block drawn from a different subsequence and mostly from that
subsequence's *reflection*; consecutive entries across a block boundary are
neither consecutive in time nor from the same data.

Four things downstream read structure rather than magnitude out of that
array — `eta1`'s autocorrelation, `taucov`'s cross-τ covariance, the
cross-variance's index-by-index pairing, and the decomposition's weights —
and `StabilityResult.coeffs_are_ordered` makes all four refuse.

**A fallback would be worse than a refusal.** `eta3` counts non-overlapping
windows as `n_terms/(span·m)`, and `n_terms` here is `6m(N−3m+1)` drawn
from about `N/3m` independent stretches of record: at long τ it over-counts
by three orders of magnitude. An error bar wrong in the optimistic
direction is worse than no error bar.

The published degrees of freedom for this family are in Riley, "Confidence
Intervals and Bias Corrections for the Stable32 Variance Functions", PTTI
2003, which this repository does not have. Recorded in `VALIDATION-DEBT.md`.

## Bias, and a disagreement with SP 1065's own table

Both MTOT and HTOT are biased estimators of the classical statistic they
mimic. Measured here — 400 realizations at N = 2048, ensemble mean of the
classical over ensemble mean of the total, at m = 8, 16, 32, 64, flat in m
over that range and flat in N from 256 to 4096:

| noise | α | MVAR/MTOT measured | SP 1065 Table 11 | HVAR/HTOT measured |
|---|---|---|---|---|
| white PM | 2 | 1.004 | 1.06 | 0.728 |
| flicker PM | 1 | 1.171 | **1.17** | 0.808 |
| white FM | 0 | 1.295 | 1.27 | 1.004 |
| flicker FM | −1 | 1.393 | 1.30 | 1.171 |
| random-walk FM | −2 | 1.473 | 1.31 | 1.295 |

**The table's caption is inverted relative to its own values.** It reads
"the ratio of the expected value of Mod Totvar to MVAR", but MTOT measures
*low* against MVAR for everything but white PM — E[MTOT]/MVAR runs 0.68 to
1.00 — so it is the reciprocal that lands on these numbers. Read as a
multiplier, which is what "should be used to correct all reported MTOT
results" has to mean, the direction is consistent. `bias.mtot_bias` returns
it that way.

**The values agree for flicker PM exactly and white FM to 2 %, and
disagree by 5 %, 7 % and 12 % for the other three.** The estimator is not in
doubt: `allantools.mtotdev`, `ttotdev` and `htotdev` return these numbers
to every printed digit on the same records, which is two independent
readings of eq. (27) agreeing. What is in doubt is whether Stable32
computes eq. (27) or something slightly different — a different half-length
convention, a different reflection, a different inner range — and only a
Stable32 run settles it. **Recorded, not reconciled.** `bias.mtot_bias`
defaults to the published table so that a report agrees with Stable32, and
`published=False` gives the measured one.

### The HTOT table is the MTOT table shifted by two in α

The three FM rows above are identical to five significant figures. That is
not a coincidence: HTOT applies to the *frequency* record the operator MTOT
applies to the *phase*, and `S_x = S_y/(2πf)²`, so a y-noise of exponent α
presents to it as an x-noise of α + 2. The two PM rows continue the
sequence to α = 3 and 4, which the modified family has no counterpart for —
and they are also the only two that still drift with m (white PM runs 0.741
to 0.723 over m = 8…64), so they are quoted at m = 32 and should be read as
indicative.

`tests/test_mtotal.py` asserts the shift, which is a cross-check on both
estimators at once: a normalization error in either would break it.

## Cost

This family is **quadratic in the record length**, unlike everything else
here. The coefficient count is `6m(N − 3m + 1)`, which peaks near `m = N/6`
at about `N²/2`; a 100 000-point record would ask for 40 GB there.
`_guard_size` catches it with the number rather than leaving it to the
allocator, and says which knob to turn.

## Validation status

**Validated as an algorithm, not as interop.** `tests/test_mtotal.py`: all
three against `allantools` to machine precision at every τ, under
`importorskip`; HTOT's `m = 1` identity with OHVAR; the α → α+2 shift
between the two bias tables; the reflection and detrend as separate exact
properties; the estimator contract (`variance = scale·mean(c²)`); and the
refusals.

`allantools` is a **numeric differential-test oracle only**, per the rule
in `references.md`: no code was read and nothing is copied, and comparing
two independent implementations of a published algorithm is what it buys.
It is *not* Stable32, so it establishes that eq. (27) was read the same way
twice, not that Stable32 agrees. **There is no Stable32 fixture for this
family in the repository**, and the bias disagreement above is exactly the
thing a fixture would settle.

## References

* NIST SP 1065 (Riley), §5.2.12–5.2.14, eq. (27)–(29); Table 11 for the
  MTOT bias.
* D. A. Howe and F. Vernotte, "Generalization of the Total Variance
  Approach to the Modified Allan Variance," Proc. 31st PTTI, 267–276, 1999.
* D. A. Howe, R. Beard, C. A. Greenhall, F. Vernotte, W. J. Riley and
  T. K. Peppler, "Enhancements to GPS operations and clock evaluations
  using a total Hadamard deviation," *IEEE Trans. UFFC* **52**, 1253–1261,
  2005 — the HTOT bias correction, **not in hand**.
* W. J. Riley, "Confidence Intervals and Bias Corrections for the Stable32
  Variance Functions," PTTI 2003 — the degrees of freedom, **not in hand**.
* `theory/totvar.md` for the single-extension member of the family.
