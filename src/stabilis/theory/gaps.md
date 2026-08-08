# Gaps, segments, and why NaN is not enough

## The convention

A missing sample is `NaN`. It propagates correctly through differencing, so
any estimator that drops non-finite terms is automatically gap-aware
without special-case code: a second difference touching a hole is simply
never counted, and the term count falls accordingly.

## The distinction NaN cannot make

Two physically different situations produce the same NaN:

**A dropped sample on a known grid.** The clock kept running, one reading
was lost. The phase either side is still on the same timescale, so
stepping over the hole is correct — only the terms that span it are
invalid.

**A counter that stopped and resumed.** The instrument was off, or
reacquired, or the cable was moved. Phase resumed with an *unknown offset*.
Stepping over this hole manufactures a phase step, which appears as an
enormous fake frequency excursion and will dominate every statistic that
spans it.

Nothing in the data distinguishes these. The user has to say which it is,
because it is a fact about the measurement, not about the numbers.

## Segments

`preprocess.segments(a)` returns half-open `[start, stop)` ranges of
contiguous finite runs, and `longest_segment` picks the largest. When the
second case above is possible, analyse each segment independently and
combine, rather than masking individual terms across the whole record.

The two treatments give the same answer for case one and different answers
for case two, which is exactly why the choice has to be explicit and
recorded in provenance.

## The `gap_policy` argument

Every estimator takes `gap_policy`, resolved by `stats.core.phase_runs` and
recorded in the step log along with the number of runs and the longest one.

| value | meaning |
|---|---|
| `"mask"` | Keep the record whole. Terms whose window touches a gap come back non-finite and are dropped. Assumes one common phase origin across the gap — the dropped-sample case. |
| `"segment"` | Analyse each contiguous run on its own origin and pool the coefficients. Costs only the terms spanning the gap. The counter-reset case. |
| `"auto"` | `"segment"` for frequency input, `"mask"` for phase. The default. |

Pooling is exact rather than approximate. The estimator is a mean of
squared coefficients, and coefficients from different runs are draws from
the same distribution, so concatenating the arrays and taking one mean is
the right combination — not a weighted average of per-run variances, which
would let a five-point run outvote a five-hundred-point one.

Two consequences follow and both are visible in the output. The algebraic
limit `m ≤ (N−1)/2` binds on the **longest run**, not the record, because no
term can straddle a gap: a heavily gapped file reports fewer long taus.
And the surviving term count, not the nominal one, is what reaches the
confidence interval.

## Why frequency data cannot use `mask`

Frequency integrates to phase by cumulative sum, and a cumulative sum
cannot step over an unknown increment — one `NaN` in `y` makes every later
phase sample `NaN`. Masking terms on that record does not drop the terms
near the gap; it drops everything after it.

Measured on 200 frequency points with a single gap at `y[100]`, before this
was fixed: ADEV at τ₀ read 0.910527 from 99 terms where the correct answer
is 0.943730 from 197. **One missing sample in two hundred moved the answer
3.5 % and silently discarded half the record** — a plausible number with no
warning and no symptom, which is the worst failure mode a metrology tool
has. Requesting `gap_policy="mask"` on gapped frequency data now raises.

This is not a limitation of the Allan variance. Absolute phase after a
frequency gap genuinely is unknown, but AVAR never needs absolute phase —
only second differences of it. A gap at `y[k]` should invalidate the `2m`
terms whose window spans it, and nothing else. It now costs exactly that
(`3m` for the Hadamard and modified families).

## Gap sentinels in files

Stable32 writes a literal `0` to mark a gap in frequency data. This reader
does **not** assume that, because a genuine zero-offset frequency
measurement is a perfectly ordinary thing to record and deleting it
silently is unacceptable. Pass `gap_value=0.0` to `read_text` to opt in.

## Effect on confidence

Dropping terms reduces the count, which reduces the degrees of freedom,
which widens the confidence interval. That is handled automatically by the
data-driven EDOF path, since it works from the surviving coefficient array.
A gapped record does not merely give a slightly noisier answer — it gives
an answer with correspondingly honest error bars, provided the term count
is propagated rather than the nominal `N - 2m` being assumed.

## Validation status

**Validated.** `tests/test_gaps.py` asserts the exact cost of one gap —
`2m` terms for `oadev`, `3m` for `ohdev` — the `3m + 2` bound across all
five estimators, bit-identical results under both policies on an ungapped
record, the pooling contract `variance = scale · mean(c²)` across runs, that
the averaging-factor limit follows the longest run, and that `mask` on
gapped frequency input raises.

`tests/test_preprocess.py` covers interior, leading, trailing and total
gaps, the `min_length` filter, and the empty cases.
`tests/test_types.py::test_gaps_are_counted_and_propagate` asserts that a
single NaN in phase reaches exactly the two straddling frequency samples.

**Not yet validated:** the effect of gaps on confidence intervals (M2 — the
term count is propagated, but no interval is computed from it yet), and gap
filling (M7).

## Gap filling

Deferred to M7, deliberately. Filling is not free: linear interpolation
inserts an interval containing no noise, which pulls ADEV *low* at τ
comparable to the gap. The estimate looks better precisely because data was
invented.

The method to be implemented is Howe & Schlossberger's *Total Imputer*,
which fills with real measured data — the preceding run, reflected and
inverted, spliced with a slope matched on a low-pass-filtered copy — so the
filled samples carry the true noise process rather than none. Until then,
`gap_policy="segment"` is the honest default. Note that imputation *raises*
the equivalent degrees of freedom, by up to 6× for white noise, so a
confidence interval computed from filled data is too narrow unless the
imputed samples are excluded from the EDOF count.

## References

* NIST SP 1065 (Riley), §4 on data gaps and editing.
* D. A. Howe and N. Schlossberger, "Characterizing Frequency Stability
  Measurements Having Multiple Data Gaps," *IEEE Trans. UFFC* **69**(2),
  468–472, 2022. doi:10.1109/TUFFC.2021.3137425. Open copy at
  `tf.nist.gov/general/pdf/3150.pdf`. Read 2026-08-08.
* D. Mondal and D. Percival, "Wavelet variance analysis for gappy time
  series," *Ann. Inst. Stat. Math.* **62**(5), 943–966, 2010 — the
  imputation-free alternative, and the natural fit for the wavelet module.
