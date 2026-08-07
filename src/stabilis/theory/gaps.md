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

**Validated for segmentation and propagation.** `tests/test_preprocess.py`
covers interior, leading, trailing and total gaps, the `min_length` filter,
and the empty cases. `tests/test_types.py::test_gaps_are_counted_and_propagate`
asserts that a single NaN in phase reaches exactly the two straddling
frequency samples.

**Not yet validated:** gap-aware term counting inside the estimators
(arrives with M1), and gap filling (M6).

## References

* NIST SP 1065 (Riley), §4 on data gaps and editing.
