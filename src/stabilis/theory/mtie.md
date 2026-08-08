# MTIE and TIE rms

The two telecom statistics, and the two that are **not variances**.

## Definitions

    MTIE(tau)    = max over all windows of length tau of ( max x - min x )
    TIE rms(tau) = sqrt( mean_i [ (x[i+m] - x[i])**2 ] )

Both are defined on phase (time error) data. A window of averaging time
`tau = m*tau0` spans `m+1` phase samples, and there are `N - m` of them.

They answer a question the Allan family cannot: **how far apart can two
clocks get**. That is what a timing-distribution or telecom specification
is written against — ITU-T masks are MTIE masks — and it is a peak
quantity, not a dispersion.

`mtie` refuses frequency input rather than integrating it. MTIE is a spread
of *time errors* in seconds, and integrating frequency introduces an
arbitrary constant that the answer would then depend on. Convert
deliberately with `types.to_phase` if that is what you mean.

## Neither is a variance, and they fail it differently

Both results carry `is_variance=False`, and three places check that flag
rather than a list of names — so a plugin estimator cannot slip past a
blacklist that never heard of it:

| consumer | why it refuses |
|---|---|
| `stats.confidence` | no chi-square sampling distribution |
| `stats.powerlaw.decompose` | the sum over power laws does not hold |
| `bench.solve` | the sum rule `σ²_meas = σ²_i + σ²_j + σ²_c` does not hold |

**MTIE is an extreme value.** The maximum of a sum is not the sum of
maxima, so for the bench solver the sum rule is not approximate but false.
Its sampling distribution converges to a Gumbel law rather than a
chi-square, and — the part worth internalising — **the spread of MTIE grows
with record length rather than shrinking**. Watch longer and you will
eventually see a bigger excursion. An error bar that narrows with more data
would say the opposite of what is true, so no interval is offered at all.

**TIE rms is a mean square, and its stochastic part *does* obey the sum
rule.** It is refused for a different reason: it retains any frequency
offset between the two sources, and an offset is deterministic. Two clocks
each 1e-12 fast can differ by 2e-12 or by nothing, depending on sign — so
the quantity does not combine in quadrature the way the solver assumes.
Calling both "extreme-value statistics" would be wrong about this one.

That TIE rms keeps the frequency offset is the definition, not a defect: a
clock fast by a fixed amount contributes `offset * tau`, which dominates at
long tau. The Allan family removes exactly that term; TIE rms keeps it,
because for timing distribution the offset is the thing you care about.

## The algorithm, and why it is not the obvious one

Naive MTIE is `O(N*m)` per averaging factor and `O(N²)` over a grid —
unusable at the record lengths MTIE is actually wanted for.

Instead, **binary doubling**. With `T_j[i]` the extreme over
`x[i : i+2**j]`,

    T_0 = x,    T_j[i] = max( T_{j-1}[i], T_{j-1}[i + 2**(j-1)] )

any window of length `L` is then covered by two overlapping blocks of
length `2**j` with `j = floor(log2(L))`:

    window_max[k] = max( T_j[k], T_j[k + L - 2**j] )

**The overlap is harmless only because max is idempotent** — `max(a,a) = a`.
The same trick applied to a sum would double-count, which is why this
structure works for extremes and nothing else.

Averaging factors are visited in increasing order and the table advances
one level at a time, so only the current level is held: `O(N)` memory and
`O(N log N)` work for the whole octave grid, in vectorized numpy with no
Python loop over samples.

Measured: **1 000 000 points over 18 octave taus in 0.15 s.**

## Gaps

A window touching a gap comes back non-finite and is dropped, the same
convention `run_estimator` uses for coefficients spoiled by a gap. It is
never silently shortened — an MTIE computed over fewer samples than it
claims would be biased low, and low is the dangerous direction for a peak
statistic used against a limit mask.

## Validation status

**Validated.** `tests/test_mtie.py`:

* **the fast algorithm against brute force**, at *every* window length from
  1 to 40, on data rounded to produce many ties and equal runs — exact
  equality, since both take the maximum of the same finite set. Ties are
  where an off-by-one in the tail index would hide;
* the incremental table reuse gives bit-identical results to computing each
  window length from scratch;
* `mtie` reproduces the definition on a real record to 1e-12 relative;
* **monotonicity in tau**, which holds of the quantity itself and not
  merely of an estimate: a longer window contains every shorter one, so its
  spread cannot fall. No amount of noise can make that flaky;
* a pure phase ramp gives `MTIE(tau) = a*tau` and
  `TIE rms(tau) = offset*tau` exactly, to 1e-13;
* the three refusals above, including one keyed on a disguised plugin
  result to prove the flag is doing the work rather than the name.

## References

* ITU-T G.810 / G.811 / G.823 for the mask definitions MTIE is measured
  against.
* NIST SP 1065 (Riley) §5 for MTIE and TIE rms in the wider family.
* S. Bregni, "Fast algorithms for TVAR and MTIE computation", for the
  binary-decomposition idea. **Not read for this implementation** — the
  doubling structure here is the standard sparse-table one and was written
  directly; the citation is where a reader should go for the survey.
