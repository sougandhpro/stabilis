# ThêoBR and ThêoH

## Definition

Thêo1 reaches 75 % of the record but is biased against AVAR for every noise
type except white FM. There are two ways to remove that bias. `stats.bias.
theo1_bias` uses SP 1065 Table 2 and **needs the noise type**:

    AVAR/Thêo1 = a + b / m^c

| noise | alpha | a | b | c |
|---|---|---|---|---|
| RW FM | −2 | 2.70 | −1.53 | 0.85 |
| F FM | −1 | 1.87 | −1.05 | 0.79 |
| W FM | 0 | 1.00 | 0.00 | 0.00 |
| F PM | 1 | 0.14 | 0.82 | 0.30 |
| W PM | 2 | 0.09 | 0.74 | 0.40 |

ThêoBR removes it **without** the noise type, which is the point of it, by
measuring the ratio off the data:

    ThêoBR(m,tau0,N) = [ 1/(n+1) sum_{i=0}^{n}
                             AVAR(m = 9+3i, tau0, N)
                           / Theo1(m = 12+4i, tau0, N) ] * Theo1(m,tau0,N)

    n = floor(N/6) - 3

SP 1065 eq. (33), §5.2.16, from Howe 2006. The two averaging factors are
paired **at equal tau** — `0.75(12+4i) = 9+3i` — and that pairing is one of
the three arguments that settled Thêo1's own tau convention, since it holds
under `tau = 0.75 m tau0` and under nothing else. See `theory/theo1.md`.

ThêoH then puts AVAR and ThêoBR on one axis, eq. (34):

    ThêoH(m,tau0,N) = AVAR(m)    for  m tau0 <= k
                      ThêoBR(m)  for  0.75 m tau0 > k,  m even

with `k` the largest available AVAR tau at or below 20 % of the record. One
curve from `tau0` to `0.75 T`, with no noise-type assumption anywhere in
it. On a 1024-point record at 1 Hz the join lands at 128 s and the curve
runs to 768 s, exactly three quarters of the record.

## The bias ratio is noisy, and the noise is common to every point

This is the part a user has to know, and it is not in the handbook.

Measured over 12 realizations of **white FM**, where the true ratio is 1.00
by definition:

| N | mean ratio | single records |
|---|---|---|
| 1024 | 1.07 | 0.53 – 1.68 |
| 2048 | 1.01 | 0.47 – 1.69 |
| 4096 | 0.91 | 0.50 – 1.23 |

The mean is right. The spread barely improves with record length, because
the average is dominated by AVAR terms at `m` up to `N/2`, and those have a
handful of degrees of freedom however long the record is.

**The consequence is structural, not statistical scatter.** ThêoBR
multiplies every tau by *one* draw from that distribution, so the error is
a common vertical shift of the whole curve. It does not average away by
reading several taus, and two ThêoBR curves from the same oscillator can
sit a factor apart while each looks internally consistent. `bias_ratio` is
public and the value is in provenance so it can be reported beside the
curve rather than hidden inside it.

Against the published asymptotes, the ensemble means are good — at N = 1024
over 12 realizations, 1.07 against `a` = 1.00 for white FM and 1.86 against
1.87 for flicker FM. Random-walk FM came out at 2.30 against 2.70, which is
within the scatter of 12 draws of that noise but is not a tight check.

Averaging the ratio term by term is what eq. (33) prints. **Pooling
instead** — `sum(AVAR)/sum(Theo1)` over the same terms — was measured at the
same mean with about half the spread: 0.82–1.32 against 0.53–1.68 at
N = 1024. That is recorded rather than substituted, because a parity claim
against Stable32 has to mean the published definition.

## Cost

The bias sum runs about `N/6` Thêo1 evaluations at averaging factors
reaching `2N/3`, so it grows as roughly `N³`:

| N | element operations | measured |
|---|---|---|
| 1025 | 1.7e7 | 0.36 s |
| 2049 | 1.3e8 | 1.8 s |
| 4097 | 1.1e9 | 8.7 s |

`theobr` estimates that cost before starting and refuses past
`MAX_BIAS_OPS` = 4e9, about half a minute here and reached near N = 6500,
rather than appearing to hang. Howe's Fast TheoBR (McGee & Howe 2008)
phase-averages the record first and is not implemented; decimating the
record is the way through today, and SP 1065 notes that phase averaging has
negligible effect on the bias removal.

## Validation status

**Validated.** `tests/test_theoh.py`: the eq. (33) term pairing lands at
equal tau; the ratio recovers the published Table 2 asymptotes for white
and flicker FM over an ensemble; ThêoBR is Thêo1 times exactly that ratio;
ThêoH is AVAR below the crossover and ThêoBR above it, point for point,
with the crossover at the largest Allan tau under 20 % of the record; the
reach is 0.75 T; frequency input is refused; and the cost guard fires
before a long record does.

`stats.bias.theo1_bias` is validated against Table 2 directly, including
that white FM is exactly 1 at every `m`.

## Not yet

* **The Thêo1 EDF**, SP 1065 Table 9 — five power-law formulae in `N` and
  `tau`. Until it is here the Thêo family has no confidence interval, since
  it reports no coefficient arrays for the data-driven path either, and the
  CLI offers no interval flags for it rather than offering ones that would
  fail.
* **Fast TheoBR**, the phase-averaged form that makes long records
  practical.
* Whether Stable32's ThêoH agrees with this one point for point. It is in
  the parity matrix and needs a session on the Windows machine.

## References

* NIST SP 1065 (Riley), §5.2.15–5.2.16, eqs. (31), (33), (34), Tables 2, 9.
* D. A. Howe, "ThêoH: a hybrid, high-confidence statistic that improves on
  the Allan deviation", Metrologia 43, S322 (2006).
* D. A. Howe and T. N. Tasset, "Thêo1: characterization of very long-term
  frequency stability", Proc. 2004 EFTF — NewThêo1, eq. (32), which differs
  from ThêoBR only in taking `n = floor(N/30) - 3`.
* J. McGee and D. A. Howe, "Fast TheoBR", IEEE UFFC 2008. **Not
  implemented.**
