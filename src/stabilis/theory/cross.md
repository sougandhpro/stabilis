# Cross-variance

## What it is

Every estimator in this library is a mean of squared coefficients:
`variance = scale · mean(c²)`. Measure the same clock through two
independent channels and replace one factor with the other channel's
coefficients:

    cross_var = scale · mean(c_a · c_b)

That is the whole estimator. Because the coefficient array is already
carried on every `StabilityResult`, cross-AVAR, cross-MVAR, cross-HVAR and
cross-TVAR are one function rather than four, and a third-party estimator
registered as a plugin gets its cross form without writing any code.

## Why it beats the auto-variance

Write each channel as the clock plus its own measurement noise,

    x_a = x_c + n_a
    x_b = x_c + n_b

The auto-variance of either is `σ²_c + σ²_n`: the instrument is a floor
you cannot get under, however long you log. The difference operator is
linear, so `d_a = d_c + e_a`, and

    E[d_a·d_b] = E[d_c²] + E[d_c·e_b] + E[e_a·d_c] + E[e_a·e_b]
               = E[d_c²]

when the two instrument noises are independent of each other and of the
clock. **The instrument term has expectation zero rather than being
small.** This is what a cross-correlation phase noise test set — the
Microchip 3120A among them — does in hardware. Neither `allantools` nor
Stable32 offers it in software.

Measured on a synthetic clock 3× below its channels' noise (white FM clock,
white PM channels, N = 8192):

| τ (s) | single-channel σ_y | cross σ_y | true clock σ_y |
|---|---|---|---|
| 1   | 2.08e-12 | 7.07e-13 | 6.98e-13 |
| 8   | 3.43e-13 | 2.44e-13 | 2.47e-13 |
| 64  | 9.48e-14 | 9.03e-14 | 9.04e-14 |

At τ = 1 s the auto-variance is a factor of three high and says nothing
about the clock; the cross estimate is within 1.4 % of truth.

## Three prices, all of them stated

### 1. The estimate is not positive

It is a mean of products, not of squares. At long τ, where few terms
survive and the common part is small, it goes negative routinely. That is
a result, not a failure: it means the shared signal is below the scatter
of the estimate.

So `CrossResult.var` is **signed** and nothing clips it or takes an
absolute value. `.dev` returns NaN where the variance is negative, rather
than zero or `sqrt(|var|)` — a log axis would otherwise draw a noise floor
that is not there, and a reader would take the shape of it seriously.

### 2. Convergence is slow

The instrument noise does not vanish; only its *estimate* averages down,
as `1/√M` in variance and hence `M^(1/4)` in amplitude. Wanting another
factor of ten in amplitude means ten thousand times the data. `rejection_db`
reports what the measurement actually achieved,

    rejection = 10·log10( sqrt(auto_a · auto_b) / sem )

a power ratio of the single-channel floor to the estimate's own standard
error. When the channels are instrument-dominated this reduces to
`5·log10(η)`, the familiar `5 log10 M` law of cross-spectrum measurement —
arrived at here as a consequence of the error bar rather than substituted
for one, so it degrades honestly when the terms overlap or the record is
short.

### 3. Correlated instrument noise breaks it silently

A shared reference, a shared power supply, one thermal environment, two
channels of the same counter: any of these leaves a term in `E[e_a·e_b]`
that does not cancel, and the answer is simply wrong. **There is no
symptom in the data.** No arrangement of the two records tests it, because
the thing being assumed is about the apparatus.

`cross_variance` therefore records the assumption as a provenance warning
on every call rather than mentioning it in a docstring. The same hazard,
at the level of the whole bench, is what `bench.common_mode_warnings`
exists for; see `theory/bench.md`.

## The error bar

The chi-square interval in `theory/ci.md` rests on `σ̂²` being a sum of
squares of a Gaussian. A sum of *products* is not: it is unbounded below,
and its mean is the quantity of interest rather than a scale. So the
interval here is a symmetric standard error, and it comes from Isserlis'
theorem. For jointly Gaussian coefficients,

    Cov(c_a[i]·c_b[i], c_a[j]·c_b[j]) = R_aa(k)·R_bb(k) + R_ab(k)·R_ba(k)

with `k = i − j`, so summing over the `M²` pairs of terms,

    Var(cross_var) = (1/M)·Σ_k (1 − |k|/M)·[R_aa(k)·R_bb(k)
                                            + R_ab(k)·R_ba(k)]

truncated at `4m` lags for the reason `eta1` truncates. **Nothing is
approximated.** All four sequences come out of the same zero-padded FFT
the EDOF machinery already uses — the only new observation is that the
*cross*-correlation of the two coefficient arrays is as computable as the
auto. `multi.cross.isserlis_var` evaluates it, and `reject` calls the same
function with more pairs.

Setting `b = a` makes the bracket `2·R(k)²` and the whole expression
`2σ⁴/η` with `η` **exactly** `eta1`. The chi-square error bar is the auto
case of this one, not a parallel construction that could drift from it —
`cross_variance(a, a)` returns the auto-variance bit for bit *and* its
interval, verified to the last bit rather than to a tolerance. The `η`
reported on the result is this relation inverted,
`η = (auto_a·auto_b + cross²)/sem²`, so it stays comparable with every
other degrees-of-freedom number in the library.

Below `MIN_TERMS_FOR_ETA1` terms, or when the truncated lag sum comes out
negative, the error bar falls back to the equal-shape closed form on `η₃` —
a lower bound on the degrees of freedom and therefore an upper bound on
the bar, which is the direction a fallback must err in. `edof_method`
records which path each τ took.

### Two estimators this replaced

Both were implemented, both were measured against Monte-Carlo scatter over
300 realizations at N = 4096, and both are still in `tests/test_cross.py`
being measured on the same realizations — so this is a live comparison
rather than a paragraph.

**A Bartlett sum over the products' own autocovariance**, with the weight
`1 − k/M` truncated at `4m`. It is the obvious thing to do and assumes no
distribution at all.

**The equal-shape closed form.** Take `R_aa`, `R_bb` and `R_ab` to have the
same shape and differ only in scale, and the double sum above collapses
onto the `Σρ²` that defines the EDOF, leaving
`Var = (auto_a·auto_b + cross²)/η` with `η` the geometric mean of the two
channels' `eta1`. Simple, and it is what `cross_sem` still computes.

As a ratio to the true scatter — one means the error bar is right, below
one means it is too narrow. Two channels, N = 4096, 300 realizations, all
three computed on the same records:

| m | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| Bartlett on the products | 0.97 | 0.98 | 1.05 | 1.00 | 0.97 | 0.91 | 0.90 | 0.85 | 0.64 |
| equal-shape closed form | 0.98 | 0.99 | 1.04 | 0.93 | 0.90 | 0.90 | 1.00 | 1.09 | 1.11 |
| exact lag sum | 0.97 | 0.98 | 1.06 | 1.02 | 1.01 | 0.98 | 1.05 | 1.12 | 1.13 |

The two failures have different causes, sit in different places, and
neither is truncation bias.

The Bartlett form decays monotonically towards the **long-τ** end, reaching
0.64 at `m = N/16` here and 0.30 at `m = N/4` on a longer grid — optimistic
by a factor of three where the error bar matters most. Lengthening the lag
limit made it worse and so did the unbiased normalization, which is what
rules out truncation. The cause is distributional: the products are
heavy-tailed where the coefficients are Gaussian, so each additional lag of
their sample autocovariance contributes more noise than signal and the sum
shrinks towards zero.

The equal-shape form dips 7–10 % in the **middle** and is fine at both
ends, and that shape is the diagnosis. At short τ the instrument dominates
and at long τ the clock does; in each limit one process sets all three
correlation shapes, and collapsing them onto one is then exact. In between
they contribute comparably, `R_aa` carries the mixture while `R_ab` carries
only the clock, and the shapes genuinely differ. The dip is systematic
rather than scatter: on M = 3 channels, where averaging three pairs
sharpens it, three independent seeds gave 0.81–0.88 across `m = 8` to `32`
against 0.96–1.04 for the exact sum on the same records.

Both failures are worse for the M-channel average than for one pair, which
is why the exact sum went in before `reject` was written rather than after:
averaging P pairs shrinks the scatter being predicted without shrinking a
systematic error in the prediction.

With asymmetric channels — white PM in one, flicker PM in the other, three
decades apart in level — the exact form runs 0 to 30 % **high**, which is
the safe direction.

## More than two channels

`cross_matrix` computes every pair at once and `reject` averages the
off-diagonal terms:

    reject = (2 / (M(M−1))) · Σ_{i<j} cross_ij

Each is an unbiased estimate of the same clock variance carrying an
independent draw of its own two channels' noise, so the average reaches
deeper than any single pair. The cost is `M` estimator runs and `M(M+1)/2`
array products — not `M²` estimator runs — because the coefficients are
computed once per channel and reused in every pair they appear in.

**How much deeper is not a rule of thumb.** It is `isserlis_var` again,
with every pair passed in instead of one: the covariance of two pair
estimates is the same lag sum with the four subscripts taken from the two
pairs, and the variance of the average is that summed over every ordered
pair of pairs, divided by `P²`. Two channels is one pair, so `reject` on
two channels returns *exactly* what `cross_variance` returns on them —
variance, error bar and η, bit for bit. That is a test, and it is what
keeps the two entry points one implementation.

The pairs are **not** independent: (1,2) and (1,3) share channel 1, and
the shared channel's own covariance appears in their cross term. Treating
`P = M(M−1)/2` pairs as `P` independent estimates claims a rejection that
is not there. Measured on four channels of a clock a factor of three under
them, the naive `5·log10(P)` gain of 3.9 dB is really 3.0 dB at τ = 1 s,
and falls to nothing by τ = 32 s — where the clock dominates, there is no
instrument noise left to reject and extra channels buy nothing at all.
Only when `A_ij ≪ A_ii` do the shared terms vanish and the gain approach
`5·log10(P·η)`.

**Why a plain average and not the best linear unbiased one.** The BLUE
weights the pairs by the inverse of that covariance matrix, and is
genuinely more efficient when the covariance is known. It is not known —
it is the matrix itself, estimated from the very data being weighted — and
inverting a noisy estimate to weight the sample it came from makes the
weights correlate with the noise they are meant to suppress, which is a
well-established way to get an answer that is both wrong and confident.
The unweighted mean has no such feedback and its variance is computed
exactly rather than assumed. When the channels differ enough for weighting
to matter, `Rejection.notes` says so and suggests dropping the loud channel
and comparing, which is a test the data can actually support.

**A dataset unions the gaps across every member, not pairwise.** A channel
with heavy dropouts therefore costs every other channel. That is the price
of a matrix that estimates *one* covariance matrix: done pairwise, entry
(1,2) and entry (1,3) would be averages over different sets of terms, and
combining them — which is exactly what the rejection does — would be adding
numbers that do not belong to the same experiment.

## Alignment, which is the part that bites

The arithmetic above pairs term `i` of one record with term `i` of the
other. Everything therefore rests on those being the same instant, and a
misalignment produces a plausible curve with no symptom — the same class of
failure as correlated channel noise, and rather more common.

`multi/align.py` is deliberately the strict part of the package.
**Nothing is interpolated onto a common grid**, here or anywhere else in
the library, so alignment means only: crop to the window the records share,
by a whole number of samples. Three refusals:

- **Rates that differ.** Not tested for equality, which float64 timestamps
  rarely satisfy, but by the *slip accumulated over the overlap*: two
  counters nominally at 1 Hz differing by 2.4e-7 — which is what an MJD
  timebase infers to, see `theory/bench.md` — drift a quarter of a sample
  apart in a million points and two and a half in ten million. Only the
  record length turns a rate difference into an answer.
- **An epoch off the grid** by a fraction of a sample. No crop fixes it and
  only interpolation would.
- **No overlap**, which usually means the wrong pair of files.

`common_support` is the second half and is just as necessary. Two records
cropped to one window still disagree about *which* samples are missing, and
an estimator drops the terms its own gaps spoil — so two coefficient arrays
from records with different dropouts have different lengths, or worse, the
same length with different terms in them. The union of the gaps is imposed
on both before either is estimated. The cost is real: a sample either
record lost is a sample both lose. It is also the only honest option, since
the product of a term with a term that was never measured is not an
observation.

## Validation status

**Validated functionally against synthetic records whose clock is known.**
The estimator itself needs no reference table: `cross_variance(a, a)`
returns the auto-variance bit for bit, which pins the arithmetic against a
path already validated to 14 correct digits against the exact-rational
oracle. What needs measuring is everything the auto case cannot show.

| claim | how it is checked | where |
|---|---|---|
| the cross form is the auto form | `cross_variance(a, a)` equals `oadev(a)` exactly, and its `sem` equals `σ²√(2/η)` with η exactly `eta1` | `tests/test_cross.py` |
| the M-channel path is the pairwise one | `reject` on two channels equals `cross_variance` bit for bit, error bar and η included | `tests/test_matrix.py` |
| it recovers a clock the auto-variance cannot see | synthetic clock 3× below its channels' noise; cross within 15 % of the record that generated it, auto 50 % high or worse | `tests/test_cross.py`, `tests/test_matrix.py` |
| the error bar is the scatter of the estimate | 300 realizations, mean `sem` over measured standard deviation, within ±20 % for every τ up to N/4; again at M = 3 and M = 4 | `tests/test_cross.py`, `tests/test_matrix.py` (slow) |
| both rejected estimators are worse, and where | same realizations, paired three ways | `tests/test_cross.py` (slow) |
| asymmetric channels fail conservatively | white PM against flicker PM, three decades apart | `tests/test_cross.py` (slow) |
| extra channels buy a real but sub-naive gain | four channels against two, gain positive and below `5 log10 P` | `tests/test_matrix.py` |
| uncorrelated records cross to zero and go negative | every point within 3 `sem` of zero | `tests/test_cross.py`, `tests/test_matrix.py` |
| alignment refuses what it cannot fix | rate slip, sub-sample epoch, no overlap, mixed domains | `tests/test_cross.py` |
| a dataset unions gaps over every member | three channels, three different dropouts | `tests/test_matrix.py` |
| `stats/` never imports `multi/` | AST scan of the whole package | `tests/test_cross.py` |

`stabilis validate` carries the rejection measurement as a `coverage` row.

**Not validated:** there is no interop check, because there is nothing to
check against — neither Stable32 nor `allantools` computes a
cross-variance, so no published table exists and none can be manufactured
honestly. Nor is there a check against real hardware; that arrives with a
two-channel capture from the 3120A. Both are recorded in
`VALIDATION-DEBT.md`.

## What is not here

The three-cornered hat and its N-channel generalization need to know which
oscillator and which instrument each record saw. That is metadata a bare
series does not carry, and it lives in `bench/` — see `theory/bench.md`,
whose *derived edge* section is the closely related result that two records
against a common reference already determine a third pair by subtraction.

The two are complementary and will meet: `Var(r₁ − r₂) = Var(r₁) + Var(r₂)
− 2·Cov(r₁, r₂)`, so the derived edge is one particular use of the
cross-covariance this module estimates directly.
