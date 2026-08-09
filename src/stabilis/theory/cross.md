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

with `k = i − j`. Take the three correlation *shapes* to be the same —
they are the same estimator at the same averaging factor, which is a far
weaker assumption than naming a noise type — and the double sum over `i`
and `j` collapses onto exactly the `Σρ²` that defines the equivalent
degrees of freedom. What is left is

    Var(cross_var) = (auto_a · auto_b + cross_var²) / η

and `sem` is its square root. Two things fall out of that expression that
are worth noticing:

- Setting `b = a` gives `sem = σ²·sqrt(2/η)`, which is the auto-variance
  result the chi-square interval is built on. The cross error bar is not a
  parallel construction; it is the same one. `cross_variance(a, a)` returns
  the auto-variance **bit for bit** and its `sem` matches, and that is a
  test.
- `η` is `eta1` from the coefficients — the noise-type-free estimator of
  `theory/edof.md` — as the geometric mean of the two channels', with
  `eta3` as its usual short-record fallback. Geometric because `η` divides
  the *product* `auto_a · auto_b`, and because that is the choice which
  makes the `b → a` limit above come out exactly.

### The estimator this replaced

The obvious alternative needs no Gaussian assumption at all: estimate the
variance of `mean(p)` directly from the products' own autocovariance, with
the Bartlett weight `1 − k/M` truncated at `4m`, which is the same
Newey–West form `eta1` embodies. It was implemented first. Measured
against Monte-Carlo scatter over 300 realizations at N = 4096:

| m | N/64 | N/32 | N/16 | N/8 | N/4 |
|---|---|---|---|---|---|
| Bartlett on the products | 0.89 | 0.73 | 0.56 | 0.37 | 0.30 |
| Isserlis on `eta1` | 0.99 | 0.99 | 1.03 | 1.05 | 0.89 |

as a ratio to the true scatter — so the first is optimistic by a factor of
three at the long-τ end, in the direction that matters. Lengthening the lag
limit made it worse, and so did the unbiased normalization, which rules out
truncation bias as the cause. The reason is distributional: the products
are heavy-tailed where the coefficients are Gaussian, so each additional
lag of their sample autocovariance contributes more noise than signal and
the sum shrinks towards zero.

`tests/test_cross.py` keeps the rejected estimator and measures both, so
the finding is a live measurement rather than a paragraph.

With asymmetric channels — white PM in one, flicker PM in the other, three
decades apart in level — the equal-shape approximation is at its worst and
the Isserlis form runs 0 to 30 % **high**, which is the safe direction.

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
| the cross form is the auto form | `cross_variance(a, a)` equals `oadev(a)` exactly, and its `sem` equals `σ²√(2/η)` | `tests/test_cross.py` |
| it recovers a clock the auto-variance cannot see | synthetic clock 3× below its channels' noise; cross within 15 % of the record that generated it, auto 50 % high or worse | `tests/test_cross.py` |
| the error bar is the scatter of the estimate | 300 realizations, mean `sem` over measured standard deviation, within ±20 % for every τ up to N/4 | `tests/test_cross.py` (slow) |
| the rejected Bartlett estimator is worse | same realizations, paired | `tests/test_cross.py` (slow) |
| asymmetric channels fail conservatively | white PM against flicker PM, three decades apart | `tests/test_cross.py` (slow) |
| uncorrelated records cross to zero and go negative | every point within 3 `sem` of zero | `tests/test_cross.py` |
| alignment refuses what it cannot fix | rate slip, sub-sample epoch, no overlap, mixed domains | `tests/test_cross.py` |
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
