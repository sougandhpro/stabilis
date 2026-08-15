# The measurement report, and deciding conformity

`stabilis validate` is evidence that the *software* computes what it
claims. `stabilis report` is evidence about *one measurement*. Both are
needed and they are not interchangeable: a good oscillator proves nothing
about the library, and a validated library proves nothing about the unit
on the bench.

This page is about the one genuinely contentious thing the report does —
turning a stability curve and a datasheet limit into a verdict.

## The point estimate is the wrong thing to compare

The obvious implementation compares `sigma_y(tau)` to the limit and prints
PASS or FAIL. In conformity-assessment language (ILAC-G8, ISO/IEC 17025
clause 7.8.6) that is **simple acceptance** with shared risk, and its
false-accept probability for a unit sitting exactly on the limit is about
50 %. That is defensible when the uncertainty is small against the
tolerance. Here it is not, for two reasons peculiar to this quantity.

**The interval is asymmetric and wide where it matters.** The sampling law
of a variance estimate is chi-square, and at the long-tau end of a record
the effective degrees of freedom fall to single figures. At `eta = 5` the
1-sigma interval on the deviation spans `0.793 sigma` to `1.559 sigma`.
A datasheet limit at the longest tau is precisely where the
measurement is least able to decide anything, and it is precisely where
customers write their specifications.

**The point estimate is biased low.** `sigma_hat` is the square root of an
unbiased variance, and the square root is concave, so

    E[sigma_hat] / sigma = c4(eta) = sqrt(2/eta) * Gamma((eta+1)/2) / Gamma(eta/2)

which is 0.886 at `eta = 2` and 0.9975 at `eta = 100`. The bias runs
*towards passing*, and it is largest exactly where the interval is widest.
Simple acceptance on a short record is therefore not merely a coin toss —
it is a coin weighted in the vendor's favour.

## The rule used here

Guarded acceptance, taken on the interval, both ways:

| condition | verdict |
|---|---|
| `sigma_hi <= limit` | **pass** — the whole interval is inside |
| `sigma_lo > limit`  | **fail** — the whole interval is outside |
| otherwise           | **indeterminate** |

`indeterminate` is not a failure and not a hedge. It is the statement that
this record is too short to resolve conformity at this confidence, and the
remedy is more data. Reporting it is the difference between a measurement
report and an opinion. A tool that has only two outcomes must assign the
straddling case to one of them, and whichever it picks it is asserting
something the data does not support.

The **guarded margin** is `limit / sigma_hi`, so a value above 1 conforms.
The report also prints `limit / sigma_hat` — the margin every other tool
shows — beside it, so the size of the difference is visible rather than
arguable.

The deviations are reported **uncorrected** for `c4`. Every other tool in
the field reports them uncorrected, so a silent correction would look like
a defect when a customer diffs the two; and the verdict rests on
`sigma_hi`, which the argument does not touch. The factor is printed per
row so a reader who wants it can apply it. See `theory/bias.md`.

## A limit is not interpolated

A datasheet states `sigma_y` at particular averaging times. The value
between them is not a specification, it is a drawing convention. So the
report evaluates a spec point only where the analysis actually produced a
point within `SPEC_TAU_RTOL` (2 %) of the stated tau, and otherwise says
so and moves on.

That would make the table mostly empty on an octave grid, which is why
`report.spec_afs` exists: it converts each spec tau to the averaging
factor `m = round(tau_spec / tau0)`, and the caller unions those with the
analysis grid before estimating. The measurement is then made where the
specification lives. Spec taus beyond the grid's `max_af` policy are left
out rather than pushing the analysis past it, and are reported as not
evaluated.

The figure follows the same rule: spec points are drawn as open markers
with no connecting line. Drawing a line would assert the interpolation the
table refuses.

## What a two-source measurement can conclude

Every phase record here is a *comparison*. Under independence,

    sigma2_meas = sigma2_dut + sigma2_ref + sigma2_channel

so `sigma_meas >= sigma_dut` always. The consequences are asymmetric and
the report states them:

* a **pass is conclusive** — the unit is at least as good as reported,
  because the reference and the channel can only have made it look worse;
* a **failure is not attributable** — it may belong to the reference or to
  the measurement channel.

Separating them needs a third oscillator, or a back-to-back run to
measure the channel floor. That is what the bench model is for; see
`theory/bench.md`. Until then, a failing report is a statement about the
*pair*, and saying otherwise is the commonest way a stability measurement
is misread.

## Two kinds of limit, kept apart

A report can carry both a manufacturer's datasheet claim (`spec`) and a
published limit (`masks`), and they are rendered in separate tables on
purpose. Both are limits; they are not interchangeable evidence. A
datasheet is a supplier's claim about a product, and a Recommendation is a
requirement an operator has to meet — merging them into one table would let
a reader lose which is which, and a conformity statement whose provenance
is ambiguous is worth less than none.

A failing mask fails the whole report: it cannot pass overall while failing
a limit it was asked to judge against. A mask whose points were all
`not_evaluated` contributes nothing, for the same reason an unevaluated
spec point does. See `stabilis docs mask` for how those verdicts are
reached, including why an extreme-value statistic is judged differently
from a variance.

## Validation status

**Validated.** `tests/test_report.py`:

* the decision rule at its two boundaries — a limit set exactly to
  `sigma_hi` passes, one a hair below it goes indeterminate, and one below
  `sigma_lo` fails;
* the guarded margin is never looser than the margin on the point
  estimate, at every evaluated tau of a real curve;
* `spec_afs` lands the analysis exactly on each spec tau, and a spec tau
  that no grid point reaches is refused rather than interpolated;
* a report built without an interval evaluates no spec point and returns
  no verdict, rather than falling back to simple acceptance;
* the digest is invariant to the operator, the title and the timestamp,
  and changes when a limit or a measured value changes;
* the rendered HTML is self-contained — no external reference of any kind
  — and escapes what it is given.

## What the report deliberately does not claim

It states no traceability chain and makes no accreditation claim. It is a
measurement report, not a calibration certificate. The reproducibility
claim it does make is narrow and checkable: the digest covers the rows and
the conformity decisions and not the timestamp, the operator or the
platform, so re-running the same analysis on the same file reproduces the
same sixteen characters.
