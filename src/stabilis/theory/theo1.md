# Thêo1

## Definition

    Theo1(m, tau0, N) = 1/(0.75 (N-m)(m tau0)^2)
        * sum_{i=1}^{N-m} sum_{d=0}^{m/2-1}  1/(m/2 - d)
              [ (x_i - x_{i-d+m/2}) + (x_{i+m} - x_{i+d+m/2}) ]^2

NIST SP 1065 eq. (30), §5.2.15. Defined for **even** `m`, `10 <= m <= N-1`,
with

    tau = 0.75 * m * tau0

**not** `m * tau0`. Thêo1 is the only estimator here whose averaging time
is not the averaging factor times the sampling interval, and getting that
wrong shifts every point along the axis while leaving a curve that looks
entirely plausible.

## Which tau, and the second SP 1065 contradiction

The handbook prints the averaging time two ways, on facing pages. Page 29,
under eq. (30): "an averaging time `tau = 0.75 (m − 1) tau0`". Page 30,
twice: "the averaging time, tau, associated with a Thêo1 value is
`tau = 0.75·m·tau0`", and under eq. (31) again. Table 3's EDF formulae take
`r = 0.75m`. This library shipped the `(m − 1)` reading first. It is wrong.

Three independent arguments, none of which requires trusting a printing:

**1. Eq. (33) only parses under `0.75 m tau0`.** ThêoBR averages the ratio
`AVAR(m = 9+3i) / Thêo1(m = 12+4i)`, which is meaningless unless the two
sit at the same tau. They do, exactly, under this reading —
`0.75(12+4i) = 9+3i` — and pair with nothing under the other.

**2. The advertised reach is exact.** At the largest `m = N−1`,
`0.75 (N−1) tau0 = 0.75 T`. The other reading falls short of the 75 % that
is the estimator's entire selling point.

**3. Measurement.** Thêo1 is unbiased against AVAR for white FM (Table 2:
`a = 1.00, b = c = 0`), so the right tau is the one where the ratio is one.
The two candidates differ by `m/(m−1)`, so this only says anything at small
`m`. 60 realizations, N = 8192:

| m | Thêo1/AVAR at `0.75(m−1)tau0` | at `0.75 m tau0` |
|---|---|---|
| 10 | 0.901 | **1.002** |
| 12 | 0.917 | **1.002** |
| 16 | 0.938 | **0.999** |
| 24 | 0.960 | **1.000** |

The `(m − 1)` column is low by precisely the factor it introduces. The two
converge as `m` grows, which is how this survived a suite that checked
unbiasedness only for `m >= 100` — the lesson being that a test for an
asymptotic property has to be run where the candidates disagree.

## Why it exists

The Allan variance stops at `tau = (N-1)/2 * tau0` — it needs two adjacent
averaging windows inside the record. Thêo1 reaches **75 % of the record**,
by averaging over a family of overlapping window pairs of different widths
instead of a single pair. The inner sum over `d` is that family; the
`1/(m/2 - d)` weight stops the wide, sparse pairs dominating.

Measured on a 4097-point phase record at 1 Hz: Thêo1's longest tau is
3072 s — exactly 0.75 of the 4096 s record — against the overlapping Allan
variance's 1024 s.

## The missing 0.75 in eq. (30)

**Equation (30) as printed omits the `0.75` in the denominator**, and this
library restores it. That is a deliberate departure from the printed
source, so here is the evidence.

Thêo1 is documented as an unbiased estimator of AVAR at large `m`. Measured
over 60 realizations of white FM at N = 8192, Thêo1 against an interpolated
overlapping AVAR *at Thêo1's own tau*:

| m | as printed | with the 0.75 restored |
|---|---|---|
| 10 | 0.749 | **0.999** |
| 16 | 0.754 | **1.005** |
| 64 | 0.755 | **1.007** |
| 100 | 0.758 | **1.010** |
| 404 | 0.753 | **1.004** |
| 1020 | 0.736 | 0.981 |

As printed it is uniformly 0.75 of AVAR — exactly the missing factor. With
the 0.75 restored it sits at one across the whole range, which is the
documented property. The factor appears in the denominator in Howe's own
formulation. (The last row drifts because AVAR itself is running out of
data there, not Thêo1; at `m = 1020` the Allan estimate at that tau rests
on a handful of terms.)

This is the **second** printing slip found in SP 1065 by measurement rather
than by reading; the tau contradiction above is the third. The first was
the flicker-PM conversion row, where the handbook prints two mutually
inconsistent forms — see `theory/powerlaw.md`. All are recorded rather than
silently corrected.

## Remaining bias, and TheoBR

With the 0.75 restored, Thêo1 is unbiased for white FM but not for the
steeper noises. Measured at `m >= 100`, N = 8192:

| noise | Thêo1 / AVAR |
|---|---|
| white FM | 1.00 |
| flicker FM | 0.55 |
| random-walk FM | 0.37 |

These live in `stats.bias.THEO1_BIAS_MEASURED` and are **measured here, not
published**. They are there to be *reported*, never silently applied: a
number divided by a ratio measured on synthetic noise is not the same thing
as one corrected by a published estimator.

**TheoBR** is that published estimator — it removes the bias by averaging
Thêo1 against AVAR over the range where both are defined — and **it is not
implemented yet**, nor is ThêoH, the hybrid that uses AVAR at short tau and
TheoBR beyond it.

## Validation status

**Validated.** `tests/test_theo1.py`:

* the averaging factors are even and at least ten, per the definition;
* `tau = 0.75 m tau0` including its scaling with `tau0`, **and which of the
  handbook's two readings that is**, measured at small `m` where they
  differ by 11 % rather than at large `m` where they agree;
* the reach: exactly 0.75 of the record, and more than 2.5x the Allan
  grid's;
* **the missing 0.75**, by the ensemble measurement above, asserting both
  that the corrected form is unbiased and that the printed form is not;
* the FM biases against their recorded values.

One trap the tests hit and now guard: Thêo1 reaches `0.75 N` while the
Allan grid stops at `N/4`, and `np.interp` **clamps** past its last point
rather than extrapolating. Comparing the two out there measures the clamp,
not the estimator, and produces a convincing-looking fall-off.

## References

* NIST SP 1065 (Riley), §5.2.15, eq. (30).
* D. A. Howe and T. K. Peppler, "Very long-term frequency stability: estimation
  using a special-purpose statistic", Proc. 2003 IEEE Int. Freq. Cont. Symp.
* D. A. Howe, "ThêoH: a hybrid, high-confidence statistic that improves on
  the Allan deviation", Metrologia 43, S322 (2006). **Not implemented.**
