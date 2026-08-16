# Dynamic Allan variance — stability as a function of when you looked

## What it is

A single σ_y(τ) curve presumes the oscillator was doing the same thing all
along. Real ones are not: they warm up, they get bumped, the air
conditioning cycles, an antenna sees a different sky. Galleani and
Tavella's dynamic Allan variance is the obvious repair — slide a window
along the record and evaluate the ordinary estimator inside it, giving a
surface σ_y(τ, t) instead of a curve.

Their estimator, eq. (9) of the FCS 2009 paper, is

    sigma^2_y[n,k] = 1/(2 k^2 tau0^2) · 1/(Nw - 2k)
                     · sum_{m = n-Nw/2}^{n+Nw/2-2k-1}
                       (x[m+2k] - 2 x[m+k] + x[m])^2

which is exactly the overlapping Allan variance of the `Nw` phase samples
centred on `n`. That is worth stating plainly, because it is what keeps
this module a wrapper rather than a second estimator: **DAVAR is not a new
statistic.** It is a windowing policy around one you already trust, and
`stabilis` implements it as literally that.

## Why it is recomputed rather than slid

The same authors' contribution in that paper is a recursion. Writing
`Δ_k[m] = x[m+2k] - 2x[m+k] + x[m]`, their eq. (13) is

    sigma^2_y[n+1,k] = sigma^2_y[n,k]
                       + 1/(2 k^2 tau0^2) · 1/(Nw - 2k)
                       · ( Δ^2_k[n + Nw/2 - 2k] - Δ^2_k[n - Nw/2] )

— add the entering term, subtract the expiring one, O(1) per step instead
of O(Nw). It is correct arithmetic and this library does not use it.

The reason is not speed and not taste. A running sum that both adds and
subtracts has **no self-correcting mechanism**: once a large term has been
added and subtracted, the bits it displaced from the accumulator do not
come back, and the error stays for the rest of the record. Measured, on
200 000 samples of white FM with `Nw = 2000` and `k = 1`, comparing the
recursion against a direct sum of the same window at every one of the
198 000 steps:

| record | worst relative error of eq. (13) |
|---|---|
| white FM, no disturbance | 3.7e-14 |
| random-walk FM, no disturbance | 1.5e-14 |
| white FM with one 1 µs phase step | 1.5e-6 |

**The hazard is dynamic range, not run length.** Two hundred thousand
slides of ordinary noise cost thirty-seven parts in a quadrillion; one
glitch costs eight orders of magnitude more than that. And it is
permanent rather than transient — the error after the step has left the
window entirely is 1.3e-6, essentially the same as while it was inside.
It scales as the square of the excursion:

| phase step passing through the window | final error, 100 000 steps later |
|---|---|
| none | 2.4e-14 |
| 1 ns | 3.5e-12 |
| 1 µs | 1.3e-6 |
| 1 ms | **28 %** |

A millisecond phase step is an ordinary counter glitch. The sliding form
would report a surface that is a quarter wrong for the remainder of the
measurement, with no symptom anywhere in it and every appearance of a
physical trend — which is precisely what a DAVAR is read to look for.
Compensated summation does not rescue this; the cancellation happens in
the subtraction, not the accumulation.

What recomputation costs instead is `window/step` times one batch
evaluation, which for any window worth plotting is a fraction of a second.
What it buys, beyond correctness:

* **`davar(x, window=len(x))` is bit-identical to `oadev(x)`** — not equal
  to 1e-15, identical, because it is the same call on the same data. That
  is a test rather than a claim, and it is only possible because nothing
  here reimplements the estimator.
* **Every estimator gets a dynamic form for free.** The surface is built
  by registry name, so `kind="ohdev"` gives a dynamic Hadamard deviation,
  `kind="mdev"` a dynamic modified one, and a third-party plugin gets its
  own without writing any code.
* **Gaps, EDOF and confidence intervals are the batch machinery's**, not a
  second implementation of them that would have to be validated again.

## What the surface says

A worked case. 8192 s of white FM at σ_y(1 s) ≈ 7e-13, with the frequency
noise multiplied by five for one thousand seconds in the middle — a
disturbance and then a recovery, which is what a real fault looks like.

The whole-record `oadev` reports

| τ | 1 s | 4 s | 16 s | 64 s |
|---|---|---|---|---|
| σ_y | 1.44e-12 | 7.00e-13 | 3.56e-13 | 1.76e-13 |

and the slope is still −1/2. Read on its own it says: *a white-FM
oscillator, twice as noisy as this one actually is*. It describes neither
state the clock was in. The surface at τ = 1 s says something else:

| window centre t (s) | 512 | 2304 | 2816 | 3584 | 4352 | 4608 | 7680 |
|---|---|---|---|---|---|---|---|
| σ_y(1 s) | 6.90e-13 | 7.04e-13 | 2.27e-12 | 3.45e-12 | 1.58e-12 | 6.92e-13 | 7.05e-13 |

The quiet level is 6.9e-13, the disturbance peaks at 3.45e-12, and the
recovery is complete. The single number 1.44e-12 is 2.1× the quiet level
and 0.42× the peak.

Note what the ridge's *width* is: the window, not the event. A 1000 s
disturbance seen through a 1024-sample window occupies about 2000 s of the
surface, because the window is contaminated from the moment it first
touches the event until it last does. Deconvolving that is not attempted
here; the window length is on the plot title so a reader can subtract it
by eye.

## The conventions this implementation fixes

**Only whole windows.** Windows run from `start = 0` in steps of `step`
while `start + window <= n`, so a truncated window is never evaluated. A
short window has fewer terms, a shorter longest τ and different degrees of
freedom, and drawing it beside the others produces an edge artefact that
reads exactly like a transient at the start of the record. The uncovered
half-window at each end is recorded in provenance as `uncovered_head_s`
and `uncovered_tail_s` rather than filled.

**The window is in samples.** Rounding a duration to samples silently is
how two runs of the same analysis end up on different τ grids.

**`t` is the centroid of the window's samples**, `epoch + (start +
(Nw-1)/2)/rate` for phase. Galleani and Tavella's `n` indexes the window
`n - Nw/2 … n + Nw/2 - 1`, whose centroid is `n - 1/2`; so this
convention sits half a sample below theirs for even `Nw`, and is defined
for odd `Nw`, which theirs is not. Half a sample is nothing on a plot and
everything when a surface is being lined up against a bench log. A
frequency sample is an interval rather than an instant, so its windows'
centres sit a further half sample later, and the code branches on that.

**A τ a window cannot estimate is NaN, not omitted.** Gaps shorten a
window's longest contiguous run, so the same τ may be available in one
window and not the next. Omitting it makes the surface ragged and
unplottable; filling it invents a number. NaN is the third option. A τ
that *no* window could estimate is dropped entirely and the count goes
into provenance — an all-NaN column is not information.

**The normalization is the surviving term count, not `Nw - 2k`.** These
agree exactly on a gap-free window, which is the case eq. (9) is written
for. Where they differ, a gap has removed terms, and dividing by the
nominal count would report a variance biased low by exactly the fraction
of the window that is missing.

## Confidence intervals

Pass `level` and each window gets one, computed while its coefficients are
still in memory. There is no way to add an interval afterwards: the
data-driven EDOF is a function of the coefficient arrays, and the surface
does not keep them — `window/step` copies of a window's coefficients is
the one allocation that would make this module expensive.

`window_result(k)` hands back one window as an ordinary `StabilityResult`
for plotting or export. It carries no coefficients, so `confidence()`
refuses it rather than guessing, which is the same refusal
`stream.online` gets and for the same reason.

## Validation status

**Validated by identity against the batch estimator, which is stronger
here than a reference table would be — it is exact rather than to six
figures.** The surface has no arithmetic of its own; what is tested is
the windowing, and the tests are written so that any shortcut in it
(re-zeroing a window, re-deriving the τ grid, reusing a coefficient
array) shows up as a value equal to 1e-15 instead of identical.

| claim | how it is checked | where |
|---|---|---|
| `window = len(record)` is the batch call | `np.array_equal` on `var`, `af`, `tau_s` and `n_terms`; again for `ohdev` and for frequency input | `tests/test_davar.py` |
| every row is the batch call on its own slice | each window re-evaluated independently, compared bit for bit | `tests/test_davar.py` |
| the identity survives gaps | a six-sample dropout, whole-record window | `tests/test_davar.py` |
| a disturbance shows as a ridge where it happened | white FM ×5 for 1000 s of 8192; peak/quiet > 4, whole-record value between the two and matching neither | `tests/test_davar.py` |
| the eq. (13) recursion loses significance permanently | the recursion reproduced and slid; a 1 µs step costs > 1e-8 and > 1000× the clean case, long after it has left the window | `tests/test_davar.py` |
| whole windows only, ends reported | last start plus window fits, next would not; `uncovered_head_s` in provenance | `tests/test_davar.py` |
| the centre time is the window's centroid, +½ sample for frequency | asserted for both domains | `tests/test_davar.py` |
| an unreachable τ is NaN, an everywhere-unreachable τ is dropped and counted | a 350-sample gap against a 512-sample window | `tests/test_davar.py` |
| intervals are per window and only when asked for | bounds bracket the estimate, `edof > 0`, `None` otherwise | `tests/test_davar.py` |
| refusals | window over the record, window < 3, step < 1, too many windows, an estimator that ignores the τ list, a grid no window can reach | `tests/test_davar.py` |

`stabilis validate` carries the window-equals-record identity as an
`identity` row, asserted at exactly zero difference.

## What is not validated

The surface inherits every property of the estimator inside it, which is
where its evidence comes from — the window-equals-record identity is bit
exact, and everything else in `theory/avar.md`, `theory/ci.md` and
`theory/gaps.md` applies unchanged inside each window. What has **no**
second implementation to differ from is the surface itself: neither
Stable32 nor `allantools` computes a DAVAR, so there is no published
table, exactly as for the cross-variance (`theory/cross.md`). The
identity against the batch estimator is what stands in for it, and it is
stronger than a table would be, being exact rather than to six figures.

Not implemented, and worth naming rather than leaving as a silence: the
DAVAR's own confidence bands *as a surface*. The per-window intervals here
are each correct for their own window and are **not** simultaneous — a
surface of `n_windows × n_taus` intervals at 68.3 % each will show
excursions outside the band somewhere with near certainty, and reading
those as events is the mistake this whole tool exists to prevent. Adjacent
windows overlapping by `1 - step/window` of their terms are also strongly
correlated, so the excursions come in runs. Treat the bands as a per-point
scale, not as a detection threshold.

## References

* L. Galleani and P. Tavella, "Fast Computation of the Dynamic Allan
  Variance," *Proc. 2009 IEEE Int. Frequency Control Symposium*, pp.
  685–687. **Read in full 2026-08-16**; eq. (9) and eq. (13) above are
  transcribed from it. The paper presents the recursion for speed and
  contains no discussion of its numerical accuracy — the words
  "accuracy", "precision", "rounding" and "error" do not appear in it in
  that sense, which is why the measurements above were made here.
* L. Galleani and P. Tavella, "The Dynamic Allan Variance," *IEEE Trans.
  UFFC* **56**(3), 450–464, March 2009. The full treatment. Not fetched;
  reference [2] of the paper above, which restates the definition this
  implementation follows.
