# Masks: deciding whether a unit meets a standard

"Does this unit meet spec" is the question a calibration report exists to
answer. A telecom standard asks it as a **mask**: a piecewise limit on
`MTIE(tau)` or `TDEV(tau)`, and a measured curve that must stay under it.

Drawing the limit line is a plotting flourish. Producing a *verdict* is
not, and three things about it are not obvious.

## 1. A maximum and a variance are not judged the same way

For a variance the rule is already settled in `theory/report.md`: the
verdict is taken on the **confidence interval**, not the point estimate.
Pass only when the whole interval is below the limit, fail only when the
whole interval is above it, otherwise `indeterminate`.

**MTIE is not a variance**, and the same rule applied to it would be both
wrong and useless. MTIE is an observed extreme, and extremes are one-sided
in a way that means the two directions carry different weight:

| observation | what follows |
|---|---|
| measured MTIE **exceeds** the limit | **fail**, conclusively. The excursion happened; no statistics are needed to believe it. |
| measured MTIE is **under** the limit | **pass over the window measured**, and no more. A record twice as long can only find a larger maximum, never a smaller one. |

So exceeding a limit fails on the spot without an interval, and staying
under one is a pass whose scope is the observation window. The window is
therefore part of the verdict rather than a footnote to it, and
`MaskCheck.reason` says so on every passing MTIE point.

This is why `evaluate` branches on `StabilityResult.is_variance` rather
than on the estimator's name — the same flag that keeps the chi-square
interval and the power-law decomposition away from MTIE and TIE rms.

## 2. A standard's measurement conditions are part of its limit

G.811 does not merely give numbers. Clause 6 also requires:

* an equivalent **10 Hz, first-order low-pass** measurement filter;
* a sampling time **`tau0` no coarser than 1/30 s**;
* for TDEV, a record of at least **twelve times** the integration period
  (`T = 12*tau`).

A curve that violates any of these has not been compared with the
standard, whatever the numbers say. So those points come back
`not_evaluated` — not `pass`. A 1 Hz record judged against G.811 produces
no verdict at all, which is the correct and surprising answer, and a much
more useful one than a confident green tick obtained under conditions the
standard does not recognise.

The filter is recorded and **not** applied: this library cannot verify what
the measurement chain did, and silently filtering the data to make it
comply would be manufacturing conformity.

## 3. A limit is never interpolated, and never extrapolated

Each mask states the tau range over which it applies, and outside it there
is no limit to compare against. G.811's TDEV mask ends at 10 000 s; beyond
that the Recommendation states no requirement, so neither does this.
Continuing the last segment would be inventing a requirement and attributing
it to the ITU.

This is the same rule `report.spec_afs` follows for datasheet points, and
for the same reason.

## The catalogue, and why it was fetched

Masks are transcribed from the standard's own text, converted to SI seconds
at transcription time so nothing downstream handles microseconds, and each
carries its citation into the printed output.

Fetching rather than recalling mattered here. The widely repeated form of
G.811's MTIE limit above 1000 s is a **flat 300 ns ceiling**. The
Recommendation says `10⁻⁵τ + 0.29 µs`, which *rises*: the two agree only at
τ = 1000 s and differ by 30 % at 10 000 s. That is exactly the class of
error `CLAUDE.md` forbids inventing, and it would have been invisible in
any test written from the same memory.

**Transcription is checked by arithmetic, not by eye.** Both G.811 masks
are continuous at their breakpoints while being written differently on
either side, so agreement there is a real check that the digits went in
correctly and a mistyped coefficient appears as a step:

| mask | breakpoint | from below | from above |
|---|---|---|---|
| MTIE | 1000 s | `0.275e-3·1000 + 0.025` = 0.3 µs | `1e-5·1000 + 0.29` = 0.3 µs |
| TDEV | 100 s | 3 ns floor | `0.03·100` = 3 ns |
| TDEV | 1000 s | `0.03·1000` = 30 ns | 30 ns ceiling |

## The margin, not only the verdict

A unit that passes by 1 % and one that passes by 40 % are not the same
unit, so `MaskResult.worst` reports the evaluated point with the least
headroom and the margin there. Points that were *not evaluated* have no
margin and cannot become the worst one — absence of a comparison is not a
narrow pass.

The overall verdict is the worst one present, with `not_evaluated` ranked
outside the ordering rather than at the bottom of it: a curve where nothing
could be compared is not a failing curve, and it is certainly not a passing
one.

## Validation status

**Validated.** `tests/test_mask.py`:

* **every verdict against hand evaluation** — five taus either side of both
  breakpoints, limits worked out in the test from the text of G.811, and
  each verdict asserted, which is the M8 milestone criterion;
* both masks continuous at all three breakpoints, and each limit against
  hand arithmetic at four taus;
* no extrapolation past a mask's stated range, in either direction;
* an exceeded MTIE limit failing without an interval, and an unexceeded one
  passing only over the measured window with the reason attached;
* a variance judged on its interval — a mask placed exactly on the measured
  points yields `indeterminate` everywhere, and a generous one passes;
* a record sampled at 1 Hz, and a record shorter than `12*tau`, both
  yielding `not_evaluated` rather than `pass`, and conditions that cannot
  be checked at all not being assumed met;
* the worst margin reported, and points with no limit excluded from it;
* a mask refusing a curve of a different statistic.

**Not validated.** The catalogue holds one Recommendation. G.812, G.813 and
G.823 are the obvious next entries and each is a straightforward
transcription of the same shape — but each needs its own fetched source,
and none is in the repository yet. Nothing here has been checked against a
commercial wander analyser's verdict on the same record, which is the
interop tier this feature would eventually want.
