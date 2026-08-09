# The bench model

Describing the experiment inside the program: which oscillator, against
which, through which channel of which instrument. Once that is written
down, four things become mechanical.

## Why a measurement is not a column of numbers

`x_ab(t)` is never just a column. It is *this* source against *that* one,
through *this* channel of *that* device. Tools that forget this cannot
check their own assumptions — Stable32 has a three-cornered hat but no
notion of where the data came from, so it cannot tell you when the hat does
not apply.

Four things fall out of knowing:

* **N-cornered hat** — needs to know that records share sources pairwise;
* **noise-floor subtraction** — needs to know which records came through
  the same hardware;
* **common-mode detection** — the same reference in several records means
  its noise is correlated across them, so they are not independent;
* **the honest refusal** — telling you a setup *cannot* separate its
  oscillators no matter how long it runs, and naming the measurement that
  would fix it. This is the most valuable of the four.

## One linear system

Under independence, a measurement of sources `i` and `j` through channel
`c` has, at each tau,

    sigma2_meas = sigma2_i + sigma2_j + sigma2_c

so the whole bench is `A @ v = b` with `v >= 0`, one solve per tau. Each
row carries `1` in two source columns and `1` in its channel column.

Three named techniques are the same solve:

| technique | what it is here |
|---|---|
| three-cornered hat | the 3-source triangle |
| N-cornered hat | any larger graph with an odd cycle |
| noise-floor subtraction | what a **back-to-back** row does |

A back-to-back run has both ends on one source. Its noise is common to both
ends and cancels, so the row is `0` across every source column and `1` in
its channel — it pins the channel and says nothing about any oscillator.

**Any estimator that is a variance works for free**, because the sum rule
holds for anything that is one and every estimator returns `.var`. The
solver refuses **MTIE and TIE rms**: they are extreme-value statistics, and
the sum rule is not approximate for them but false.

Every variance is non-negative by physics, so the constrained solution is
exactly non-negative least squares — `_nnls`, Lawson–Hanson, exact and
terminating, with a KKT certificate on every solve. No optimizer, no
starting guess, no convergence tolerance.

## Identifiability is a graph property

Sources are vertices, measurements are edges. The source block of `A` is
the **unsigned** incidence matrix, whose rank is

    rank = n_sources - (number of bipartite connected components)

which gives the result the whole module exists for:

> **A connected set of measurements determines its sources' individual
> variances if and only if it contains an odd cycle.**

*That is why it is called* three*-cornered hat*: a triangle is the smallest
odd cycle.

The parity argument: an unsigned incidence row is `e_i + e_j`. Two-colour
the vertices `+1`/`−1`; in a bipartite component every edge joins opposite
colours, so the colour vector is orthogonal to every row. That null
direction is exactly the ambiguity "raise everything on one side and lower
everything on the other" — invisible to every measurement, forever. An odd
cycle makes the two-colouring impossible and the null direction with it.

### The consequences people get wrong

* **A ring of four determines nothing.** A–B, B–C, C–D, D–A is a 4-cycle,
  hence bipartite. However long it runs, it cannot give you any individual
  oscillator. One diagonal fixes it, and either diagonal will do.
* **A chain is a tree**, also bipartite, also hopeless.
* **A bench of only back-to-back runs** is perfectly well determined in its
  channel block and completely useless in its source block. It measures the
  instrument, not the clocks.
* **A declared channel floor is not a free unknown.** Three sources through
  one channel is four unknowns against three rows — underdetermined. The
  classical three-cornered hat resolves this by *assuming* the channel
  contributes nothing; here you write `Channel(floor=0.0)` and the
  assumption becomes visible. Better still, add a back-to-back run and
  measure the floor instead of assuming it.

`stabilis bench check` runs all of this **before any data is collected**,
which is when the answer is still cheap to act on, and names the single
extra measurement that would fix a singular setup.

## Where the method actually fails: correlation

The whole system rests on the off-diagonal terms vanishing. The bench knows
enough to check the common causes, and each warning carries the evidence
that triggered it.

1. **Two channels of one device.** Shared timebase, LO, supply and thermal
   environment, so instrument noise is *common* and does not cancel in the
   hat. **This is the dangerous case, because the answer still looks
   reasonable.**
2. **Two records on the same channel.** Maximally correlated — they are one
   record, not independent replicates.
3. **A shared source across many edges.** The physics may be clean, but the
   *estimates* are then correlated, which inflates the uncertainty on the
   solution without biasing it. Not yet whitened for; treat the spread
   between taus as optimistic.
4. **Negative variances are the symptom, and the only honest one.** The
   unconstrained solution is computed alongside the constrained one and
   returned signed. A negative source variance is the system telling you an
   assumption is violated — the sources are not independent, or one is
   misattributed, or a channel is worse than believed. Clipping it to zero
   destroys the single most useful diagnostic the method produces.

### Cross-correlation and the hat want opposite things

Worth stating plainly, because it is a real fork in experiment design:

| setup | cross-correlation | N-cornered hat |
|---|---|---|
| two channels, **one** device, one source | **good** — rejects the shared instrument noise you want gone | **bad** — needs independent instruments |
| two **different** devices | worse — less common noise to reject | **good** — the independence the hat assumes |

So the technique should be recommended *from the graph*, not picked
blindly.

## `bench.json`

Plain JSON, schema-versioned, human-editable, and it **references data
rather than containing it**: every measurement carries a path plus the
SHA-256 of the file it names. Loading with `verify_hashes=True` refuses a
bench whose data has changed underneath it — a measurement whose bytes
changed is a different measurement.

Verification is off by default because benches are routinely written
*before* the data exists, which is the point of `bench check`. It is on
when a solve is about to happen.

`Source.spec_adev` holds datasheet points straight off a vendor's
specification. They are a **claim, not a measurement**, are never mixed
into a fit, and the validation-integrity rule applies to them exactly as it
does to published reference tables.

## Two records are enough: the derived edge

The odd-cycle theorem above is a statement about what the **variances
alone** determine. It is not a statement about what the **records**
determine, and the difference is the whole of ordinary practice.

Measure A and B against a common reference C. Then

    x_ab(t) = x_ac(t) - x_bc(t)

*identically* — the reference cancels term by term — so the third edge of
the triangle costs a subtraction rather than a counter. And that edge
closes an odd cycle, which is exactly what the bipartite component was
missing. Two records determine three clocks:

| | equation |
|---|---|
| measured | `Var(AC) = σ²_A + σ²_C` |
| measured | `Var(BC) = σ²_B + σ²_C` |
| derived  | `Var(AC − BC) = σ²_A + σ²_B` |
| ⇒ | `σ²_C = (Var(AC) + Var(BC) − Var(AB)) / 2` |

This is not a trick. **It is how the three-cornered hat is normally run**:
two channels against the house clock, the third pair formed in software.
Nor does the shared reference have to be special — C can be the local clock
everything is plugged into, and the hat still returns `σ_C`. The reference
is an unknown like any other, which is the point of the method.

The derived row is a genuinely new *equation* while being no new *data*.
Equivalently, and more revealingly: `Var(r₁ − r₂) = Var(r₁) + Var(r₂) −
2·Cov(r₁, r₂)`, so forming the difference is precisely a way of using the
**cross-covariance** between the two records — the information the solver
throws away when it looks at each record's own variance in isolation. Full
cross-variance estimation arrives with `multi/`; differencing is the part
available today, and it is exact.

`derivable_edges` reports these, and `check` marks them separately from the
pairs that genuinely need new data, because "subtract two columns" and "buy
a counter and log for a fortnight" should not print as the same advice. A
four-clock ring turns out to need no new data either: both diagonals are
derivable.

Two conditions the graph cannot check, so the report states them:

* **Simultaneity.** The subtraction means nothing unless the two records
  overlap in time on a common timebase. `Measurement` carries an epoch
  string and no duration, so verifying that here would be inventing a
  capability — the same reason `_overlapping_epochs` trusts only exact
  equality.
* **The channel term.** A derived record carries the noise of *both*
  contributing channels: `Var = σ²_a + σ²_b + σ²_c1 + σ²_c2`. A
  `Measurement` names one channel, so the design row would count one.
  Declare the floors — zero is the classical hat's silent assumption, and
  writing it down is what makes it visible — or give the derived row its own
  channel whose floor is the sum. Note that when the two channels share an
  instrument their common-mode part *cancels* in the difference, so
  `σ²_c1 + σ²_c2` is an upper bound there.

One numerical footnote. The cancellation is exact in algebra and to about
`1e-25 s` in float64 on pedestal-free phase; on columns each carrying a
0.35 s pedestal it is exact only to `5.5e-17 s`, because the subtraction
happens at the pedestal's magnitude rather than the signal's. That is
`1.8e-5` of ADEV at `tau0 = 1 s` for a 3e-12 clock — negligible, but it is
the same float64 hazard `precision.md` describes, and it is why the library
removes the pedestal before scaling units.

## Building one interactively

`stabilis gui`, stage 4, is the same model with the graph drawn as you edit
it. Clocks and instruments on the left, one measurement per row on the
right, and the identifiability verdict recomputed on every keystroke —
which is the whole point, because that verdict is worth most **while the
experiment is still being planned**, not after a fortnight of logging.

The one design decision worth recording is where the file-reading lives.
`bench.solve` takes a `{measurement id -> StabilityResult}` mapping and
never opens a file; it has no notion of a column, a unit or a header, and
must not grow one. So the page strips each measurement down to its data
column itself — expression, pedestal, units, crop — and by the time
anything in `bench/` is called, a measurement is an array with a sample
rate.

That is why `Measurement` carries no column index. The alternative was to
add one and bump the schema, and it would have been the wrong trade: the
column belongs to *this file*, the bench describes *the experiment*, and a
`bench.json` that recorded both would be the only place the two could
disagree. What the page prints instead is the full recipe — `read_table`,
the numpy expression, the pedestal, the unit factor, the estimator call —
so the session reproduces exactly without the mouse, and
`tests/test_gui_bench.py` runs that recipe by hand and demands identical
numbers back.

Two things the page reports that the rank alone would misrepresent:

* **An empty bench is trivially identifiable.** Rank 0 of 0 unknowns is
  consistent and says nothing about any clock, so the measurement count
  comes back beside the verdict and the page says so in words.
* **A clock nobody measures has no column in the solve.** Not an error —
  benches get written before data is taken — but it is named rather than
  silently dropped.

### Six counters, six formats, none of them clean

Stage 4 is the only part of the page that looks at several files at once,
and the one that must assume least about them. Every other stage examines a
file the user just chose and is already looking at. A bench is six exports
from six instruments: six delimiters, six ideas of a timestamp, six sets of
dropouts, and a glitch at the moment somebody opened the rack door.

So the delimiter, the units, the timebase and format, the kept stretches
and the outlier threshold are **per measurement**, not per bench. The
record is drawn — min/max envelope, so a single-sample spike in two million
points survives decimation — before it is solved, and stretches are kept by
dragging rather than by typing indices at a record you cannot see. Several
kept stretches become **one gapped record**, never two curves, because a
measurement is one row of the bench; nothing between them is interpolated.

What each file needed tolerating to be read at all, how many points were
gaps, how many an outlier threshold removed, and how far its timetags
wander from a clean grid are all reported against the row they belong to.
"Something was tolerated somewhere" is not actionable when the something
belongs to one of six counters.

Two consequences worth writing down:

* **A timebase in MJD needs an explicit rate.** Day number 59000 has a
  float64 spacing of 7.3e-12 d = 6.3e-7 s, so a 1 Hz grid infers as
  0.99999976 Hz. `af/rate` then misses an exact 1.000 s by far more than
  the nanosecond tolerance `_common_taus` allows, and against a row given
  an exact 1 Hz the two share *no* averaging time at all. The solver is
  right to refuse — it interpolates nothing — but the refusal alone does
  not say why two counters that both ran at 1 Hz disagree. The page adds
  the rates it used, how each was obtained, and the fix. This is the same
  float64 hazard as `precision.md`, met on the timestamp column instead of
  the phase column.
* **The file cache is budgeted in bytes, not in files.** A count was right
  when the page looked at one record at a time; a bench wants all of its
  files on every solve, so a four-slot cache evicted the one it was about
  to need. A count also mismeasures the risk, since four 2 GB records is
  not four 2 MB ones. One table is always kept however large, or a record
  bigger than the budget would be re-read on every keystroke.

The page reads bench files and never writes one: saving hands the JSON to
the browser's downloads. The server binds to localhost because a file
reader is already worth confining, and a file writer would be a different
conversation. What comes down is `store.to_dict` output, round-tripped
through the library on the last check, so a bench the model would reject
cannot be saved at all.

## Independence from `stats/`

**`stats/` must never import `bench/`.** An AST test over the whole `stats`
package enforces it, the same discipline the certified oracle has. The
estimators must stay usable, and testable, with no notion of where their
data came from; if that ever inverts, the bench stops being an optional
layer and becomes a dependency of every ADEV anyone computes.

## Validation status

**Validated.** `tests/test_bench.py`:

* the graph theorem, in both directions — the triangle is identifiable, the
  4-cycle is not, and its null direction is asserted to be exactly the
  alternating one;
* `suggest_edge` returns **both** diagonals of the ring, and adding either
  makes the bench identifiable;
* the hat recovers three known white-FM oscillators against `h/(2 tau)` —
  a truth in closed form, not one taken from the fit;
* the raw pair measurement is shown to equal the sum of the two recovered
  variances to 5%, which is what the hat separates and a single measurement
  cannot;
* a channel floor is recovered from a back-to-back run to 40%;
* deliberately correlated sources drive the unconstrained solution negative
  and it is reported signed while the constrained answer stays physical;
* MTIE, mixed estimators, unidentifiable benches, benches with no common
  tau, unknown schema versions and changed data files are each refused with
  a directed message.

`tests/test_gui_bench.py` covers the interactive path over real HTTP: the
graph verdicts again through the endpoint, the hat recovering three clocks
of known level, the printed recipe run by hand for bit-identical numbers,
`tau0_s` on the bench overriding a rate typed on the page, and every
refusal above surfacing as a message rather than a traceback.

### How accurate is the hat, really

The validation report carries the measurement, and its tolerance is
**propagated rather than chosen**. Each measured variance has
`Var(sigma2_hat) = 2 sigma^4 / eta`, so the solution's covariance is
`A+ diag(...) A+^T` and every source's standard error follows from the
geometry of the bench and the degrees of freedom of its rows.

Three sources at `h = 4e-24`, `1e-24` and `9e-24` — deliberately 30 apart
— measured pairwise, at `tau <= 64 s`:

| source | `h` | measured error | predicted |
|---|---|---|---|
| A | 4e-24 | 21 % | 19 % |
| B | 1e-24 | **53 %** | **77 %** |
| C | 9e-24 | 11 % | 9 % |

Worst 2.5 standard errors. The lesson is in the middle row: **the quietest
oscillator is the one the hat determines worst**, because there it is a
small difference of large numbers, and its error is dominated by the
*other* sources' noise. A 53 % error there is not a defect — it is what
error propagation predicts, and the estimate can and does go negative.

That is why a flat percentage tolerance would be the wrong test: it would
score the choice of oscillators rather than the method. And it is why the
signed diagnostic matters more than the constrained answer when one source
is much quieter than its partners.

**Not yet:** the correlated-estimate whitening of point 3 above, and
cross-variance rejection, which arrive with `multi/`.
