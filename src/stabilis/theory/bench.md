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
