# Realtime ingest and the online Allan variance

## Why this came last

The acceptance test for a streaming estimator is an identity, not a
comparison:

    online_result(feed(acc, replay(x))) == oadev(x)

and that identity is worthless until `oadev` is trusted. Building the
streaming path first would have produced two implementations of unknown
correctness that agree with each other. So `stream/` was written after the
batch core had its analytic identities, its exact-rational oracle and its
measured coverage — and then inherits all of it for free, since agreeing
with `oadev` is agreeing with everything `oadev` agrees with.

## The online estimator

At sample index `k`, exactly one overlapping-Allan term *ends*:

    d = (x[k] - 2 x[k-m]) + x[k-2m]
    c = d / (m tau0)

for each averaging factor `m` with `k >= 2m`. So the accumulator keeps one
running sum of `c²` and one count per `m`, plus the last `2·max(m)+1`
phase samples in a ring buffer. Memory is O(max τ) and independent of how
long the measurement runs; cost is one small vector operation per sample.

Two details make the identity hold rather than nearly hold.

**The association order is copied from the batch expression.** numpy
evaluates `x[2m:] - 2.0*x[m:-m] + x[:-2m]` as `(x[i+2m] - 2·x[i+m]) +
x[i]`, and the accumulator forms `(x[k] - 2·ring[k-m]) + ring[k-2m]` in
that same order. Every coefficient is therefore **bit-identical**, not
close, and the only difference between the two paths is how the
coefficients are summed.

**The sum is Neumaier-compensated**, because an accumulator cannot sum
pairwise — it never holds more than one term at a time. Measured against
the batch estimator over octave τ:

| N | max relative difference | in ulp | of which bit-identical |
|---|---|---|---|
| 1 024 | 2.9e-16 | 1.3 | 6 / 9 |
| 4 096 | 2.3e-16 | 1.0 | 6 / 11 |
| 16 384 | 2.0e-16 | 0.9 | 8 / 13 |
| 65 536 | 2.2e-16 | 1.0 | 13 / 15 |

One ulp, and **flat in N** — which is the compensation's whole point. The
same accumulator with a plain sequential sum gives 15 ulp at N = 4096 and
74 ulp at N = 65536, growing as O(N) the way an uncompensated sum must.
Note the direction of the residual: the compensated sum is the *more*
accurate of the two, so what the table measures is mostly the batch path's
pairwise summation error, not the streaming path's.

**Gaps need no special case.** A NaN sample is stored in the ring like any
other, so it poisons exactly the terms whose window contains it and is
dropped by the same `isfinite` test the batch estimator applies to its
coefficient array. That is not a parallel rule that happens to agree; it
is the same arithmetic reached by the same route, which is why the gapped
identity holds as tightly as the clean one.

**No coefficients, therefore no confidence interval.** Retaining the
coefficient arrays is the one thing an O(max τ) accumulator cannot do, and
the data-driven EDOF is a function of exactly those arrays. So
`confidence()` refuses an online result rather than substituting a model
that assumes a power law nobody has checked for. Collect the record and
run in batch, or use `davar`, when an interval is needed.

## Putting arriving samples back on the grid

This is the part that is longer than its arithmetic, and the reason is one
failure mode.

Every estimator here assumes `x[i]` is the phase at `epoch + i/rate`. A
stream arrives as `(t, value)` pairs and carries no such guarantee. The
natural implementation — append each arriving value to a list — is wrong
in the worst available way: a counter that drops fifty readings hands you
a record that is fifty samples shorter and **perfectly smooth**. Every
sample after the outage has slid earlier by fifty intervals, the timebase
is wrong from there on, and σ_y(τ) comes back *better* than the oscillator
is. Nothing in the data says so.

So the rule is: **a sample that never arrived becomes a NaN in its own
slot, never the disappearance of a slot.** `on_grid` computes

    slot   = (t - epoch) · rate
    index  = round(slot)
    offset = slot - index        [samples]

fills every index between the last placed sample and this one with NaN,
and hands the result to machinery that already treats NaN honestly
(`theory/gaps.md`). The acceptance test is that replaying a gapped record
with the gaps *omitted from the stream entirely* reconstructs the original
record exactly, NaN for NaN.

Three things it refuses rather than repairs:

* **A timestamp more than a quarter-sample off grid.** The declared rate is
  then not the rate the data was taken at, and placing the sample anyway
  is interpolation — inventing data to make an inconsistency go away.
  A quarter of an interval is well past any plausible logging jitter and
  well short of the half-sample at which the assignment is ambiguous.
* **A timestamp that goes backwards or repeats one already placed.**
  Sorting it into position would take the record's order from the
  transport's scheduling rather than from the instrument. UDP reorders
  datagrams; that is a property of the transport and must not become a
  property of the measurement.
* **A gap longer than `MAX_GAP_SAMPLES`.** An hour's outage at 1 MHz is
  three billion NaNs. Past that length it is an outage rather than a gap
  and the two sides are separate records.

`offset` is carried out of the placement and the worst one goes into
provenance, because it is the only diagnostic that cannot be recovered
afterwards: a record whose samples all land 0.2 of an interval late was
taken at a different rate from the declared one, and once placed it looks
perfect.

### The file with no timestamp column

`parse_lines` will synthesize `t` from arrival order when the file has no
time column, which is what most logging scripts produce. The samples land
in the same places. What changes is the claim the record may make:
**a dropped sample is then undetectable**, because the line that would have
carried it is simply absent and nothing in the remaining text says so.

`collect` records which of the two it used, and when the timestamps were
synthesized it says so as a warning rather than a note, with the gap count
described as a floor rather than a count. That distinction is the whole
value of a timestamp column and it should not be something a reader has to
work out from the absence of a column they cannot see.

## Sources

A source is a generator. That is the entire abstraction: no base class, no
`start`/`stop`/`read` protocol, no object to keep alive. `for line in
tail_file(path):` is the interface, `break` is the shutdown, and a closed
generator has released its handle because `finally` ran.

`replay` yields `(t, value)` pairs, since it knows both. `tail_file`,
`tcp_lines` and `udp_lines` yield text lines, since a byte stream has no
opinion about what a sample is — turning a line into a pair is
`parse_lines`, where the column layout belongs. All three buffer partial
lines, because a read can land mid-line and a half-line parsed as a number
is a wrong answer rather than an error.

**Reconnection is deliberately absent.** A reconnect loses an unknown
number of samples; a source that papers over that gap hands the estimator
a record whose timebase is wrong in a way nothing downstream can see.
Catch the exception, note the outage, start a new stream on a new epoch.

**`serial` is absent too**, and for a plainer reason: it needs hardware to
test against, and shipping an untested source while listing it in the
feature table is how a library acquires a function nobody has ever run.
It arrives with the first instrument, along with the `.TFF` reader.

## Validation status

**Validated by identity against the batch estimator, plus one
reconstruction test that the natural wrong implementation fails.** The
online estimator has no independent reference and needs none: it computes
the same quantity as `oadev`, so agreeing with it to one ulp inherits
every identity, oracle comparison and coverage measurement `oadev` has.

| claim | how it is checked | where |
|---|---|---|
| a replayed record gives the batch answer | `var` within 2e-15 relative, `af` and `n_terms` equal | `tests/test_stream.py` |
| the disagreement is 1 ulp and flat in N | measured at N = 1024 and N = 16384, both under 4 ulp; an uncompensated sum gives 15 and 74 | `tests/test_stream.py` |
| gaps need no special case | a 5-sample and a 1-sample dropout; `n_terms` and `dropped_terms` equal to batch | `tests/test_stream.py` |
| chunking cannot change the answer | 37-sample chunks against one push per sample, bit for bit | `tests/test_stream.py` |
| **a stream that omits its dropped samples rebuilds the record** | `replay(skip_gaps=True)`; NaN mask and surviving values equal to the original array | `tests/test_stream.py` |
| a fed gapped stream matches the gapped batch call | 4-sample dropout, end to end | `tests/test_stream.py` |
| a feed resumes where the last stopped | two streams in sequence equal one, bit for bit | `tests/test_stream.py` |
| a uniform offset is an epoch, not jitter | 0.2-sample shift absorbed; alternating ±jitter reported instead | `tests/test_stream.py` |
| refusals | wrong rate, out-of-order sample, duplicate slot, outage past the limit, empty stream, τ history past the memory bound | `tests/test_stream.py` |
| an online result has no interval | `confidence()` raises rather than modelling one | `tests/test_stream.py` |
| synthesized timestamps are declared as such | provenance says "synthesized" and the gap count is called a floor | `tests/test_stream.py` |
| sources yield whole lines only | a write split mid-line, over a file, a TCP socket and a UDP datagram | `tests/test_stream.py` |
| text on disk to batch answer, end to end | a file parsed, gridded, accumulated, compared to `oadev` | `tests/test_stream.py` |

`stabilis validate` carries the replay identity as an `identity` row.

## What is not validated

* **No live source has ever run.** `replay` drives the test suite;
  `tail_file` has a hermetic test against a file written from the same
  process; TCP and UDP have loopback tests. None of that is an instrument
  on a bench, and the first appending text file from the user's rig is
  what closes it.
* **The identity is against this library's own batch path**, which is the
  right comparison — it is the *same* estimator, so any disagreement is
  the streaming machinery and nothing else — but it is not an independent
  implementation. Nor is one available: `allantools` carries a `realtime`
  fixture directory and so presumably a matching module, and it is
  LGPL-3.0, which under this repo's licence position may be called as a
  numeric oracle but not read (see `references.md`). Nothing has been
  differential-tested against it here.
* **The grid tolerance is a judgement.** A quarter sample is defensible
  and has not been calibrated against a real logger's jitter, because no
  real logger's timestamps have been through this yet.
