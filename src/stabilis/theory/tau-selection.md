# Averaging-factor and tau selection

## Why this has its own page

Two tools computing identical mathematics will disagree if they chose
different tau grids or different maximum averaging factors, and neither
says which it used. This is reportedly the largest single source of
"why doesn't this match Stable32". So the choice is explicit here, and
every estimator records it in provenance.

## Definition

An averaging factor `m` is an integer number of samples. The corresponding
averaging time is

    tau = m * tau0 = m / rate_hz

## Grids as implemented

`tau.averaging_factors(n, mode, max_af, n_log)`:

| mode | sequence |
|---|---|
| `all` | every m from 1 to max_af |
| `octave` | 1, 2, 4, 8, 16, ... |
| `decade` | 1, 2, 4, 10, 20, 40, 100, ... |
| `log` | `n_log` log-spaced values, deduplicated |

Note that `decade` is **1-2-4 per decade**, not 1-2-5. That is the Stable32
convention and it is a common source of mismatch.

## The maximum averaging factor

Two separate limits apply and they are easy to conflate.

**The algebraic limit** is a property of the estimator. Overlapping AVAR on
`N` phase points forms `N - 2m` terms, so it needs `m <= (N-1)/2`. Hadamard
needs `m <= (N-1)/3`. Exceeding it is not "noisy", it is undefined.

**The policy limit** is a judgement about when a value stops being worth
reporting. The default here is `m <= n/4`. Beyond roughly that point an
overlapping estimator has very few independent terms, the confidence
interval is wider than the value is informative, and a plotted point
invites over-interpretation. Stable32's practical guidance is similar
(`m ~ N/5`).

The policy limit is a default, not a law: pass `max_af` to override it. It
is always recorded, so a reader of the provenance can tell whether a curve
stops because the data ran out or because the policy said so.

A more principled rule, planned once EDOF lands: include a point only where
the estimated degrees of freedom exceed a threshold (4 is a sensible
default). That adapts to the noise type — white FM tolerates a larger `m`
than random-walk FM — instead of applying one fraction to every case.

## Validation status

**Validated.** `tests/test_tau.py` pins every grid exactly, including the
1-2-4 decade sequence, the `n/4` default, clamping to `n-1`, deduplication
in log mode, and the empty-grid-not-a-crash case for very short records.

## References

* NIST SP 1065 (Riley), §5 and §9.
* Stable32 user manual, tau selection.
