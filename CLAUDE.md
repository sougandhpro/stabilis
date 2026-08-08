# Project context

## Who this is for

The author is a PhD physicist in time and frequency metrology (timescales,
White Rabbit sub-nanosecond sync). Assume full fluency in metrology, noise
analysis, power-law noise types, and uncertainty budgets. Do not explain
Allan deviation. Do explain software-engineering choices when they are
load-bearing.

This is a distributable library where correctness *is* the product, so it
carries packaging, tests and type hints — unlike `gpu-power-metrology`,
which is a research ladder of flat scripts and correctly has none of those.

## Code style contract

These are rules, not aspirations. The author must be able to read and
change every line.

- **One estimator = one function, ≤ 40 lines**, fully vectorized numpy.
- **The defining equation goes in the docstring, using the same variable
  names as the code below it.** Formula and implementation must be
  diffable by eye.
- **No inheritance anywhere.** No ABCs, no mixins, no plugin base classes.
  Frozen dataclasses for data; plain functions for behaviour.
- **No `**kwargs` passthrough chains.** Explicit arguments only.
- **No metaprogramming** except the single registry dict in `registry.py`.
- **Module cap ~400 lines.** Split by family rather than letting one
  module accrete.
- **Variances internally, deviations derived.** Never sqrt a value the
  caller will square. Result objects store `var`; `.dev` is a property.
- Every public function returns a frozen dataclass carrying `Provenance`.

## Hard constraints

- **The core is numpy-only.** scipy and matplotlib are optional, lazily
  imported, and never required for a correct answer. There is a test for
  this and it must keep passing — it is a selling point, and it regresses
  silently.
- **No pickle support, ever.** `pickle.load` on a file arriving from a
  customer's rig is arbitrary code execution. `.npy` covers the need.
- **Provenance and the step log are not optional.** They exist from the
  first commit because the paid layer prints signed reports, and
  retrofitting them after the API has users is painful.
- **float64 phase in seconds is a hazard.** At accumulated phase of
  10³–10⁶ s, float64 spacing is 10⁻¹³–10⁻¹⁰ s, which silently quantizes
  exactly the picoseconds this field cares about. Phase is stored as
  offsets from an explicit epoch, and construction warns when spacing
  approaches the resolvable deviation.

## Validation conventions

### Reference data is evidence. Do not touch it.

Absolute rules when validating against a dataset. Each has already been
violated once, which is why they are written down.

- **Never invent a number.** An expected value comes from a published
  source or from the exact-rational oracle, and the two are labelled
  differently because they mean different things. A value produced by this
  library and then written into a reference table makes the assertion a
  statement about itself.
- **Never edit a published entry to match our output.** If a table
  disagrees with the oracle, record it in `reference.KNOWN_DISCREPANCIES`
  with the published value, the certified value and the reason, and leave
  the table alone. A test asserts every recorded disagreement stays inside
  `INTEROP_RTOL`, so a real conflict still fails the suite.
- **Never fill a gap in validation data.** No interpolation, no imputation,
  no dropping to the longest segment, no quiet padding. Validate on the
  record as it came. Gap *filling* is a feature under test (M7, Howe &
  Schlossberger) and must never be applied to the fixtures that test it —
  the imputer is validated against records where the removed points are
  known, by comparing the filled result to the original.
- **A missing expected value stays missing.** Absence is data about what
  has actually been checked. Skip the assertion and say so; do not
  interpolate a neighbouring tau or average two estimators.
- **Assert at the tolerance the source supports, no tighter.** Seven
  published figures means `rel=1e-6`. Tightening beyond what the source
  claims tests our transcription, not our arithmetic.

### General

- Every new estimator must add a theory page, a test, and a row in the
  validation report before its milestone counts as passed.
- Prefer analytic identities over reference tables. The best single test
  in the suite is linear frequency drift: `avar(τ) = D²τ²/2` exactly and
  `hvar(τ) = 0` exactly, which validates two normalizations and the drift
  convention at once.
- Reference fixtures are labelled as *interop* assertions, not correctness
  assertions, in the test docstrings. That honesty is itself a credibility
  asset when a customer's metrologist reads the suite.
- Do not vendor `allantools` fixtures — it is LGPL and this is MPL. Use it
  as a test-only dependency for differential testing, and generate our own
  Stable32 references.

## Environment

- Repo lives on the G: drive, editable from WSL and Windows.
- `wine` is not installed, but `/mnt/c` is reachable, so **Stable32 runs
  natively through WSL interop** when generating reference outputs. Wine is
  the fallback only if this ever moves to a Linux-only machine.
- scipy, matplotlib and pytest are not in the system Python. Work in
  `.venv`.
