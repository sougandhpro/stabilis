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
