# stabilis — frequency stability analysis

## Thesis

Stable32 is the de-facto tool for frequency stability analysis. It is
Windows-only, closed, GUI-driven, unscriptable, and no longer developed.
TimeLab is a hobby project. `allantools` is a library rather than a
workflow, with no uncertainty or reporting layer. Nothing available today
is a modern, cross-platform, embeddable engine that a calibration lab can
put in a production test rig and get a defensible number out of.

Two things here are not a Stable32 reimplementation.

**The Haar wavelet variance is the Allan variance.** For a fractional
frequency series, `AVAR(τ) = 2·ν²_Haar(τ)`. Classical AVAR is the K=1 case
of one framework. Higher-order Daubechies filters buy vanishing moments —
D4 annihilates linear frequency drift with no pre-detrending — and
convergence for steeper noise (`α > −2K−1` rather than AVAR's `α > −3`, so
random-run and flicker-walk FM stop diverging).

**Confidence intervals that do not require guessing the noise type
first.** Degrees of freedom are estimated from the estimator's own
coefficient autocovariance. Stable32 looks the noise type up in a table
and has no answer at all when the noise is not one of its five power laws.

## Validation

Every published reference ADEV table was produced by Stable32 — NIST
SP 1065 Table 31 is Riley's own tool's output, and `allantools`' expected
values are documented as Stable32 calculations. Matching them proves
*interoperability*, not correctness. So validation stands on three legs,
in this order of priority:

1. **Analytic identities** at machine precision, needing no external data.
   This is where correctness actually lives.
2. **Monte-Carlo confidence-interval coverage** — measured containment
   against nominal. No competing tool publishes this.
3. **Interop fixtures** — proves we match the incumbent.

`stabilis validate` runs the whole hierarchy and emits a versioned
evidence report with pass/fail per identity, coverage tables, seeds, input
hashes and a report hash. It is designed to be reproducible by a third
party from a clean checkout, and to be capable of failing.

## Install

```bash
pip install -e ".[dev,plot]"
```

The core needs only numpy. `scipy` and `matplotlib` are optional and are
never required for a correct answer — there is a test asserting the
package imports and computes with both absent.

## Status

M0 (skeleton) in progress. See `docs/` for the theory pages, which are
also readable from the CLI:

```bash
stabilis docs --list
stabilis docs avar
```

## Licence

MPL-2.0. You may embed this in closed products, including commercial test
rigs and instrument firmware; modifications to these files come back.
