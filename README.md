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
pip install -e ".[dev,plot]"          # working on stabilis itself
pip install /path/to/stabilis         # using it from another project
```

The core needs only numpy. `scipy` and `matplotlib` are optional and are
never required for a correct answer — there is a test asserting the
package imports and computes with both absent.

## Using it

Everything normally needed is re-exported from the top level, so consuming
code never has to know which submodule something lives in. **The names in
`stabilis.__all__` are the public API** and will not move without a version
bump; anything reached through a submodule path may.

```python
from stabilis import load_phase, oadev, confidence, identify

series = load_phase("run.dat", rate_hz=1.0)
result = oadev(series, taus="octave")
ci = confidence(result)

print(identify(series).name)                     # 'flicker FM'
for tau, dev, lo, hi in zip(result.tau_s, result.dev, ci.dev_lo, ci.dev_hi):
    print(f"{tau:8.3g} s  {dev:.4e}  [{lo:.4e}, {hi:.4e}]")
```

Every result carries a `Provenance` recording what was done and every
decision taken on your behalf — the tau grid, the gap policy, the degrees-
of-freedom estimator actually used, any assumed bandwidth. That is what
makes a number defensible later.

Two subsystems stay off the top level on purpose. `stabilis.plot` needs
matplotlib, so importing it here would drag an optional dependency into
every `import stabilis`. `stabilis.bench` is the measurement-setup model,
and `stats/` is forbidden to import it — keeping it a separate import keeps
that independence visible.

## Status

**M0–M2 complete, most of M3, and M5's bench model.** 648 tests;
`stabilis validate` 31/31.

M2 was the v0.1 line — the first version from which an uncertainty may
legitimately be quoted. Since then:

- **Greenhall EDOF**, so Stable32's `Alpha` / `Min Sigma` / `Max Sigma`
  columns can be reproduced. Selected explicitly, never by `auto`.
- **Exact non-negative least squares** (Lawson–Hanson, with a KKT
  certificate), which two very different features both need.
- **Amplitude-calibrated synthesis** — `h_alpha` is a settable SI
  parameter, checked against four closed forms exact at every averaging
  factor.
- **Power-law decomposition** — a measured σ(τ) curve back into the five
  IEEE 1139 noise levels plus drift, stacking Allan and modified rows
  because the two PM noises do not separate from AVAR alone.
- **The bench model** — sources, devices, channels. Three-cornered hat,
  N-cornered hat and instrument noise-floor subtraction become one
  non-negative solve per τ, and `stabilis bench check` refuses a setup
  that cannot separate its oscillators, naming the one extra measurement
  that would fix it, *before any data is collected*.
- **MTIE and TIE rms**, by binary doubling: 10⁶ points over 18 octave τ in
  0.15 s, where the naive form is O(N·m) per τ.
- **Plots** — σ(τ) with asymmetric error bars, power-law slope guides,
  phase, frequency, and the bench solution.

Built on top of M0/M1: pure-numpy special functions (regularized incomplete
gamma, Wichura AS241 normal quantile, chi-square inverse), data-driven
degrees of freedom, chi-square confidence intervals, power-law noise
identification by two independent methods, the deviation bias correction,
and `stabilis validate`.

Measured coverage of the confidence intervals, over 800 realizations of
N = 512 at each power law:

| nominal | measured |
|---|---|
| 68.27 % (1σ) | 0.61 – 0.77 |
| 95.45 % (2σ) | 0.90 – 0.99 |

No competing tool publishes that measurement.

```bash
stabilis validate --out validation-report.html
```

runs the whole hierarchy — analytic identities, certified values against
exact rational arithmetic, Monte-Carlo coverage, and Stable32 interop — and
writes a versioned evidence report. Exit status is the verdict, so it works
as a gate in a lab's own qualification pipeline. `--self-test` injects a
check designed to fail, because a validation report that cannot fail is
worthless. A committed example is at `docs/validation-report.html`.

<details><summary>Earlier milestones</summary>

**M0 and M1 complete.** 154 tests, three commits. Built: immutable
phase/frequency containers with a float64 precision guard, provenance with
a step log, the plugin registry, tau selection, Kasdin-Walter synthesis,
delimited-text reading, a CLI, the five classical estimators (`adev`,
`oadev`, `mdev`, `tdev`, `ohdev`), the exact-rational certified-value
oracle, and the Tier 0 identity suite.

All five estimators agree with exact rational arithmetic to 14.1-16.3
correct digits. See `docs/stabilis-progress.pdf`.

</details>

**Next** — the rest of M3: TOTVAR with its bias correction, Thêo1/ThêoH,
and the MODWT wavelet family, whose acceptance test is bit-agreement with
M1's AVAR.

## A GUI, for when flags are tedious

```bash
stabilis gui                     # opens http://127.0.0.1:8765
stabilis gui --root ~/data       # confine the file browser to one tree
```

A local page in three stages, which is the order the questions arrive in.

**1 · Data.** Pick a file. It is read **once**, and everything after that is
a slice of an array already in memory. If it has a header those names are
filled in; if not the columns are numbered and you name them — either way
what you type wins, and the expressions in stage 2 follow. The record is
drawn as a min/max envelope, one pair per pixel column, so a single-sample
glitch in two million points is still visible where stride sampling would
have thrown it away with 99.9% probability. Drag to keep a stretch; drag
again to keep another. Zooming asks for finer detail over the visible span
rather than magnifying the picture.

**2 · Prepare.** Say what the numbers *are*: phase or frequency, in which
units, at what rate or off which timebase, and what arithmetic turns the
file's columns into the quantity of interest.

**3 · Analyse.** Run an estimator over what stage 2 produced and read the
answer off an interactive log-log curve, all series on one axis or each on
its own.

Every selection prints **the equivalent non-interactive call**, so the
session is reproducible and doubles as how you learn the API:

```
series = edit.select_range(series, 3000, 50000, by='index', reason='after warm-up')
series = edit.drop_outliers(series, threshold=6.0, reason='counter glitch')
```

An exclusion that exists only as a mouse gesture is not evidence — and the
full step log is on the page beside the numbers, not just carried in the
result object.

Each series is a **column expression**, so a derived channel needs no
preprocessing step of its own:

| expression | meaning |
|---|---|
| `NCS` | that column as it stands |
| `NCS - VSL` | one channel against another |
| `-VSL` | the negated column |
| `(NCS - VSL) / 2` | scaled by a constant — a divider ratio backed out |
| `#2 * 1e9` | the third column by position, into nanoseconds |

Columns may be named by header, by the name you gave them, or by `#index`.
**There is no evaluator behind the box** — a hand-written tokenizer and a
closed grammar of numbers, columns, `+ - * /` and parentheses, with no name
lookup that can reach a builtin and no call syntax at all. The tests throw
`__import__('os').system(...)` at it among other things. It is the same
reasoning that makes this library refuse pickle: a file, or an expression,
arriving from someone else's rig is untrusted input.

Marking several windows has two honest readings and the page offers both:
each window analysed as its own record and drawn as its own curve, or one
record spanning them with the stretches between marked as gaps for the
documented gap policy to handle. **Neither interpolates.** Filling a gap is
a later milestone, validated against records where the removed points are
known, and improvising one here would invent exactly the data whose
exclusion was the point.

Nothing about a timebase is guessed. An MJD column ticks by 1/86400 per
sample at 1 Hz, so reading it as seconds would put every tau out by that
factor, plausibly and silently; where the values fit more than one format
the page refuses and asks rather than choosing. Units are the same story: a
`ns` file analysed as `s` is wrong by a factor of a billion and looks
entirely reasonable on a log axis.

It computes nothing you would quote. Every variance, degree of freedom and
confidence bound comes from `stats/`, through the same calls the CLI makes,
so the page and the command line cannot disagree — there is a test that
runs the page's own printed recipe by hand and demands identical numbers.
The whole of `gui/` imports from `stabilis` and nothing in `stabilis`
imports from it, so it is a client that could be lifted out entirely.

It uses `http.server` from the standard library and your existing browser,
so nothing new is installed; the plots are drawn on a canvas from numbers,
so **`stabilis gui` needs no matplotlib either**. It **binds to 127.0.0.1
only** — the page reads local files, so a wider bind would be handing a
stranger a file reader. `--root` narrows that further, and paths trying to
climb out of it are refused rather than normalized back in.

This is a convenience for the author's own use, not the workbench in the
roadmap; that is a separate product built on top of this engine.

The theory pages are readable from the CLI:

```bash
stabilis docs --list
stabilis docs avar
```

## Licence

MPL-2.0. You may embed this in closed products, including commercial test
rigs and instrument firmware; modifications to these files come back.

## Progress report

A report covering what is built, the measured results, and the design
decisions is regenerated from a live run -- every figure in it is computed
at generation time rather than typed in:

```bash
python3 docs/make_report.py --pdf
```

Output lands in `docs/stabilis-progress.pdf`. The PDF is rendered by driving
the Windows Edge binary headless through WSL interop, so no PDF library is
needed; without Edge, open the HTML and print it.
