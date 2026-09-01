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

The core needs only numpy. `scipy` and `matplotlib` are optional and are
never required for a correct answer — there is a test asserting the
package imports and computes with both absent.

### From another project, on any machine

```bash
pip install "stabilis @ git+https://github.com/sougandhpro/stabilis.git@main"
pip install "stabilis[plot] @ git+https://github.com/sougandhpro/stabilis.git@main"
```

That is the whole install. There is no PyPI release yet, so the git URL is
the distribution channel; it needs no credentials because the repository is
public.

**Pin it when the number has to be reproducible.** A measurement quoted in
a report should name the code that produced it, and `@main` moves:

```bash
pip install "stabilis @ git+https://github.com/sougandhpro/stabilis.git@6a548f2"
```

Any commit-ish works — a tag, a branch, a full or short SHA. `Provenance`
records what was done to the data; the pin records what did it.

### Tracking the tip, on a machine that will follow along

Two ways, and they differ in what "update" costs.

```bash
# A checkout that follows: `git pull` is the whole update, no reinstall.
git clone https://github.com/sougandhpro/stabilis.git
cd stabilis && pip install -e ".[dev,plot]"
git pull                              # every later update, and that is all
```

An editable install puts a link on `sys.path` rather than a copy in
`site-packages`, so the next `import stabilis` in any project on that
machine already sees the pulled commit. Nothing to reinstall, nothing to
remember. This is the right shape for a development machine, and for any
machine where you will also want to run the test suite or `stabilis
validate` against a specific commit.

```bash
# A plain install that is re-pulled on demand.
pip install --upgrade --force-reinstall --no-deps \
    "stabilis @ git+https://github.com/sougandhpro/stabilis.git@main"
```

`--force-reinstall` is not superstition: pip treats a direct URL whose
version string has not changed as already satisfied, and this project's
version moves far more slowly than its commits do, so a plain `--upgrade`
can silently leave you on last month's code. `--no-deps` keeps numpy where
it is instead of dragging the whole tree through a reinstall.

Whichever route, ask the installed package what it is rather than assuming:

```bash
python3 -c "import stabilis; print(stabilis.__version__)"
stabilis validate                     # 41/41, and a digest over the evidence
```

`stabilis validate` is the honest check that a remote machine got a working
copy: it re-runs the analytic identities, the exact-rational oracle, the
Monte-Carlo coverage and the Stable32 interop fixtures on *that* machine,
and its exit status is the verdict. A copy that installed but computes
differently — a broken numpy, a truncated download — fails it.

### Working on stabilis itself

```bash
pip install -e ".[dev,plot]"
python3 -m pytest -q
```

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

Three subsystems stay off the top level on purpose. `stabilis.plot` needs
matplotlib, so importing it here would drag an optional dependency into
every `import stabilis`. `stabilis.bench` is the measurement-setup model,
and `stats/` is forbidden to import it — keeping it a separate import keeps
that independence visible. `stabilis.report` is the output layer, and its
figure path wants matplotlib for the same reason as `plot`.

## Documentation

[`docs/README.md`](docs/README.md) is the map. In short:

| Want to know | Read |
| --- | --- |
| Why an estimator is defined the way it is | `stabilis docs --list`, then `stabilis docs <topic>` |
| What a function takes and returns | [`docs/api/index.md`](docs/api/index.md) |
| What a command does | [`docs/cli-reference.md`](docs/cli-reference.md) |
| What is proven, and how | `stabilis validate` |
| What is *not* yet proven | [`VALIDATION-DEBT.md`](VALIDATION-DEBT.md) |
| How to use this from another project, or from an LLM | [`llms.txt`](llms.txt) |

The theory pages ship inside the package, so they survive a `pip
install` and are readable on a machine that never saw this repository. The
API and CLI references are generated from the source tree — `python3
docs/make_api.py` and `python3 docs/make_cli.py` — and a test fails if
either is stale, so a signature can never drift away from its description.

## Status

**M0–M3, M5 and M6 complete, plus the measurement report, the fitting
layer, the export layer, and most of M8.** 1532 tests; `stabilis validate` 41/41.

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
  that would fix it, *before any data is collected* — and saying which of
  those you already have, since two clocks against a common reference are
  one subtraction away from the edge that closes the triangle.
  `stabilis bench recommend` goes further and says *which technique* the
  setup supports, since the hat and cross-correlation are not alternatives:
  the hat separates the sources, cross-correlation removes the channel
  term, and a bench with both needs neither assumption.
- **Cross-variances and multi-channel noise rejection** — channels
  watching one clock, so the instruments' own noise cancels in expectation
  instead of forming a floor you cannot get under. Neither Stable32 nor
  `allantools` has this. Because every estimator already returns its
  coefficient array, cross-AVAR, cross-MVAR, cross-HVAR and cross-TVAR are
  one function: the auto case is `mean(c²)` and the cross case is
  `mean(c_a·c_b)`. `cross_matrix` does every pair at once and `reject`
  averages them, which reaches deeper still. The estimate is signed and
  goes negative at long τ, reported rather than clipped, and its error bar
  is not chi-square but an exact Isserlis lag sum that reduces to the
  chi-square one — bit for bit — when the two channels are one channel.
- **MTIE and TIE rms**, by binary doubling: 10⁶ points over 18 octave τ in
  0.15 s, where the naive form is O(N·m) per τ.
- **TOTVAR** with its published bias correction, and **Thêo1 / ThêoBR /
  ThêoH** — extending the usable τ range to 75 % of the record.
- **MTOT, TTOT and HTOT**, the total variances built on 3m subsequences,
  which keep 6m terms at every τ instead of collapsing at the long end.
  Cross-checked against a second implementation of the published equations
  to machine precision. They report **no confidence interval**: their terms
  are not one time-ordered sequence, so no coefficient-based EDOF applies,
  and the conservative fallback would over-count by three orders of
  magnitude in the optimistic direction. Refusing is the honest answer
  until the published degrees of freedom are implemented.
- **The MODWT wavelet family**, four filters, with confidence intervals.
  The Haar case returns the overlapping Allan variance *and* its error
  bar, which is a cross-check of two independent code paths rather than a
  special case. The longer filters are anchored on their *level*, not only
  their shape: `E[ν_j²] = ‖a_j ⊛ h‖²` is computed exactly from the
  generator's impulse response, so the truth side of that comparison
  simulates nothing.
- **Model fitting under the χ² likelihood**, with the correlation *between*
  τ measured rather than assumed away. The exact likelihood turns out to be
  a sum of squares, so Levenberg–Marquardt minimizes it with no Gaussian
  approximation; the covariance between τ comes from the same coefficient
  arrays through Isserlis' theorem, and a sandwich estimator carries it
  into the parameter uncertainties. Ignoring it leaves the parameters
  roughly unbiased and their error bars a quarter to a third too narrow —
  measured, in `docs/validation-report.html`, as a slope interval that
  covers 0.53 of the time against a nominal 0.68, where the
  correlation-aware one covers 0.67. Built-in models include a
  Gauss–Markov *knee*, which is a correlation time no power-law fit can
  produce. The fit refuses to quote parameters whose deviance is outside
  what its effective degrees of freedom admit, in either direction.
- **The measurement report** — see below.
- **Export** — the record back out as self-describing ASCII (the file
  carries its own rate and epoch, so a round trip needs no arguments and
  cannot be given the wrong ones), the curve as CSV or JSON, and the block
  Stable32 prints. That last one is byte-checked: all 6461 data rows in the
  committed Stable32 1.62 exports are reproduced character for character,
  the preamble with them, and the σ column computed here agrees to the last
  digit the format carries. Plus `stabilis export bundle`, a directory
  holding the data, the curve, the report and a SHA-256 for each, whose
  digest is over the *measurement* rather than the files — so two copies
  compare in sixteen characters however long apart they were made.
- **Power spectra** — Welch with the window's noise-equivalent bandwidth
  divided out, `S_x`/`S_y`/`S_phi` and `L(f)`. The carrier frequency is
  required rather than assumed, because a spectrum converted with a guessed
  one is wrong by a constant nothing downstream can detect. `L(f)` is
  **withheld** where the phase excursion is too large for `L = S_phi/2` —
  which is exactly the near-carrier region a datasheet gets written from.
- **Limit masks with a real verdict** — ITU-T G.811 transcribed from the
  Recommendation itself, with its *measurement conditions* enforced as part
  of the limit: a record sampled at 1 Hz, or shorter than the `12τ` the
  standard demands, yields `not evaluated` rather than a green tick.
- **Spur detection and notching** — tones found against a running-median
  floor and then measured by fitting the record, so a tone between bins is
  exact without a flat-top window. Every removal is recorded, because a
  record with a tone taken out describes an oscillator that does not exist.
- **Dynamic stability and realtime ingest** — σ(τ) as a *surface* over
  time, and an Allan variance that runs while the measurement does. See
  below.
- **Plots** — σ(τ) with asymmetric error bars, power-law slope guides,
  phase, frequency, spectra, `L(f)`, the bench solution, and the dynamic
  stability surface.

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

**Next** — the Thêo family's confidence intervals, and the
imputation-free gap-tolerant wavelet variance. Instrument readers wait on
paired vendor exports, and binding live sources to bench channels waits on
a real appending file.

## Gap filling

```bash
stabilis impute run.dat --kind frequency --rate 1 \
    --reason "receiver reboot 04:12" --out filled.dat --mask-out imputed.txt
```

Howe & Schlossberger's *Total Imputer*. Interpolating across a gap inserts
an interval containing **no noise**, so ADEV improves precisely because
data was invented. This fills instead with real measured data — the
preceding run, reversed, spliced with a slope matched on a low-pass-filtered
copy — so the filled samples carry the true noise process.

Three things it does that a fill usually does not:

- **it requires a reason**, records the imputed fraction as a warning
  rather than a note, and hands back the mask of what it made up. The
  authors ask that gaps and their cause be stated with the results, and a
  default reason would let that slide;
- **imputed samples buy no degrees of freedom.** Imputation raises the
  naive EDOF — the paper measures up to 6× for white noise — so
  `confidence(result, live_terms=live_terms(result, mask))` scales η by the
  fraction of terms free of invented data. Where no term is free, **no
  interval is quoted at all**: a χ² quantile at zero degrees of freedom is
  not a wide interval, it is no interval;
- **it refuses** a gap reaching either end of the record (trim it instead —
  it bridges nothing), a gap wider than the live data on either side, and a
  phase record, because the phase *across* a gap is not in the data even
  once the frequency samples are.

Validated against the paper's own Fig. 4 acceptance criterion. Two printed
ambiguities had to be resolved in code and both are argued in
`theory/impute.md` — including which series it fills, which is settled by
measurement: on phase, random-walk FM comes back **74× high**.

## Cross-variances

```bash
stabilis cross channel_a.dat channel_b.dat --rate 1
```

Two channels watching one clock. Write each as the clock plus that
channel's own measurement noise,

    x_a = x_c + n_a          x_b = x_c + n_b

and the auto-variance of either is `σ²_c + σ²_n` — the instrument is a
floor no amount of logging gets under. The cross term is not: the
difference operator is linear, so `E[d_a·d_b] = E[d_c²]` when the two
channels' noise is independent. **The instrument contribution has
expectation zero rather than being small.** That is what a
cross-correlation phase noise test set does in hardware; here it applies to
any two-channel capture.

On a synthetic clock a factor of three under its channels' noise, at
τ = 1 s: the single-channel answer reads 2.08e-12, the cross reads
7.07e-13, and the clock that generated both is 6.98e-13.

Three prices, printed rather than hidden:

- **The estimate is signed.** A mean of products, not of squares, so at
  long τ it goes negative — a real result meaning the shared part is below
  the scatter. Nothing clips it, and the deviation column prints `--`
  rather than `sqrt(|var|)`, which would draw a noise floor that is not
  there.
- **Convergence is slow.** Only the *estimate* of the instrument noise
  averages down, as `M^(1/4)` in amplitude. The table's `rej dB` column is
  the rejection actually achieved, and it reduces to the familiar
  `5·log10(M)` law as a consequence of the error bar rather than as a
  substitute for one.
- **Correlated channels break it silently.** A shared reference, a shared
  supply, one thermal environment: the answer is simply wrong with no
  symptom in the data, and no arrangement of the two records tests it. The
  assumption is a provenance warning on every call.

The error bar cannot be the chi-square one — a sum of products is unbounded
below. It comes from Isserlis' theorem instead: an exact sum over the four
lagged covariances of the two coefficient arrays, approximating nothing,
which reduces to `σ²√(2/η)` with η exactly `eta1` when the two channels are
one channel. So `cross_variance(a, a)` returns the auto-variance **and its
interval** bit for bit — the chi-square error bar is the auto case of this
one, not a parallel construction. `stabilis docs cross` tabulates the two
estimators that were tried first and rejected, and where each fails.

### Hat or cross-correlation, from the graph

```bash
stabilis bench recommend bench.json
```

They are not alternatives — they remove different unknowns from the same
design matrix:

|  | separates the sources | removes the channel term |
|---|---|---|
| **N-cornered hat** (odd cycle among different sources) | **yes** | no: declare the floor, or measure it back-to-back |
| **cross-correlation** (one edge measured twice, through different channels) | no | **yes, in expectation**, declaring nothing |

So they compose, and the composition is worth having. Measure A against B
twice through two *independent* channels, and the cross-variance of those
records is `σ²_A + σ²_B` with the channel noise rejected — a hat row with
no channel term. **That is the classical three-cornered hat's missing
row.** Three sources through one channel is four unknowns against three
rows, and the textbook fix is to assert `σ²_channel = 0`, which nobody
measured. One repeated edge closes the same system having measured it
instead, and `recommend` says so by adding the derived row to a
hypothetical bench and re-running the identifiability check.

One correction worth stating, because this library had it backwards in a
warning string until it was checked: two channels of **one** instrument are
not the good case for cross-correlation. It rejects the part of the channel
noise the two channels do not share; the part they do — one timebase, one
LO, one supply, one oven — passes straight through, and the residual is a
floor rather than something more data lowers. Independent instruments are
better for *both* techniques. `recommend` reports the same-device case as
capped rather than clean, and does not let it close a system.

### More channels

```bash
stabilis reject ch_a.dat ch_b.dat ch_c.dat ch_d.dat --rate 1
```

Every pair is an unbiased estimate of the same clock, so averaging them
reaches deeper than any one. How much deeper is the same Isserlis sum with
every pair passed in — and the pairs are **not** independent, since (a,b)
and (a,c) share a channel. Four channels is six pairs, which would be
3.9 dB if they were; measured on a clock a factor of three down it is
3.0 dB at τ = 1 s, falling to nothing by τ = 32 s where the clock dominates
and there is no instrument noise left to reject. Two channels is one pair,
so `reject` on two returns exactly what `cross_variance` does — the two
entry points are one implementation, and there is a test that says so.

## The measurement report

```bash
stabilis report dut.dat --rate 1 \
    --spec 1:2e-12 --spec 10:8e-13 --spec 100:3e-13 \
    --subject "OCXO 42 vs H-maser" --out report.html
```

One self-contained HTML document per measurement: the σ(τ) table with its
intervals, the noise type at each τ, the record's own description and
hash, the full provenance step log, and a conformity verdict.

The verdict is the part that differs from every other tool. **It is taken
on the confidence interval, not on the point estimate.** A point passes
only when its whole interval lies below the limit and fails only when the
whole interval lies above it; an interval straddling the limit is reported
`indeterminate`, meaning the record is too short to decide and the remedy
is more data. Comparing the point estimate instead — what the field
normally does — is simple acceptance with a ~50 % false-accept risk for a
unit sitting on the limit, and it is worse than that here: at low degrees
of freedom the point estimate is itself biased *low* (11.4 % at η = 2), so
the error runs towards passing exactly where the evidence is thinnest.

Nor is a limit interpolated. A datasheet states σ(τ) at particular τ; the
tau grid is extended to land on each of them so the comparison is made
where the specification actually lives, and a spec τ the record cannot
reach is reported as not evaluated rather than guessed at.

It also states what it cannot conclude: a two-source comparison bounds the
device under test from *above*, so a pass is conclusive while a failure
may belong to the reference or to the measurement channel. Separating
those is what the bench model does. See `stabilis docs report`.

This is a measurement report, not a calibration certificate — it asserts
no traceability chain and makes no accreditation claim. `stabilis
validate` is the separate, and different, question of whether the software
is right.

## Getting it back out

```bash
# the record itself; the file carries its rate and epoch
stabilis export data dut.dat --rate 1 --out clean.dat
stabilis stats clean.dat              # no --rate needed, it says so itself

# the curve, for a spreadsheet, a program, or Stable32
stabilis export table dut.dat --rate 1 --format csv --out curve.csv
stabilis export table dut.dat --rate 1 --format stable32 \
    --edof-method greenhall --alpha 0 --out curve.txt

# everything, with a manifest of what it all hashes to
stabilis export bundle dut.dat --rate 1 --spec 1:2e-12 --out ./run-2026-08-15/
```

Three decisions worth knowing about. Values are written at **17 significant
digits**, because `%.15e` — the width people reach for — returns a different
double for 42 % of them; a record written and re-read is bit-identical.
Gaps are written as gaps, and the Stable32 format, which has no spelling for
a missing sample, **refuses** rather than writing the zero its own
convention would suggest: a genuine zero-offset measurement and a missing
one must not become the same bytes. And a bundle's digest is taken over the
*measurement* — the curve, the interval, and the hash of the data they came
from — not over the files, since the provenance embedded in them carries a
timestamp and a file-hash digest would change on every run.

The Stable32 writer is checked against Stable32's own output rather than
against ours: all 6461 data rows in the committed 1.62 exports are
reproduced character for character, and the σ column computed here from the
same input agrees to the last digit the format prints. What no test here
shows is that Stable32 can *read* what we write — that needs Stable32, and
it is item 8 of `VALIDATION-DEBT.md` until somebody runs it. See
`stabilis docs export`.

## Spectra, masks and spurs

```bash
stabilis psd dut.dat --rate 1 --spectrum S_y
stabilis psd dut.dat --rate 1 --script-l --nu0 10e6      # L(f) in dBc/Hz
stabilis mask dut.dat --mask itu-t-g.811-tdev            # exits 1 on failure
stabilis spur dut.dat --rate 1 --notch-out clean.dat
```

Three refusals worth knowing about, each because the mistake is invisible
afterwards.

**`L(f) = S_phi/2` is withheld where it is not true.** It is the
small-angle limit, and close to the carrier of a noisy source it fails —
which is exactly the region a datasheet gets written from. Those bins come
back NaN and the plot leaves a gap rather than joining across it.

**A standard's measurement conditions are part of its limit.** G.811 wants
a sampling time no coarser than 1/30 s and, for TDEV, a record of at least
`12τ`. A curve violating either has not been compared with the standard,
so those points read `not evaluated` — never `pass`. Judging a 1 Hz record
against G.811 therefore yields no verdict at all, which is the correct
answer and a more useful one than a confident tick.

**A mask is never extrapolated.** G.811's TDEV limit ends at 10 000 s and
so does ours; continuing the last segment would invent a requirement and
attribute it to the ITU. The limits were fetched from the Recommendation
rather than recalled, which caught one: the widely repeated form of the
MTIE limit above 1000 s is a flat 300 ns ceiling, and the document says
`10⁻⁵τ + 0.29 µs`, which rises.

Spur amplitudes are measured by fitting the record rather than reading the
spectrum, so a tone between bins is exact without a flat-top window —
0.001 % on amplitude and about a nanohertz on frequency. Notching records
every tone it removed: a record with a tone taken out describes an
oscillator that does not exist, and the file has to say so.

## A GUI, for when flags are tedious

```bash
stabilis gui                     # opens http://127.0.0.1:8765
stabilis gui --root ~/data       # confine the file browser to one tree
```

A local page in four stages, which is the order the questions arrive in.

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

**4 · Bench.** Describe the *setup* rather than the file — which
oscillators, which counters and channels, and which record is which pair —
and get the N-cornered hat out of it. The identifiability verdict is
recomputed on every edit, **before any data is read**, because "go and
measure B against D" is cheap advice on a Tuesday and expensive advice
after a fortnight of logging. A chain and a four-clock ring are told they
determine nothing however long they run, and both diagonals are offered as
the fix. The classical three-cornered hat is shown for what it is: four
unknowns against three rows until you either declare the channel floor —
the assumption the textbook method makes silently — or add the
back-to-back run that measures it.

Crucially, the advice distinguishes **the pairs you must go and measure
from the pairs you already have**. The odd-cycle theorem is about the
variances, not about the records: `x_ab = x_ac − x_bc` identically, so two
clocks measured against a common reference are one subtraction away from
the edge that closes the triangle. That is how the hat is normally run —
two channels against the house clock, the third pair formed in software —
and the reference need not be special, since C can be the local clock and
the hat still returns `σ_C`. A four-clock ring turns out to need no new
data either. `stabilis bench check` names the two rows to subtract, and
states the two conditions the graph cannot check: the records must overlap
in time, and a derived row carries both channels' noise.

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

Stage 4 is the only stage that reads several files at once, and it assumes
the least about them. The others examine a file you just picked and are
already looking at; a bench is six exports from six instruments, so the
delimiter, units, timebase, kept stretches and outlier threshold are all
**per measurement**. The record is drawn before it is solved — min/max
envelope, so a one-sample glitch survives — and stretches are kept by
dragging rather than by typing indices at a record you cannot see. Several
stretches become one gapped record, never two curves, because a measurement
is one row of the bench. What each file needed tolerating, how many points
were gaps, how many the outlier threshold removed and how far its timetags
wander are reported against the row they belong to.

One trap that costs an afternoon: a timebase in **MJD needs an explicit
rate**. Day number 59000 has a float64 spacing of 6.3e-7 s, so a 1 Hz grid
infers as 0.99999976 Hz, and its taus then miss round numbers by more than
the solver's tolerance — against a counter given an exact 1 Hz the two
share no averaging time at all. The refusal is correct, since nothing is
interpolated onto a common grid; the page adds the rates it used, how each
was obtained, and the fix.

Stage 4 needs no new field on `Measurement` for the columns it assigns,
because `bench.solve` is handed *results* and never opens a file. The page
strips each measurement to its data column itself — expression, pedestal,
units, crop — and hands over a mapping of finished results. The column
belongs to the file; the bench describes the experiment; recording both in
`bench.json` would create the one place they could disagree. Saving hands
the JSON to your browser's downloads: this server reads your disk and never
writes to it.

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

## Was it doing the same thing all along?

```bash
stabilis davar run.dat --rate 1 --window 1024 --step 256
```

A single σ(τ) curve presumes the oscillator was stationary. Real ones warm
up, get bumped, and sit next to an air conditioner. Slide a window along
the record and evaluate the same estimator inside it, and you get a
surface instead of a curve.

On 8192 s of white FM with the noise multiplied by five for a thousand
seconds in the middle, the whole-record `oadev` reports 1.44e-12 at τ = 1 s
with a clean −1/2 slope — *a white-FM oscillator twice as noisy as this one
actually is*, describing neither state it was in. The surface says:

```
         t (s)           oadev
        2047.5    6.929e-13     <- quiet
        2815.5    2.273e-12
        3583.5    3.454e-12     <- the disturbance
        4351.5    1.578e-12
        4607.5    6.921e-13     <- recovered
```

`davar(x, window=len(x))` returns **bit for bit** what `oadev(x)` returns,
because it *is* that call — the surface is built by evaluating the
registered estimator on each window rather than reimplementing it, so
`--estimator ohdev` gives a dynamic Hadamard deviation and a plugin gets
its own for free.

The literature's O(1) recursion, which adds the entering term and
subtracts the expiring one, is deliberately not used. It has no
self-correcting mechanism, and the error it accumulates is **permanent**:
measured over 200 000 slides, ordinary noise costs 4e-14, one 1 µs phase
step costs 1.3e-6 *long after the step has left the window*, and a 1 ms
step — an ordinary counter glitch — leaves the surface 28 % wrong for the
rest of the measurement, looking exactly like a physical trend.
`stabilis docs davar` has the table.

## A measurement that has not finished

```bash
stabilis stream live.dat --rate 1 --time-column 0 --column 1 \
         --expect 100000 --follow
```

The overlapping Allan variance maintained sample by sample, in O(max τ)
memory however long the run lasts. It agrees with the batch estimator to
**one ulp, flat in record length** — the coefficients are bit-identical
because the arithmetic's association order is copied from the batch
expression, and the sum is Neumaier-compensated because an accumulator
cannot sum pairwise. The same accumulator with a plain running sum drifts
to 74 ulp at N = 65536.

The load-bearing part is not the arithmetic. It is this:

> **A sample that never arrived becomes a NaN in its own slot. It never
> becomes the disappearance of a slot.**

Append arriving samples to a list — the obvious implementation — and a
counter that drops fifty readings hands you a record fifty samples
shorter and *perfectly smooth*. Every later sample has slid earlier, the
timebase is wrong from the outage onward, and σ_y(τ) comes back better
than the oscillator is, with nothing in the data to say so. Samples are
placed by timestamp instead, and a stream that omits its dropouts entirely
rebuilds the original record NaN for NaN.

Three things are refused rather than repaired: a timestamp more than a
quarter-sample off grid (the declared rate is wrong, and placing it anyway
is interpolation), one that goes backwards or repeats (that would take the
record's order from the network's scheduling), and an outage too long to
materialize. And a file with **no timestamp column cannot report a dropped
sample at all** — so the provenance says the timestamps were synthesized
and calls the gap count a floor rather than a count.

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
