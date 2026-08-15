# Export: getting the numbers back out without changing them

Reading a file is a problem of tolerance — real exports are ragged, and
`theory/gaps.md` and `io/text.py` are largely about surviving that. Writing
one is the opposite problem. Nothing is malformed and nothing has to be
guessed, so every defect in an export is a decision somebody took on
purpose. This page is the record of four of them.

## 1. Seventeen digits, because fifteen is not enough

A float64 has 53 bits of mantissa, which is between 15 and 17 decimal
digits. Seventeen significant digits is the width at which *every* double
survives the trip out to text and back; sixteen also suffices in practice;
fifteen does not, and the failure is not rare.

Measured on 200 000 normal deviates: `%.15e` returns a different double for
**42 %** of them. That is a bit or two in the last place, which is nothing
next to any real measurement uncertainty and everything next to a test that
asserts two analyses of the same record agree. So the default is `%.17g`,
and `tests/test_io_write.py::test_fifteen_digits_would_not_have_been_enough`
keeps the measurement in the suite rather than the claim in a comment.

The exception is deliberate. `tests/data/stable32/generate.py` still writes
its fixtures at `%.15e`, because those files and their SHA-256s are the
inputs Stable32 was actually run on: changing a digit would silently
invalidate every committed export in `outputs/`. The text on disk is the
reference, not the doubles that produced it.

The Stable32 σ(τ) *table* is a separate matter — it prints four decimals,
so exporting a curve to it is lossy by construction. That is the format's
choice, not ours, and it is why the table is an interchange format and not
an archive format.

## 2. An export carries its own rate and epoch

`PhaseSeries` stores phase as offsets from an explicit `epoch`, for the
reason set out in `theory/precision.md`: at accumulated phase of 10³–10⁶ s
the float64 spacing is 10⁻¹³–10⁻¹⁰ s, which is exactly the resolution this
field cares about. A file holding only the offsets has thrown away the
anchor, and a file holding only the numbers has thrown away the sample
interval too.

So the `stabilis` and `csv` formats begin with a comment block:

    # phase_s
    # stabilis phase export, format stabilis
    # rate_hz = 1.0
    # epoch_unix = 1700000000.0

and `load_phase` reads `rate_hz` and `epoch_unix` back. Three properties
make that safe rather than a new guessing surface:

* only those two keys are recognised, and only when they parse as floats;
* anything else in the block is ignored, so a malformed header can never
  stop a file being read;
* an explicit `rate_hz` from the caller always wins.

The column-name line comes *first* because `read_table` inspects only the
first line of a file for names. Putting a provenance banner above it would
quietly demote every column to `column 0`.

**The rate and the timetags are not cross-checked**, and that is deliberate.
The reader has no unit information — it differences the time column and
assumes seconds — so an MJD column, which is days, disagrees with the header
by a factor of 86 400 while both are perfectly correct. A check that cannot
tell a real conflict from a unit teaches its reader to ignore it.

## 3. Timetags that cannot separate two samples are refused

A timetag column is optional, and the default is not to write one: the rate
in the header is exact, and a column of times is a second copy of
information that can then disagree with the first.

When one is asked for, `elapsed` counts from zero and stays exact. `unix`
and `mjd` do not. MJD at a present-day date is ~6 × 10⁴ days, where float64
spacing is ~7 × 10⁻¹² days, about **0.6 µs**. That is comfortable on a
one-second grid, marginal at a microsecond, and on a nanosecond grid
consecutive timetags *print as the same number*.

The writer therefore checks the printed strings — not a model of their
precision, since the strings are what a reader receives — and:

* **refuses** when consecutive tags collide, because a column claiming two
  samples were simultaneous would make our own reader infer an infinite
  rate from our own file;
* **records a warning step** when they are merely coarse, at the same
  threshold (10⁻³ of the sample interval) at which `io/text.py` warns about
  timetag jitter on the way in. The writer warns exactly when the reader
  would, so a file this library writes never surprises the library reading
  it.

Only the last pair needs testing: float64 spacing grows with magnitude and
the grid is uniform, so if the final two tags survive as distinct text then
every earlier pair does.

## 4. A gap is written as a gap, or not at all

NaN is this library's gap representation throughout, and the `stabilis` and
`csv` formats simply write `nan` — which `io/text.py` reads back as a gap,
through `np.loadtxt`'s fast path, with no defect reported.

The `stable32` format has no spelling for a missing sample. The format's own
convention is that zero means a gap in frequency data, and that convention
is precisely what `io/text.py` refuses to assume on the way in:

> Stable32 treats a zero as a gap marker in frequency data, which is
> convenient right up until a genuine zero-offset measurement is silently
> deleted.

The same asymmetry applies on the way out, and worse for phase, where zero
is an ordinary phase value. So writing a gapped record in `stable32` format
**refuses** unless the caller names a sentinel explicitly — and a sentinel
that already occurs in the data is refused too, since reading that file back
would delete real samples.

## 5. The curve: three shapes, and what each one drops

A σ(τ) curve leaves by one of three doors, and each loses something the one
above it kept.

**JSON** is the complete record: `af`, `tau_s`, `var` *and* `dev`, the
interval, the per-τ EDOF and the method that produced each one, the points
where Greenhall declined and why, and the provenance behind all of it. It
carries `var` as well as `dev` because the library stores variances and
derives deviations — an export that shipped only the deviation would force
every consumer to square it back, which is the round trip the code style
forbids internally.

The provenance written is the *interval's*, not the result's. Computing an
interval appends steps, and taking the shorter record would silently drop
the EDOF method from the audit trail of a number that depends on it.

**CSV** is a table for a spreadsheet and is lossy by construction. A
rectangular grid cannot hold a per-τ method string or a decline reason, so
those stay in the JSON. The column budget is hard rather than tasteful:
`io/text.py` refuses a file wider than `MAX_COLUMNS = 8`, so a ninth column
would produce a file this library could not read back. A column with
nothing to put in it is *absent* rather than padded — an empty cell reads
back as a gap and a zero reads back as a measurement.

**Stable32 text** is an interchange format, not an archive format. It
prints σ to four decimals, so a curve written there has shed eleven digits
it had. What it buys is that Stable32 and TimeLab can open it.

### The Stable32 block is transcribed, not derived

Every one of the **6461 data rows** in `tests/data/stable32/outputs/` is
reproduced character for character by one format string:

    {af:5d}{tau:13.4e}{n:9d}{alpha:6d}{lo:13.4e}{sig:14.4e}{hi:15.4e}

75 characters, contiguous, no separators, CRLF-terminated. The eight
preamble lines reproduce exactly too, with `Maximum`, `Minimum` and
`Average` being `preprocess.basic_stats` at `%.6e` — which incidentally
confirms our descriptive statistics agree with Stable32's on the same file.

The *column header* lines are hand-tuned strings that do **not** align to
the data beneath them, and there are four of them because the middle column
is renamed after the statistic (`Sigma`, `Mod Sigma`, `Hadamard Sig`,
`Time Sigma`). They are stored verbatim rather than generated from a width.
This is the same fact that makes `validate/stable32.py` match rows by arity
and never by header text.

`export/` does not import `validate/`, in either direction: importing
`stabilis.validate` executes its `__init__` and pulls the whole
evidence-report chain, which an export has no business dragging in. The two
modules' names for the five statistics are asserted equal by test instead.

### Four refusals, because the format has no honest encoding

* **An estimator Stable32 never named** — TOTDEV, the Thêo family, MTIE,
  the wavelets. There is no Sigma Type for them.
* **A frequency series.** Every committed export says `Phase Data Points`.
  No export of a frequency file has been captured, so the preamble for one
  would have to be guessed.
* **A gapped record.** `Points 1 thru N of N` is a contiguity claim.
* **A result with no α.** The Alpha column is an integer and the format has
  no way to say "unknown". Writing 0 asserts white FM — and the data-driven
  EDOF methods leave `alpha` as `None` *because* not assuming a noise type
  is the entire point of them. Filling it in here would undo that where
  nobody would look.

### What the comparison actually shows

Against the committed Stable32 1.62 export of `alphap0.dat`, computing the
curve here from the same input file:

| part of the block | agreement |
|---|---|
| preamble and column header | byte-identical |
| `AF`, `Tau`, `#`, `Alpha` | byte-identical |
| `Sigma` | byte-identical at the four decimals printed; worst relative difference 4.9 × 10⁻⁵ over 6458 points |
| `Min Sigma`, `Max Sigma` | differ in the last digit or two — Stable32's implied EDOF runs up to 1.9 % above ours at small m |

That last row is a **known, recorded discrepancy**, not a defect of the
writer: it is `VALIDATION-DEBT.md` item 1, and it is measured every run by
`stabilis validate`. If the bounds ever agreed exactly, something would
have been fitted to the reference rather than derived.

## 6. The bundle, and what its digest is over

A bundle is one measurement as a directory: the record itself, the curve in
two or three forms, the report, the provenance, and a SHA-256 for each.

    run/
      MANIFEST.json      schema, digest, and a hash for every file below
      SHA256SUMS         the same hashes, in the format `sha256sum -c` reads
      curve.csv          eight columns
      curve.json         the complete record
      curve.s32.txt      the Stable32 block, when the format admits it
      data/<name>        the source record, byte for byte
      provenance.json
      report.html
      sigma_tau.png      when matplotlib is installed

The source record is **copied, not re-serialized**, so the copy's SHA-256
still equals `provenance.source_sha256` and the chain from the bytes that
arrived to the number that was reported is unbroken. A re-serialized copy
would hash differently and quietly sever it. `--no-data` leaves the record
out for a large or licence-encumbered file; the *hash* stays either way,
so the bundle is still anchored to what was analysed.

### The digest is over the measurement, not over the files

Hashing the files is the obvious construction and it is wrong. `curve.json`
embeds the provenance, provenance records *when* each step happened, and a
digest that changes on every run of the same analysis says nothing at all.

So the digest covers the curve (`kind`, `af`, `tau_s`, `var`, `n_terms`),
the interval (`level`, `edof`, `var_lo`, `var_hi`, `method`, `alpha`), and
the SHA-256 of the data they came from. That last term matters: without it,
two different records that happened to yield the same curve would collide.

This is `report.build`'s rule one level up — its digest already excludes the
operator, the title and the timestamp so that two copies of one report
compare by sixteen characters.

**The bytes of a bundle are therefore not reproducible, and that is not a
defect.** Provenance is a record of when work was done; a bundle that could
be made byte-identical on another day would be one that had thrown that
away. What is reproducible is the digest — and the per-file hashes in
`MANIFEST.json` and `SHA256SUMS` are for a different job, detecting
tampering with *this* copy.

The figure is the one optional member. The core is numpy-only and that is a
selling point, so a bundle built without matplotlib carries every number
and simply says in provenance that the picture is missing.

## What a round trip does and does not prove

| assertion | what it establishes |
|---|---|
| written → read → identical doubles | **Self-consistency.** This library's writer and reader agree. Enough to use an export as an intermediate; not evidence about any other tool. |
| gaps, epoch and rate survive | Self-consistency, same standing. |
| the printed timetag check | A property of the file itself, independent of any reader. |
| the CSV re-read through `read_table` | Self-consistency. |
| σ(τ) table → `validate/stable32.parse` | Self-consistency, and only to four figures, because that is all the format carries. |
| **6461 committed rows rebuilt exactly** | **Interop, formatting.** The reference is Stable32's own bytes. Establishes the layout, not our arithmetic — the values came from our parser. |
| **the preamble rebuilt exactly** | **Interop, formatting**, plus agreement of the descriptive statistics. |
| **σ computed here vs σ Stable32 printed** | **Interop, numbers and formatting together.** Everything but the reference is ours. The strongest assertion available. |
| **no test in this repo** | **That Stable32 or TimeLab can open a file we wrote.** |

That last row is the honest one. `VALIDATION-DEBT.md` item 3 states the rule
it follows from — *a reader validated only against its own writer is
validated against nothing* — and the mirror image holds for writers. It is
not falsifiable in CI, because the falsifying tool does not run there. It is
discharged by one addition to the manual run sheet at
`tests/data/stable32/README.md`: export `alphap0.dat` back out through this
library, open *that* file in Stable32 1.62, and check its statistics against
the export already committed for the original.

## Validation status

**Validated.** `tests/test_io_write.py`:

* a written series reads back bit for bit, in both self-describing formats,
  including signed zero, a subnormal, the top of the float64 range and NaN;
* gaps come back as gaps and are not reported as defects the reader had to
  tolerate;
* the rate and the epoch are recovered without being supplied again, and an
  explicit rate from the caller overrides the header;
* a malformed or unrecognised header line is ignored rather than fatal, and
  a comment below the data is not read as metadata;
* `stable32` format refuses a gapped record, refuses a sentinel that occurs
  in the data, and writes no comment lines at all;
* MJD timetags on a nanosecond grid are refused and on a microsecond grid
  are recorded as coarse; elapsed timetags are exact at any epoch;
* the returned record's hash is the hash of the bytes on disk, and line
  endings are LF whichever platform wrote them;
* the fifteen-digit measurement above, so the default has evidence rather
  than a rationale.

`tests/test_export_table.py`:

* the CSV never exceeds the width our own reader accepts, reads back
  through `read_table` with its column names and no defects, and keeps
  every digit of the curve;
* a curve with no interval writes four columns rather than columns of
  filler, and the α column appears only when a power law was assumed;
* the JSON carries `var` beside `dev`, takes the interval's provenance
  rather than the result's, says whether the quantity is a variance, and
  keeps what the CSV cannot — the per-τ method and the decline reasons;
* **the command line's `--json` output is byte-for-byte this module's**,
  which is the proof that the CLI no longer keeps a second hand-written
  copy of the result schema, and that every key it used to emit is still
  there.

`tests/test_export_stable32.py` — the three **interop** assertions in the
table above, plus: the absent-bound convention survives in both directions;
our five statistic names are the reader's map inverted; the line endings
are CRLF; and each of the four refusals.

`tests/test_export_bundle.py`:

* the same analysis gives the same digest, a different estimator gives a
  different one, and the digest does not move although the evidence files
  carry wall-clock times — which is the measurement showing that a
  file-hash digest would have been the wrong construction;
* the copied record is the original bytes and hashes to what provenance
  recorded, and leaving it out changes neither the digest nor the anchor;
* an in-memory series is serialized instead and *said* to have been;
* the manifest hashes every file it lists and lists every file except
  itself and `SHA256SUMS`, whose format `sha256sum -c` accepts;
* `verify` returns every changed file rather than the first, and reports a
  missing one rather than raising;
* a bundle built without matplotlib is complete but for the picture;
* it refuses to write over a bundle unless asked.

**Not validated.** That anything outside this repository can read what it
writes. See the table above and `VALIDATION-DEBT.md`.
