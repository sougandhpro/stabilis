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

## What a round trip does and does not prove

| assertion | what it establishes |
|---|---|
| written → read → identical doubles | **Self-consistency.** This library's writer and reader agree. Enough to use an export as an intermediate; not evidence about any other tool. |
| gaps, epoch and rate survive | Self-consistency, same standing. |
| the printed timetag check | A property of the file itself, independent of any reader. |
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

**Not validated.** That anything outside this repository can read what it
writes. See the table above and `VALIDATION-DEBT.md`.
