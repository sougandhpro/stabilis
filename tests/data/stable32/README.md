# Stable32 parity session — run sheet

**Status: waiting on one manual session. Nothing here is committed output
yet.** This is item 1 of `VALIDATION-DEBT.md`, the highest-priority open
item, and it is the one thing in this project that cannot be automated: it
needs Stable32 driven by hand on Windows.

Until it happens, this library's claim is *"implements the published
Greenhall algorithm"*, not *"matches Stable32's output"* — and
`theory/greenhall.md`, the validation report and the progress report all
say exactly that. Your session is what changes the claim.

Budget about an hour. Everything that could be prepared in advance has
been.

---

## What this session is for

Stable32 prints `Alpha`, `Min Sigma`, `Sigma`, `Max Sigma`. The sigma
columns test the estimators; the `Alpha`, `Min` and `Max` columns test the
Greenhall EDOF path, which is the headline interoperability claim, because
an accredited lab replacing Stable32 will diff exactly those columns.

We already know the algorithm agrees with an independent implementation of
the same paper across 18 627 parameter combinations to 1e-9. What is
untested is agreement with the actual program.

---

## Before you start

1. **Get Stable32 1.62.** Free since IEEE UFFC took ownership in Dec 2017:
   <https://ieee-uffc.org/software>, or the mirror at
   `wriley.com/162Stable32.exe`. Install on Windows normally — `wine` is
   not needed and is not installed here.
2. **The inputs are already generated**, in `inputs/`, with their SHA-256
   in `inputs/MANIFEST.md`. They are reachable from Windows at
   `G:\projects\stabilis\tests\data\stable32\inputs\`.
   Do not regenerate them unless you mean to: `generate.py` is
   deterministic, but if the bytes change the manifest hashes must be
   updated in the same commit.
3. **Read `inputs/MANIFEST.md` once.** It lists what each file is and the
   expected σ(τ) — including two *systematic* short-τ offsets (random-walk
   FM is 22% high in deviation at τ₀, flicker FM 20%) that are properties
   of discretely sampled noise and are in the data. If you see those,
   nothing is wrong. Not knowing about them is the likeliest way to
   conclude there is a disagreement when there is not.

---

## The runs

### A. Our own records — the ones that matter most

Six files in `inputs/`. Each is **phase data in seconds, τ₀ = 1 s, one
column, 4097 points**.

For each file, in Stable32:

1. `File → Open`, select the `.dat`. Set **Data Type: Phase**, **τ₀ = 1**.
   Confirm it reads 4097 points; if it reads 4096 or 4098, stop and note
   it — that is a real finding about the reader, not a nuisance.
2. `Analysis → Stability`. Set:
   - **Sigma Type: Overlapping Allan** (do the whole set again for
     `Modified Allan` and `Overlapping Hadamard`, below);
   - **Tau Mode: Octave**, then repeat with **All** and **Decade**;
   - **Confidence: on**, at the **68.27%** (1σ) setting;
   - **Bias correction: off** if it is offered as a separate toggle — we
     want the raw estimator, and the bias correction is tested separately.
3. Export the results table: `File → Save` or the table's own export, to
   `outputs/<file>_<sigma>_<taumode>.txt`, e.g.
   `outputs/alphap0_oadev_octave.txt`. **Save the literal output,
   header and all** — the header carries the version and settings, which
   is evidence.
4. Note anything Stable32 says that we would not: warnings, refusals,
   points it drops.

Sigma types to cover, in priority order — if time runs short, stop after
the first:

| priority | Stable32 sigma type | our estimator |
|---|---|---|
| 1 | Overlapping Allan | `oadev` |
| 2 | Modified Allan | `mdev` |
| 3 | Overlapping Hadamard | `ohdev` |
| 4 | Normal Allan | `adev` |
| 5 | Time deviation | `tdev` |

**`drift.dat` is the one to run first.** It is pure linear drift, where
`σ_y(τ) = D·τ/√2` **exactly** with `D = 1e-13/s` — no statistics involved.
If Stable32's sigma column does not reproduce `7.0711e-14` at τ = 1 s and
`9.0510e-12` at τ = 128 s, something is wrong with the file, the import
settings or the τ₀, and every other run in the session would be worthless.
It is a two-minute check that protects the other fifty-eight.

### B. The classic public datasets

Fetch these on the Windows side (URLs recorded in `references.md`):

- `wriley.com/nbs.dat` — 9 frequency points, the NBS Monograph 140 set.
  **Data Type: Frequency**, τ₀ = 1.
- `wriley.com/tst_suit.dat` — 1000 frequency points, the Stable32
  pseudo-random test suite. **Data Type: Frequency**, τ₀ = 1.
- `PHASE.DAT` — from the allantools test directory. **Phase**, τ₀ = 1.

Same procedure. These matter because SP 1065 Tables 30 and 31 publish
expected values for the first two, which `reference.py` already asserts
against — so this run closes the loop between the printed table, the
program that produced it, and us.

**Do not commit `PHASE.DAT`, `nbs.dat` or `tst_suit.dat` themselves** —
only Stable32's output over them. Their licence status is unresolved
(`references.md`, "Third-party code we do not use"), and the outputs are
facts about Stable32's behaviour rather than copies of anyone's dataset.

### C. Your own hardware, if you have a record to hand

One real WR or OCXO phase record is worth more than all of the above for
the *credibility* of the evidence bundle, because it is the only case
where the data was not generated by the tool being validated. Any length.
Same procedure; record the instrument and the measurement setup in
`PROVENANCE.md`.

---

## As you go

Fill in `PROVENANCE.md` — it is a blank template, and blank fields are
better than guessed ones. The fields exist because an accredited lab's
metrologist will ask exactly those questions.

Put every exported file in `outputs/`, named
`<input>_<sigma>_<taumode>.txt`. Do not reformat, re-save through a
spreadsheet, or strip headers. If Stable32 writes something odd, that is
data.

---

## Afterwards — what I do with it

Once `outputs/` and a filled `PROVENANCE.md` are committed, an `interop`
tier gets added that asserts our `Min Sigma` / `Sigma` / `Max Sigma`
against Stable32's, **at the tolerance the exported significant figures
support and no tighter**. Then item 1 moves out of `VALIDATION-DEBT.md`
and into `docs/validation-report.html`, and the claim becomes "matches
Stable32".

If a column disagrees, it gets recorded the way every other source
disagreement in this repo has been — published value, our certified value,
and the reason — and **not** fixed by changing our number to match. See
`reference.KNOWN_DISCREPANCIES`, whose one historical entry turned out to
vindicate exactly that discipline.
