# Provenance of the Stable32 reference outputs

**Template. Every field below is blank and must be filled from the actual
session — not from memory, not from what it "should" be.** A guessed
provenance line is worse than a missing one: it makes an unreproducible
result look reproducible. Leave anything you did not observe blank and say
so.

See `README.md` in this directory for the run sheet.

---

## The session

| field | value |
|---|---|
| Stable32 version (from `Help → About`, verbatim) | |
| Executable SHA-256 | |
| Where the executable came from (URL or media) | |
| Operating system and version | |
| Date of session (ISO 8601) | |
| Operator | |

## Global settings

Settings that applied to every run. Anything left at its default should
say "default", with the default's value if it is visible.

| field | value |
|---|---|
| Confidence interval setting | |
| Confidence level used | |
| Bias correction on/off | |
| Drift removal on/off | |
| Outlier/deletion settings | |
| Anything else changed from a fresh install | |

## Runs

One row per exported file. `Input SHA-256` for our own files comes from
`inputs/MANIFEST.md`; for fetched files, compute it on the file as
downloaded (`certutil -hashfile <file> SHA256` on Windows).

| output file | input file | input SHA-256 | data type | tau0 (s) | points read | sigma type | tau mode | notes |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

## Fetched inputs

For anything downloaded rather than generated here. The files themselves
are deliberately not committed — see `README.md`.

| file | URL | date fetched | SHA-256 |
|---|---|---|---|
| | | | |

## Own-hardware records, if any

| field | value |
|---|---|
| Source oscillator(s) | |
| Measurement instrument | |
| Channel / configuration | |
| tau0 and record length | |
| When the data was taken | |
| Anything known to be wrong with it (gaps, resets, temperature events) | |

## Observations

Anything Stable32 did that was surprising, refused, warned about, or
silently changed. Points it dropped. Columns it left blank. Disagreements
you noticed on the spot.

Write them down even if they look like operator error — especially then.

-
-
-

## Not done

What the run sheet asked for and this session did not cover, so that the
gap is visible rather than inferred from an absence.

-
-
