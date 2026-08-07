# float64 and sub-nanosecond phase

## The hazard

Phase is stored in seconds as float64, which carries roughly 15-16
significant decimal digits. The spacing between representable numbers
scales with magnitude:

| phase magnitude | float64 spacing |
|---|---|
| 1 s | 2.2e-16 s |
| 1e3 s | 1.1e-13 s |
| 1e5 s | 1.5e-11 s |
| 1e6 s | 1.2e-10 s |

A White Rabbit link is characterised at the picosecond level. If phase is
stored as absolute seconds derived from an MJD epoch, the representation
alone quantizes away the entire measurement — silently, with no error, no
warning, and a plot that looks plausible.

## The convention

`PhaseSeries.x` holds **offsets from an explicit `epoch`**, never absolute
time. The epoch is a separate field and is not used in any arithmetic that
touches the phase values.

## The guard as implemented

`types.precision_ratio(x)` computes

    ratio = spacing(max|x|) / min(|diff x| > 0)

that is, the float64 spacing at the largest magnitude present, divided by
the smallest step the data actually resolves. A ratio near 1 means the
representation has consumed the measurement entirely. Construction warns
above `PRECISION_WARN_RATIO = 1e-3`, and the warning is recorded as a step
in provenance rather than only printed, so it survives into a report.

The function returns 0.0 when the question is not answerable — fewer than
two finite points, or a perfectly constant series — rather than guessing.

## What it does not catch

The ratio uses the smallest observed step as a proxy for instrument
resolution. If the data are already quantized coarser than the instrument
resolution (for example a counter exporting fewer digits than it measured),
the proxy is the export precision, not the measurement, and the ratio will
look healthy. Provenance records the source file hash, which is the thing
that lets you go back and check.

## A related, separate effect

Converting phase -> frequency -> phase does not return the original array
bit for bit. A cumulative sum of differences does not telescope exactly in
floating point. Measured accumulation is 0.2 ULP at n=10, 1.0 at n=1e3 and
2.5 at n=1e5 — negligible, but real, and bounded by a test rather than
assumed. The integration constant is separately unrecoverable, so
`to_phase` pins `x[0] = 0` and records that it did.

## Validation status

**Validated.** `tests/test_types.py`:

* `test_precision_ratio_quiet_for_well_conditioned_data` — picosecond steps
  around zero.
* `test_precision_ratio_detects_epoch_quantization` — the same steps riding
  on a 1e6 s offset.
* `test_precision_warning_is_recorded_in_provenance`.
* `test_phase_frequency_roundtrip_is_accurate_to_a_few_ulp` — bound of
  8 ULP, derived from the measured growth above rather than tuned.
